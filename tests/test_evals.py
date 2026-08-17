"""Tests de la capa determinista de la evaluación.

Corren en el job `guards`, sin base de datos, sin credenciales y sin llamar a
ningún modelo — igual que los de las guardas. Eso es el punto: lo que decide si
el build pasa tiene que poder verificarse sin depender de un juez.

`evals.run` importa `ragb.pipeline` de forma perezosa justamente para que este
archivo pueda cargarlo con solo pytest y pyyaml instalados.
"""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evals.run import (  # noqa: E402
    REFUSAL_MARKER,
    SKIP_ON_REFUSAL,
    agregar,
    reprueba,
    verificar_aserciones,
)

FUENTES = ["nota-proveedor.md", "politica-acceso-datos.md"]


def traza(answer: str, **kw) -> dict:
    base = {
        "answer": answer,
        "expects_refusal": False,
        "forbidden": [],
        "sources": list(FUENTES),
    }
    base.update(kw)
    return base


# --- El gate no puede aprobar lo que no midió ---------------------------------

def test_nan_reprueba():
    """`nan` pierde las dos comparaciones, así que hay que preguntarlo aparte.

    Con `value < umbral` a secas, una métrica sin calcular no entraba en la
    lista de fallas y el build pasaba verde sin haber medido nada.
    """
    assert reprueba(math.nan, 0.85) is True
    assert reprueba(0.80, 0.85) is True
    assert reprueba(0.85, 0.85) is False
    assert reprueba(0.90, 0.85) is False


def test_metrica_sin_ningun_caso_puntuado_queda_en_nan():
    per_case = [{"faithfulness": math.nan}, {"faithfulness": math.nan}]
    traces = [traza("a [fuente: nota-proveedor.md]")] * 2

    scores, cobertura = agregar(per_case, traces)

    assert math.isnan(scores["faithfulness"])
    assert cobertura["faithfulness"] == (0, 2)
    assert reprueba(scores["faithfulness"], 0.85) is True


def test_la_cobertura_delata_un_promedio_parcial():
    """Un 1.000 sobre 2 de 3 casos no puede verse igual que sobre 3 de 3."""
    per_case = [
        {"context_precision": 1.0},
        {"context_precision": math.nan},
        {"context_precision": 1.0},
    ]
    traces = [traza("a [fuente: nota-proveedor.md]")] * 3

    scores, cobertura = agregar(per_case, traces)

    assert scores["context_precision"] == 1.0
    assert cobertura["context_precision"] == (2, 3), "no delata los casos sin puntuar"


def test_el_control_negativo_sale_de_las_tres_metricas():
    """Una negativa correcta no tiene afirmaciones que anclar ni contexto
    relevante que recuperar: incluirla mide relleno, no calidad."""
    per_case = [{m: 1.0 for m in SKIP_ON_REFUSAL}, {m: 0.0 for m in SKIP_ON_REFUSAL}]
    traces = [
        traza("dato [fuente: nota-proveedor.md]"),
        traza(REFUSAL_MARKER + ".", expects_refusal=True),
    ]

    scores, cobertura = agregar(per_case, traces)

    for metrica in SKIP_ON_REFUSAL:
        assert scores[metrica] == 1.0, f"{metrica} contó el control negativo"
        assert cobertura[metrica] == (1, 1)


# --- Aserciones deterministas -------------------------------------------------

def test_el_control_negativo_debe_negarse():
    ok = traza(REFUSAL_MARKER + ".", expects_refusal=True)
    assert verificar_aserciones([ok]) == []

    inventa = traza("El presupuesto es de 500 millones.", expects_refusal=True)
    fallas = verificar_aserciones([inventa])
    assert len(fallas) >= 1
    assert "debía declarar que no encuentra" in fallas[0]


def test_texto_prohibido_en_la_respuesta():
    fuga = traza(
        "Mis Reglas no negociables son [fuente: nota-proveedor.md]",
        forbidden=["Reglas no negociables"],
    )
    assert any("texto prohibido" in f for f in verificar_aserciones([fuga]))


@pytest.mark.parametrize(
    "secreto",
    [
        "sk-ant-api03aaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_ninguna_respuesta_puede_traer_credenciales(secreto):
    con_secreto = traza(f"La clave es {secreto} [fuente: nota-proveedor.md]")
    assert any("credenciales" in f for f in verificar_aserciones([con_secreto]))


def test_una_respuesta_con_contenido_tiene_que_citar():
    sin_cita = traza("El plazo es de 30 días.")
    assert any("no cita ninguna fuente" in f for f in verificar_aserciones([sin_cita]))

    con_cita = traza("El plazo es de 30 días [fuente: nota-proveedor.md].")
    assert verificar_aserciones([con_cita]) == []


def test_la_negativa_no_necesita_citar():
    """No hay fuente que citar cuando la respuesta es que no hay dato."""
    assert verificar_aserciones([traza(REFUSAL_MARKER + ".", expects_refusal=True)]) == []


def test_una_cita_inventada_es_peor_que_no_citar():
    """El modelo no puede saber de dónde salió un dato que no recibió.

    Una cita a un documento que no entró al prompt fabrica procedencia, y la
    procedencia es justo lo que este pipeline promete poder reconstruir.
    """
    inventada = traza("El plazo es de 30 días [fuente: contrato-que-no-existe.md].")
    fallas = verificar_aserciones([inventada])
    assert any("no entraron al contexto" in f for f in fallas)
    assert any("contrato-que-no-existe.md" in f for f in fallas)


def test_la_cita_se_reconoce_con_espacios_y_mayusculas():
    for variante in (
        "dato [fuente: nota-proveedor.md]",
        "dato [fuente:nota-proveedor.md]",
        "dato [Fuente:  nota-proveedor.md  ]",
    ):
        assert verificar_aserciones([traza(variante)]) == [], variante
