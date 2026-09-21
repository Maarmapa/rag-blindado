"""El juez SLM sin torch: umbral, orden, la config del entrenamiento, y que
sin modelo LANZA.

El scoring real (`_score_pairs`) se reemplaza por un doble. Lo que se prueba
es el contrato con `graph_flow`: un bool por fragmento en el mismo orden, el
umbral leído del entorno en cada llamada, y que la ausencia de modelo —o una
carpeta que no es un juez entrenado, o fragmentos de un tamaño que el juez
nunca vio— sea una excepción, que el grafo convierte en `degraded=True`, y
nunca un veredicto inventado.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from ragb import grade_slm, graph_flow
from slm import train_grader

CHUNKS = [
    {"text": "El plazo de severidad 1 es de 1 hora.", "source": "sla.md"},
    {"text": "La cafetería abre a las 9.", "source": "avisos.md"},
    {"text": "Severidad 2: plazo de 4 horas.", "source": "sla.md"},
]

GRADER_CONFIG = {
    "base_model": "intfloat/multilingual-e5-small",
    "pair_format": "plain",
    "max_length": 256,
    "chunk_chars": 320,
    "default_threshold": 0.5,
}


def carpeta_de_juez(tmp_path, config: dict | None = GRADER_CONFIG):
    """Una carpeta que parece un modelo guardado por train_grader.py."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    if config is not None:
        (tmp_path / grade_slm.CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


@pytest.fixture
def modelo_falso(tmp_path, monkeypatch):
    """La carpeta de arriba más un scoring determinista."""
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path)))
    monkeypatch.delenv(grade_slm.ENV_THRESHOLD, raising=False)
    llamadas: list[tuple] = []

    def doble(path, question, texts):
        llamadas.append((path, question, list(texts)))
        return [0.9, 0.1, 0.5][: len(texts)]

    monkeypatch.setattr(grade_slm, "_score_pairs", doble)
    return llamadas


def test_umbral_por_defecto_y_orden(modelo_falso):
    assert grade_slm.grade("¿plazo de severidad 1?", CHUNKS) == [True, False, True]
    ((path, question, texts),) = modelo_falso
    assert question == "¿plazo de severidad 1?"
    assert texts == [c["text"] for c in CHUNKS]  # mismo orden, solo el texto


def test_umbral_desde_el_entorno_se_lee_en_cada_llamada(modelo_falso, monkeypatch):
    monkeypatch.setenv(grade_slm.ENV_THRESHOLD, "0.6")
    assert grade_slm.grade("q", CHUNKS) == [True, False, False]
    monkeypatch.setenv(grade_slm.ENV_THRESHOLD, "0.05")
    assert grade_slm.grade("q", CHUNKS) == [True, True, True]
    # Igual al umbral cuenta como relevante (>=), igual que en las métricas.
    monkeypatch.setenv(grade_slm.ENV_THRESHOLD, "0.5")
    assert grade_slm.grade("q", CHUNKS) == [True, False, True]


def test_sin_variable_manda_el_umbral_del_entrenamiento(tmp_path, monkeypatch):
    """`default_threshold` de grader_config.json no es decorativo: si el
    entorno no dice nada, es el umbral que se usa."""
    cfg = {**GRADER_CONFIG, "default_threshold": 0.6}
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path, cfg)))
    monkeypatch.delenv(grade_slm.ENV_THRESHOLD, raising=False)
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9, 0.1, 0.5])
    assert grade_slm.grade("q", CHUNKS) == [True, False, False]


def test_los_dos_defaults_de_umbral_coinciden():
    # El entrenamiento mide con train_grader.DEFAULT_THRESHOLD y lo guarda; el
    # juez cae a grade_slm.DEFAULT_THRESHOLD si el archivo no lo trae. Si se
    # desalinean, lo medido y lo desplegado dejan de ser lo mismo.
    assert train_grader.DEFAULT_THRESHOLD == grade_slm.DEFAULT_THRESHOLD
    assert train_grader.PAIR_FORMAT == grade_slm.PAIR_FORMAT


def test_score_expone_los_puntajes_crudos(modelo_falso):
    assert grade_slm.score("q", CHUNKS) == [0.9, 0.1, 0.5]


@pytest.mark.parametrize("valor", ["1.5", "-0.1", "alto", "0", "0.0", "1", "1.0"])
def test_umbral_invalido_lanza(modelo_falso, monkeypatch, valor):
    # 0 y 1 también: 0 aprueba todo (un no-op disfrazado de juez) y 1 rechaza todo.
    monkeypatch.setenv(grade_slm.ENV_THRESHOLD, valor)
    with pytest.raises(ValueError):
        grade_slm.grade("q", CHUNKS)


def test_sin_fragmentos_no_carga_ni_llama(modelo_falso):
    assert grade_slm.grade("q", []) == []
    assert modelo_falso == []


def test_sin_ruta_configurada_lanza(monkeypatch):
    monkeypatch.delenv(grade_slm.ENV_PATH, raising=False)
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda *a: pytest.fail("no debía llegar al scoring"))
    with pytest.raises(RuntimeError, match="SLM_GRADER_PATH no configurada"):
        grade_slm.grade("q", CHUNKS)


def test_ruta_sin_modelo_lanza(tmp_path, monkeypatch):
    monkeypatch.setenv(grade_slm.ENV_PATH, str(tmp_path))  # existe, pero vacía
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda *a: pytest.fail("no debía llegar al scoring"))
    with pytest.raises(RuntimeError, match="config.json"):
        grade_slm.grade("q", CHUNKS)


def test_encoder_base_sin_grader_config_no_es_un_juez(tmp_path, monkeypatch):
    """La carpeta del modelo de embeddings del repo tiene config.json pero no
    cabezal entrenado. Sin grader_config.json se rechaza ANTES de tocar
    torch: un cabezal inventado da ~0.5 en todo y no lanza solo."""
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path, config=None)))
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda *a: pytest.fail("no debía llegar al scoring"))
    with pytest.raises(RuntimeError, match="grader_config.json"):
        grade_slm.grade("q", CHUNKS)


@pytest.mark.parametrize(
    ("config", "mensaje"),
    [
        ({**GRADER_CONFIG, "pair_format": "e5-prefixed"}, "pair_format"),
        ({k: v for k, v in GRADER_CONFIG.items() if k != "max_length"}, "max_length"),
        ({k: v for k, v in GRADER_CONFIG.items() if k != "chunk_chars"}, "chunk_chars"),
    ],
)
def test_grader_config_incompleto_o_de_otro_formato_lanza(tmp_path, monkeypatch, config, mensaje):
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path, config)))
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda *a: pytest.fail("no debía llegar al scoring"))
    with pytest.raises(RuntimeError, match=mensaje):
        grade_slm.grade("q", CHUNKS)


def test_fragmento_mas_largo_que_el_entrenamiento_lanza(modelo_falso):
    """Producción fragmenta a 900 por defecto; el juez se entrenó a 320. No
    califica lo que nunca vio: lanza con los números y el grafo degrada."""
    largo = [{"text": "x" * 481, "source": "a.md"}]  # 1.5 x 320 = 480
    with pytest.raises(RuntimeError, match="481 caracteres.*CHUNK_CHARS=320"):
        grade_slm.grade("q", largo)
    assert modelo_falso == []  # se rechazó antes de puntuar
    assert grade_slm.grade("q", [{"text": "x" * 480, "source": "a.md"}]) == [True]


def test_cantidad_de_puntajes_distinta_lanza(modelo_falso, monkeypatch):
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9])
    with pytest.raises(RuntimeError, match="1 puntajes para 3"):
        grade_slm.grade("q", CHUNKS)


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="con torch instalado _load intenta cargar")
def test_sin_torch_load_lanza_con_instruccion(tmp_path, monkeypatch):
    """La rama que corre en el job `guards` de CI, sin doble: llega a `_load`
    y el error dice qué instalar en vez de un ImportError de diez líneas."""
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path)))
    grade_slm._cache.clear()
    with pytest.raises(RuntimeError, match="falta torch.*slm/REQUIREMENTS.txt"):
        grade_slm.grade("q", CHUNKS)


def test_enchufado_en_graph_flow_sin_modelo_degrada_sin_mentir(monkeypatch):
    """El contrato de punta a punta: el juez lanza, el grafo no se cae, los
    fragmentos pasan sin calificar y la traza dice por qué."""
    monkeypatch.delenv(grade_slm.ENV_PATH, raising=False)
    graph = graph_flow.build_graph(
        retrieve=lambda q: [CHUNKS[0]],
        grade=grade_slm.grade,
        rewrite=lambda q: q,
        generate=lambda q, c: {
            "text": "1 hora [sla.md]",
            "sources": ["sla.md"],
            "stop_reason": "end_turn",
            "model": "fake",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    out = graph_flow.run(graph, "¿plazo de severidad 1?")
    assert out["degraded"] is True
    assert out["rounds"][0]["graded"] is False
    assert out["rounds"][0]["error"].startswith("RuntimeError: SLM_GRADER_PATH no configurada")
    assert out["used"] == 1  # pasó sin calificar, no se descartó


def test_puntaje_no_numerico_o_fuera_de_rango_lanza(modelo_falso, monkeypatch):
    """`nan >= 0.5` es False: sin esta guarda, pesos corruptos se verían como
    "nada relevante", graph_flow marcaría graded=True y el generador diría
    NO_ENCONTRADO sin `degraded`. Un puntaje que no es probabilidad es un
    modelo roto, y eso es excepción, no veredicto."""
    for malo in ([float("nan"), 0.1, 0.9], [0.9, float("inf"), 0.1], [1.5, 0.1, 0.1], [-0.01, 0.1, 0.1]):
        monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t, malo=malo: malo)
        with pytest.raises(RuntimeError, match="puntaje inválido"):
            grade_slm.grade("q", CHUNKS)
        with pytest.raises(RuntimeError, match="puntaje inválido"):
            grade_slm.score("q", CHUNKS)


@pytest.mark.parametrize("clave", ["max_length", "chunk_chars"])
@pytest.mark.parametrize("valor", [0, -1, "abc", 10**9, True, 2.5])
def test_grader_config_con_tipo_o_rango_invalido_lanza(tmp_path, monkeypatch, clave, valor):
    """La guarda de tamaño se calcula desde chunk_chars: con 10**9 un fragmento
    de 50.000 caracteres pasaba. El archivo lo escribe train_grader, pero la
    guarda existe para negarse a calificar fuera de régimen y un valor absurdo
    la anulaba sin aviso."""
    cfg = {**GRADER_CONFIG, clave: valor}
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path, cfg)))
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9] * len(t))
    with pytest.raises(RuntimeError, match=clave):
        grade_slm.grade("q", [{"text": "x" * 50_000, "source": "a.md"}])
    with pytest.raises(RuntimeError, match=clave):
        grade_slm.grade("q", CHUNKS)


def test_chunk_chars_hasta_el_maximo_se_acepta(tmp_path, monkeypatch):
    cfg = {**GRADER_CONFIG, "chunk_chars": grade_slm.MAX_CHUNK_CHARS}
    monkeypatch.setenv(grade_slm.ENV_PATH, str(carpeta_de_juez(tmp_path, cfg)))
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9] * len(t))
    assert grade_slm.grade("q", CHUNKS) == [True, True, True]


def test_ruta_relativa_se_resuelve_contra_la_raiz_del_repo(tmp_path, monkeypatch):
    """Un servicio arrancado desde otra carpeta no debe perder al juez porque
    SLM_GRADER_PATH era relativa al cwd de quien la escribió."""
    (tmp_path / "out" / "juez").mkdir(parents=True)
    juez = carpeta_de_juez(tmp_path / "out" / "juez")
    monkeypatch.setattr(grade_slm, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path / "out")  # cwd distinto de la raíz a propósito
    monkeypatch.setenv(grade_slm.ENV_PATH, "out/juez")
    monkeypatch.delenv(grade_slm.ENV_THRESHOLD, raising=False)
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9, 0.1, 0.5])
    assert grade_slm.grade("q", CHUNKS) == [True, False, True]
    assert grade_slm.describe()["path"] == str(juez)


def test_describe_expone_ruta_config_y_umbral(modelo_falso, monkeypatch):
    info = grade_slm.describe()
    assert info["config"] == GRADER_CONFIG
    assert info["threshold"] == 0.5
    monkeypatch.setenv(grade_slm.ENV_THRESHOLD, "0.7")
    assert grade_slm.describe()["threshold"] == 0.7
    assert modelo_falso == []  # describe no puntúa nada


def test_score_raw_salta_la_guarda_de_largo_pero_grade_no(modelo_falso, monkeypatch):
    """`raw=True` existe solo para que slm/smoke.py mida cuánto se equivocaría
    el juez a 900 caracteres. `grade`, que es lo que ve el grafo, sigue lanzando."""
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9] * len(t))
    largo = [{"text": "x" * 900, "source": "a.md"}]
    assert grade_slm.score("q", largo, raw=True) == [0.9]
    with pytest.raises(RuntimeError, match="900 caracteres"):
        grade_slm.score("q", largo)
    with pytest.raises(RuntimeError, match="900 caracteres"):
        grade_slm.grade("q", largo)


def test_con_chunk_chars_de_produccion_el_grafo_degrada_en_cada_ronda(modelo_falso, monkeypatch):
    """El límite que el README tiene que decir en voz alta: con el CHUNK_CHARS
    por defecto del repo (900) el juez entrenado a 320 rechaza TODOS los
    fragmentos, el grafo degrada en cada ronda y el juez SLM es un no-op
    permanente. No es un bug del juez —es fail-closed— pero tiene que ser
    visible, no descubierto. Indexar con CHUNK_CHARS=320 lo evita."""
    from ragb.config import settings

    assert settings.chunk_chars == 900  # si esto cambia, el README y este test se revisan
    monkeypatch.setattr(grade_slm, "_score_pairs", lambda p, q, t: [0.9] * len(t))
    fragmento_de_produccion = [{"text": "y" * 845, "source": "politica.md"}]
    graph = graph_flow.build_graph(
        retrieve=lambda q: fragmento_de_produccion,
        grade=grade_slm.grade,
        rewrite=lambda q: q + " (reformulada)",
        generate=lambda q, c: {
            "text": "respuesta [politica.md]",
            "sources": ["politica.md"],
            "stop_reason": "end_turn",
            "model": "fake",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    out = graph_flow.run(graph, "¿pregunta?")
    assert out["degraded"] is True
    assert all(r["graded"] is False for r in out["rounds"])
    assert all("845 caracteres" in r["error"] and "CHUNK_CHARS=320" in r["error"] for r in out["rounds"])
    assert modelo_falso == []  # nunca llegó a puntuar: la guarda es previa
    # Con fragmentos del tamaño del entrenamiento, el mismo grafo califica.
    corto = [{"text": "y" * 320, "source": "politica.md"}]
    graph_ok = graph_flow.build_graph(
        retrieve=lambda q: corto,
        grade=grade_slm.grade,
        rewrite=lambda q: q,
        generate=lambda q, c: {
            "text": "respuesta [politica.md]",
            "sources": ["politica.md"],
            "stop_reason": "end_turn",
            "model": "fake",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    out_ok = graph_flow.run(graph_ok, "¿pregunta?")
    assert out_ok["degraded"] is False
    assert out_ok["rounds"][0]["graded"] is True
