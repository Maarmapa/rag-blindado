"""La traza se mira: registro, alerta y persistencia.

Estos tests existen por una razón concreta: la versión anterior devolvía una
traza perfecta que nadie leía. Un módulo de observabilidad sin tests repite ese
error un nivel más arriba — queda escrito, nadie comprueba que se dispare, y se
descubre el día que hacía falta.

Corren sin credenciales, sin base y sin red: es lo que el job `guards` de CI
puede ejecutar.
"""

from __future__ import annotations

import json
import logging

import pytest

from ragb import observe


def _traza(**cambios) -> dict:
    base = {
        "text": "El plazo de severidad 1 es de 1 hora.",
        "sources": ["sla.md"],
        "stop_reason": "end_turn",
        "model": "claude-opus-5",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "question": "¿cuál es el plazo de severidad 1?",
        "retrieved": 3,
        "used": 3,
        "quarantined": [],
        "contexts": ["[fuente: sla.md]\ntexto"],
    }
    base.update(cambios)
    return base


# --- Alertas: lo que merece atención, y lo que no ---------------------------

def test_una_consulta_normal_no_alerta():
    """El control. Si todo alerta, nada alerta."""
    assert observe.alertas(_traza()) == []


def test_la_cuarentena_alerta_con_su_motivo():
    """Una guarda que se dispara mil veces no puede verse igual que una que no.

    Es el caso que motiva el módulo entero: la inyección se detectó y se
    excluyó correctamente, y aun así nadie se enteraba nunca.
    """
    traza = _traza(
        quarantined=[{"source": "nota.md", "reason": "instruccion_embebida"}]
    )
    (alerta,) = observe.alertas(traza)

    assert alerta["tipo"] == "cuarentena"
    assert alerta["nivel"] == logging.WARNING
    assert alerta["motivos"] == ["instruccion_embebida"]


def test_si_las_guardas_dejan_la_consulta_sin_contexto_es_error_no_aviso():
    """Distinguir la guarda funcionando de la guarda dejando a alguien sin respuesta.

    Excluir un fragmento de cinco es el sistema haciendo su trabajo. Excluir
    los cinco es un usuario recibiendo un "no encuentro" que el corpus sí
    cubría — y eso hay que mirarlo hoy, no en el reporte del mes.
    """
    traza = _traza(
        retrieved=5,
        used=0,
        sources=[],
        quarantined=[{"source": "x.md", "reason": "instruccion_embebida"}] * 5,
    )
    tipos = {a["tipo"]: a for a in observe.alertas(traza)}

    assert tipos["contexto_vacio_por_guardas"]["nivel"] == logging.ERROR
    assert "cuarentena" in tipos


def test_no_recuperar_nada_alerta():
    """Una colección vacía o mal indexada se ve exactamente igual que una
    pregunta fuera del corpus. Las dos merecen que alguien mire."""
    traza = _traza(retrieved=0, used=0, sources=[])
    assert any(a["tipo"] == "sin_recuperacion" for a in observe.alertas(traza))


def test_el_rechazo_del_modelo_alerta():
    traza = _traza(stop_reason="refusal", used=0, retrieved=0, sources=[])
    assert any(a["tipo"] == "rechazo_del_modelo" for a in observe.alertas(traza))


def test_respuesta_con_fragmentos_pero_sin_fuentes_es_error():
    """La contradicción que todo el diseño intenta hacer imposible: hay
    respuesta, se usaron fragmentos, y la cita no respalda nada."""
    traza = _traza(sources=[])
    alerta = next(
        a for a in observe.alertas(traza) if a["tipo"] == "respuesta_sin_fuentes"
    )
    assert alerta["nivel"] == logging.ERROR


# --- Registro: qué se escribe, y sobre todo qué NO -------------------------

def test_por_defecto_el_registro_no_lleva_el_corpus_ni_la_respuesta():
    """El log no es lugar para el contenido de los documentos.

    Si este test empieza a fallar, alguien invirtió el default y el corpus de
    un cliente está viajando a donde vayan los logs.
    """
    reg = observe.registro(_traza())

    assert "respuesta" not in reg
    assert "pregunta" not in reg
    assert reg["largo_respuesta"] > 0
    assert reg["largo_pregunta"] > 0
    # Lo que sí debe estar, porque es lo auditable.
    assert reg["fuentes"] == ["sla.md"]
    assert reg["recuperados"] == 3


def test_el_texto_entra_solo_si_alguien_lo_pide_a_sabiendas():
    reg = observe.registro(_traza(), incluir_texto=True)
    assert reg["pregunta"] == "¿cuál es el plazo de severidad 1?"
    assert reg["respuesta"].startswith("El plazo")


def test_la_variable_de_entorno_activa_el_texto(monkeypatch):
    monkeypatch.setenv("TRACE_INCLUDE_TEXT", "1")
    assert "respuesta" in observe.registro(_traza())


# --- mirar(): el camino completo -------------------------------------------

def test_mirar_emite_las_alertas_al_log(caplog):
    traza = _traza(quarantined=[{"source": "n.md", "reason": "instruccion_embebida"}])
    with caplog.at_level(logging.WARNING, logger="ragb.observe"):
        observe.mirar(traza)

    assert any("cuarentena" in r.getMessage() for r in caplog.records)


def test_mirar_persiste_una_linea_json_por_consulta(tmp_path, monkeypatch):
    destino = tmp_path / "trazas" / "runtime.jsonl"
    monkeypatch.setenv("TRACE_LOG_PATH", str(destino))

    observe.mirar(_traza())
    observe.mirar(_traza(retrieved=0, used=0, sources=[]))

    lineas = destino.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 2
    primero = json.loads(lineas[0])
    assert primero["evento"] == "consulta"
    assert primero["alertas"] == []
    segundo = json.loads(lineas[1])
    assert segundo["alertas"][0]["tipo"] == "sin_recuperacion"


def test_sin_ruta_configurada_no_escribe_nada(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACE_LOG_PATH", raising=False)
    observe.mirar(_traza())
    assert list(tmp_path.iterdir()) == []


def test_la_observabilidad_jamas_tumba_la_consulta(monkeypatch, caplog):
    """La regla dura del módulo.

    Perder la traza cuesta auditoría; perder la respuesta cuesta el servicio.
    Se rompe el registro a propósito y `mirar` tiene que tragárselo.
    """
    def explota(*_a, **_k):
        raise RuntimeError("disco lleno")

    monkeypatch.setattr(observe, "registro", explota)
    with caplog.at_level(logging.ERROR, logger="ragb.observe"):
        assert observe.mirar(_traza()) == {}
    assert "no se pudo registrar" in caplog.text


def test_una_ruta_imposible_tampoco_tumba_la_consulta(monkeypatch):
    monkeypatch.setenv("TRACE_LOG_PATH", "/proc/no/se/puede/escribir.jsonl")
    assert observe.mirar(_traza()) == {}


# --- El cableado: que el pipeline REALMENTE la mire ------------------------

def test_query_llama_a_mirar(monkeypatch):
    """El test que impide repetir el bug original.

    Da igual lo bueno que sea `observe` si `pipeline.query` no lo llama: la
    traza volvería a morir en el diccionario de retorno, que es exactamente el
    defecto que esto vino a arreglar.
    """
    from ragb import pipeline
    from ragb.guards import Permissions

    vistas: list[dict] = []
    monkeypatch.setattr(pipeline.embeddings, "embed_one", lambda _q: [0.0])
    monkeypatch.setattr(
        pipeline.store,
        "search",
        lambda *_a, **_k: [{"source": "sla.md", "chunk_index": 0, "text": "1 hora"}],
    )
    monkeypatch.setattr(
        pipeline.generate,
        "answer",
        lambda *_a, **_k: {
            "text": "1 hora",
            "sources": ["sla.md"],
            "stop_reason": "end_turn",
            "model": "modelo-de-prueba",
        },
    )
    monkeypatch.setattr(pipeline.observe, "mirar", lambda t: vistas.append(t) or {})

    salida = pipeline.query(Permissions(tenant="t1"), "¿plazo sev 1?")

    assert len(vistas) == 1, "la traza volvió a morir sin que nadie la mirara"
    assert vistas[0]["used"] == 1
    # Y la traza sigue devolviéndose al llamador, como antes.
    assert salida["used"] == 1


@pytest.mark.parametrize("campo", ["retrieved", "used", "quarantined"])
def test_la_traza_que_recibe_mirar_trae_lo_que_alertas_necesita(campo):
    """Contrato entre las dos mitades: si `query` deja de poner uno de estos
    campos, las alertas se apagan en silencio y nadie se entera."""
    assert campo in _traza()
