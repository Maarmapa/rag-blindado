"""Mirar la traza: registro, persistencia y alerta de cada consulta.

`pipeline.query` ya devolvía una traza completa —qué se recuperó, qué se usó,
qué quedó en cuarentena y por qué—, y esa traza **moría en el diccionario de
retorno**. Sin registro, sin persistencia y sin alerta, una guarda que se
dispara mil veces se ve exactamente igual que una que no se disparó nunca.

El propio repositorio ya había entendido el problema, pero solo del lado de
las evaluaciones: *"sin saber cuál afirmación falló, un umbral que no se
alcanza es indistinguible de un juez que se equivoca"* (`evals/run.py`). Esto
es el mismo razonamiento aplicado a runtime.

Tres decisiones que conviene no deshacer sin pensarlo:

1. **Nunca revienta.** La observabilidad es lo primero que se cae y lo último
   que puede tumbar una consulta. Todo el trabajo va dentro de un `try`, y un
   fallo acá se queda en un log de nivel ERROR: el usuario recibe su respuesta
   igual. Perder la traza cuesta auditoría; perder la respuesta cuesta el
   servicio.

2. **Por defecto no registra el contenido de los documentos.** La traza lleva
   fragmentos del corpus y el texto de la respuesta, o sea exactamente lo que
   no debería terminar en un archivo de log compartido. Se registran conteos,
   nombres de fuente y motivos de cuarentena; el texto solo si alguien activa
   `TRACE_INCLUDE_TEXT` a sabiendas.

3. **Sin dependencias.** `logging` y un JSONL opcional. Un pipeline de
   observabilidad que exige infraestructura nueva no se instala, y una
   observabilidad que no se instala no existe.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time

logger = logging.getLogger("ragb.observe")

_VERDADERO = {"1", "true", "yes", "on", "si", "sí"}


def _flag(nombre: str) -> bool:
    return os.getenv(nombre, "").strip().lower() in _VERDADERO


def _ruta_jsonl() -> str:
    return os.getenv("TRACE_LOG_PATH", "").strip()


def alertas(traza: dict) -> list[dict]:
    """Lo que merece despertar a alguien, separado de lo que es solo ruido.

    Cada alerta lleva su nivel de `logging` porque la diferencia importa: que
    una inyección quede en cuarentena es la guarda funcionando (WARNING, hay
    que mirarlo); que *todo* lo recuperado quede en cuarentena es una consulta
    que se quedó sin contexto por culpa de las guardas (ERROR, alguien recibió
    un "no encuentro" que no corresponde al corpus).
    """
    fuera: list[dict] = []
    recuperados = traza.get("retrieved", 0)
    usados = traza.get("used", 0)
    cuarentena = traza.get("quarantined") or []
    motivo_corte = traza.get("stop_reason")

    if cuarentena:
        fuera.append(
            {
                "tipo": "cuarentena",
                "nivel": logging.WARNING,
                "detalle": f"{len(cuarentena)} fragmento(s) excluidos por las guardas",
                "motivos": sorted({q.get("reason", "?") for q in cuarentena}),
            }
        )

    if recuperados > 0 and usados == 0:
        fuera.append(
            {
                "tipo": "contexto_vacio_por_guardas",
                "nivel": logging.ERROR,
                "detalle": (
                    f"se recuperaron {recuperados} fragmentos y las guardas "
                    "descartaron todos: la respuesta salió sin contexto"
                ),
            }
        )

    if recuperados == 0:
        fuera.append(
            {
                "tipo": "sin_recuperacion",
                "nivel": logging.WARNING,
                "detalle": (
                    "la búsqueda no devolvió nada: o el corpus no cubre la "
                    "pregunta, o la colección está vacía o mal indexada"
                ),
            }
        )

    if motivo_corte == "refusal":
        fuera.append(
            {
                "tipo": "rechazo_del_modelo",
                "nivel": logging.WARNING,
                "detalle": "los filtros de seguridad rechazaron la consulta",
            }
        )

    # Contradicción interna: hay respuesta pero no hay de dónde salió. Si esto
    # aparece, el anclaje se rompió en algún punto y la cita ya no respalda
    # nada — es el escenario que todo el diseño intenta hacer imposible.
    if usados > 0 and not traza.get("sources"):
        fuera.append(
            {
                "tipo": "respuesta_sin_fuentes",
                "nivel": logging.ERROR,
                "detalle": (
                    f"se usaron {usados} fragmentos pero la respuesta no "
                    "declara ninguna fuente"
                ),
            }
        )

    return fuera


def registro(traza: dict, *, incluir_texto: bool | None = None) -> dict:
    """Arma el registro de auditoría a partir de la traza de `query`.

    Es deliberadamente más pobre que la traza: conteos, fuentes y motivos.
    El contenido de los documentos y el texto de la respuesta solo entran con
    `TRACE_INCLUDE_TEXT` activado (o `incluir_texto=True`, para pruebas).
    """
    if incluir_texto is None:
        incluir_texto = _flag("TRACE_INCLUDE_TEXT")

    cuarentena = traza.get("quarantined") or []
    reg = {
        "ts": time.time(),
        "evento": "consulta",
        "recuperados": traza.get("retrieved", 0),
        "usados": traza.get("used", 0),
        "en_cuarentena": len(cuarentena),
        "motivos_cuarentena": sorted({q.get("reason", "?") for q in cuarentena}),
        "fuentes_cuarentena": sorted({q.get("source", "?") for q in cuarentena}),
        "fuentes": traza.get("sources") or [],
        "stop_reason": traza.get("stop_reason"),
        "modelo": traza.get("model"),
        "uso_tokens": traza.get("usage"),
    }

    if incluir_texto:
        reg["pregunta"] = traza.get("question")
        reg["respuesta"] = traza.get("text")
    else:
        # Longitudes en vez de contenido: sirven para detectar una respuesta
        # vacía o una desproporcionada sin volcar el corpus a un log.
        reg["largo_pregunta"] = len(traza.get("question") or "")
        reg["largo_respuesta"] = len(traza.get("text") or "")

    return reg


def _persistir(reg: dict, ruta: str) -> None:
    destino = pathlib.Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(reg, ensure_ascii=False, default=str) + "\n")


def mirar(traza: dict) -> dict:
    """Registra, alerta y (si está configurado) persiste una traza de consulta.

    Devuelve el registro emitido, para que un llamador pueda encadenarlo con su
    propia telemetría. Nunca lanza: ver el módulo.
    """
    try:
        reg = registro(traza)
        disparadas = alertas(traza)
        reg["alertas"] = [
            {"tipo": a["tipo"], "detalle": a["detalle"]} for a in disparadas
        ]

        for a in disparadas:
            logger.log(a["nivel"], "[%s] %s", a["tipo"], a["detalle"])

        # Una línea JSON por consulta: legible por una persona con `grep` y por
        # un colector de logs sin parser propio.
        logger.info("consulta %s", json.dumps(reg, ensure_ascii=False, default=str))

        ruta = _ruta_jsonl()
        if ruta:
            _persistir(reg, ruta)

        return reg
    except Exception:  # noqa: BLE001 — ver el módulo: jamás tumba la consulta
        logger.exception("no se pudo registrar la traza de la consulta")
        return {}
