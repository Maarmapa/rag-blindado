"""Selección del juez de relevancia para el ciclo CRAG de `graph_flow`.

Vive en su propio módulo, y no en `graph_flow.py`, por una razón de proceso:
`graph_flow.py` es un archivo que este stream no puede tocar (lo integra el
orquestador). Acá se ofrece la elección del juez sin modificarlo; para
cablearlo basta con reemplazar la operación `grade` que `default_ops` ya
devuelve:

    ops = graph_flow.default_ops(perms)
    ops["grade"] = graders.make_grader()          # o make_grader("jev")
    graph = graph_flow.build_graph(**ops)

Decisiones:

- **El nombre desconocido es un error, no un default.** Caer a `llm` en
  silencio ante un typo en `RAGB_GRADER` escondería en producción que el juez
  que se creía configurado no es el que corre.
- **Jev se construye al elegirlo, no en la primera pregunta.** `JevGrader`
  falla cerrado sin `TYPESAFE_API_KEY`; que falle al arrancar el proceso, que
  es donde alguien lo ve, y no a las tres de la mañana en una consulta.
- **`llm` importa `grade.grade_chunks` perezosamente**: ese módulo existe en
  la rama CRAG; si no está, `graph_flow.node_grade` atrapa el error y degrada
  a la vista (`degraded=True`), que es el contrato de todo juez del grafo.
"""

from __future__ import annotations

import os
from typing import Callable

Chunk = dict
Grader = Callable[[str, list[Chunk]], list[bool]]

GRADERS = ("llm", "jev")


def make_grader(name: str | None = None) -> Grader:
    """Elige el juez de relevancia. `name` o `RAGB_GRADER`; por defecto `llm`.

    - `llm`: el LLM generativo de `grade.grade_chunks` (rama CRAG).
    - `jev`: `grade_jev.JevGrader`, el modelo no generativo de TypeSafe AI,
      hablado por `urllib` (sin SDK ni dependencias nuevas).
    """
    elegido = (name or os.getenv("RAGB_GRADER", "llm")).strip().lower()
    if elegido == "jev":
        from .grade_jev import JevGrader

        return JevGrader()
    if elegido == "llm":

        def grade(question: str, chunks: list[Chunk]) -> list[bool]:
            from .grade import grade_chunks  # existe en la rama CRAG; si no, degrada

            return grade_chunks(question, chunks)

        return grade
    raise ValueError(f"RAGB_GRADER desconocido: {elegido!r}; opciones: {', '.join(GRADERS)}")
