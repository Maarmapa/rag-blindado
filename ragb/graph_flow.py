"""Flujo CRAG como grafo de estados con LangGraph.

Por qué existe este módulo: el pipeline lineal (`ragb.pipeline.query`) recupera
una vez y responde. Cuando lo recuperado no sirve, responde mal con confianza.
El ciclo correctivo (Corrective RAG) califica lo recuperado, reescribe la
pregunta si nada es relevante, y reintenta un número acotado de veces.

Decisiones, escritas para que un refactor no las deshaga en silencio:

1. **No hay caída a búsqueda web.** El paper de CRAG la propone; acá se
   descarta a propósito: rompería el anclaje al corpus y reabriría LLM01
   (inyección desde contenido externo) y LLM09. Si no se encuentra, se dice
   que no se encontró.
2. **Si el juez se cae, se degrada sin mentir**: los fragmentos pasan sin
   calificar y la traza lo marca (`degraded=True`). Perder al juez cuesta
   precisión, nunca disponibilidad. Lo mismo que hace `observe.mirar`.
3. **Sin fragmentos relevantes no se llama al modelo**: la respuesta es un
   "no encontrado" determinista. Fail-closed, igual que `generate.answer`
   cuando no hay contexto.
4. **Las cuatro operaciones se inyectan** (`retrieve`, `grade`, `rewrite`,
   `generate`). Así el grafo se prueba entero sin credenciales, sin base y sin
   red, que es lo que el job `guards` de CI puede ejecutar. El cableado por
   defecto (`default_ops`) usa los módulos reales.
5. **La sanitización va en el nodo de recuperación**, antes del juez: un
   fragmento con inyección no llega ni a calificarse (LLM01).

LangGraph aporta lo que un bucle a mano no da gratis: el estado es explícito
y tipado, cada nodo es una función pura sobre ese estado, la arista
condicional queda declarada en un solo lugar, y el grafo compilado se puede
dibujar, interrumpir y reanudar con un checkpointer. Aquí se usa sin
checkpointer para mantener el módulo sin dependencias de infraestructura.
"""

from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from .guards import sanitize_chunks

# --- Tipos ------------------------------------------------------------------

Chunk = dict  # {"text": str, "source": str, ...} como los devuelve store.search
Retriever = Callable[[str], list[Chunk]]
Grader = Callable[[str, list[Chunk]], list[bool]]
Rewriter = Callable[[str], str]
Generator = Callable[[str, list[Chunk]], dict]

NO_ENCONTRADO = (
    "No encontré en el corpus información suficiente para responder con "
    "fundamento. No invento: si el dato existe, no está en los documentos "
    "indexados o la pregunta necesita otra formulación."
)


class Round(TypedDict):
    question: str
    retrieved: int
    accepted: int
    quarantined: int
    relevant: int
    graded: bool
    error: str | None


class RagState(TypedDict, total=False):
    question: str  # pregunta vigente (puede haber sido reescrita)
    original_question: str  # la que hizo la persona; se responde a ESTA
    chunks: list[Chunk]  # recuperados y saneados en la ronda vigente
    relevant: list[Chunk]  # los que el juez aprobó
    rounds: list[Round]  # traza por ronda, para auditoría
    max_rounds: int
    degraded: bool  # el juez falló en alguna ronda
    answer: dict


# --- Nodos ------------------------------------------------------------------


def _make_nodes(retrieve: Retriever, grade: Grader, rewrite: Rewriter, generate: Generator):
    def node_retrieve(state: RagState) -> RagState:
        raw = retrieve(state["question"])
        accepted, quarantined = sanitize_chunks(raw)
        rounds = list(state.get("rounds", []))
        rounds.append(
            {
                "question": state["question"],
                "retrieved": len(raw),
                "accepted": len(accepted),
                "quarantined": len(quarantined),
                "relevant": 0,
                "graded": False,
                "error": None,
            }
        )
        return {"chunks": accepted, "rounds": rounds}

    def node_grade(state: RagState) -> RagState:
        chunks = state.get("chunks", [])
        rounds = list(state.get("rounds", []))
        current = dict(rounds[-1])
        degraded = state.get("degraded", False)
        if not chunks:
            relevant: list[Chunk] = []
            current["graded"] = True
        else:
            try:
                verdicts = grade(state["question"], chunks)
                if len(verdicts) != len(chunks):
                    raise ValueError(f"el juez devolvió {len(verdicts)} veredictos para {len(chunks)} fragmentos")
                relevant = [c for c, ok in zip(chunks, verdicts) if ok]
                current["graded"] = True
            except Exception as exc:  # noqa: BLE001 — degradar, nunca tumbar
                relevant = list(chunks)
                current["graded"] = False
                current["error"] = f"{type(exc).__name__}: {exc}"
                degraded = True
        current["relevant"] = len(relevant)
        rounds[-1] = current
        return {"relevant": relevant, "rounds": rounds, "degraded": degraded}

    def node_rewrite(state: RagState) -> RagState:
        nueva = rewrite(state["question"])
        if not isinstance(nueva, str) or not nueva.strip():
            nueva = state["question"]
        return {"question": nueva.strip()}

    def node_generate(state: RagState) -> RagState:
        relevant = state.get("relevant", [])
        pregunta = state.get("original_question") or state["question"]
        if not relevant:
            return {
                "answer": {
                    "text": NO_ENCONTRADO,
                    "sources": [],
                    "stop_reason": "no_context",
                    "model": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            }
        return {"answer": generate(pregunta, relevant)}

    return node_retrieve, node_grade, node_rewrite, node_generate


def _decide(state: RagState) -> str:
    """Arista condicional: seguir a generar, o reescribir y reintentar."""
    if state.get("relevant"):
        return "generate"
    if len(state.get("rounds", [])) >= state.get("max_rounds", 2):
        return "generate"
    return "rewrite"


# --- Grafo ------------------------------------------------------------------


def build_graph(
    *,
    retrieve: Retriever,
    grade: Grader,
    rewrite: Rewriter,
    generate: Generator,
):
    """Compila el grafo CRAG. Las operaciones se inyectan para poder probarlo
    sin credenciales; `default_ops()` entrega las reales."""
    node_retrieve, node_grade, node_rewrite, node_generate = _make_nodes(retrieve, grade, rewrite, generate)
    g = StateGraph(RagState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("rewrite", node_rewrite)
    g.add_node("generate", node_generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _decide, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def run(graph, question: str, *, max_rounds: int = 2) -> dict:
    """Ejecuta el grafo y devuelve la traza completa, con la misma forma que
    `pipeline.query` para que `observe.mirar` la entienda."""
    if max_rounds < 1:
        raise ValueError("max_rounds debe ser >= 1")
    final: RagState = graph.invoke(
        {
            "question": question,
            "original_question": question,
            "rounds": [],
            "max_rounds": max_rounds,
            "degraded": False,
        }
    )
    answer = final.get("answer", {})
    relevant = final.get("relevant", [])
    return {
        **answer,
        "question": question,
        "final_question": final.get("question"),
        "retrieved": sum(r["retrieved"] for r in final["rounds"]),
        "used": len(relevant),
        "quarantined": [],
        "contexts": [f"[fuente: {c['source']}]\n{c['text']}" for c in relevant],
        "rounds": final["rounds"],
        "degraded": final.get("degraded", False),
    }


# --- Cableado real -----------------------------------------------------------


def default_ops(
    perms,
    *,
    collection: str | None = None,
    top_k: int | None = None,
    grader: str | None = None,
) -> dict:
    """Operaciones reales sobre los módulos del repo. Se importa perezosamente
    para que `graph_flow` se pueda importar y probar sin psycopg ni anthropic.

    `grader` elige el juez de relevancia (`llm` | `jev`, ver `ragb.graders`);
    si es None manda la variable `RAGB_GRADER`."""
    from . import embeddings, generate, graders, store
    from .config import settings

    coll = collection or settings.collection

    def retrieve(question: str) -> list[Chunk]:
        return store.search(perms, coll, embeddings.embed_one(question), top_k=top_k)

    grade = graders.make_grader(grader)

    def rewrite(question: str) -> str:
        from .grade import rewrite_question

        return rewrite_question(question)

    def generate_(question: str, chunks: list[Chunk]) -> dict:
        return generate.answer(question, chunks)

    return {
        "retrieve": retrieve,
        "grade": grade,
        "rewrite": rewrite,
        "generate": generate_,
    }
