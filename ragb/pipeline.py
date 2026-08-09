"""Pipeline RAG completo: ingesta y consulta."""

from __future__ import annotations

import pathlib

from . import embeddings, generate, store
from .chunking import chunk_text
from .config import settings
from .grade import grade_chunks, rewrite_query
from .guards import Permissions, sanitize_chunks


def ingest_path(
    perms: Permissions,
    path: str | pathlib.Path,
    *,
    collection: str | None = None,
    patterns: tuple[str, ...] = ("*.md", "*.txt"),
) -> dict:
    """Indexa un archivo o directorio. Respeta perms.dry_run."""
    collection = collection or settings.collection
    root = pathlib.Path(path)
    files = (
        [root]
        if root.is_file()
        else sorted(f for p in patterns for f in root.rglob(p))
    )

    rows: list[dict] = []
    for f in files:
        pieces = chunk_text(f.read_text(encoding="utf-8"))
        vectors = embeddings.embed(pieces)
        rows.extend(
            {
                "source": f.name,
                "chunk_index": i,
                "text": piece,
                "embedding": vec,
            }
            for i, (piece, vec) in enumerate(zip(pieces, vectors))
        )

    written = store.upsert(perms, collection, rows)
    return {
        "files": len(files),
        "chunks": len(rows),
        "written": written,
        "dry_run": perms.dry_run,
    }


def query(
    perms: Permissions,
    question: str,
    *,
    collection: str | None = None,
    top_k: int | None = None,
    corrective: bool | None = None,
    grader=None,
    rewriter=None,
) -> dict:
    """Recupera, sanea, califica y responde. Devuelve traza completa para auditoría.

    Con `corrective` activo (por defecto, ver CRAG_ENABLED) el pipeline deja de
    ser lineal: si tras sanear no queda contexto relevante, reformula la
    pregunta y vuelve a buscar, hasta CRAG_MAX_ROUNDS veces. Si aun así no
    encuentra nada, responde que no lo encuentra — nunca completa con
    conocimiento propio ni sale a internet (ver ragb/grade.py).

    `grader` y `rewriter` se pueden inyectar para probar el ciclo sin credenciales.
    """
    collection = collection or settings.collection
    corrective = settings.crag_enabled if corrective is None else corrective
    max_rondas = max(1, settings.crag_max_rounds) if corrective else 1

    consulta = question
    rondas: list[dict] = []
    aceptados: list[dict] = []
    cuarentena_total: list[dict] = []
    recuperados_total = 0

    for i in range(max_rondas):
        recuperados = store.search(
            perms, collection, embeddings.embed_one(consulta), top_k=top_k
        )
        recuperados_total += len(recuperados)
        limpios, cuarentena = sanitize_chunks(recuperados)
        cuarentena_total.extend(cuarentena)

        if corrective:
            veredicto = grade_chunks(question, limpios, grader=grader)
            aceptados = veredicto["relevant"]
            rondas.append({
                "query": consulta,
                "rewritten": i > 0,
                "retrieved": len(recuperados),
                "quarantined": len(cuarentena),
                "relevant": len(aceptados),
                "graded": veredicto["graded"],
                "grader_error": veredicto["error"],
            })
        else:
            aceptados = limpios
            rondas.append({
                "query": consulta,
                "rewritten": i > 0,
                "retrieved": len(recuperados),
                "quarantined": len(cuarentena),
                "relevant": len(aceptados),
                "graded": False,
                "grader_error": None,
            })

        if aceptados or i == max_rondas - 1:
            break

        # Nada relevante: reformular SIEMPRE desde la pregunta original, para
        # que las rondas no se vayan alejando del sentido inicial.
        nueva = rewrite_query(question, rewriter=rewriter)
        if nueva == consulta:
            break  # sin reescritura útil, otra vuelta daría lo mismo
        consulta = nueva

    result = generate.answer(question, aceptados)

    return {
        **result,
        "question": question,
        "retrieved": recuperados_total,
        "used": len(aceptados),
        "quarantined": [
            {"source": q["source"], "reason": q["quarantine_reason"]}
            for q in cuarentena_total
        ],
        "contexts": [c["text"] for c in aceptados],
        "corrective": {
            "enabled": corrective,
            "rounds": rondas,
            "final_query": consulta,
            "gave_up": not aceptados,
        },
    }
