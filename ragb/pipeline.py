"""Pipeline RAG completo: ingesta y consulta."""

from __future__ import annotations

import pathlib

from . import embeddings, generate, store
from .chunking import chunk_text
from .config import settings
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
) -> dict:
    """Recupera, sanea y responde. Devuelve traza completa para auditoría."""
    collection = collection or settings.collection

    retrieved = store.search(
        perms, collection, embeddings.embed_one(question), top_k=top_k
    )
    accepted, quarantined = sanitize_chunks(retrieved)
    result = generate.answer(question, accepted)

    return {
        **result,
        "question": question,
        "retrieved": len(retrieved),
        "used": len(accepted),
        "quarantined": [
            {"source": q["source"], "reason": q["quarantine_reason"]}
            for q in quarantined
        ],
        "contexts": [c["text"] for c in accepted],
    }
