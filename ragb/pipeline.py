"""Pipeline RAG completo: ingesta y consulta."""

from __future__ import annotations

import pathlib

from . import embeddings, generate, observe, store
from .chunking import chunk_text
from .config import settings
from .guards import Permissions, redact_secrets, sanitize_chunks


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
    redactados: dict[str, tuple[str, ...]] = {}
    for f in files:
        # LLM02 sobre el documento entero, antes de fragmentar. Redactar recién
        # al recuperar no alcanza: si el corte de un fragmento cae dentro de una
        # llave, el fragmento siguiente arranca sin el prefijo y ningún patrón
        # la reconoce. Pasa en un JSON de una sola línea, donde no hay «\n» ni
        # «. » para cortar limpio. De paso, la llave no queda guardada en la
        # base ni convertida en embedding.
        texto, tipos = redact_secrets(f.read_text(encoding="utf-8"))
        if tipos:
            redactados[str(f)] = tipos
        pieces = chunk_text(texto)
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
        # Qué archivos traían credenciales y de qué tipo. La base queda limpia,
        # pero el archivo de origen no: alguien tiene que sacarlas de ahí.
        "redacted": redactados,
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

    traza = {
        **result,
        "question": question,
        "retrieved": len(retrieved),
        "used": len(accepted),
        "quarantined": [
            {"source": q["source"], "reason": q["quarantine_reason"]}
            for q in quarantined
        ],
        # El fragmento va CON su fuente, igual que lo recibe el modelo dentro
        # de <documento fuente="…"> en generate.build_context.
        #
        # Antes viajaba como texto pelado, y eso rompía la evaluación de forma
        # silenciosa: el juez de fidelidad extrae "la fuente de este dato es
        # nota-proveedor.md" como una afirmación más de la respuesta y la
        # marcaba como no sostenida, porque el nombre del archivo no aparecía
        # por ningún lado en lo que le mostrábamos. Medido: dos casos con
        # respuestas correctas y citadas quedaron clavados en 0.667 (2 de 3)
        # durante cuatro corridas por esta única razón.
        #
        # El juez tenía razón: la culpa era de mostrarle menos contexto del que
        # tuvo el modelo. Y de paso la traza de auditoría ahora dice de qué
        # documento salió cada fragmento, que es lo que el README promete poder
        # reconstruir.
        "contexts": [f"[fuente: {c['source']}]\n{c['text']}" for c in accepted],
    }

    # La traza deja de morir acá. `mirar` registra, alerta sobre lo que merece
    # atención y persiste si hay ruta configurada; nunca lanza, así que la
    # respuesta sale igual aunque la observabilidad falle.
    observe.mirar(traza)

    return traza
