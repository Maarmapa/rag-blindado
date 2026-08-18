"""Vector store sobre Postgres + pgvector.

Toda consulta filtra por `tenant` a nivel de SQL. Ese filtro es la defensa
contra OWASP LLM08 (Vector and Embedding Weaknesses): sin él, un embedding
de un tenant puede recuperarse en el contexto de otro, y el modelo filtra
información entre clientes sin que ninguna regla de prompt lo impida.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from .config import settings
from .guards import Permissions


@contextmanager
def connection():
    with psycopg.connect(settings.require_database()) as conn:
        register_vector(conn)
        yield conn


def asegurar_extension() -> None:
    """CREATE EXTENSION en su propia conexión, antes de registrar el tipo.

    `register_vector()` busca `vector` en el catálogo de Postgres y falla si
    no está, así que no puede correr sobre una base donde la extensión todavía
    no existe. Crearla desde dentro de `connection()` es imposible: para
    entonces `register_vector` ya reventó.

    En una base ya usada nadie nota el problema —la extensión está puesta de
    antes— y por eso vivió sin verse. En una base recién creada, que es la que
    levanta CI en cada corrida, es el primer error.

    Tolerante a propósito: si el CREATE falla (típicamente permisos en una base
    gestionada donde la extensión ya viene instalada), se sigue igual y
    `register_vector` queda como la verificación real. Nunca deja las cosas
    peor de lo que estaban.
    """
    try:
        with psycopg.connect(settings.require_database()) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    except psycopg.Error as e:
        print(f"[store] no se pudo crear la extensión vector: {e}", file=sys.stderr)


def init_schema(dim: int) -> None:
    """Crea la tabla y el índice HNSW. Idempotente."""
    asegurar_extension()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id          BIGSERIAL PRIMARY KEY,
                tenant      TEXT NOT NULL,
                collection  TEXT NOT NULL,
                source      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                embedding   VECTOR({dim}) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (tenant, collection, source, chunk_index)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS documents_tenant_collection_idx "
            "ON documents (tenant, collection)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        conn.commit()


def upsert(perms: Permissions, collection: str, rows: list[dict]) -> int:
    """Inserta fragmentos. Requiere permiso de escritura explícito."""
    perms.assert_can_write(collection)
    if perms.dry_run:
        return 0

    with connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents
                (tenant, collection, source, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant, collection, source, chunk_index)
            DO UPDATE SET text = EXCLUDED.text,
                          embedding = EXCLUDED.embedding,
                          created_at = now()
            """,
            [
                (
                    perms.tenant,
                    collection,
                    r["source"],
                    r["chunk_index"],
                    r["text"],
                    r["embedding"],
                )
                for r in rows
            ],
        )
        conn.commit()
        return len(rows)


def search(
    perms: Permissions,
    collection: str,
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[dict]:
    """Búsqueda por similitud coseno, siempre acotada al tenant autorizado."""
    perms.assert_can_read(collection)
    k = top_k or settings.top_k

    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, chunk_index, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM documents
            WHERE tenant = %s AND collection = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, perms.tenant, collection, query_embedding, k),
        )
        return [
            {"source": s, "chunk_index": i, "text": t, "score": float(sc)}
            for s, i, t, sc in cur.fetchall()
        ]
