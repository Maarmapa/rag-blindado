-- Esquema del vector store. Equivalente SQL de ragb.store.init_schema(),
-- para desplegar por migración en vez de por código.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    tenant      TEXT NOT NULL,
    collection  TEXT NOT NULL,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL,   -- all-MiniLM-L6-v2
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant, collection, source, chunk_index)
);

CREATE INDEX IF NOT EXISTS documents_tenant_collection_idx
    ON documents (tenant, collection);

-- HNSW sobre distancia coseno: los embeddings se guardan normalizados.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

-- Aislamiento por tenant a nivel de motor, además del filtro en la query.
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
