"""Configuración por variables de entorno. Ningún secreto vive en el código."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    generation_model: str = os.getenv("GENERATION_MODEL", "claude-opus-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-opus-5")

    # Corrective RAG: calificar el contexto y, si no sirve, reformular y
    # volver a buscar. Apagarlo deja el pipeline lineal de siempre.
    crag_enabled: bool = os.getenv("CRAG_ENABLED", "true").lower() not in ("0", "false", "no")
    crag_max_rounds: int = int(os.getenv("CRAG_MAX_ROUNDS", "2"))

    collection: str = os.getenv("COLLECTION", "default")
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Umbrales de calidad que la CI exige para aprobar (0.0 - 1.0).
    min_faithfulness: float = float(os.getenv("MIN_FAITHFULNESS", "0.85"))
    min_answer_relevancy: float = float(os.getenv("MIN_ANSWER_RELEVANCY", "0.75"))
    min_context_precision: float = float(os.getenv("MIN_CONTEXT_PRECISION", "0.70"))

    def require_database(self) -> str:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL no configurada (ver .env.example)")
        return self.database_url

    def require_anthropic(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no configurada (ver .env.example)")
        return self.anthropic_api_key


settings = Settings()
