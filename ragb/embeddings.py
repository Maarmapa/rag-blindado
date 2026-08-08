"""Embeddings con modelo open source local.

Decisión de arquitectura: los embeddings corren localmente
(sentence-transformers) en vez de una API remota. Tres razones:

  1. Costo: el corpus se re-indexa completo sin costo por token.
  2. Privacidad: los documentos de la firma nunca salen de la infraestructura.
  3. CI: el pipeline de evaluación corre sin credenciales de embeddings.

La generación sí usa un modelo frontera (ver generate.py). Ese es el reparto
razonable: barato y local donde el volumen es alto, caro y capaz donde
importa la calidad del razonamiento.
"""

from __future__ import annotations

import functools

from .config import settings


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Vectoriza una lista de textos. Normaliza para usar distancia coseno."""
    if not texts:
        return []
    vectors = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def dimension() -> int:
    return int(_model().get_sentence_embedding_dimension())
