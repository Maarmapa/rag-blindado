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


def _prefix(kind: str) -> str:
    """Prefijo que la familia E5 espera para distinguir pregunta de documento.

    E5 se entrenó con "query: " y "passage: " delante del texto, y omitirlos
    degrada la recuperación en silencio: no falla, solo recupera peor. Se
    aplica concatenando texto y no con el parámetro `prompt` de `encode()`,
    para no depender de la firma de una versión concreta de
    sentence-transformers.

    Cualquier otro modelo va sin prefijo: en un modelo que no lo espera, es
    ruido que empeora el vector.
    """
    if "e5" not in settings.embedding_model.lower():
        return ""
    return f"{kind}: "


def embed(texts: list[str], *, kind: str = "passage") -> list[list[float]]:
    """Vectoriza una lista de textos. Normaliza para usar distancia coseno.

    `kind` distingue un documento indexado ("passage") de una pregunta
    ("query"). El default es "passage" porque la ingesta es quien llama en
    volumen; las preguntas entran por `embed_one`.
    """
    if not texts:
        return []
    prefix = _prefix(kind)
    vectors = _model().encode(
        [prefix + t for t in texts] if prefix else texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    """Vectoriza una pregunta: es el lado 'query' de la búsqueda."""
    return embed([text], kind="query")[0]


def dimension() -> int:
    return int(_model().get_sentence_embedding_dimension())
