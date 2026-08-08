"""Fragmentación de documentos. Sin dependencias pesadas a propósito: así los
tests deterministas corren en CI sin instalar el stack de ML.
"""

from __future__ import annotations

from .config import settings


def chunk_text(
    text: str, *, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """Fragmenta por caracteres respetando cortes de párrafo cuando puede."""
    size = size or settings.chunk_chars
    overlap = overlap or settings.chunk_overlap
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Corte limpio hacia atrás: párrafo, salto de línea, u oración.
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]
