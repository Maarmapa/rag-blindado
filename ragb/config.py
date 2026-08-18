"""Configuración por variables de entorno. Ningún secreto vive en el código."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Multilingüe a propósito: el corpus y las preguntas están en español, y el
    # anterior (all-MiniLM-L6-v2) declara un solo idioma, `en`. Con él, una
    # respuesta exacta —"1 hora" a "¿cuál es el plazo de severidad 1?"— sacaba
    # 0.312 de answer_relevancy. Eso medía un modelo fuera de su idioma, no la
    # calidad de la respuesta.
    #
    # Se prefirió e5-small sobre paraphrase-multilingual-MiniLM-L12-v2 por el
    # límite de secuencia: 512 tokens contra 128. Con 128, la mitad de cada
    # fragmento de 900 caracteres se truncaría en silencio.
    #
    # Misma dimensión que el anterior (384): el esquema no cambia. El espacio
    # vectorial sí, así que una base ya indexada hay que re-indexarla.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
    )
    generation_model: str = os.getenv("GENERATION_MODEL", "claude-opus-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-opus-5")

    # Juez aparte solo para `faithfulness`. Vacío = usar `judge_model`.
    #
    # Esa métrica es una razón sobre pocas afirmaciones, así que en respuestas
    # cortas un error del juez cuesta un tercio del puntaje. Medido con Haiku:
    # una respuesta que cita el documento casi palabra por palabra —"las
    # credenciales se gestionan exclusivamente mediante variables de entorno y
    # bóveda de secretos", contra "…exclusivamente por variables de entorno y
    # bóveda de secretos" en el corpus— sacaba 0.667. No había nada que
    # recortar: era error de medición.
    #
    # Las otras dos métricas no mostraron ese problema, así que el modelo caro
    # se paga solo donde se demostró que hace falta.
    #
    # ⚠ Ojo al usarla: Ragas fija `temperature` en cada llamada al juez, y hay
    # modelos que rechazan ese parámetro con 400. Con claude-sonnet-5 los ocho
    # jobs de fidelidad fallaron y la métrica quedó sin medir. La salida es
    # `LangchainLLMWrapper(..., bypass_temperature=True)`.
    faithfulness_judge_model: str = os.getenv("FAITHFULNESS_JUDGE_MODEL", "")

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
