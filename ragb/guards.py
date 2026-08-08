"""Controles de seguridad para el pipeline RAG.

Cada control está mapeado al riesgo correspondiente de OWASP Top 10 for LLM
Applications, para que la decisión de diseño sea auditable:

  LLM01 Prompt Injection      -> detect_injection() sobre el texto recuperado
  LLM02 Sensitive Info Disc.  -> redact_secrets() antes de mandar al modelo
  LLM06 Excessive Agency      -> Permissions (mínimo privilegio, dry-run)
  LLM08 Vector/Embedding Weak.-> filtros de tenant en la búsqueda (store.py)

Principio rector: el contenido recuperado es DATO, nunca INSTRUCCIÓN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Patrones de instrucción incrustada en documentos. No pretenden ser
# exhaustivos: son una primera barrera barata y determinista. La defensa
# estructural real es el delimitado del contexto en generate.py.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_instructions", re.compile(
        r"\b(ignora|ignore|olvida|forget|disregard)\b[^.\n]{0,40}"
        r"\b(anterior|previous|above|instruc\w+|prompt|reglas?|rules?)\b", re.I)),
    ("role_hijack", re.compile(
        r"\b(you are now|ahora eres|act as|actúa como|new (system )?prompt|"
        r"nuevo prompt( de sistema)?)\b", re.I)),
    ("system_tag_spoof", re.compile(
        r"</?\s*(system|assistant|tool_result|instructions?)\s*>", re.I)),
    ("exfiltration", re.compile(
        r"\b(revela|reveal|imprime|print|muestra|show|repite|repeat)\b"
        r"[^.\n]{0,40}\b(system prompt|prompt de sistema|api[ _-]?key|token|"
        r"credential\w*|contraseñ\w+|password)\b", re.I)),
    ("tool_coercion", re.compile(
        r"\b(llama|call|ejecuta|execute|run|invoca|invoke)\b[^.\n]{0,30}"
        r"\b(tool|herramienta|función|function|endpoint|curl|http)\b", re.I)),
)

# Secretos que nunca deben viajar al modelo aunque estén en el corpus.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
    ("pg_dsn", re.compile(r"postgres(?:ql)?://[^\s:]+:[^\s@]+@[^\s]+")),
)


@dataclass(frozen=True)
class InjectionVerdict:
    """Resultado del análisis de un fragmento recuperado."""

    flagged: bool
    rules: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return ", ".join(self.rules) if self.rules else "clean"


def detect_injection(text: str) -> InjectionVerdict:
    """LLM01: marca texto recuperado que intenta dar órdenes al modelo."""
    hits = tuple(name for name, pat in _INJECTION_PATTERNS if pat.search(text))
    return InjectionVerdict(flagged=bool(hits), rules=hits)


def redact_secrets(text: str) -> tuple[str, tuple[str, ...]]:
    """LLM02: sustituye credenciales por un marcador antes de armar el prompt.

    Devuelve el texto saneado y los tipos de secreto encontrados.
    """
    found: list[str] = []
    out = text
    for name, pat in _SECRET_PATTERNS:
        out, n = pat.subn(f"[REDACTED:{name}]", out)
        if n:
            found.append(name)
    return out, tuple(found)


@dataclass(frozen=True)
class Permissions:
    """LLM06: mínimo privilegio explícito, negado por defecto.

    `dry_run=True` es el default deliberado: el pipeline calcula la escritura
    y la reporta, pero no la ejecuta. Habilitarla es una decisión consciente
    de quien despliega, no un accidente de configuración.
    """

    tenant: str
    can_read: bool = True
    can_write: bool = False
    dry_run: bool = True
    allowed_collections: frozenset[str] = field(default_factory=frozenset)

    def assert_can_read(self, collection: str) -> None:
        if not self.can_read:
            raise PermissionError("lectura no autorizada para este principal")
        if self.allowed_collections and collection not in self.allowed_collections:
            raise PermissionError(
                f"colección '{collection}' fuera del alcance autorizado"
            )

    def assert_can_write(self, collection: str) -> None:
        self.assert_can_read(collection)
        if not self.can_write:
            raise PermissionError("escritura no autorizada para este principal")


def sanitize_chunks(
    chunks: list[dict], *, drop_flagged: bool = True
) -> tuple[list[dict], list[dict]]:
    """Aplica LLM01 + LLM02 a los fragmentos recuperados.

    Devuelve (aceptados, cuarentena). Los aceptados llevan el texto ya
    redactado; los de cuarentena conservan el motivo para auditoría.
    """
    accepted: list[dict] = []
    quarantined: list[dict] = []

    for chunk in chunks:
        verdict = detect_injection(chunk["text"])
        clean, secrets = redact_secrets(chunk["text"])
        enriched = {**chunk, "text": clean, "redacted": secrets}

        if verdict.flagged and drop_flagged:
            quarantined.append({**enriched, "quarantine_reason": verdict.reason})
            continue

        if verdict.flagged:
            enriched["injection_flags"] = verdict.rules
        accepted.append(enriched)

    return accepted, quarantined
