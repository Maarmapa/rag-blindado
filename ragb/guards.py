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


def _paragraphs(text: str) -> list[str]:
    """Parte por líneas en blanco, conservando el texto de cada párrafo."""
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


@dataclass(frozen=True)
class Excision:
    """Resultado de sacar los párrafos con instrucciones de un fragmento.

    `safe` en False significa que la escisión no se pudo hacer con garantías y
    quien llama debe descartar el fragmento completo.
    """

    kept: str
    removed: tuple[str, ...]
    rules: tuple[str, ...]
    safe: bool


def excise_injection(text: str) -> Excision:
    """LLM01 a nivel de párrafo: saca la instrucción, conserva el dato.

    Botar el fragmento entero porque un párrafo trae una inyección también bota
    la información legítima que lo acompaña. En el corpus de demo eso es
    literal: `nota-proveedor.md` cabe en un solo fragmento, así que la
    inyección se llevaba consigo el plazo de pago y el horario de soporte, y el
    pipeline respondía "no encuentro esa información" a dos preguntas que el
    corpus sí contesta.

    Fail-closed en dos puntos, porque partir el texto puede esconder un ataque:

      1. Si el escaneo del fragmento completo detecta una regla que ningún
         párrafo individual reproduce, esa regla calza CRUZANDO el corte. La
         escisión por párrafo no la eliminaría, así que se descarta todo.
      2. Lo que sobrevive se vuelve a escanear. Si todavía marca, se descarta.

    Nunca deja pasar texto que el detector marque: en el peor caso se comporta
    igual que botar el fragmento entero.
    """
    full = detect_injection(text)
    if not full.flagged:
        return Excision(kept=text, removed=(), rules=(), safe=True)

    paragraphs = _paragraphs(text)
    kept_parts: list[str] = []
    removed: list[str] = []
    attributed: set[str] = set()

    for para in paragraphs:
        verdict = detect_injection(para)
        if verdict.flagged:
            removed.append(para)
            attributed.update(verdict.rules)
        else:
            kept_parts.append(para)

    # (1) Reglas que solo aparecen mirando el fragmento entero: el patrón cruza
    # el límite entre párrafos y sacar párrafos no lo neutraliza.
    if set(full.rules) - attributed:
        return Excision(kept="", removed=tuple(paragraphs), rules=full.rules, safe=False)

    kept = "\n\n".join(kept_parts).strip()

    # (2) Verificación sobre el resultado, no sobre la intención.
    if not kept or detect_injection(kept).flagged:
        return Excision(kept="", removed=tuple(paragraphs), rules=full.rules, safe=False)

    return Excision(
        kept=kept, removed=tuple(removed), rules=tuple(sorted(attributed)), safe=True
    )


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
    redactado y sin los párrafos con instrucciones; la cuarentena conserva el
    material excluido con su motivo, para auditoría.

    La cuarentena es de lo excluido, no del fragmento: si un fragmento traía
    una inyección entre datos legítimos, los datos siguen adelante y solo el
    párrafo ofensor queda registrado. Cuando la escisión no se puede hacer con
    garantías (ver `excise_injection`), cae al comportamiento anterior y se
    descarta el fragmento completo.
    """
    accepted: list[dict] = []
    quarantined: list[dict] = []

    for chunk in chunks:
        verdict = detect_injection(chunk["text"])

        if not verdict.flagged:
            clean, secrets = redact_secrets(chunk["text"])
            accepted.append({**chunk, "text": clean, "redacted": secrets})
            continue

        if not drop_flagged:
            # Modo auditoría: nada se saca, todo se marca.
            clean, secrets = redact_secrets(chunk["text"])
            accepted.append(
                {
                    **chunk,
                    "text": clean,
                    "redacted": secrets,
                    "injection_flags": verdict.rules,
                }
            )
            continue

        excision = excise_injection(chunk["text"])

        if not excision.safe:
            clean, secrets = redact_secrets(chunk["text"])
            quarantined.append(
                {
                    **chunk,
                    "text": clean,
                    "redacted": secrets,
                    "quarantine_reason": verdict.reason,
                    "scope": "chunk",
                }
            )
            continue

        clean, secrets = redact_secrets(excision.kept)
        accepted.append(
            {
                **chunk,
                "text": clean,
                "redacted": secrets,
                "excised": len(excision.removed),
            }
        )
        for para in excision.removed:
            redacted_para, _ = redact_secrets(para)
            quarantined.append(
                {
                    **chunk,
                    "text": redacted_para,
                    "quarantine_reason": ", ".join(excision.rules),
                    "scope": "paragraph",
                }
            )

    return accepted, quarantined
