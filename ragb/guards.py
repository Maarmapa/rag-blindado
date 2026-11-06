"""Controles de seguridad para el pipeline RAG.

Cada control está mapeado al riesgo correspondiente de OWASP Top 10 for LLM
Applications, para que la decisión de diseño sea auditable:

  LLM01 Prompt Injection      -> detect_injection() sobre el texto recuperado
  LLM02 Sensitive Info Disc.  -> redact_secrets() al indexar y antes de mandar al modelo
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
#
# Un patrón por tipo de secreto, para que `redact_secrets` no repita tipos.
#
# Ningún patrón exige un límite de palabra delante. En un corpus real la llave
# rara vez va entre espacios: llega detrás de un escape JSON (\n, \t), de un
# %20 o un %3D de una URL, del =3D de un correo quoted-printable, de la «m» de
# un color ANSI, de texto chino sin espacios o del «-» de la línea quitada de
# un diff. Un \b o un (?<!\w) delante deja pasar justo esos casos.
#
# La prosa que se parece a una llave tiene dos formas, y cada una se ataja por
# su lado:
#
#   1. La cola de una palabra: «ta|sk-ant-colony-…», «ri|sk-…». Delante de
#      «sk-» hay dos letras y la segunda es minúscula. Ahí la llave cuenta solo
#      si el cuerpo trae una mayúscula, como «API Keysk-…» cuando se extrae el
#      texto de una tabla HTML: la prosa que calza es un slug en minúsculas.
#   2. Un slug, una ruta o un placeholder que empieza con el prefijo:
#      «sk-proj-template-for-…», «docs/github_pat_rotation.md»,
#      «sk-proj-YOUR_API_KEY_HERE». Pasa en los formatos cuyo cuerpo admite
#      guiones o guiones bajos, y en esos el cuerpo tiene que traer una
#      mayúscula y un dígito, que una llave real, por ser aleatoria, trae.


# Hasta dónde mira un lookahead o una contraseña. Sin tope, la redacción se
# vuelve cuadrática con un texto como «task-task-task-…» o «postgres://» repetido
# sin «@»: cada aparición del prefijo recorre de nuevo toda la racha. Con 80 KB
# eso eran seis segundos, y la ingesta redacta documentos enteros. 200 cubre el
# cuerpo completo de las llaves de hoy: la más larga, sk-proj-, trae 156.
_TOPE = 200


def _trae(clase: str, *requisitos: str) -> str:
    """Lookahead: la racha de `clase` que sigue trae cada uno de `requisitos`.

    Recorre la misma clase que el cuerpo de la llave, así que mira justo lo que
    se va a redactar y no el texto que viene después.
    """
    return "".join(f"(?={clase}{{0,{_TOPE}}}{r})" for r in requisitos)


def _sk(resto: str, clase: str) -> str:
    """«sk-» seguido de `resto`, con la guarda de la forma 1."""
    return f"(?:(?<![A-Za-z][a-z])|(?=sk-{_trae(clase, '[A-Z]')}))sk-{resto}"


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Anthropic va antes que OpenAI: las dos empiezan con «sk-», y si el patrón
    # de OpenAI se ensancha, una llave sk-ant- no debe quedar rotulada como suya.
    ("anthropic_key", re.compile(_sk(r"ant-[A-Za-z0-9_\-]{20,}", r"[A-Za-z0-9_\-]"))),
    # OpenAI: las de proyecto, cuenta de servicio y administración traen guiones
    # y guiones bajos en el cuerpo (forma 2); el formato viejo, solo
    # alfanuméricos (forma 1).
    ("openai_key", re.compile(
        r"sk-(?:proj|svcacct|admin)-" + _trae(r"[A-Za-z0-9_\-]", "[A-Z]", "[0-9]")
        + r"[A-Za-z0-9_\-]{32,}|" + _sk(r"[A-Za-z0-9]{32,}", "[A-Za-z0-9]"))),
    # GitHub: los clásicos (ghp_, gho_, ghu_, ghs_, ghr_) y los de grano fino
    # (github_pat_), cuyo cuerpo admite guiones bajos (forma 2).
    ("github_token", re.compile(
        r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_"
        + _trae("[A-Za-z0-9_]", "[A-Z]", "[0-9]") + r"[A-Za-z0-9_]{30,}")),
    # AWS: AKIA es la llave de larga duración y ASIA la temporal de STS. Asia
    # también es una palabra, así que ASIA se acota más: el cuerpo usa el
    # alfabeto real del ID (base32: A-Z y 2-7), detrás no puede seguir otra
    # mayúscula ni dígito, y delante tampoco, salvo que sea el final de un
    # escape %XX o =XX (X-Amz-Credential%3DASIA… en una URL prefirmada).
    ("aws_key", re.compile(
        r"AKIA[0-9A-Z]{16}"
        r"|(?:(?<=[%=][0-9A-Fa-f]{2})|(?<![A-Z0-9]))ASIA[A-Z2-7]{16}(?![A-Z0-9])")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
    # La contraseña va acotada por la misma razón que los lookaheads (ver _TOPE).
    ("pg_dsn", re.compile(r"postgres(?:ql)?://[^\s:]+:[^\s@]{1,256}@[^\s]+")),
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
