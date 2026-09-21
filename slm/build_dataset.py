"""Construye el dataset de pares (pregunta, fragmento) -> relevante / no relevante.

Sin modelo, sin red, determinista: dos corridas sobre el mismo corpus y el
mismo dataset dorado producen bytes idénticos. Es lo que permite que
`tests/test_slm_dataset.py` afirme conteos exactos en CI sin torch.

Fuentes:
  - corpus/*.md            los documentos, partidos con `ragb.chunking`
  - evals/dataset.yaml     los casos dorados (question, reference, expects_refusal)

Cómo se etiqueta (la heurística, escrita para que se pueda discutir):

1. Los fragmentos pasan por `ragb.guards.sanitize_chunks` ANTES de etiquetar.
   Un párrafo con inyección no entra al dataset ni como negativo: si el juez
   aprendiera que "IGNORA TODAS LAS INSTRUCCIONES" es un fragmento normal de
   cualquier clase, estaríamos entrenando sobre el ataque (LLM01).

2. Se fragmenta más fino que en producción (`CHUNK_CHARS=320` contra 900).
   Con 900 caracteres el corpus entero son cuatro fragmentos, uno por
   documento: un clasificador aprendería a reconocer documentos, no
   relevancia, y no existirían negativos difíciles. Con 320 hay trece, y cada
   documento tiene fragmentos que responden y fragmentos que no.

3. Un fragmento es POSITIVO para un caso cuando cubre "la mayor parte" de los
   términos de contenido: se promedia la cobertura de los términos de la
   referencia y la de los términos de la pregunta, y se exige >= 0.5. Por qué
   se mezcla la pregunta con la referencia: hay referencias que traen texto
   sobre lo que la respuesta NO debe contener (el caso de la inyección); ahí la
   cobertura de la referencia sola se diluye, y el fragmento que contesta
   literalmente la pregunta quedaría como negativo. Los términos de contenido
   son las palabras sin acentos, sin stopwords, con un stemming mínimo (quitar
   plural) y con los números en palabras ("treinta") convertidos a cifra,
   porque el corpus dice "30 días" y la referencia "Treinta días". "una"/"uno"
   se convierten SOLO delante de una unidad de tiempo ("una hora" -> "1 hora"):
   como artículo ("una acción") no es un número, y convertirlo hacía que
   cualquier fragmento cubriera el "1" de "severidad 1".

   3b. Referencias que ENUMERAN (ítems separados por ";"): un fragmento también
   es positivo si cubre >= 0.75 de los términos de CUALQUIER ítem con al menos
   tres términos de contenido. Sin esta regla, la referencia "cuatro
   condiciones" repartida en cuatro fragmentos dejaba tres de ellos como
   negativos difíciles —y el juez aprendía a rechazar la respuesta— porque la
   cobertura promedio de la referencia entera se diluía. El mínimo de tres
   términos evita que un ítem corto ("Una hora") convierta en positivo
   cualquier fragmento que diga "hora".

4. NEGATIVO: todo fragmento que no alcance el umbral. Se distingue el
   `kind` para poder medir por separado: `negative` si viene de otro
   documento, `hard_negative` si viene del mismo documento que algún positivo
   (comparte vocabulario y tema, pero no contesta).

5. Un caso con `expects_refusal` produce SOLO negativos: la respuesta correcta
   es "no está en el corpus", así que ningún fragmento es relevante.

6. Aumento por reglas, sin modelo: cada pregunta genera paráfrasis
   deterministas (sin signos, "¿Cuál es X?" -> "Dime X", etc.). Cada
   paráfrasis hereda las etiquetas de su pregunta original.

7. Split train/valid por hash SHA-256 de la pregunta ORIGINAL (hashlib, no
   `hash()`, que Python sala por proceso). Las paráfrasis siguen a su
   original: si cayeran en splits distintos, valid estaría lleno de
   casi-duplicados de train y la métrica mentiría.

Salida: slm/data/train.jsonl y slm/data/valid.jsonl, una fila por par, más
slm/data/manifest.json con los conteos y los parámetros que los produjeron.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Ejecutado como script (`python slm/build_dataset.py`), sys.path[0] es slm/ y
# `ragb` no se encuentra; como módulo (`python -m slm.build_dataset`) sí. Se
# agrega la raíz del repo para que las dos formas funcionen igual.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ragb.chunking import chunk_text  # noqa: E402
from ragb.guards import sanitize_chunks  # noqa: E402

CORPUS_DIR = REPO_ROOT / "corpus"
DATASET_YAML = REPO_ROOT / "evals" / "dataset.yaml"
OUT_DIR = REPO_ROOT / "slm" / "data"

# Ver punto 2 del docstring.
CHUNK_CHARS = 320
CHUNK_OVERLAP = 60

# Ver punto 3 del docstring.
POSITIVE_THRESHOLD = 0.5
# Ver punto 3b: cobertura mínima de un ítem enumerado, y cuántos términos de
# contenido necesita el ítem para contar como tal.
ITEM_THRESHOLD = 0.75
ITEM_MIN_TERMS = 3

# Ver punto 7. 100 cubetas y las < VALID_BUCKETS van a valid. El número se
# fijó mirando dónde caen los ocho hashes del dataset dorado: con 25 quedaba
# un solo caso en validación; con 30 quedan dos (severidad 1 y condiciones
# del agente), ambos con positivos, que es lo mínimo para que la métrica de
# valid mida generalización a preguntas no vistas y no a una sola. Si el
# dataset dorado crece, el reparto cambia solo y los conteos exactos de
# `tests/test_slm_dataset.py` hay que actualizarlos a mano: es a propósito.
VALID_BUCKETS = 30

_STOPWORDS = frozenset(
    """
    a al ante antes bajo como con cual cuales cuando cuanta cuanto cuantos de
    debe deben del desde donde e el ella ellos en entre es esta estas este
    estos hasta la las le les lo los mas ni no o para por que quien se ser si
    sin sobre son su sus tiempo todo toda todos u un una unas uno unos y
    ademas nunca
    """.split()
)

# Números en palabras que aparecen en las referencias. El corpus los escribe
# en cifras, así que sin esta tabla "Veinticuatro meses" no cubriría "24 meses".
_NUMBER_WORDS = {
    "dos": "2",
    "tres": "3",
    "cuatro": "4",
    "cinco": "5",
    "veinticuatro": "24",
    "treinta": "30",
    "cuarenta": "40",
}


# "una"/"uno"/"un" son artículos casi siempre; solo delante de una unidad de
# tiempo son el número 1 (punto 3 del docstring).
_TIME_UNITS = r"(?:hora|horas|dia|dias|semana|semanas|mes|meses|minuto|minutos|ano|anos)"
_ONE_BEFORE_UNIT = re.compile(rf"\b(?:una|uno|un)\b(?=\s+{_TIME_UNITS}\b)")


def _normalize(text: str) -> str:
    """Minúsculas, sin acentos, números en palabras -> cifras."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    plain = plain.lower()
    plain = _ONE_BEFORE_UNIT.sub("1", plain)
    for word, digits in _NUMBER_WORDS.items():
        plain = re.sub(rf"\b{word}\b", digits, plain)
    return plain


def content_terms(text: str) -> frozenset[str]:
    """Términos de contenido: sin stopwords, sin tokens cortos, plural quitado."""
    out: set[str] = set()
    for tok in re.findall(r"[a-z0-9]+", _normalize(text)):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 3 and not tok.isdigit():
            continue
        if len(tok) > 4:
            tok = re.sub(r"(es|s)$", "", tok)
        out.add(tok)
    return frozenset(out)


def coverage(needle: frozenset[str], haystack: frozenset[str]) -> float:
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def relevance_score(question: str, reference: str, chunk_text_: str) -> float:
    """Promedio de cobertura de la referencia y de la pregunta (punto 3)."""
    chunk_terms = content_terms(chunk_text_)
    return (coverage(content_terms(reference), chunk_terms) + coverage(content_terms(question), chunk_terms)) / 2


def reference_items(reference: str) -> list[frozenset[str]]:
    """Ítems de una referencia que enumera (punto 3b). Una referencia sin ";"
    no tiene ítems: la regla 3b no se aplica y manda solo el promedio."""
    parts = [p.strip() for p in reference.split(";")]
    if len(parts) < 2:
        return []
    items = [content_terms(p) for p in parts]
    return [t for t in items if len(t) >= ITEM_MIN_TERMS]


def item_score(reference: str, chunk_text_: str) -> float:
    """Mayor cobertura de un ítem enumerado de la referencia; 0 si no enumera."""
    chunk_terms = content_terms(chunk_text_)
    return max((coverage(item, chunk_terms) for item in reference_items(reference)), default=0.0)


def is_positive(question: str, reference: str, chunk_text_: str) -> bool:
    """Puntos 3 y 3b juntos: es la única función que decide una etiqueta."""
    return (
        relevance_score(question, reference, chunk_text_) >= POSITIVE_THRESHOLD
        or item_score(reference, chunk_text_) >= ITEM_THRESHOLD
    )


# --- Paráfrasis por reglas ----------------------------------------------------

# Cada regla es (patrón sobre la pregunta sin signos, reemplazo). Se aplican
# todas las que calcen; cada una produce una paráfrasis distinta.
_PARAPHRASE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^¿?Cuál es (.+)$", re.I), r"Dime \1"),
    (re.compile(r"^¿?Cuánto tiempo (.+)$", re.I), r"Dime cuánto tiempo \1"),
    (re.compile(r"^¿?Qué (.+)$", re.I), r"Indica qué \1"),
    (re.compile(r"^¿?Dónde (.+)$", re.I), r"Dime dónde \1"),
    (re.compile(r"^¿?En qué (.+)$", re.I), r"Dime en qué \1"),
)


def paraphrases(question: str) -> list[str]:
    """La pregunta original primero, después las variantes. Sin duplicados,
    en orden estable."""
    base = question.strip()
    sin_signos = base.strip("¿?").strip()
    out = [base, sin_signos]
    for pattern, repl in _PARAPHRASE_RULES:
        m = pattern.match(sin_signos)
        if m:
            out.append(pattern.sub(repl, sin_signos))
    seen: list[str] = []
    for q in out:
        if q and q not in seen:
            seen.append(q)
    return seen


# --- Split -------------------------------------------------------------------


def split_for(question: str) -> str:
    """Determinista entre procesos y máquinas: SHA-256, no `hash()`."""
    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "valid" if bucket < VALID_BUCKETS else "train"


# --- Construcción ---------------------------------------------------------------


def load_chunks_with_quarantine(
    corpus_dir: Path = CORPUS_DIR, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> tuple[list[dict], int]:
    """Fragmentos saneados del corpus, en orden estable por nombre de archivo,
    y cuántas entradas dejó fuera la cuarentena. El conteo se devuelve para
    que el manifest lo diga: un documento futuro que se cuarentene entero
    achicaría el dataset, y eso tiene que verse, no inferirse.

    `size`/`overlap` existen para que `slm/smoke.py` mida el juez con el
    tamaño de producción usando ESTA misma secuencia (glob -> chunk_text ->
    sanitize) en vez de una copia; el dataset siempre se construye con los
    defaults del módulo."""
    raw: list[dict] = []
    for path in sorted(corpus_dir.glob("*.md")):
        for text in chunk_text(path.read_text(encoding="utf-8"), size=size, overlap=overlap):
            raw.append({"text": text, "source": path.name})
    accepted, quarantined = sanitize_chunks(raw)
    # Solo lo que el pipeline dejaría pasar: texto ya redactado y sin los
    # párrafos con instrucciones. La cuarentena se descarta a propósito.
    return [{"text": c["text"], "source": c["source"]} for c in accepted], len(quarantined)


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    return load_chunks_with_quarantine(corpus_dir)[0]


def load_cases(dataset_yaml: Path = DATASET_YAML) -> list[dict]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    return list(data["cases"])


def build_pairs(cases: list[dict], chunks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for case_id, case in enumerate(cases):
        question = case["question"].strip()
        reference = case.get("reference", "")
        refusal = bool(case.get("expects_refusal", False))
        split = split_for(question)

        # Etiquetas se calculan UNA vez sobre la pregunta original y se
        # heredan: una paráfrasis no cambia qué fragmento es relevante.
        labels: list[tuple[int, str]] = []
        if refusal:
            labels = [(0, "refusal")] * len(chunks)
        else:
            positives = [is_positive(question, reference, c["text"]) for c in chunks]
            positive_sources = {c["source"] for c, p in zip(chunks, positives) if p}
            for chunk, positive in zip(chunks, positives):
                if positive:
                    labels.append((1, "positive"))
                elif chunk["source"] in positive_sources:
                    labels.append((0, "hard_negative"))
                else:
                    labels.append((0, "negative"))

        for variant_idx, q in enumerate(paraphrases(question)):
            for chunk, (label, kind) in zip(chunks, labels):
                rows.append(
                    {
                        "case_id": case_id,
                        "question": q,
                        "paraphrase": variant_idx > 0,
                        "chunk_text": chunk["text"],
                        "source": chunk["source"],
                        "label": label,
                        "kind": kind,
                        "split": split,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Conteos que el manifest y los tests comparten."""
    summary: dict = {"total": len(rows), "by_split": {}, "by_kind": {}}
    for split in ("train", "valid"):
        subset = [r for r in rows if r["split"] == split]
        summary["by_split"][split] = {
            "rows": len(subset),
            "positives": sum(r["label"] for r in subset),
            "negatives": sum(1 - r["label"] for r in subset),
            "cases": len({r["case_id"] for r in subset}),
        }
    for kind in ("positive", "hard_negative", "negative", "refusal"):
        summary["by_kind"][kind] = sum(1 for r in rows if r["kind"] == kind)
    return summary


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(
    *,
    corpus_dir: Path = CORPUS_DIR,
    dataset_yaml: Path = DATASET_YAML,
    out_dir: Path | None = OUT_DIR,
) -> dict:
    """Construye el dataset y, si `out_dir` no es None, lo escribe. Devuelve
    el manifest (conteos + parámetros) para poder afirmarlo en tests."""
    chunks, quarantined = load_chunks_with_quarantine(corpus_dir)
    cases = load_cases(dataset_yaml)
    rows = build_pairs(cases, chunks)
    manifest = {
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "positive_threshold": POSITIVE_THRESHOLD,
        "item_threshold": ITEM_THRESHOLD,
        "item_min_terms": ITEM_MIN_TERMS,
        "valid_buckets": VALID_BUCKETS,
        "chunks": len(chunks),
        "quarantined": quarantined,
        "cases": len(cases),
        **summarize(rows),
    }
    if out_dir is not None:
        write_jsonl([r for r in rows if r["split"] == "train"], out_dir / "train.jsonl")
        write_jsonl([r for r in rows if r["split"] == "valid"], out_dir / "valid.jsonl")
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--dataset", type=Path, default=DATASET_YAML)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    manifest = build(corpus_dir=args.corpus, dataset_yaml=args.dataset, out_dir=args.out)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
