"""Humo del juez entrenado sobre las preguntas doradas, medido a nivel FRAGMENTO.

Un humo que solo mira "¿pasó la fuente correcta?" puede dar verde mientras el
juez descarta tres de las cuatro condiciones que la referencia exige. Acá se
mide lo que importa río abajo: qué parte de los términos de la referencia
cubren los fragmentos que PASARON (el generador no ve los demás), y para las
referencias que enumeran, cuántos de sus ítems quedaron cubiertos.

Dos modos, por `--chunk-chars`:

- 320 (el del dataset): lo que el juez vio en entrenamiento. Pasa por
  `grade_slm.score`, con la guarda de largo puesta, como en producción.
- 900 (el CHUNK_CHARS por defecto de producción): un régimen que el juez nunca
  vio. La guarda de `grade_slm` lo rechaza y el humo lo imprime; con `--raw`
  se puntúa igual (`score(raw=True)`) SOLO para medir cuánto se equivocaría.
  Los números de `--raw` no son los de producción: producción lanza.

Los fragmentos salen de `build_dataset.load_chunks_with_quarantine`, la misma
secuencia que produjo el dataset: si el humo fragmentara por su cuenta, podría
medir sobre fragmentos que el juez nunca vio y atribuirle el error al modelo.

⚠ Las preguntas son las del dataset dorado (seis están en train): esto mide
CABLEADO y cobertura de referencia, no generalización. La generalización es
el F1 de valid en metrics.json.

Necesita torch y transformers. Uso:

    SLM_GRADER_PATH=slm/out/e5-small-12ep python -m slm.smoke
    SLM_GRADER_PATH=slm/out/e5-small-12ep python -m slm.smoke --chunk-chars 900 --raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ragb import grade_slm  # noqa: E402
from slm import build_dataset as bd  # noqa: E402


def evaluate(case: dict, chunks: list[dict], *, raw: bool, threshold: float) -> dict:
    t0 = time.time()
    scores = grade_slm.score(case["question"], chunks, raw=raw)
    ms = round((time.time() - t0) * 1000)
    passed = [c for c, s in zip(chunks, scores) if s >= threshold]
    passed_terms = frozenset().union(*(bd.content_terms(c["text"]) for c in passed)) if passed else frozenset()
    ref_terms = bd.content_terms(case.get("reference", ""))
    items = bd.reference_items(case.get("reference", ""))
    return {
        "question": case["question"],
        "expects_refusal": bool(case.get("expects_refusal")),
        "passed": len(passed),
        "of": len(chunks),
        "sources": sorted({c["source"] for c in passed}),
        "reference_coverage": round(bd.coverage(ref_terms, passed_terms), 3),
        "items_covered": (
            f"{sum(1 for it in items if bd.coverage(it, passed_terms) >= bd.ITEM_THRESHOLD)}/{len(items)}"
            if items
            else "-"
        ),
        "scores": [round(s, 3) for s in scores],
        "ms": ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-chars", type=int, default=bd.CHUNK_CHARS)
    parser.add_argument("--overlap", type=int, default=None, help="default: 60 con 320, 150 con otro tamaño")
    parser.add_argument("--raw", action="store_true", help="salta la guarda de largo (solo para medir)")
    args = parser.parse_args(argv)
    if args.overlap is not None:
        overlap = args.overlap
    else:
        # 150 es el CHUNK_OVERLAP por defecto de producción (ragb/config.py).
        overlap = bd.CHUNK_OVERLAP if args.chunk_chars == bd.CHUNK_CHARS else 150

    juez = grade_slm.describe()
    threshold = juez["threshold"]
    chunks, _ = bd.load_chunks_with_quarantine(size=args.chunk_chars, overlap=overlap)
    print(
        json.dumps(
            {
                "model": juez["path"],
                "chunk_chars": args.chunk_chars,
                "overlap": overlap,
                "chunks": len(chunks),
                "chunk_lengths": [len(c["text"]) for c in chunks],
                "raw": args.raw,
                "threshold": threshold,
                "trained_chunk_chars": juez["config"]["chunk_chars"],
            },
            ensure_ascii=False,
        )
    )
    print("| pregunta | pasan | fuentes | cobertura ref. | ítems | ms |")
    print("|---|---|---|---|---|---|")
    results = []
    for case in bd.load_cases():
        try:
            r = evaluate(case, chunks, raw=args.raw, threshold=threshold)
        except RuntimeError as exc:
            # La guarda de largo lanza en la primera pregunta: eso ES el
            # resultado del humo en el régimen de producción, no un fallo del humo.
            print(json.dumps({"guard": "LANZA", "error": str(exc)}, ensure_ascii=False))
            return 0
        results.append(r)
        print(
            f"| {r['question'][:55]} | {r['passed']}/{r['of']} | {', '.join(r['sources']) or '—'} | "
            f"{r['reference_coverage']} | {r['items_covered']} | {r['ms']} |"
        )
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
