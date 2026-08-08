"""Evaluación automatizada de calidad del RAG, integrable a CI/CD.

Mide tres dimensiones sobre un dataset dorado (evals/dataset.yaml):

  faithfulness      ¿cada afirmación de la respuesta se sostiene en el contexto?
  answer_relevancy  ¿la respuesta contesta la pregunta que se hizo?
  context_precision ¿lo recuperado era efectivamente relevante?

Las tres se calculan con Ragas usando Claude como modelo juez. El script
termina con exit code 1 si alguna métrica cae bajo el umbral configurado,
para que un pull request que degrada la calidad no pueda mergearse.

Uso:
    python -m evals.run                 # dataset completo
    python -m evals.run --limit 5       # subconjunto rápido
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

from ragb.config import settings
from ragb.guards import Permissions
from ragb.pipeline import query

DATASET = pathlib.Path(__file__).parent / "dataset.yaml"

THRESHOLDS = {
    "faithfulness": settings.min_faithfulness,
    "answer_relevancy": settings.min_answer_relevancy,
    "context_precision": settings.min_context_precision,
}


def collect(limit: int | None) -> list[dict]:
    """Corre el pipeline sobre cada pregunta del dataset dorado."""
    cases = yaml.safe_load(DATASET.read_text(encoding="utf-8"))["cases"]
    if limit:
        cases = cases[:limit]

    perms = Permissions(tenant="eval", can_read=True, can_write=False)
    rows = []
    for case in cases:
        result = query(perms, case["question"])
        rows.append(
            {
                "user_input": case["question"],
                "response": result["text"],
                "retrieved_contexts": result["contexts"],
                "reference": case["reference"],
            }
        )
        print(f"  · {case['question'][:64]}", file=sys.stderr)
    return rows


def score(rows: list[dict]) -> dict[str, float]:
    """Puntúa con Ragas. Claude actúa como juez de las métricas."""
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
    )

    judge = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.judge_model,
            api_key=settings.require_anthropic(),
            max_tokens=2048,
        )
    )

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    report = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[
            Faithfulness(llm=judge),
            AnswerRelevancy(llm=judge, embeddings=judge_embeddings),
            ContextPrecision(llm=judge),
        ],
    )
    scores = report._repr_dict if hasattr(report, "_repr_dict") else dict(report)
    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="eval-report.json")
    args = parser.parse_args()

    print("Ejecutando pipeline sobre el dataset dorado…", file=sys.stderr)
    rows = collect(args.limit)

    print("Puntuando con Ragas…", file=sys.stderr)
    scores = score(rows)

    failures = [
        (name, value, THRESHOLDS[name])
        for name, value in scores.items()
        if name in THRESHOLDS and value < THRESHOLDS[name]
    ]

    report = {"scores": scores, "thresholds": THRESHOLDS, "passed": not failures}
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Resultados ---")
    for name, value in sorted(scores.items()):
        threshold = THRESHOLDS.get(name)
        mark = "" if threshold is None else (" OK" if value >= threshold else " FALLA")
        limit = "" if threshold is None else f" (mínimo {threshold:.2f})"
        print(f"{name:20s} {value:.3f}{limit}{mark}")

    if failures:
        print(f"\n{len(failures)} métrica(s) bajo el umbral. Build rechazado.")
        return 1

    print("\nTodas las métricas sobre el umbral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
