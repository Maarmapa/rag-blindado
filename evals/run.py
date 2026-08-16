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
import math
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


def collect(limit: int | None) -> tuple[list[dict], list[dict]]:
    """Corre el pipeline sobre cada pregunta del dataset dorado.

    Devuelve (filas_para_ragas, trazas). Las trazas van aparte a propósito:
    `EvaluationDataset.from_list` valida las claves que recibe, así que los
    metadatos de auditoría no pueden viajar dentro de las filas.
    """
    cases = yaml.safe_load(DATASET.read_text(encoding="utf-8"))["cases"]
    if limit:
        cases = cases[:limit]

    perms = Permissions(tenant="eval", can_read=True, can_write=False)
    rows: list[dict] = []
    traces: list[dict] = []
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
        traces.append(
            {
                "question": case["question"],
                "retrieved": result["retrieved"],
                "used": result["used"],
                "quarantined": len(result["quarantined"]),
                "stop_reason": result.get("stop_reason"),
                "answer": result["text"],
            }
        )
        print(
            f"  · [{result['retrieved']}→{result['used']}] {case['question'][:56]}",
            file=sys.stderr,
        )
    return rows, traces


def score(rows: list[dict]) -> tuple[dict[str, float], list[dict]]:
    """Puntúa con Ragas. Claude actúa como juez de las métricas.

    Devuelve (promedios, puntaje_por_caso). El detalle por caso es lo que
    permite distinguir "el pipeline respondió mal" de "el juez no pudo
    calificar": en el promedio las dos cosas se ven igual.
    """
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
    )
    from ragas.run_config import RunConfig

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

    # El default de Ragas (timeout 180 s, 16 hilos) satura el límite de tasa de
    # la API y los jobs que se pasan del tiempo mueren con TimeoutError: sus
    # tokens ya se pagaron y el resultado se descarta. Menos paralelismo y más
    # paciencia hacen las MISMAS llamadas y desperdician menos.
    run_config = RunConfig(timeout=300, max_workers=4)

    report = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[
            Faithfulness(llm=judge),
            AnswerRelevancy(llm=judge, embeddings=judge_embeddings),
            ContextPrecision(llm=judge),
        ],
        run_config=run_config,
    )
    scores = report._repr_dict if hasattr(report, "_repr_dict") else dict(report)
    per_case = [dict(s) for s in report.scores]
    return (
        {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))},
        per_case,
    )


def reprueba(value: float, threshold: float) -> bool:
    """Decide si una métrica reprueba. Un `nan` reprueba.

    `nan` pierde TODAS las comparaciones: `nan < 0.85` es False, y
    `nan >= 0.85` también. Comparando a secas, una métrica que no se pudo
    calcular no entraba en la lista de fallas y el build pasaba verde sin
    haber medido nada — justo en el modo de falla más probable, que es que el
    juez se caiga o no devuelva nada que parsear.

    Una barrera que se abre sola cuando la medición falla no es una barrera.
    Sin número no hay aprobación.
    """
    return math.isnan(value) or value < threshold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="eval-report.json")
    args = parser.parse_args()

    print("Ejecutando pipeline sobre el dataset dorado…", file=sys.stderr)
    rows, traces = collect(args.limit)

    # Cortocircuito antes de gastar el juez. Si NINGÚN caso recuperó contexto,
    # todas las respuestas son el "no encuentro esa información" de
    # `generate.answer`, y Ragas devolvería `nan` en las métricas que necesitan
    # algo que juzgar. Pagar el juez para medir eso es tirar tokens: el
    # problema está en la indexación o en la búsqueda, no en la calidad.
    if not any(row["retrieved_contexts"] for row in rows):
        print(
            "\nNingún caso recuperó contexto. La evaluación no puede medir "
            "calidad sobre un corpus vacío: revisa que la ingesta haya escrito "
            "en el mismo tenant y colección que consulta `evals.run`.",
            file=sys.stderr,
        )
        for trace in traces:
            print(f"  · [{trace['retrieved']}→{trace['used']}] {trace['question'][:64]}")
        return 1

    print("Puntuando con Ragas…", file=sys.stderr)
    scores, per_case = score(rows)

    failures = [
        (name, value, THRESHOLDS[name])
        for name, value in scores.items()
        if name in THRESHOLDS and reprueba(value, THRESHOLDS[name])
    ]

    report = {
        "scores": scores,
        "thresholds": THRESHOLDS,
        "passed": not failures,
        "cases": [
            {**trace, "scores": per_case[i] if i < len(per_case) else {}}
            for i, trace in enumerate(traces)
        ],
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # El detalle va a stdout, no al artifact solamente: si el artifact no se
    # puede descargar, el log tiene que bastar para diagnosticar la corrida.
    print("\n--- Detalle por caso ---")
    for i, trace in enumerate(traces):
        case = per_case[i] if i < len(per_case) else {}
        # `f"{float('nan'):.3f}"` ya imprime "nan": no hace falta un caso aparte.
        marcas = " ".join(
            f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in sorted(case.items())
        )
        print(
            f"\n[{i + 1}] recuperados={trace['retrieved']} usados={trace['used']} "
            f"cuarentena={trace['quarantined']} stop={trace['stop_reason']}"
        )
        print(f"    P: {trace['question'][:96]}")
        print(f"    R: {trace['answer'][:96]}")
        print(f"    {marcas}")

    print("\n--- Resultados ---")
    for name, value in sorted(scores.items()):
        threshold = THRESHOLDS.get(name)
        if threshold is None:
            print(f"{name:20s} {value:.3f}")
            continue
        if math.isnan(value):
            # Distinto de "bajo el umbral": acá no hay medición que comparar.
            mark = " SIN MEDIR"
        else:
            mark = " OK" if value >= threshold else " FALLA"
        print(f"{name:20s} {value:.3f} (mínimo {threshold:.2f}){mark}")

    if failures:
        sin_medir = [n for n, v, _ in failures if math.isnan(v)]
        print(f"\n{len(failures)} métrica(s) no aprobaron. Build rechazado.")
        if sin_medir:
            print(
                f"  {len(sin_medir)} sin medir ({', '.join(sin_medir)}): el juez no "
                "devolvió puntaje. Revisa el detalle por caso de más arriba."
            )
        return 1

    print("\nTodas las métricas sobre el umbral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
