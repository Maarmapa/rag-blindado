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
import re
import sys

import yaml

# `ragb.pipeline` se importa dentro de `collect`, no acá: arrastra
# sentence-transformers y psycopg. Importándolo perezosamente, este módulo se
# puede cargar con solo pytest y pyyaml instalados, y así el job `guards` —que
# corre en TODO push, sin credenciales y sin costo— puede testear la capa
# determinista. Sin esto, lo único que la probaba eran dobles en una sesión.
from ragb.config import settings
from ragb.guards import Permissions

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
    from ragb.pipeline import query

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
                "expects_refusal": bool(case.get("expects_refusal")),
                "forbidden": list(case.get("forbidden") or []),
                # Documentos que entraron al prompt. Toda cita de la respuesta
                # tiene que resolver a uno de estos.
                "sources": list(result["sources"]),
            }
        )
        print(
            f"  · [{result['retrieved']}→{result['used']}] {case['question'][:56]}",
            file=sys.stderr,
        )
    return rows, traces


def extraer_traza_juez(report, metrica: str) -> list[dict]:
    """Rescata el razonamiento interno del juez, que Ragas calcula y descarta.

    Para `faithfulness`, Ragas hace dos llamadas por caso: primero parte la
    respuesta en afirmaciones atómicas, después pregunta una por una si el
    contexto las sostiene. El puntaje es la razón entre aprobadas y totales —de
    ahí que un 0.667 sea "2 de 3"— pero cuál fue la tercera no se guarda en
    ninguna parte, y sin eso un umbral que no se alcanza es indistinguible de
    un juez que se equivoca.

    Cuesta cero llamadas extra: el dato ya se produjo. Solo se estaba tirando.

    Se guarda únicamente la SALIDA de cada llamada. La entrada es el prompt
    completo con todos los contextos, que ya está en el reporte por otro lado
    y multiplicaría el tamaño del artifact sin agregar nada.
    """
    salida: list[dict] = []
    for traza_caso in getattr(report, "traces", []) or []:
        por_prompt = (traza_caso or {}).get(metrica, {}) or {}
        salida.append(
            {
                nombre: _serializable(datos.get("output"))
                for nombre, datos in por_prompt.items()
            }
        )
    return salida


def _serializable(valor):
    """Los objetos de Ragas son modelos Pydantic; el reporte es JSON."""
    if hasattr(valor, "model_dump"):
        return valor.model_dump()
    if isinstance(valor, dict):
        return {k: _serializable(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_serializable(v) for v in valor]
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    return str(valor)


def score(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Puntúa con Ragas. Claude actúa como juez de las métricas.

    Devuelve (puntaje por caso, traza del juez de faithfulness). El promedio no
    se calcula acá: eso lo hace `agregar`, que sabe qué casos entran en qué
    métrica. El detalle por caso es además lo que permite distinguir "el
    pipeline respondió mal" de "el juez no pudo calificar" — en un promedio las
    dos cosas se ven igual.
    """
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
    )
    from ragas.cost import get_token_usage_for_anthropic
    from ragas.run_config import RunConfig

    def construir_juez(model: str):
        # max_tokens holgado a propósito: si el modelo razona antes de
        # responder, ese presupuesto sale del mismo tope, y un juez truncado no
        # devuelve un puntaje bajo — devuelve `nan`. Los tokens no gastados no
        # se cobran, así que el margen es gratis.
        return LangchainLLMWrapper(
            ChatAnthropic(
                model=model,
                api_key=settings.require_anthropic(),
                max_tokens=4096,
            )
        )

    judge = construir_juez(settings.judge_model)
    faithfulness_model = settings.faithfulness_judge_model or settings.judge_model
    judge_faithfulness = (
        judge if faithfulness_model == settings.judge_model
        else construir_juez(faithfulness_model)
    )
    print(
        f"Jueces: {faithfulness_model} para faithfulness · "
        f"{settings.judge_model} para el resto",
        file=sys.stderr,
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
            Faithfulness(llm=judge_faithfulness),
            AnswerRelevancy(llm=judge, embeddings=judge_embeddings),
            ContextPrecision(llm=judge),
        ],
        run_config=run_config,
        # Contabiliza los tokens del juez. Sin esto, `total_tokens()` no tiene
        # de dónde sacarlos y el costo del gate queda invisible.
        token_usage_parser=get_token_usage_for_anthropic,
    )

    # Tokens, no dólares: los tokens son un hecho de la corrida, los precios
    # cambian y un precio hardcodeado desactualizado miente con más
    # convicción que no poner nada.
    try:
        uso = report.total_tokens()
        usos = uso if isinstance(uso, list) else [uso]
        for u in usos:
            print(
                f"Juez ({u.model or 'sin modelo'}): {u.input_tokens:,} tokens de "
                f"entrada, {u.output_tokens:,} de salida",
                file=sys.stderr,
            )
    except Exception as e:  # noqa: BLE001
        # El conteo es informativo: si falla, no puede tumbar la evaluación.
        print(f"[evals] no se pudo contabilizar tokens: {e}", file=sys.stderr)

    try:
        traza = extraer_traza_juez(report, "faithfulness")
    except Exception as e:  # noqa: BLE001
        # Diagnóstico, no resultado: si falla, no puede tumbar la evaluación.
        print(f"[evals] no se pudo extraer la traza del juez: {e}", file=sys.stderr)
        traza = []

    return [dict(s) for s in report.scores], traza


# Frase exacta que el system prompt de generate.py obliga a usar cuando el
# contexto no alcanza, y que `generate.answer` devuelve cuando no hay contexto.
REFUSAL_MARKER = "No encuentro esa información en los documentos disponibles"

# Métricas que no se le pueden exigir a un caso cuya respuesta correcta es
# negarse. Las tres, y cada una por su motivo:
#
#   answer_relevancy   Ragas puntúa cerca de cero toda respuesta evasiva.
#   context_precision  no hay contexto relevante que recuperar: esa es la
#                      premisa del caso.
#   faithfulness       es una razón de afirmaciones sostenidas sobre
#                      afirmaciones totales, y una negativa no contiene
#                      afirmaciones que anclar al contexto. Medido: al acortar
#                      la negativa a su frase canónica, el caso pasó de 0.545
#                      a 0.000. No mejoró ni empeoró la respuesta; simplemente
#                      no hay nada que la métrica pueda calificar.
#
# Estos casos no quedan sin verificar: los cubre `verificar_aserciones`, que es
# más estricta que las tres métricas juntas para lo que se les pide.
SKIP_ON_REFUSAL = ("answer_relevancy", "context_precision", "faithfulness")


# Formato de cita que el system prompt de generate.py obliga a usar.
CITA = re.compile(r"\[fuente:\s*([^\]]+?)\s*\]", re.I)


def verificar_aserciones(traces: list[dict]) -> list[str]:
    """Comprobaciones deterministas, sin juez y sin costo.

    Esta es la capa que conviene que bloquee. Las tres métricas de Ragas las
    calcula un modelo, y un modelo con un mal día mueve el puntaje de una
    respuesta corta en un tercio —pasó, y costó cinco corridas averiguarlo—.
    Lo de acá no puntúa: verifica propiedades, en binario, y da el mismo
    resultado siempre.

    Nada de umbrales inventados: cada comprobación es una propiedad que el
    repositorio ya declara y que antes solo estaba escrita en prosa.
    """
    from ragb.guards import redact_secrets

    fallas: list[str] = []
    for i, trace in enumerate(traces, start=1):
        answer = trace["answer"]

        if trace["expects_refusal"] and REFUSAL_MARKER not in answer:
            fallas.append(
                f"[{i}] debía declarar que no encuentra el dato y respondió: "
                f"{answer[:80]!r}"
            )

        for needle in trace["forbidden"]:
            if needle.lower() in answer.lower():
                fallas.append(f"[{i}] la respuesta contiene texto prohibido: {needle!r}")

        _, secrets = redact_secrets(answer)
        if secrets:
            fallas.append(f"[{i}] la respuesta contiene credenciales: {', '.join(secrets)}")

        # --- Disciplina de citas -------------------------------------------
        # El README promete trazabilidad "que permita reconstruir qué documentos
        # sustentaron cada respuesta". Sin esto, esa promesa no se verifica.
        citadas = {c.strip() for c in CITA.findall(answer)}
        disponibles = set(trace["sources"])

        if not trace["expects_refusal"] and not citadas:
            fallas.append(
                f"[{i}] la respuesta no cita ninguna fuente: {answer[:80]!r}"
            )

        # Una cita a un documento que no entró al prompt es peor que no citar:
        # fabrica procedencia. El modelo no puede saber de dónde salió un dato
        # que no recibió.
        inventadas = citadas - disponibles
        if inventadas:
            fallas.append(
                f"[{i}] cita fuentes que no entraron al contexto: "
                f"{', '.join(sorted(inventadas))} "
                f"(disponibles: {', '.join(sorted(disponibles)) or 'ninguna'})"
            )

    return fallas


def agregar(
    per_case: list[dict], traces: list[dict]
) -> tuple[dict[str, float], dict[str, tuple[int, int]]]:
    """Promedia cada métrica sobre los casos a los que sí les corresponde.

    Se agrega acá en vez de usar el promedio de Ragas porque los casos con
    `expects_refusal` quedan fuera de dos métricas (ver SKIP_ON_REFUSAL). Un
    `nan` suelto se ignora —es un caso que el juez no pudo puntuar, no un
    cero—; si NINGÚN caso puntuó, la métrica queda en `nan` y `reprueba` la
    trata como falla.

    Devuelve además la COBERTURA por métrica: (casos puntuados, casos que le
    correspondían). Ignorar los `nan` en silencio deja un promedio que se ve
    idéntico calculado sobre 7 casos o sobre 3, y eso es el mismo agujero que
    el `nan` que aprobaba el gate, solo que un nivel más abajo. Acá se ignora,
    pero se dice.
    """
    scores: dict[str, float] = {}
    cobertura: dict[str, tuple[int, int]] = {}
    nombres = {k for case in per_case for k in case}

    for name in sorted(nombres):
        aplicables = [
            case.get(name)
            for case, trace in zip(per_case, traces)
            if not (trace["expects_refusal"] and name in SKIP_ON_REFUSAL)
        ]
        valores = [
            v
            for v in aplicables
            if isinstance(v, (int, float)) and not math.isnan(v)
        ]
        scores[name] = sum(valores) / len(valores) if valores else math.nan
        cobertura[name] = (len(valores), len(aplicables))

    return scores, cobertura


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
    per_case, traza_juez = score(rows)

    # Agregación propia: los casos negativos salen de las métricas que no
    # saben calificarlos y se verifican con `verificar_aserciones`.
    scores, cobertura = agregar(per_case, traces)
    aserciones = verificar_aserciones(traces)

    failures = [
        (name, value, THRESHOLDS[name])
        for name, value in scores.items()
        if name in THRESHOLDS and reprueba(value, THRESHOLDS[name])
    ]

    report = {
        "scores": scores,
        "coverage": {k: {"scored": s, "applicable": a} for k, (s, a) in cobertura.items()},
        "thresholds": THRESHOLDS,
        "passed": not failures and not aserciones,
        "assertion_failures": aserciones,
        "cases": [
            {
                **trace,
                "scores": per_case[i] if i < len(per_case) else {},
                # Afirmaciones que el juez extrajo y su veredicto sobre cada
                # una. Es lo que convierte un 0.667 de dato opaco en algo
                # revisable: sin esto, "2 de 3" no dice cuál fue la tercera.
                "faithfulness_trace": traza_juez[i] if i < len(traza_juez) else {},
            }
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
        etiqueta = " (control negativo)" if trace["expects_refusal"] else ""
        print(
            f"\n[{i + 1}]{etiqueta} recuperados={trace['retrieved']} "
            f"usados={trace['used']} cuarentena={trace['quarantined']} "
            f"stop={trace['stop_reason']}"
        )
        print(f"    P: {trace['question'][:96]}")
        print(f"    R: {trace['answer'][:96]}")
        print(f"    {marcas}")

        # El razonamiento del juez de fidelidad, para poder revisar un 0.667 en
        # vez de especular con él. Va al log además del artifact: el artifact no
        # siempre se puede descargar.
        for nombre, salida in (
            traza_juez[i] if i < len(traza_juez) else {}
        ).items():
            texto = json.dumps(salida, ensure_ascii=False)
            print(f"    juez·{nombre}: {texto[:400]}")

    print("\n--- Resultados ---")
    parciales: list[str] = []
    for name, value in sorted(scores.items()):
        threshold = THRESHOLDS.get(name)
        puntuados, aplicables = cobertura.get(name, (0, 0))
        # La cobertura va SIEMPRE, no solo cuando falta algo: un promedio no
        # dice sobre cuántos casos se calculó, y 1.000 sobre 4 de 7 no es lo
        # mismo que 1.000 sobre 7 de 7.
        alcance = f"  [{puntuados}/{aplicables} casos]"
        if puntuados < aplicables:
            alcance += " ⚠"
            parciales.append(f"{name} ({aplicables - puntuados} sin puntuar)")

        if threshold is None:
            print(f"{name:20s} {value:.3f}{alcance}")
            continue
        if math.isnan(value):
            # Distinto de "bajo el umbral": acá no hay medición que comparar.
            mark = " SIN MEDIR"
        else:
            mark = " OK" if value >= threshold else " FALLA"
        print(f"{name:20s} {value:.3f} (mínimo {threshold:.2f}){mark}{alcance}")

    if parciales:
        print(
            "\n⚠ Promedios calculados sobre menos casos de los que correspondían: "
            + "; ".join(parciales)
            + ".\n  El juez no devolvió puntaje en esos casos. El promedio los "
            "ignora, así que es menos representativo de lo que aparenta."
        )

    print("\n--- Aserciones deterministas ---")
    if aserciones:
        for falla in aserciones:
            print(f"  FALLA {falla}")
    else:
        negativos = sum(1 for t in traces if t["expects_refusal"])
        prohibidos = sum(1 for t in traces if t["forbidden"])
        print(
            f"  OK · {negativos} control(es) negativo(s) se negaron correctamente · "
            f"{prohibidos} caso(s) sin texto prohibido · sin credenciales en ninguna "
            "respuesta"
        )

    if failures or aserciones:
        if failures:
            sin_medir = [n for n, v, _ in failures if math.isnan(v)]
            print(f"\n{len(failures)} métrica(s) no aprobaron. Build rechazado.")
            if sin_medir:
                print(
                    f"  {len(sin_medir)} sin medir ({', '.join(sin_medir)}): el juez "
                    "no devolvió puntaje. Revisa el detalle por caso de más arriba."
                )
        if aserciones:
            print(f"\n{len(aserciones)} aserción(es) de seguridad fallaron.")
        return 1

    print("\nTodas las métricas sobre el umbral y las aserciones en verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
