"""Fine-tuning de un encoder chico como juez de relevancia (cross-encoder).

Entrada: los pares de `slm/data/{train,valid}.jsonl` que produce
`slm/build_dataset.py`. Cada par entra como los dos segmentos del tokenizer
(`tokenizer(pregunta, fragmento)`; con e5-small, que usa el tokenizer de
XLM-R, eso es `<s> pregunta </s></s> fragmento </s>`) y un cabezal de
clasificación binaria decide si el fragmento es relevante. Es un cross-encoder
y no un bi-encoder a propósito: el juez ve pregunta y fragmento juntos, que es
lo que un modelo grande hace cuando califica, y lo que una similitud de
embeddings no puede hacer (la recuperación ya usó embeddings; el juez existe
para corregirla).

Decisiones:

- **Modelo por defecto `intfloat/multilingual-e5-small`**: ya es el modelo de
  embeddings del repo (ragb/config.py), así que quien despliega no descarga
  una familia nueva; 118M parámetros, cabe en CPU. Se cambia con `--model`.
  El cabezal de clasificación arranca aleatorio: por eso se fija la semilla.
- **Pérdida ponderada por clase**: el dataset tiene ~12 negativos por
  positivo. Sin ponderar, "todo es irrelevante" da 92% de accuracy y un juez
  inútil. El peso del positivo es `n_neg / n_pos` del split de entrenamiento.
- **Las métricas se calculan aparte, con umbral 0.5 sobre la probabilidad de
  relevante**, el mismo umbral por defecto de `ragb.grade_slm`. Lo que se mide
  es lo que se despliega. Se informa además el mejor umbral sobre valid, solo
  como dato: elegirlo con el mismo valid que lo mide sería hacer trampa.
- **`--dry-run` no importa torch ni transformers**: valida el dataset (campos,
  etiquetas, fuga entre splits, clases presentes, cuarentena) y la
  configuración, e imprime lo que haría. Sirve para CI y para el Mini sin GPU.
- **La guarda LLM01 se aplica también aquí, no solo en `build_dataset`**: cada
  fragmento pasa por `ragb.guards.detect_injection` antes de entrenar. Los
  archivos versionados ya vienen limpios, pero `--data-dir` acepta cualquier
  carpeta, y un dataset ajeno con una inyección etiquetada como positivo
  entrenaría al juez a aprobar el ataque. La regla "la cuarentena no entra ni
  como negativo" tiene que cumplirse en el consumidor, no solo en el productor.
- **Los imports pesados son perezosos** y si faltan el mensaje dice qué
  instalar, en vez de un ImportError de diez líneas.
- **Sin prefijos `query:`/`passage:`** aunque e5 se preentrenó con ellos
  (ragb/embeddings.py los pone). Acá el cabezal es nuevo y se ajustan todos
  los pesos sobre pares: el prefijo servía para que un embedding sin ajustar
  distinguiera consulta de pasaje, y con fine-tuning completo esa señal la
  aportan los segmentos del tokenizer. No se midió la variante con prefijos;
  es una prueba pendiente, no una decisión cerrada.

Salida: `slm/out/<nombre>/` con el modelo y el tokenizer (`save_pretrained`),
`grader_config.json` (formato de par, `max_length` y `chunk_chars` del
dataset, que `ragb.grade_slm` exige para tokenizar IGUAL que en
entrenamiento y para negarse a calificar fragmentos de otro tamaño) y
`metrics.json`.

Uso:
    python -m slm.train_grader --dry-run
    python -m slm.train_grader                      # defaults: 12 épocas, lote 8, lr 5e-5
    python -m slm.train_grader --model distilbert-base-multilingual-cased
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Como script (`python slm/train_grader.py`) sys.path[0] es slm/ y `ragb` no
# se encuentra; como módulo sí. Igual que en build_dataset.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ragb.guards import detect_injection  # noqa: E402

DATA_DIR = REPO_ROOT / "slm" / "data"
OUT_DIR = REPO_ROOT / "slm" / "out"

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
# Mismo valor que ragb/grade_slm.DEFAULT_THRESHOLD (un test lo afirma); se
# guarda en grader_config.json y el juez lo usa si SLM_GRADER_THRESHOLD falta.
DEFAULT_THRESHOLD = 0.5
REQUIRED_FIELDS = {"question", "chunk_text", "label"}

# Formato del par. Se guarda en grader_config.json para que el juez en
# producción tokenice exactamente igual; cambiarlo acá sin reentrenar no
# tiene sentido y el archivo de config es lo que lo hace visible.
PAIR_FORMAT = "plain"  # tokenizer(question, chunk_text) sin prefijos


# --- Dataset ---------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"no existe {path}; genera el dataset con `python -m slm.build_dataset`")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n} no es JSON válido: {exc}") from exc
    return rows


def load_chunk_chars(data_dir: Path) -> int:
    """El tamaño de fragmento con que se construyó el dataset, del manifest.
    Va a grader_config.json: el juez en producción se niega a calificar
    fragmentos de otro tamaño, y para eso tiene que saber cuál fue."""
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no existe {manifest}; genera el dataset con `python -m slm.build_dataset`")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if "chunk_chars" not in data:
        raise ValueError(f"{manifest} sin `chunk_chars`: regenera el dataset")
    return int(data["chunk_chars"])


def validate_dataset(train: list[dict], valid: list[dict]) -> dict:
    """Falla temprano y con nombre: entrenar sobre un dataset roto cuesta
    minutos de CPU y devuelve métricas que parecen reales."""
    for name, rows in (("train", train), ("valid", valid)):
        if not rows:
            raise ValueError(f"el split {name} está vacío")
        for i, r in enumerate(rows):
            missing = REQUIRED_FIELDS - set(r)
            if missing:
                raise ValueError(f"{name}[{i}] sin campos {sorted(missing)}")
            if r["label"] not in (0, 1):
                raise ValueError(f"{name}[{i}] label={r['label']!r}, se esperaba 0 o 1")
            if not str(r["question"]).strip() or not str(r["chunk_text"]).strip():
                raise ValueError(f"{name}[{i}] con pregunta o fragmento vacío")
            verdict = detect_injection(str(r["chunk_text"]))
            if verdict.flagged:
                # No se muestra el fragmento: el mensaje de error va a logs y
                # el texto marcado es, por definición, el que no queremos
                # propagar. La fila y el motivo bastan para encontrarlo.
                raise ValueError(
                    f"{name}[{i}] tiene un fragmento marcado por la guarda de inyección "
                    f"({verdict.reason}); la cuarentena no entra ni como negativo"
                )
        labels = {r["label"] for r in rows}
        if labels != {0, 1}:
            raise ValueError(f"el split {name} tiene una sola clase: {sorted(labels)}")

    # Fuga: un par idéntico en los dos splits infla valid sin que se note.
    pares_train = {(r["question"], r["chunk_text"]) for r in train}
    fugados = sum(1 for r in valid if (r["question"], r["chunk_text"]) in pares_train)
    if fugados:
        raise ValueError(f"{fugados} pares de valid también están en train")

    n_pos = sum(r["label"] for r in train)
    n_neg = len(train) - n_pos
    return {
        "train_rows": len(train),
        "train_positives": n_pos,
        "train_negatives": n_neg,
        "valid_rows": len(valid),
        "valid_positives": sum(r["label"] for r in valid),
        "valid_negatives": sum(1 - r["label"] for r in valid),
        "positive_class_weight": round(n_neg / n_pos, 4),
    }


# --- Métricas ---------------------------------------------------------------------


def binary_metrics(labels: list[int], probs: list[float], threshold: float) -> dict:
    """Accuracy, precision, recall y F1 con divisiones por cero definidas en 0.
    Escrito a mano para no arrastrar scikit-learn por cuatro fórmulas."""
    tp = fp = tn = fn = 0
    for y, p in zip(labels, probs):
        pred = 1 if p >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "n": total,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def best_threshold(labels: list[int], probs: list[float]) -> dict:
    """Informativo. Barre 0.05..0.95 y devuelve el umbral con mejor F1."""
    best = None
    for k in range(1, 20):
        t = k / 20
        m = binary_metrics(labels, probs, t)
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


# --- Entrenamiento ---------------------------------------------------------------


def _import_stack():
    try:
        import torch
        import transformers
    except ImportError as exc:  # mensaje accionable en vez de traceback
        raise SystemExit(
            f"falta {exc.name}: instala el extra opcional con "
            "`pip install -r slm/REQUIREMENTS.txt` (torch CPU: "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`)"
        ) from exc
    return torch, transformers


def _predict_probs(model, tokenizer, rows: list[dict], *, max_length: int, batch_size: int, torch) -> list[float]:
    """Probabilidad de la clase 1 por fila, sin gradientes, por lotes."""
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            enc = tokenizer(
                [r["question"] for r in batch],
                [r["chunk_text"] for r in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            out.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return out


def train(args: argparse.Namespace) -> dict:
    torch, transformers = _import_stack()
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    valid_rows = load_jsonl(args.data_dir / "valid.jsonl")
    summary = validate_dataset(train_rows, valid_rows)
    chunk_chars = load_chunk_chars(args.data_dir)

    transformers.set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict]):
            self.enc = tokenizer(
                [r["question"] for r in rows],
                [r["chunk_text"] for r in rows],
                truncation=True,
                max_length=args.max_length,
            )
            self.labels = [int(r["label"]) for r in rows]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    class_weights = torch.tensor([1.0, float(summary["positive_class_weight"])])

    class WeightedTrainer(Trainer):
        # `num_items_in_batch` llega en transformers >= 4.46; se acepta y se
        # ignora para no romper con versiones anteriores.
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=Pairs(train_rows),
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
    )
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    predict = lambda rows: _predict_probs(  # noqa: E731
        model, tokenizer, rows, max_length=args.max_length, batch_size=args.batch_size, torch=torch
    )
    valid_probs = predict(valid_rows)
    train_probs = predict(train_rows)
    valid_labels = [int(r["label"]) for r in valid_rows]
    train_labels = [int(r["label"]) for r in train_rows]

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "name": args.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "dataset": summary,
        "train_loss": round(float(train_result.training_loss), 4),
        "train_seconds": round(elapsed, 1),
        "valid": binary_metrics(valid_labels, valid_probs, DEFAULT_THRESHOLD),
        "train": binary_metrics(train_labels, train_probs, DEFAULT_THRESHOLD),
        "valid_best_threshold_informativo": best_threshold(valid_labels, valid_probs),
        "valid_probs": [round(p, 4) for p in valid_probs],
    }

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "grader_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "pair_format": PAIR_FORMAT,
                "max_length": args.max_length,
                "chunk_chars": chunk_chars,
                "default_threshold": DEFAULT_THRESHOLD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


# --- CLI --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el juez de relevancia SLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--name", default=None, help="carpeta bajo --out (default: último segmento del modelo)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    # Defaults medidos, no adivinados (las cifras están en slm/RESULTADOS.md,
    # que se regenera con cada corrida): con 234 filas, 3 épocas a lote 16 son
    # 45 pasos y el cabezal aleatorio no aprende nada (F1 0.0 en valid, pérdida
    # 0.67 ~ ln 2). Con 12 épocas a lote 8 son 360 pasos, ~3.5 min en 8 núcleos
    # de CPU, y el cabezal aprende. Si el dataset crece, bajar épocas antes que
    # subir el lote: lo que importa son los pasos.
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="valida dataset y config; no descarga ni entrena")
    args = parser.parse_args(argv)
    if args.name is None:
        args.name = args.model.rstrip("/").split("/")[-1]
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("epochs y batch-size deben ser >= 1; max-length >= 8")
    if not args.lr > 0:  # `not >` y no `<=` para que NaN también caiga
        parser.error(
            f"lr debe ser > 0 (recibido {args.lr}); un learning rate <= 0 no entrena, y TrainingArguments no lo rechaza"
        )
    return args


def dry_run(args: argparse.Namespace) -> dict:
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    valid_rows = load_jsonl(args.data_dir / "valid.jsonl")
    summary = validate_dataset(train_rows, valid_rows)
    steps = -(-len(train_rows) // args.batch_size) * args.epochs
    return {
        "dry_run": True,
        "model": args.model,
        "out": str(args.out / args.name),
        "chunk_chars": load_chunk_chars(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "optimizer_steps": steps,
        "dataset": summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = dry_run(args) if args.dry_run else train(args)
    printable = {k: v for k, v in result.items() if k != "valid_probs"}
    json.dump(printable, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
