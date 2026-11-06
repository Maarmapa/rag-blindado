"""Juez de relevancia con un modelo pequeño propio, para `graph_flow`.

Implementa `Grader = Callable[[str, list[Chunk]], list[bool]]`: un veredicto
por fragmento, en el mismo orden. Se enchufa así:

    ops = graph_flow.default_ops(perms)
    ops["grade"] = grade_slm.grade
    graph = graph_flow.build_graph(**ops)

Configuración por entorno (ningún path ni umbral vive en el código):

    SLM_GRADER_PATH        carpeta que dejó `slm/train_grader.py`. Si es
                           relativa se resuelve contra la raíz del repo, no
                           contra el cwd del proceso: un servicio arrancado
                           desde otra carpeta no debe perder al juez.
    SLM_GRADER_THRESHOLD   probabilidad mínima para "relevante"; si falta, la
                           que dejó el entrenamiento en grader_config.json

Decisiones:

- **Si el modelo no está o falla, LANZA.** No devuelve "todo relevante" ni
  "nada relevante": cualquiera de las dos sería un veredicto inventado.
  `graph_flow.node_grade` atrapa la excepción, pasa los fragmentos sin
  calificar y marca `degraded=True` con el error en la traza. El juez no
  miente; el grafo degrada y lo dice.
- **Un puntaje que no es un número también lanza.** `nan` comparado con
  cualquier umbral da False, así que pesos corruptos o un desborde numérico
  se verían como "nada relevante" y el grafo respondería NO_ENCONTRADO sin
  marcar `degraded`. Un puntaje fuera de [0, 1] o no finito es un modelo
  roto, y eso es una excepción, no un veredicto.
- **`grader_config.json` es obligatorio y se valida por tipo y rango.** Solo
  lo escribe `train_grader.py`, así que su ausencia significa "esta carpeta
  no es un juez entrenado": una carpeta con el encoder base sin cabezal pasa
  la comprobación de `config.json` y transformers le inventa un cabezal
  aleatorio que emite probabilidades ~0.5 sin lanzar. Ese archivo, más la
  comprobación de que el cabezal cargó de verdad y tiene dos clases, es lo
  que impide servir un juez sin entrenar. Y como la guarda de tamaño de
  fragmento se calcula desde `chunk_chars`, un valor absurdo (0, negativo,
  texto, o 10**9) la anularía en silencio: se exigen enteros positivos,
  `chunk_chars` acotado a `MAX_CHUNK_CHARS` y `max_length` a `MAX_MAX_LENGTH`.
- **La tokenización es la del entrenamiento**: `grader_config.json` dice
  `max_length` y `pair_format`; si el formato no es el que este módulo sabe
  producir, se lanza en vez de tokenizar distinto en silencio.
- **El tamaño de fragmento también es del entrenamiento**: el juez se entrenó
  con fragmentos de `chunk_chars` caracteres (320 en el dataset del repo;
  producción fragmenta a 900 por defecto). Un fragmento bastante más largo que
  eso es un régimen que el juez nunca vio, y calificarlo sería inventar. Se
  lanza con el número, para que el operador re-indexe con `CHUNK_CHARS` igual
  al del entrenamiento o reentrene con el tamaño de producción.
- **El umbral se lee en cada llamada**, no al importar: así un test o un
  operador lo cambia sin reiniciar, y un valor inválido falla en la llamada
  que lo usa, no en un import lejano. 0 y 1 son inválidos: 0 aprueba todo y
  1 rechaza todo, y ninguno de los dos es un juez.
- **Ruta y config se resuelven UNA vez por llamada** (`_resolve`) y se pasan
  hacia abajo: antes `grade` y `score` leían la variable y parseaban el JSON
  cada uno, dos accesos a disco por ronda sin motivo.
- **El modelo se cachea por ruta absoluta, y el caché no se invalida solo.**
  Reentrenar en la misma carpeta NO refresca un proceso que ya lo cargó: hay
  que reiniciarlo. Es a propósito: mirar el mtime de 470 MB de pesos en cada
  llamada, o recargar a mitad de una ronda, son peores que un reinicio
  explícito.
- **torch y transformers se importan perezosamente** dentro de `_load`, para
  que este módulo se importe y se pruebe en CI sin el stack de ML.
- **Sin lista vacía no se carga nada**: cero fragmentos devuelve `[]` sin
  tocar el modelo.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ENV_PATH = "SLM_GRADER_PATH"
ENV_THRESHOLD = "SLM_GRADER_THRESHOLD"
# Mismo valor que slm/train_grader.DEFAULT_THRESHOLD; tests/test_grade_slm.py
# lo afirma para que no se desalineen. Solo se usa si grader_config.json no
# trae `default_threshold`.
DEFAULT_THRESHOLD = 0.5
CONFIG_FILE = "grader_config.json"
PAIR_FORMAT = "plain"  # tokenizer(question, chunk_text) sin prefijos, como en train_grader
# Un fragmento más largo que esto x chunk_chars del entrenamiento está fuera
# del régimen: con CHUNK_CHARS=320 el corte queda en 480; los fragmentos de
# producción a 900 miden 529-845 y todos lo superan.
MAX_CHUNK_RATIO = 1.5
# Techo de `chunk_chars` en grader_config.json. Ningún encoder chico ve más
# de ~512 tokens; 4096 caracteres ya es más del doble de eso en español. Un
# valor mayor solo puede ser un archivo editado a mano para saltarse la guarda.
MAX_CHUNK_CHARS = 4096
# Techo de `max_length` (tokens). Los encoders de esta familia topan en 512;
# 8192 deja margen para uno más largo sin aceptar un número que nadie entrenó.
MAX_MAX_LENGTH = 8192
BATCH_SIZE = 16
REPO_ROOT = Path(__file__).resolve().parent.parent

_cache: dict[str, tuple] = {}


def _model_path() -> Path:
    raw = os.getenv(ENV_PATH, "").strip()
    if not raw:
        raise RuntimeError(f"{ENV_PATH} no configurada: no hay juez SLM que cargar")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_dir() or not (path / "config.json").exists():
        raise RuntimeError(f"{ENV_PATH}={raw} no contiene un modelo guardado (falta config.json en {path})")
    if not (path / CONFIG_FILE).exists():
        raise RuntimeError(
            f"{ENV_PATH}={raw} no es un juez entrenado: falta {CONFIG_FILE} (solo lo escribe slm/train_grader.py)"
        )
    return path


def _positive_int(cfg: dict, key: str, *, maximum: int | None = None) -> int:
    value = cfg[key]
    # bool es subclase de int; `true` en el JSON no es un tamaño.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"{CONFIG_FILE} `{key}`={value!r} no es un entero positivo: reentrena con slm/train_grader.py"
        )
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{CONFIG_FILE} `{key}`={value} supera el máximo {maximum}: ningún juez se entrenó así")
    return value


def _grader_config(path: Path) -> dict:
    """Lee y valida grader_config.json (presencia, tipo y rango). Sin torch."""
    cfg = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
    for key in ("max_length", "pair_format", "chunk_chars"):
        if key not in cfg:
            raise RuntimeError(f"{CONFIG_FILE} sin `{key}`: reentrena con slm/train_grader.py")
    if cfg["pair_format"] != PAIR_FORMAT:
        raise RuntimeError(f"{CONFIG_FILE} pair_format={cfg['pair_format']!r}; este juez solo tokeniza {PAIR_FORMAT!r}")
    _positive_int(cfg, "max_length", maximum=MAX_MAX_LENGTH)
    _positive_int(cfg, "chunk_chars", maximum=MAX_CHUNK_CHARS)
    return cfg


def _threshold(cfg: dict) -> float:
    raw = os.getenv(ENV_THRESHOLD, "").strip()
    if not raw:
        value = float(cfg.get("default_threshold", DEFAULT_THRESHOLD))
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{ENV_THRESHOLD}={raw!r} no es un número") from exc
    if not 0.0 < value < 1.0:
        raise ValueError(f"{ENV_THRESHOLD}={value} fuera de (0, 1): 0 aprueba todo y 1 rechaza todo")
    return value


def _resolve() -> tuple[Path, dict]:
    """Ruta validada y config validada, una sola vez por llamada."""
    path = _model_path()
    return path, _grader_config(path)


def _check_chunk_lengths(cfg: dict, texts: list[str]) -> None:
    limit = MAX_CHUNK_RATIO * int(cfg["chunk_chars"])
    longest = max(len(t) for t in texts)
    if longest > limit:
        raise RuntimeError(
            f"fragmento de {longest} caracteres; el juez se entrenó con CHUNK_CHARS={cfg['chunk_chars']} "
            f"(máximo aceptado {limit:.0f}). Re-indexa con ese CHUNK_CHARS o reentrena con el de producción"
        )


def _load(path: Path):
    """Carga (y cachea por ruta) tokenizer, modelo y la config de tokenización."""
    key = str(path.resolve())
    if key in _cache:
        return _cache[key]
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"falta {exc.name}: el juez SLM necesita torch y transformers (pip install -r slm/REQUIREMENTS.txt)"
        ) from exc
    cfg = _grader_config(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model, info = AutoModelForSequenceClassification.from_pretrained(str(path), output_loading_info=True)
    # transformers inicializa al azar las claves que faltan y solo lo dice por
    # log. Un cabezal que no vino en los pesos es un juez sin entrenar.
    faltantes = [k for k in info.get("missing_keys", []) if "classifier" in k]
    if faltantes:
        raise RuntimeError(f"{path} no trae el cabezal de clasificación entrenado: faltan {faltantes}")
    if int(model.config.num_labels) != 2:
        raise RuntimeError(f"{path} tiene num_labels={model.config.num_labels}; el juez es binario")
    model.eval()
    loaded = (tokenizer, model, int(cfg["max_length"]), torch)
    _cache[key] = loaded
    return loaded


def _score_pairs(path: Path, question: str, texts: list[str]) -> list[float]:
    """Probabilidad de 'relevante' por fragmento. Es la función que un test
    reemplaza por un doble: todo lo demás (umbral, orden, errores, la config
    del entrenamiento, los largos, los puntajes inválidos) se prueba sin torch."""
    tokenizer, model, max_length, torch = _load(path)
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                [question] * len(batch),
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return scores


def _score_with(path: Path, cfg: dict, question: str, chunks: list[dict], *, raw: bool) -> list[float]:
    texts = [c["text"] for c in chunks]
    if not texts:
        return []
    if not raw:
        _check_chunk_lengths(cfg, texts)
    scores = _score_pairs(path, question, texts)
    if len(scores) != len(texts):
        raise RuntimeError(f"el juez devolvió {len(scores)} puntajes para {len(texts)} fragmentos")
    out: list[float] = []
    for i, s in enumerate(scores):
        value = float(s)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"el juez devolvió un puntaje inválido para el fragmento {i}: {s!r} (no es una probabilidad)"
            )
        out.append(value)
    return out


def describe() -> dict:
    """Lo que un operador o `slm/smoke.py` necesita saber del juez configurado,
    sin cargar pesos: ruta resuelta, config del entrenamiento y umbral vigente."""
    path, cfg = _resolve()
    return {"path": str(path), "config": cfg, "threshold": _threshold(cfg)}


def score(question: str, chunks: list[dict], *, raw: bool = False) -> list[float]:
    """Puntajes crudos, mismo orden que `chunks`. Útil para calibrar el umbral.

    `raw=True` salta la guarda de tamaño de fragmento. Existe SOLO para medir
    cuánto se equivocaría el juez fuera de su régimen (`slm/smoke.py`); `grade`
    nunca lo usa, así que producción sigue lanzando."""
    path, cfg = _resolve()
    return _score_with(path, cfg, question, chunks, raw=raw)


def grade(question: str, chunks: list[dict]) -> list[bool]:
    """Grader para `graph_flow`: True si la probabilidad de relevante alcanza
    el umbral. Lanza si no hay modelo configurado o si algo falla."""
    path, cfg = _resolve()
    threshold = _threshold(cfg)
    return [s >= threshold for s in _score_with(path, cfg, question, chunks, raw=False)]
