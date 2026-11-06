# `slm/` — un modelo pequeño propio como juez de relevancia del ciclo CRAG

## Qué es

El ciclo CRAG de `ragb/graph_flow.py` califica cada fragmento recuperado
—¿sirve para responder esta pregunta, sí o no?— antes de generar. Ese juez es
un `Grader = Callable[[str, list[dict]], list[bool]]`. En esta rama
`default_ops` lo cablea a `ragb.grade` (el juez del modelo grande de la rama
CRAG), que **no existe aquí**: sin enchufar nada, el grafo degrada en cada
ronda y pasa los fragmentos sin calificar. Esta carpeta pone en ese lugar un
**cross-encoder chico entrenado sobre el propio corpus**: pregunta y fragmento
entran como los dos segmentos del tokenizer (`tokenizer(pregunta, fragmento)`;
con `multilingual-e5-small`, que usa el tokenizer de XLM-R, eso es
`<s> pregunta </s></s> fragmento </s>`) y un cabezal binario decide.

| archivo | qué hace | necesita torch |
|---|---|---|
| `slm/build_dataset.py` | pares (pregunta, fragmento, etiqueta) deterministas desde `corpus/` y `evals/dataset.yaml`, sin modelo | no |
| `slm/train_grader.py` | fine-tuning con `transformers.Trainer`, apto para CPU; `--dry-run` valida sin descargar nada | sí (salvo `--dry-run`) |
| `ragb/grade_slm.py` | el `Grader` que carga el modelo entrenado y devuelve `list[bool]` | sí, perezoso |
| `slm/smoke.py` | humo del juez entrenado sobre las preguntas doradas, medido a nivel fragmento (cobertura de la referencia, ítems cubiertos) | sí |
| `slm/data/` | el dataset versionado (`train.jsonl`, `valid.jsonl`, `manifest.json`) | — |
| `slm/out/<nombre>/` | `metrics.json` y `grader_config.json` de cada corrida; los pesos no se versionan | — |

Los tests (`tests/test_slm_dataset.py`, `tests/test_grade_slm.py`,
`tests/test_train_grader.py`) corren sin torch, sin base y sin red: son parte
del job `guards` de CI. Dos de ellos existen para atrapar deriva:
`test_los_archivos_versionados_son_los_que_produce_el_codigo` ata
`slm/data` al código, y `test_metrics_versionadas_coinciden_con_el_manifest`
ata `slm/out/*/metrics.json` a `slm/data`. Regenerar el dataset sin reentrenar
deja el segundo en rojo: ya pasó una vez, y las métricas viejas se exhibieron
como si fueran del dataset nuevo.

## Por qué un SLM como juez

- **Costo.** El juez se llama una vez por fragmento por ronda: con `top_k=5` y
  dos rondas son hasta diez llamadas al modelo grande por pregunta, más que la
  generación misma. Un encoder de 118M parámetros califica los diez en
  milisegundos en CPU y no cuesta tokens.
- **Latencia.** Sin viaje de red por fragmento: la calificación deja de ser el
  paso lento del ciclo.
- **Los datos no salen.** Pregunta y fragmentos se califican en la misma
  máquina que los indexó. Para un corpus con documentos internos, el juez es
  la única llamada del ciclo que ve TODOS los fragmentos recuperados, incluidos
  los que después se descartan; que no salga de la máquina importa más ahí que
  en la generación (LLM02).
- **Fail-closed sin mentir.** Si el modelo no está, el fragmento es más largo
  que lo que el juez vio en entrenamiento, la config está corrupta o el
  puntaje no es una probabilidad, `grade_slm.grade` lanza; `graph_flow`
  degrada, pasa los fragmentos sin calificar y lo marca en la traza. Nunca
  devuelve un veredicto inventado.

## Cómo se etiqueta el dataset (sin modelo)

Está escrito en detalle en el docstring de `slm/build_dataset.py`. En corto:

1. Los fragmentos pasan por `ragb.guards.sanitize_chunks` antes de etiquetar:
   la cuarentena no entra ni como negativo. `train_grader.validate_dataset`
   vuelve a pasar `detect_injection` sobre cada fila, para que la regla valga
   también con un `--data-dir` que no generó `build_dataset`.
2. Se fragmenta a 320 caracteres (no 900): con 900 el corpus son cuatro
   fragmentos y el juez aprendería documentos, no relevancia.
3. Positivo = cubre la mayor parte de los términos de contenido (promedio de
   cobertura de la referencia y de la pregunta ≥ 0.5, con números en palabras
   convertidos a cifra y plural quitado). **3b**: si la referencia enumera
   (ítems separados por `;`), también es positivo el fragmento que cubre
   ≥ 0.75 de los términos de cualquier ítem con al menos tres términos de
   contenido (`ITEM_THRESHOLD`, `ITEM_MIN_TERMS`).
4. Negativo de otro documento (`negative`), negativo difícil del mismo
   documento (`hard_negative`), y `expects_refusal` → todos negativos.
5. Paráfrasis por reglas (sin signos, "¿Cuál es X?" → "Dime X", …) que heredan
   etiquetas.
6. Split train/valid por SHA-256 de la pregunta original; las paráfrasis siguen
   a su original para que valid no sea train con otra puntuación.

Resultado hoy (`slm/data/manifest.json`): 13 fragmentos, 8 casos, 312 pares
(36 positivos, 27 negativos difíciles, 210 negativos, 39 de negativa); train
234 filas (18 positivos / 216 negativos, 6 casos), valid 78 filas (18
positivos / 60 negativos, 2 casos). **Las filas de valid no son independientes**:
son 26 pares únicos (2 preguntas originales × 13 fragmentos) × 3 paráfrasis,
o sea 6 positivos distintos repetidos tres veces. Un F1 sobre valid mide
generalización a *dos* preguntas no vistas y a sus reformulaciones, nada más.

## Cómo entrenar en el Mini

```bash
# Extra opcional; no va en requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r slm/REQUIREMENTS.txt

python -m slm.build_dataset            # slm/data/{train,valid}.jsonl + manifest.json
python -m slm.train_grader --dry-run   # valida dataset y config, no descarga nada
python -m slm.train_grader             # e5-small, 12 épocas, CPU: ~3,5 min en 8 núcleos
```

Deja `slm/out/<nombre>/` (por defecto `multilingual-e5-small`, el último
segmento del modelo; la corrida documentada en `RESULTADOS.md` usó
`--name e5-small-12ep`) con el modelo, el tokenizer, `grader_config.json`
(cómo tokenizar y con qué `chunk_chars`, para que producción tokenice igual
que el entrenamiento y se niegue a calificar fragmentos de otro tamaño) y
`metrics.json` (accuracy, precision, recall, F1 sobre valid y train, con el
umbral 0.5 que usa el juez, más las probabilidades de valid).

Opciones: `--model` (p. ej. `distilbert-base-multilingual-cased`), `--name`,
`--epochs`, `--batch-size`, `--lr` (> 0), `--max-length`, `--seed`, `--out`.

Los resultados de la última corrida real están en `slm/RESULTADOS.md`.

Después, el humo a nivel fragmento (necesita torch):

```bash
SLM_GRADER_PATH=$PWD/slm/out/e5-small-12ep python -m slm.smoke                       # 320 chars, como producción bien indexada
SLM_GRADER_PATH=$PWD/slm/out/e5-small-12ep python -m slm.smoke --chunk-chars 900     # muestra la guarda lanzando
SLM_GRADER_PATH=$PWD/slm/out/e5-small-12ep python -m slm.smoke --chunk-chars 900 --raw  # cuánto se equivocaría; NO es producción
```

## Cómo enchufarlo en `graph_flow`

```python
from ragb import graph_flow, grade_slm

ops = graph_flow.default_ops(perms)
ops["grade"] = grade_slm.grade          # reemplaza al juez de default_ops
graph = graph_flow.build_graph(**ops)
out = graph_flow.run(graph, "¿Cuál es el plazo de pago?")
```

Variables de entorno:

```
SLM_GRADER_PATH=/ruta/absoluta/a/slm/out/e5-small-12ep   # obligatoria; sin ella el juez lanza y el grafo degrada
SLM_GRADER_THRESHOLD=0.5                                  # opcional; probabilidad mínima para "relevante"
CHUNK_CHARS=320                                           # OBLIGATORIO al indexar: ver abajo
CHUNK_OVERLAP=60
```

**El índice tiene que estar fragmentado con el mismo `CHUNK_CHARS` del
entrenamiento.** El repo indexa a 900 por defecto (`ragb/config.py`), el juez
se entrenó a 320 y rechaza cualquier fragmento de más de 480 caracteres (1,5 ×
320); los fragmentos de producción a 900 miden entre 529 y 845, así que con la
configuración por defecto el juez lanza en TODAS las preguntas, el grafo
degrada en cada ronda y en la traza aparece:

```
rounds[i].graded == False
rounds[i].error  == "RuntimeError: fragmento de 845 caracteres; el juez se entrenó con CHUNK_CHARS=320 (máximo aceptado 480). Re-indexa con ese CHUNK_CHARS o reentrena con el de producción"
```

Es fail-closed correcto —no califica lo que nunca vio— pero convierte al juez
en un no-op permanente. Las dos salidas: re-indexar con `CHUNK_CHARS=320
CHUNK_OVERLAP=60` (`python -m ragb.cli --tenant <t> ingest corpus/ --write` con
esas variables en el entorno), o
entrenar el juez con el tamaño de producción cambiando `CHUNK_CHARS` en
`build_dataset.py` (con 900 el corpus de ejemplo queda en cuatro fragmentos,
demasiado poco para aprender relevancia). `tests/test_grade_slm.py::
test_con_chunk_chars_de_produccion_el_grafo_degrada_en_cada_ronda` afirma este
comportamiento para que no haya que redescubrirlo.

Dos detalles operativos más:

- `SLM_GRADER_PATH` relativa se resuelve contra la raíz del repo, no contra el
  cwd; de todos modos conviene absoluta.
- El modelo se cachea por ruta en el proceso. **Reentrenar en la misma carpeta
  no refresca un servicio que ya lo cargó: hay que reiniciarlo.**

`grade_slm.score(question, chunks)` devuelve las probabilidades crudas, para
calibrar el umbral mirando la traza; `grade_slm.describe()` dice qué juez está
configurado (ruta, config, umbral) sin cargar pesos.

## Límites (léelos antes de confiar en el número de F1)

- **El dataset es chico**: ocho preguntas, trece fragmentos, dos preguntas en
  validación (26 pares únicos × 3 paráfrasis). Crecer el dataset dorado es lo
  que mejora al juez, no cambiar de modelo.
- **Las etiquetas salen de una heurística léxica**, no de una persona. La regla
  3b cubre las referencias que enumeran, pero tiene su propio límite: un ítem
  con menos de tres términos de contenido no cuenta, y un ítem que el corpus
  parafrasea sin repetir sus palabras queda sin cubrir. El ruido que queda es
  el de cualquier cobertura de términos: un fragmento que comparte vocabulario
  con la referencia sin contestarla puede quedar positivo. Un juez entrenado
  con esas etiquetas aprende también ese error.
- **Las paráfrasis son de reglas**: aumentan filas, no diversidad real. Un
  usuario que pregunte de otra forma sigue siendo terreno no visto.
- **Un cross-encoder no sabe lo que no vio**: un corpus nuevo exige regenerar
  el dataset y reentrenar. El juez está atado al corpus por diseño, y al
  `CHUNK_CHARS` del entrenamiento por la guarda de arriba.
- **La pérdida se pondera 12:1** por el desbalance; con otro corpus el peso
  cambia solo (sale de los conteos), pero el umbral 0.5 puede dejar de ser el
  mejor. `metrics.json` trae el mejor umbral sobre valid como dato
  informativo; no se usa por defecto porque elegirlo con el mismo valid que lo
  mide es hacer trampa.
