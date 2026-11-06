# Resultados reales del entrenamiento del juez SLM

**Fecha:** 2026-09-21, `trained_at 2026-09-21T22:45:02+00:00`. **Dónde:** sandbox
remoto Linux, 8 núcleos de CPU, 7 GB de RAM, sin GPU. Stack: Python 3.11.6,
`torch 2.14.0+cpu`, `transformers 5.17.0`. El repo se clonó de
`Maarmapa/rag-blindado` `main` (`0d0fcd1`) y se copiaron encima, verificados por
md5 contra el árbol local, `slm/__init__.py`, `slm/build_dataset.py`
(`30a1341b…`), `slm/train_grader.py` (`d563e332…`), `slm/smoke.py`
(`89b374c1…`) y `ragb/grade_slm.py` (`2b06161b…`). Los archivos compartidos
(`corpus/`, `evals/dataset.yaml`, `ragb/chunking.py`, `ragb/guards.py`,
`ragb/config.py`) son idénticos entre `main` y esta rama (`git diff` vacío), así
que el clon de `main` es fiel.

## Dataset

`python -m slm.build_dataset` en el sandbox produjo bytes idénticos a
`slm/data/` del repo: `train.jsonl` md5 `42abd10629e2a2217b82dd35c18e4701`,
`valid.jsonl` md5 `816b506cc288c01b073032f934632099`. Es el dataset **con la
regla 3b** (ítems enumerados), la que hoy está en `build_dataset.py`.

| | filas | positivos | negativos | casos | pares únicos |
|---|---|---|---|---|---|
| train | 234 | 18 | 216 | 6 | 78 (6 preguntas × 13 fragmentos) × 3 paráfrasis |
| valid | 78 | 18 | 60 | 2 (severidad 1, condiciones del agente) | 26 (2 × 13) × 3 paráfrasis |

Por tipo: 36 positivos, 27 negativos difíciles, 210 negativos, 39 de negativa.
13 fragmentos saneados (320 caracteres, 1 en cuarentena), 8 casos dorados.
Peso de la clase positiva: 12.0. **Los 18 positivos de valid son 6 pares
distintos repetidos en 3 paráfrasis**: los conteos de las tablas no son
observaciones independientes.

## Corrida — `intfloat/multilingual-e5-small`, 12 épocas, lote 8, lr 5e-5, semilla 42 (los defaults)

360 pasos, 226,0 s. Pérdida final 0.1419. `--dry-run` previo: mismo plan
(360 pasos, chunk_chars 320).

| split | accuracy | precision | recall | F1 | tp/fp/tn/fn |
|---|---|---|---|---|---|
| **valid** | 0.8718 | 1.0 | 0.4444 | **0.6154** | 8/0/60/10 |
| train | 1.0 | 1.0 | 1.0 | 1.0 | 18/0/216/0 |

Umbral 0.5 (el que usa `ragb.grade_slm` por defecto). El barrido informativo
sobre valid no arregla nada: el mejor F1 es 0.6207 en 0.05, con 2 falsos
positivos y todavía 9 falsos negativos.

**Los 10 falsos negativos**, en orden de `valid.jsonl` (índice, caso, fuente, probabilidad):

- `36` caso 0 *"Dime el plazo de respuesta para un incidente de severidad 1"* ×
  fragmento 10 de `procedimiento-incidentes.md`: 0.1885. La misma pareja da
  0.93 con la pregunta original y 0.72 sin signos: la forma imperativa le
  cuesta más que la interrogativa.
- `40-42`, `53-55`, `66-68` caso 2 (*condiciones del agente*, sus tres
  variantes) × fragmentos 1, 2 y 3 de `estandar-agentes.md`: **0.0085-0.0086,
  las nueve.** Son exactamente los nueve positivos que la regla 3b agregó al
  dataset, y el juez los puntúa como cualquier negativo.

Por qué: **la regla 3b cambió etiquetas solo en valid.** `train.jsonl` es
byte-idéntico al de la corrida anterior (mismo md5 `42abd106…`), porque el
único caso con referencia enumerada —el caso 2— cae en valid por el hash de su
pregunta. Con el mismo train y la misma semilla, los pesos son los mismos
(misma pérdida 0.1419, mismas 78 probabilidades de valid) y lo que cambió es la
vara con que se miden. En train no hay ningún ejemplo de "una pregunta con
cuatro fragmentos relevantes del mismo documento": el juez aprendió que a cada
pregunta le corresponde un fragmento, y en valid se lo penaliza por eso. Es la
métrica honesta de este dataset, no un fallo del entrenamiento.

Los 60 negativos de valid, incluidos los 9 negativos difíciles de
`procedimiento-incidentes.md`, quedaron todos bajo 0.26.

Las probabilidades completas de valid están en
`slm/out/e5-small-12ep/metrics.json` (md5 `d8155339c42849cfcc8f36c312f3571b`,
copiado del sandbox byte a byte junto con `grader_config.json`, md5
`78087046e06d745bab65adfd0e549c8f`; los pesos, 449 MB, no se traen:
`slm/out/.gitignore` deja pasar esos dos archivos).
`tests/test_train_grader.py::test_metrics_versionadas_coinciden_con_el_manifest`
recalcula la tabla de valid desde esas probabilidades y las etiquetas
versionadas: si alguien regenera el dataset sin reentrenar, ese test lo dice.

### Lo que pasó con el 0.94 anterior

La primera versión de este archivo reportaba valid F1 0.9412 (8/0/69/1). Ese
número era real, pero **medido sobre el dataset anterior a la regla 3b** (27
positivos / 36 negativos difíciles, 9 positivos en valid, `valid.jsonl` md5
`90b54669…`), que se regeneró después de entrenar y sin reentrenar. Con los
mismos pesos y las etiquetas de hoy, el número es 0.6154. Se deja constancia
porque es la deriva exacta que el test nuevo existe para atrapar.

## Humo de punta a punta con el modelo entrenado (`slm/smoke.py`, mismo sandbox)

Medido a nivel fragmento: qué parte de los términos de la referencia cubren los
fragmentos que PASARON el juez, y para la referencia enumerada, cuántos ítems
quedaron cubiertos. Umbral 0.5, `grader_config.json` leído del modelo.

### A 320 caracteres (el régimen del entrenamiento; 13 fragmentos)

```
SLM_GRADER_PATH=slm/out/e5-small-12ep python -m slm.smoke
```

| pregunta | pasan | fuentes | cobertura ref. | ítems | ms |
|---|---|---|---|---|---|
| ¿Cuál es el plazo de respuesta para un incidente de sev | 2/13 | procedimiento-incidentes.md | 1.0 | - | 7003 (incluye cargar el modelo) |
| ¿Cuánto tiempo se retienen los registros de auditoría d | 1/13 | politica-acceso-datos.md | 1.0 | - | 159 |
| ¿Qué condiciones debe cumplir un agente antes de habili | 1/13 | estandar-agentes.md | **0.273** | **1/4** | 157 |
| ¿Dónde se guardan las credenciales de acceso a sistemas | 1/13 | politica-acceso-datos.md | 0.9 | - | 165 |
| ¿Cuál es el plazo de pago acordado con el proveedor? | 2/13 | nota-proveedor.md | 1.0 | - | 143 |
| ¿Qué permisos recibe un principal por defecto? | 1/13 | politica-acceso-datos.md | 1.0 | - | 159 |
| ¿Cuál es el presupuesto anual del área de tecnología? | 0/13 | — | 0.0 | - | 140 |
| ¿En qué horario atiende el soporte del proveedor? | 1/13 | nota-proveedor.md | 0.357 | - | 140 |

La fuente correcta pasa en las siete preguntas con respuesta y cero fragmentos
en la de negativa (fail-closed del grafo). Pero **el caso 2 deja pasar uno de
los cuatro fragmentos que hoy el dataset llama relevantes**: el generador vería
una condición de cuatro, y la cobertura de la referencia es 0.273. Es el mismo
hecho que el recall 0.44 de valid, visto desde el otro lado. ~140-165 ms por
pregunta para 13 pares en CPU, sin tokens de API.

La cobertura 0.357 del horario de soporte no es un fallo del juez: la
referencia menciona la inyección que la respuesta NO debe repetir, y esos
términos no están (ni deben estar) en ningún fragmento saneado.

### A 900 caracteres (el `CHUNK_CHARS` por defecto de producción; 4 fragmentos de 845/350/697/806)

```
SLM_GRADER_PATH=slm/out/e5-small-12ep python -m slm.smoke --chunk-chars 900
{"guard": "LANZA", "error": "fragmento de 845 caracteres; el juez se entrenó con CHUNK_CHARS=320 (máximo aceptado 480). Re-indexa con ese CHUNK_CHARS o reentrena con el de producción"}
```

Es lo que ve `graph_flow` con el índice por defecto: `degraded=True` en cada
ronda, el juez es un no-op. El README lo dice en "Cómo enchufarlo".

### A 900 con `--raw` (salta la guarda; NO es producción, solo mide cuánto se equivocaría)

| pregunta | pasan | fuentes | cobertura ref. | ítems |
|---|---|---|---|---|
| ¿Cuál es el plazo de respuesta para un incidente de sev | 1/4 | **nota-proveedor.md** | **0.0** | - |
| ¿Cuánto tiempo se retienen los registros de auditoría d | 1/4 | politica-acceso-datos.md | 1.0 | - |
| ¿Qué condiciones debe cumplir un agente antes de habili | 1/4 | estandar-agentes.md | 0.864 | 4/4 |
| ¿Dónde se guardan las credenciales de acceso a sistemas | 1/4 | politica-acceso-datos.md | 0.9 | - |
| ¿Cuál es el plazo de pago acordado con el proveedor? | 1/4 | nota-proveedor.md | 1.0 | - |
| ¿Qué permisos recibe un principal por defecto? | 1/4 | politica-acceso-datos.md | 1.0 | - |
| ¿Cuál es el presupuesto anual del área de tecnología? | 0/4 | — | 0.0 | - |
| ¿En qué horario atiende el soporte del proveedor? | 1/4 | nota-proveedor.md | 0.429 | - |

Fuera de régimen el juez se equivoca de documento en la primera pregunta
(aprueba `nota-proveedor.md` con 0.759 y rechaza `procedimiento-incidentes.md`
con 0.099): cobertura de la referencia 0.0. Es la razón por la que la guarda
existe y por la que producción lanza en vez de puntuar.

⚠ El humo usa las preguntas del dataset dorado (seis están en train): verifica
el **cableado** y la cobertura río abajo, no la generalización. La
generalización es el F1 de valid, y vale lo que valen dos preguntas.

## Lo que no se verificó

- **Un dataset con referencias enumeradas en train.** Es la única forma de que
  el juez aprenda que varios fragmentos del mismo documento pueden ser
  relevantes a la vez; hoy el único caso así cae en valid. Mover el reparto
  (`VALID_BUCKETS`) para que caiga en train sería elegir el split mirando el
  resultado: no se hizo. Lo que corresponde es crecer `evals/dataset.yaml`.
- **Generalización a preguntas de otra forma**: el falso negativo con "Dime…"
  sugiere que las paráfrasis por reglas no cubren el espacio de formulaciones.
- **Otro modelo base**: no se entrenó `distilbert-base-multilingual-cased`;
  e5-small cupo en tiempo y aprendió el patrón de train, así que no hizo falta
  bajar de tamaño. Cambiar de modelo no arregla lo de arriba.
- **Reproducibilidad entre máquinas**: dos corridas en el mismo sandbox, con
  el mismo train y la misma semilla, dieron la misma pérdida y las mismas 78
  probabilidades de valid. En otra máquina, torch en CPU con distinto número
  de hilos puede diferir en el tercer decimal; no se comprobó.
