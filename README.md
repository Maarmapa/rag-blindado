# rag-blindado

**RAG auditable: recuperación aumentada con controles de seguridad explícitos
y evaluación de calidad bloqueante en CI/CD.**

Un pipeline RAG completo sobre Postgres + pgvector, donde cada decisión de
seguridad está mapeada a un riesgo de [OWASP Top 10 for LLM Applications](https://genai.owasp.org/)
y cada despliegue debe superar umbrales medidos de fidelidad antes de pasar a
producción.

*(English summary below / Resumen en inglés más abajo.)*

---

## Qué resuelve

Un RAG que funciona en el demo y falla en producción suele fallar por tres
cosas, no por el modelo:

1. **Un documento del corpus da órdenes al agente.** Si cualquiera puede subir
   un PDF al repositorio indexado, cualquiera puede escribir instrucciones que
   el modelo obedece. Acá el contenido recuperado se trata como dato, se analiza
   antes de llegar al prompt, y lo sospechoso va a cuarentena con su motivo
   registrado.
2. **Nadie mide si las respuestas son ciertas.** Acá tres métricas corren en
   cada push y un pull request que baja la fidelidad no puede mergearse.
3. **El agente puede hacer más de lo que necesita.** Acá el permiso de escritura
   está denegado por defecto y la ingesta corre en `dry-run` salvo decisión
   explícita.

## Arquitectura

```
documentos → fragmentación → embeddings locales → pgvector (por tenant)
                                                        ↓
pregunta → embedding → búsqueda coseno ─────────────────┘
                            ↓
                    guardas (LLM01 · LLM02)
                            ↓
                    contexto delimitado → Claude → respuesta con fuentes
                            ↓
                    evals (Ragas) → umbral → CI/CD
```

**Reparto de modelos deliberado.** Los embeddings corren con un modelo open
source local: el corpus se re-indexa sin costo por token y los documentos nunca
salen de la infraestructura. La generación usa un modelo frontera vía API,
donde la calidad del razonamiento sí importa. Barato y local donde el volumen
es alto; capaz donde importa la respuesta.

## Controles de seguridad

| Riesgo OWASP | Control | Dónde |
|---|---|---|
| **LLM01** Prompt Injection | Detección de instrucciones incrustadas en el texto recuperado + delimitado estructural del contexto en bloques `<documento>` | `ragb/guards.py`, `ragb/generate.py` |
| **LLM02** Sensitive Information Disclosure | Redacción de credenciales (claves de API, tokens, DSN) antes de armar el prompt | `ragb/guards.py` |
| **LLM06** Excessive Agency | Mínimo privilegio: lectura acotada por colección, escritura denegada por defecto, `dry_run` como default | `ragb/guards.py` |
| **LLM08** Vector and Embedding Weaknesses | Filtro por `tenant` a nivel de SQL en toda búsqueda, más RLS en el esquema | `ragb/store.py`, `sql/schema.sql` |
| **LLM09** Misinformation | Fail-closed: sin contexto suficiente el sistema declara que no sabe, en vez de completar | `ragb/generate.py` |

El corpus de demo incluye `corpus/nota-proveedor.md`, un documento con una
inyección real incrustada entre datos legítimos. El pipeline responde el dato
correcto (el horario de soporte) y pone en cuarentena **el párrafo** con la
instrucción, no el documento entero: descartar el fragmento completo se llevaría
por delante el plazo de pago y el horario de soporte, que son legítimos. La
escisión falla cerrado — si un patrón calza cruzando el corte entre párrafos,
sacar párrafos no lo neutralizaría, así que se descarta todo.

## Evaluación en CI

Tres métricas sobre un dataset dorado (`evals/dataset.yaml`), calculadas con
[Ragas](https://docs.ragas.io/) usando Claude como modelo juez:

| Métrica | Qué mide | Umbral |
|---|---|---|
| `faithfulness` | Cada afirmación de la respuesta se sostiene en el contexto recuperado | 0.85 |
| `answer_relevancy` | La respuesta contesta la pregunta que se hizo | 0.75 |
| `context_precision` | Lo recuperado era efectivamente relevante | 0.70 |

`python -m evals.run` termina con exit code 1 si alguna cae bajo el umbral. El
workflow de GitHub Actions corre los tests de guardas en **todo** push (sin
credenciales, sin costo) y las evals completas contra un Postgres con pgvector
levantado como servicio.

Una métrica que no se pudo calcular **no aprueba**. Parece obvio, pero en Python
`nan` pierde todas las comparaciones —`nan < 0.85` es falso y `nan >= 0.85`
también—, así que una barrera escrita con una sola comparación se abre sola
justo cuando el juez falla. Acá un `nan` se reporta como `SIN MEDIR` y rechaza
el build.

### La capa que no depende de un juez

Las tres métricas las calcula un modelo. Eso significa que el umbral no separa
"respuesta buena" de "respuesta mala": separa "el juez la aprobó" de "el juez no
la aprobó". `faithfulness` es una razón sobre pocas afirmaciones, así que en una
respuesta corta un error del juez mueve el puntaje un tercio — medido en este
repo: una respuesta que cita el corpus casi palabra por palabra puntuó 0.667.

Por eso hay una segunda capa, **determinista**: sin juez, sin llamadas a la API
y sin varianza. No puntúa, verifica propiedades en binario:

| Propiedad | Por qué |
|---|---|
| El control negativo declara que no encuentra el dato | Que se niegue es la conducta correcta, y nadie la verificaba |
| Ninguna respuesta contiene el system prompt | El corpus trae una inyección que lo pide |
| Ninguna respuesta contiene credenciales | LLM02 sobre la salida, no solo sobre el prompt |
| Toda respuesta con contenido cita su fuente | La trazabilidad que el repo promete |
| Toda cita resuelve a un documento que entró al contexto | Una cita inventada fabrica procedencia, y es peor que no citar |

Estas comprobaciones corren en el job `guards` —**en todo push, sin
credenciales y sin costo**— y también sobre las respuestas reales en el job de
evals. Un pipeline que filtra el system prompt o se inventa una fuente rechaza
el build **aunque las tres métricas estén en verde**.

Un caso marcado `expects_refusal` en el dataset queda fuera de las tres
métricas: una negativa correcta no tiene afirmaciones que anclar al contexto ni
contexto relevante que recuperar, y las métricas la puntúan cerca de cero por
diseño. Se verifica con la aserción, que es más exigente que las tres juntas
para lo que se le pide.

## Uso

```bash
pip install -r requirements.txt
cp .env.example .env        # completar DATABASE_URL y ANTHROPIC_API_KEY

# Indexar (dry-run por defecto; --write ejecuta la escritura)
python -m ragb.cli --tenant demo ingest corpus/ --write

# Consultar
python -m ragb.cli --tenant demo ask "¿Cuál es el plazo de respuesta de severidad 1?"

# Evaluar
python -m evals.run
```

Cualquier Postgres con la extensión `vector` sirve: Supabase, Neon, RDS o local.

## Stack

Python 3.12 · Postgres + pgvector · sentence-transformers (embeddings open
source) · Anthropic Claude (generación y juez) · Ragas (evaluación) · pytest ·
GitHub Actions.

---

## English summary

**Auditable RAG: retrieval-augmented generation with explicit security controls
and quality gates that block deployment.**

A complete RAG pipeline on Postgres + pgvector where every security decision
maps to an [OWASP Top 10 for LLM Applications](https://genai.owasp.org/) risk
and every deploy must clear measured faithfulness thresholds before reaching
production.

Retrieved content is treated as **data, never instructions**: embedded prompt
injections are detected and quarantined with a logged reason, credentials are
redacted before the prompt is assembled, tenant isolation is enforced in SQL,
and write access is denied by default with ingestion running in dry-run unless
explicitly enabled. Three Ragas metrics — faithfulness, answer relevancy and
context precision — run against a golden dataset in CI and fail the build below
threshold.

Embeddings run locally on an open-source model (no per-token cost, documents
never leave the infrastructure); generation uses a frontier model where
reasoning quality matters.

## Licencia

MIT.
