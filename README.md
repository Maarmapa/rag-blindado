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
correcto (el horario de soporte) y pone el fragmento malicioso en cuarentena.

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

## Uso

```bash
pip install -r requirements.txt
cp .env.example .env        # completar DATABASE_URL y ANTHROPIC_API_KEY

# Indexar (dry-run por defecto; --write ejecuta la escritura)
python -m ragb.cli ingest corpus/ --tenant demo --write

# Consultar
python -m ragb.cli ask "¿Cuál es el plazo de respuesta de severidad 1?" --tenant demo

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
