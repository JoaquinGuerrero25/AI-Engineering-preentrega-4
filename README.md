# Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone

Módulo de recuperación escalable en Python que ejecuta el flujo de recuperación completo de un sistema RAG en la nube:

1. **Infraestructura**: crea (o valida) un índice **Pinecone Serverless** de 1536 dimensiones.
2. **Ingesta**: carga documentación técnica en **Markdown, JSON y PDF**, la fragmenta con
   `RecursiveCharacterTextSplitter` y la sube a Pinecone con **metadatos avanzados** (fuente,
   página, sección, categoría, etiquetas) y el **texto original dentro de la metadata**.
3. **Recuperador híbrido**: la clase `RAGSystem` combina **BM25** (léxico) y **Pinecone**
   (semántico) con un `EnsembleRetriever` de LangChain y devuelve los **top-5** chunks.
4. **Evaluación**: `evaluate.py` mide **Recall@5** y **Precision@5** (más MRR) sobre un
   *Golden Set* de 5 preguntas y compara BM25, vectorial e híbrido.

El dataset es documentación en español de módulos de la biblioteca estándar de Python
(`asyncio`, `concurrent.futures`, `functools`, `itertools`, `dataclasses`, `logging`,
`pathlib`), un FAQ de buenas prácticas en JSON y una guía de entornos virtuales en PDF.

## Estructura

```
├── data/                          # Dataset: 7 .md + 1 .json (5 entradas) + 1 .pdf (3 páginas)
├── config.py                      # Variables de entorno, Pinecone, embeddings y chunking
├── setup_index.py                 # 1) Crea el índice Serverless si no existe y valida su dimensión
├── ingest.py                      # 2) Pipeline de ingesta: carga, chunking, embeddings, upsert
├── rag_system.py                  # 3) PineconeRetriever + BM25 + EnsembleRetriever -> RAGSystem
├── evaluate.py                    # 4) Recall@5 / Precision@5 / MRR sobre el golden set
├── golden_set.json                #    Benchmark: {"pregunta", "documento_id_esperado"}
├── tests/test_offline.py          #    18 tests sin API keys (Pinecone falso en memoria)
├── tools/generar_pdf.py           #    Script que generó el PDF de ejemplo
├── requirements.txt
└── .env.example                   #    Plantilla de variables (sin claves)
```

## Cómo replicar el índice paso a paso

### 1. Requisitos y entorno

- Python 3.11+ (probado con 3.14).
- Una cuenta gratuita de [Pinecone](https://app.pinecone.io) (plan Starter).
- Una API key de embeddings: **Gemini** (gratis, por defecto) u **OpenAI**.

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Variables de entorno

```bash
cp .env.example .env
```

Completá en `.env` (el archivo está en `.gitignore` y nunca se sube al repo):

| Variable | Descripción |
|---|---|
| `PINECONE_API_KEY` | Consola de Pinecone → **API Keys** → *Create API key* |
| `INDEX_NAME` | Nombre del índice (por defecto `python-docs-rag`) |
| `PINECONE_NAMESPACE` | Namespace del dataset (por defecto `python-stdlib-es`) |
| `EMBEDDING_PROVIDER` | `gemini` (por defecto) u `openai` |
| `GOOGLE_API_KEY` | Si usás Gemini: [Google AI Studio](https://aistudio.google.com/apikey) → *Create API key* |
| `OPENAI_API_KEY` | Si usás OpenAI (`text-embedding-3-small`) |

Si falta alguna clave, los scripts fallan enseguida con un mensaje que dice cuál completar.

### 3. Crear el índice Serverless

```bash
python setup_index.py
```

Verifica si `INDEX_NAME` existe. Si no existe, lo crea con `ServerlessSpec(cloud="aws",
region="us-east-1")`, **dimensión 1536** y métrica **coseno**, y espera a que esté listo. Si ya
existe, valida que la dimensión y la métrica coincidan con el modelo configurado. Si no coinciden,
se detiene antes de subir nada (así evita el error de *mismatch* de dimensiones).

### 4. Ingesta

```bash
python ingest.py --dry-run   # (opcional) muestra el chunking sin llamar a ninguna API
python ingest.py             # sube los documentos al namespace
python ingest.py --rebuild   # vacía el namespace y reindexa todo
```

La ingesta es **idempotente**: cada vector guarda el hash del documento y el modelo de
embeddings. Si se vuelve a correr sin cambios, no se re-embebe nada. Si un documento cambió,
se borran sus chunks viejos (listados por prefijo de ID) y se suben los nuevos.

### 5. Consultar el recuperador híbrido

```bash
python rag_system.py "¿Qué hace el parámetro maxsize de lru_cache?"
python rag_system.py "corrutinas en paralelo" --modo vectorial      # hibrido | vectorial | bm25
python rag_system.py "agrupar elementos" --categoria programacion-funcional
```

Desde código:

```python
from rag_system import RAGSystem

rag = RAGSystem.from_pinecone()
docs = rag.search("¿Cuándo conviene ProcessPoolExecutor?")   # top-5 (BM25 + Pinecone)
for d in docs:
    print(d.metadata["rank"], d.metadata["source"], d.metadata["section"])
```

### 6. Evaluar

```bash
python evaluate.py                                  # k=5, compara los 3 modos
python evaluate.py --guardar resultados/metricas.json
```

### 7. Tests offline (no usan ninguna API)

```bash
python -m pytest -q
```

Usan un Pinecone falso en memoria y embeddings deterministas para probar el flujo completo:
creación del índice, detección de *mismatch* de dimensiones, ingesta idempotente, esquema de
metadata, deduplicación del ensemble, filtros por categoría y cálculo de métricas.

## Diseño

### Infraestructura y variables de entorno

- Las claves se leen solo desde `.env` con `python-dotenv`. El repo solo incluye
  `.env.example`, con los valores vacíos (hay un test que lo verifica).
- **Namespace**: todos los vectores van a `PINECONE_NAMESPACE`. Si en el mismo índice se
  agregan otros datasets (u otros inquilinos), las búsquedas no se mezclan.
- **Dimensión única**: `EMBEDDING_DIMENSION=1536` se usa para crear el índice y para pedir los
  embeddings. Con OpenAI, `text-embedding-3-small` ya es de 1536. Con Gemini,
  `gemini-embedding-001` (3072 nativas) se recorta a 1536 con `output_dimensionality`
  (representación *Matryoshka*). Además la ingesta valida el largo de cada vector antes del upsert.

### Pipeline de ingesta y metadatos

| Formato | Cómo se carga | Metadata específica |
|---|---|---|
| Markdown | Frontmatter YAML (`id`, `title`, `category`, `tags`) + cuerpo | `section` = último encabezado `#`/`##`/`###` |
| JSON | Cada entrada `{id, title, category, tags, content}` es un documento | `source` = archivo JSON |
| PDF | Una unidad por página con `pypdf`. Categoría y tags salen de *Subject*/*Keywords* | `page` = número de página |

**Chunking**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con **600 tokens** y **90 de
solapamiento** (15 %), dentro del rango recomendado de 500 a 800. Los separadores priorizan la
estructura (`\n## `, `\n### `, bloques de código, párrafos, oraciones). Los fragmentos de menos
de 120 tokens (por ejemplo, una introducción corta) se **fusionan con su vecino**, con un
máximo de 800 tokens, para que no queden chunks sin contexto. Resultado: 26 chunks de entre 245
y 668 tokens.

**Esquema de cada vector**:

```jsonc
{
  "id": "functools#c0",                    // PDF: "guia-entornos-virtuales#p2-c0"
  "values": [/* 1536 floats */],           // embedding de "Título | Sección" + texto
  "metadata": {
    "text": "...texto original del chunk...",  // evita una base relacional extra
    "doc_id": "functools", "chunk_id": "functools#c0",
    "source": "functools.md", "source_type": "markdown",
    "title": "functools: funciones de orden superior y caché",
    "section": "lru_cache: memoización con límite de tamaño",
    "category": "programacion-funcional",
    "tags": ["functools", "lru_cache", "cache", "..."],
    "page": 2,                             // solo en PDFs
    "chunk_index": 0, "n_tokens": 615,
    "content_hash": "sha256...", "embedding_id": "gemini:models/gemini-embedding-001:1536"
  }
}
```

- **Texto en la metadata**: la respuesta de Pinecone ya trae el contenido, sin consultar otra
  base de datos. El índice BM25 también se arma a partir de esa metadata, así que Pinecone es la
  única fuente de verdad.
- **Encabezado contextual**: lo que se embebe es `Título | Sección` + chunk. Así un fragmento no
  pierde su tema aunque quede separado del título.
- **IDs con prefijo** (`doc_id#...`): permiten listar y borrar los chunks de un documento con
  `index.list(prefix=...)`.
- `category` permite **filtrar** búsquedas (`{"category": {"$eq": ...}}`) y `embedding_id`
  permite detectar si el namespace se indexó con otro modelo.

### Recuperador híbrido (`RAGSystem`)

```
                         ┌─> BM25Retriever (rank_bm25, top-10) ───────┐
consulta ─> RAGSystem ───┤                                            ├─> EnsembleRetriever ─> top-5
                         └─> PineconeRetriever (coseno, top-10) ──────┘   (RRF ponderado,
                              embed_query + index.query(namespace))        dedup por chunk_id)
```

- **PineconeRetriever**: es un `BaseRetriever` de LangChain sobre el **SDK nativo** de Pinecone
  (la consigna admite `PineconeVectorStore` o el SDK nativo). `langchain-pinecone` todavía no
  publica versiones compatibles con Python 3.14. Embebe la consulta con el mismo modelo de la
  ingesta y consulta el namespace con filtros opcionales.
- **BM25Retriever**: usa un tokenizador pensado para documentación técnica. Pasa todo a
  minúsculas y quita las tildes (`caché` = `cache`), conserva identificadores completos y sus
  partes (`asyncio.gather` → `asyncio.gather`, `asyncio`, `gather`) y descarta stopwords en
  español. También indexa título, sección y tags.
- **EnsembleRetriever**: aplica *Reciprocal Rank Fusion* ponderado (`peso / (rango + 60)`,
  pesos 0.5 / 0.5 configurables). Cada recuperador aporta 10 candidatos. Los duplicados se
  colapsan por `chunk_id` (`id_key`), así que un chunk que aparece en ambas listas suma el
  puntaje de las dos y sube en el ranking. `RAGSystem.search()` devuelve los 5 primeros.
- **Por qué híbrido**: BM25 resuelve bien los nombres exactos (`ProcessPoolExecutor`,
  `cache_info`) aunque aparezcan pocas veces. La búsqueda vectorial encuentra paráfrasis
  ("aislar las librerías de cada proyecto" → entornos virtuales) que no comparten palabras con
  el documento.

## Evaluación

`golden_set.json` define 5 preguntas con su documento fuente conocido. A propósito, combina
preguntas **léxicas** (identificadores exactos) y **semánticas** (paráfrasis sin palabras clave),
y cubre los tres formatos (MD, JSON y PDF).

| # | Pregunta | Documento esperado | Tipo |
|---|---|---|---|
| 1 | ¿Qué pasa cuando se llena la caché de lru_cache y cómo veo cuántos aciertos tuvo? | `functools` | léxica |
| 2 | ¿Cómo hago para que varias funciones async corran a la vez y esperar a que terminen todas? | `asyncio` | semántica |
| 3 | ¿Cuándo conviene ProcessPoolExecutor en lugar de ThreadPoolExecutor? | `concurrent-futures` | léxica |
| 4 | ¿Por qué es peligroso capturar todas las excepciones con un except vacío? | `faq-except-generico` (JSON) | mixta |
| 5 | ¿Cómo aíslo las librerías de cada proyecto para que no choquen entre sí? | `guia-entornos-virtuales` (PDF) | semántica |

**Métricas** (por pregunta, luego promediadas):

- **Recall@5**: ¿el documento correcto está entre los 5 chunks recuperados? (1 o 0).
- **Precision@5**: fracción de los 5 chunks recuperados que pertenecen al documento correcto.
- **P@5 máx.**: techo alcanzable. Si el documento esperado tiene solo 1 chunk (las entradas
  del FAQ), la precisión nunca puede superar 0,20. Se informa para interpretar bien la
  Precision@5.
- **MRR** (extra): 1 / posición del primer chunk relevante.

### Resultados

Salida de `python evaluate.py` contra el índice real (Pinecone Serverless aws/us-east-1,
`gemini-embedding-001` a 1536 dims, 26 chunks en el namespace `python-stdlib-es`):

| Modo | Recall@5 | Precision@5 | P@5 máx. | MRR |
|---|---|---|---|---|
| BM25 (léxico) | 1.00 | 0.52 | 0.52 | 1.00 |
| Pinecone (semántico) | 1.00 | 0.52 | 0.52 | 1.00 |
| **Híbrido (Ensemble)** | **1.00** | **0.52** | 0.52 | **1.00** |

Detalle del modo híbrido:

| # | Documento esperado | Recall@5 | Precision@5 (máx.) | Top-5 recuperado |
|---|---|---|---|---|
| 1 | `functools` | 1 | 0.60 (0.60) | functools ×3, itertools, dataclasses |
| 2 | `asyncio` | 1 | 0.60 (0.60) | asyncio ×3, functools ×2 |
| 3 | `concurrent-futures` | 1 | 0.60 (0.60) | concurrent-futures ×3, asyncio, faq-fstrings |
| 4 | `faq-except-generico` | 1 | 0.20 (0.20) | faq-except-generico, logging, faq-encoding-archivos, faq-argumento-mutable, asyncio |
| 5 | `guia-entornos-virtuales` | 1 | 0.60 (0.60) | guia-entornos-virtuales ×3, logging, concurrent-futures |

**Interpretación**

- **Recall@5 = 1.00**: en las 5 preguntas el documento correcto aparece entre los 5 recuperados,
  y además siempre en la **primera posición** (MRR = 1.00).
- **Precision@5 = 0.52**, que es exactamente el **techo alcanzable**: en cada pregunta el top-5
  trae *todos* los chunks del documento correcto y recién después completa con otros. Por ejemplo,
  el FAQ del `except` tiene un único chunk, así que 0.20 es la precisión perfecta para esa pregunta.
- Con este corpus chico (26 chunks) y un golden set de 5 preguntas, los tres modos llegan al
  techo y el benchmark no alcanza para diferenciarlos. El valor del híbrido aparece con más
  documentos parecidos entre sí: BM25 cubre los nombres exactos que el embedding diluye y el
  vector cubre las paráfrasis que BM25 no ve. Para medir esa diferencia habría que ampliar el
  golden set con preguntas más difíciles.

## Errores comunes y cómo se evitan

| Error | Mitigación en este repo |
|---|---|
| Mismatch de dimensiones | `setup_index.py` valida dimensión y métrica del índice existente; la ingesta verifica el largo de cada vector; `RAGSystem` rechaza namespaces indexados con otro modelo. |
| Ignorar el namespace | Todos los upserts, consultas, listados y borrados usan `PINECONE_NAMESPACE`. |
| Subestimar el chunking | 600 tokens con 15 % de solapamiento, cortes por estructura Markdown y fusión de fragmentos menores a 120 tokens. |
| Claves en el repo | `.env` en `.gitignore`; `.env.example` sin valores (verificado por test). |
| Consistencia eventual de Serverless | La ingesta espera a que `describe_index_stats` refleje todos los vectores antes de terminar. |
