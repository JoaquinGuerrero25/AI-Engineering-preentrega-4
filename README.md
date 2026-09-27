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
  el documento. La justificación completa, con evidencia, está en
  [Por qué un recuperador híbrido](#por-qué-un-recuperador-híbrido).

## Evaluación

`golden_set.json` define 5 preguntas con su documento fuente conocido. Las preguntas se eligieron
para ser **difíciles** y exigir a cada recuperador por separado. Algunas describen un síntoma
sin nombrar la API, otras mencionan solo un parámetro puntual, y varias tienen un **documento
distractor** que habla de un tema parecido. Cubren los tres formatos (MD, JSON y PDF).

| # | Pregunta | Documento esperado | Qué la hace difícil |
|---|---|---|---|
| 1 | ¿Cómo evito que todas las instancias de mi clase de datos compartan la misma lista? | `dataclasses` | Distractor: el FAQ de argumentos mutables en funciones trata el mismo problema |
| 2 | ¿Qué diferencia hay entre typed=True y typed=False? | `functools` | Solo nombra un parámetro, sin mencionar `lru_cache` ni el módulo |
| 3 | Mi script que usa todos los núcleos se queda creando procesos sin parar en Windows, ¿por qué pasa? | `concurrent-futures` | Describe el síntoma, no menciona `ProcessPoolExecutor` ni `__main__` |
| 4 | Leo un CSV en Windows y las eñes aparecen como símbolos raros, ¿cómo lo soluciono? | `faq-encoding-archivos` (JSON) | Distractor: `pathlib` también habla de `encoding` y `cp1252` |
| 5 | ¿Cómo aíslo las librerías de cada proyecto para que no choquen entre sí? | `guia-entornos-virtuales` (PDF) | Paráfrasis: no menciona `venv` ni `pip` |

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
| BM25 (léxico) | 1.00 | 0.32 | 0.48 | **1.00** |
| Pinecone (semántico) | 1.00 | **0.40** | 0.48 | 0.77 |
| **Híbrido (Ensemble)** | **1.00** | **0.40** | 0.48 | **1.00** |

Posición del documento correcto y Precision@5 de cada modo, por pregunta:

| # | Documento esperado | BM25 | Pinecone | Híbrido |
|---|---|---|---|---|
| 1 | `dataclasses` | 1.º · P=0.40 | **2.º** (1.º el FAQ distractor) · P=0.40 | 1.º · P=0.40 |
| 2 | `functools` | 1.º · P=0.20 | **3.º** (detrás de 2 chunks de dataclasses) · P=0.20 | 1.º · P=0.20 |
| 3 | `concurrent-futures` | 1.º · **P=0.20** | 1.º · P=0.60 | 1.º · P=0.60 |
| 4 | `faq-encoding-archivos` | 1.º · P=0.20 | 1.º · P=0.20 | 1.º · P=0.20 |
| 5 | `guia-entornos-virtuales` | 1.º · P=0.60 | 1.º · P=0.60 | 1.º · P=0.60 |

**Interpretación**

- **Recall@5 = 1.00 en los tres modos**: el corpus es chico (26 chunks), así que el documento
  correcto siempre entra en el top-5. La diferencia está en *dónde* aparece y en *cuánto ruido*
  lo acompaña.
- **Pinecone falla en el orden (MRR 0.77)**. En la pregunta 1, el embedding confunde dos temas
  cercanos y pone primero el FAQ de argumentos mutables. En la 2, `typed=True` es un término
  demasiado puntual y el embedding lo diluye. BM25 encuentra ese término exacto y los pone primero.
- **BM25 falla en la precisión (0.32)**. En la pregunta 3 ("procesos sin parar en Windows") solo
  recupera 1 de los 3 chunks de `concurrent-futures`, porque la pregunta no comparte palabras con
  el texto. El embedding sí entiende la paráfrasis.
- **El híbrido se queda con lo mejor de cada uno**: tiene el **MRR de BM25 (1.00)** y la
  **Precision@5 de Pinecone (0.40)**. Es el único modo que no pierde en ninguna métrica, que es
  justamente lo que justifica usar el `EnsembleRetriever`.
- La Precision@5 del híbrido (0.40) queda por debajo del techo (0.48) por la pregunta 2: el top-5
  trae 1 de los 3 chunks de `functools` y completa con otros documentos que mencionan `True` y
  `False`.

## Por qué un recuperador híbrido

### Dos buscadores que fallan en casos opuestos

| | BM25 (léxico) | Pinecone (semántico) |
|---|---|---|
| **Qué compara** | Términos exactos, ponderando los poco frecuentes (IDF) | Significado (similitud coseno entre embeddings) |
| **Acierta con** | Nombres técnicos y parámetros: `ProcessPoolExecutor`, `typed=True`, `cache_info` | Paráfrasis y síntomas: "procesos sin parar en Windows", "aislar las librerías" |
| **Falla con** | Preguntas que no comparten palabras con el texto | Términos muy puntuales que el embedding diluye; temas cercanos que confunde |

La documentación técnica tiene las dos cosas: identificadores exactos **y** preguntas escritas con
otras palabras. Ninguno de los dos buscadores alcanza por sí solo.

### Cómo se fusionan: Reciprocal Rank Fusion

El `EnsembleRetriever` combina los dos rankings con RRF ponderado:

```
score(chunk) = Σ  peso_i / (posición_i + 60)
```

Ejemplo con pesos 0.5 / 0.5:

| Chunk | BM25 | Pinecone | Score RRF |
|---|---|---|---|
| A | 1.º | 3.º | 0.5/61 + 0.5/63 = **0.0161** |
| B | – | 1.º | 0.5/61 = **0.0082** |

A queda por encima de B aunque Pinecone lo ubique más abajo, porque **los dos buscadores coinciden**
en que es relevante. Ese acuerdo entre dos señales independientes es lo que el método premia.

### Decisiones de diseño

| Decisión | Motivo |
|---|---|
| **RRF en vez de sumar scores** | BM25 devuelve puntajes sin tope (por ejemplo, 12.4) y el coseno está entre 0 y 1. Sumarlos exigiría normalizar escalas incomparables. RRF usa solo las posiciones. |
| **Pesos 0.5 / 0.5** | Es el punto de partida neutral y se configura en `.env` (`BM25_WEIGHT`, `VECTOR_WEIGHT`). Ajustarlos con 5 preguntas sería sobreajustar al benchmark. Con un golden set más grande se buscarían los pesos que maximicen MRR o nDCG. |
| **10 candidatos por recuperador para devolver 5** | Si cada uno trajera solo 5, un chunk que queda 6.º en ambas listas se perdería, aunque la fusión lo pondría arriba. |
| **Deduplicación por `chunk_id`** | Permite sumar el puntaje de un mismo chunk en ambas listas sin comparar textos completos. |
| **El corpus de BM25 sale de Pinecone** | Hay una sola fuente de verdad: los dos recuperadores ven exactamente los mismos chunks, sin otra base de datos. |

### Evidencia

El [golden set](#evaluación) tiene preguntas pensadas para que cada buscador falle en algún caso:

| Modo | Precision@5 | MRR |
|---|---|---|
| BM25 | 0.32 ❌ | 1.00 |
| Pinecone | 0.40 | 0.77 ❌ |
| **Híbrido** | **0.40** | **1.00** |

- **Donde falla Pinecone:** en *"¿Qué diferencia hay entre typed=True y typed=False?"*, el
  embedding ubica `functools` 3.º, detrás de dos chunks de `dataclasses`. BM25 encuentra el término
  poco frecuente `typed` y el híbrido lo sube al 1.º.
- **Donde falla BM25:** en *"Mi script se queda creando procesos sin parar en Windows"*, BM25
  recupera solo 1 de los 3 chunks de `concurrent-futures` porque la pregunta no comparte palabras
  con el texto. Pinecone entiende la paráfrasis y el híbrido conserva sus 3 chunks.

El híbrido es el único modo que no pierde en ninguna métrica: tiene el MRR de BM25 y la
Precision@5 de Pinecone.

Para reproducir el caso de `typed` en los tres modos:

```bash
python rag_system.py "¿Qué diferencia hay entre typed=True y typed=False?" --modo vectorial
python rag_system.py "¿Qué diferencia hay entre typed=True y typed=False?" --modo bm25
python rag_system.py "¿Qué diferencia hay entre typed=True y typed=False?"
```

### Alternativas y límites

- **Búsqueda híbrida nativa de Pinecone (sparse-dense).** Es una alternativa válida, pero
  requiere un índice con métrica `dotproduct` y generar vectores sparse durante la ingesta. Se
  eligió `BM25Retriever` + `EnsembleRetriever` porque es lo que pide la consigna y porque permite
  evaluar cada recuperador por separado.
- **Escalabilidad de BM25 en memoria.** Con miles de chunks funciona sin problemas. Con millones,
  cargar el corpus completo al iniciar sería lento; en ese caso convendría la búsqueda sparse de
  Pinecone o un motor como Elasticsearch. Como `RAGSystem` combina retrievers intercambiables,
  basta con reemplazar el recuperador léxico sin tocar el resto.
- **Recall@5 = 1.00 en todos los modos.** Con 26 chunks, el documento correcto siempre entra en el
  top-5. Por eso las diferencias se ven en MRR (en qué posición aparece) y en Precision@5 (cuánto
  ruido lo acompaña).

## Errores comunes y cómo se evitan

| Error | Mitigación en este repo |
|---|---|
| Mismatch de dimensiones | `setup_index.py` valida dimensión y métrica del índice existente; la ingesta verifica el largo de cada vector; `RAGSystem` rechaza namespaces indexados con otro modelo. |
| Ignorar el namespace | Todos los upserts, consultas, listados y borrados usan `PINECONE_NAMESPACE`. |
| Subestimar el chunking | 600 tokens con 15 % de solapamiento, cortes por estructura Markdown y fusión de fragmentos menores a 120 tokens. |
| Claves en el repo | `.env` en `.gitignore`; `.env.example` sin valores (verificado por test). |
| Consistencia eventual de Serverless | La ingesta espera a que `describe_index_stats` refleje todos los vectores antes de terminar. |
