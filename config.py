"""Configuración compartida: variables de entorno, Pinecone y modelo de embeddings.

Todo lo que tiene que coincidir entre la ingesta y la consulta vive acá:
  * el modelo de embeddings y su DIMENSIÓN (debe ser igual a la del índice),
  * el nombre del índice y el NAMESPACE,
  * los parámetros de chunking.

Proveedores de embeddings soportados (EMBEDDING_PROVIDER en .env):
  * gemini  (por defecto, tiene plan gratuito) -> gemini-embedding-001 recortado a 1536 dims
  * openai                                     -> text-embedding-3-small (1536 dims nativas)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# La consola de Windows no siempre usa UTF-8: evita caracteres rotos en tildes y eñes
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GOLDEN_SET_PATH = BASE_DIR / "golden_set.json"

# ------------------------------- Pinecone ---------------------------------- #
INDEX_NAME = os.getenv("INDEX_NAME", "python-docs-rag")
# Namespace: separa este dataset de otros dentro del mismo índice (búsqueda menos ruidosa)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "python-stdlib-es")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # región del plan gratuito
PINECONE_METRIC = "cosine"

# ------------------------------ Embeddings --------------------------------- #
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))

PROVIDERS = {
    "gemini": {
        "api_key_vars": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "embeddings": "models/gemini-embedding-001",
    },
    "openai": {
        "api_key_vars": ("OPENAI_API_KEY",),
        "embeddings": "text-embedding-3-small",
    },
}
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()
if EMBEDDING_PROVIDER not in PROVIDERS:
    raise ValueError(
        f"EMBEDDING_PROVIDER debe ser uno de {list(PROVIDERS)}, no '{EMBEDDING_PROVIDER}'"
    )
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", PROVIDERS[EMBEDDING_PROVIDER]["embeddings"])
# Se guarda en la metadata de cada vector para detectar índices creados con otro modelo
EMBEDDING_ID = f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}:{EMBEDDING_DIMENSION}"

# ------------------------------- Chunking ---------------------------------- #
# Punto medio recomendado (~500-800 tokens): 600 tokens con 15 % de solapamiento
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 600))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 90))
# Chunks más chicos que esto pierden contexto: se fusionan con su vecino (hasta 800 tokens)
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", 120))
MAX_MERGED_TOKENS = int(os.getenv("MAX_MERGED_TOKENS", 800))
TOKEN_ENCODING = "cl100k_base"

# ------------------------------ Recuperación ------------------------------- #
TOP_K = int(os.getenv("TOP_K", 5))
# Cuántos candidatos trae CADA recuperador antes de fusionarlos con RRF
CANDIDATES_PER_RETRIEVER = int(os.getenv("CANDIDATES_PER_RETRIEVER", 10))
# Pesos del EnsembleRetriever: [BM25 (léxico), Pinecone (semántico)]
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.5))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", 0.5))


def _require_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        f"Falta la variable {names[0]}. Copiá .env.example a .env y completala "
        "(el archivo .env está en .gitignore y nunca se sube al repo)."
    )


def get_pinecone_client():
    from pinecone import Pinecone

    return Pinecone(api_key=_require_env("PINECONE_API_KEY"))


def get_index(pc=None):
    """Cliente de datos del índice. Supone que el índice ya existe (ver setup_index.py)."""
    pc = pc or get_pinecone_client()
    return pc.Index(INDEX_NAME)


def get_embeddings():
    """Única fuente del modelo de embeddings, usada tanto al indexar como al consultar."""
    _require_env(*PROVIDERS[EMBEDDING_PROVIDER]["api_key_vars"])
    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # gemini-embedding-001 genera 3072 dims; con output_dimensionality lo recortamos
        # a 1536 para que coincida con el índice (Matryoshka). La métrica coseno
        # normaliza, así que no hace falta re-normalizar los vectores recortados.
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMENSION
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION)
