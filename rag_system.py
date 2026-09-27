"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) fusionados con EnsembleRetriever.

Uso:
    python rag_system.py "¿Cómo limito el tamaño de la caché de lru_cache?"
    python rag_system.py "corrutinas en paralelo" --modo vectorial
    python rag_system.py "ThreadPoolExecutor" --categoria concurrencia

Diseño:
  * PineconeRetriever: retriever de LangChain sobre el SDK nativo de Pinecone (el paquete
    langchain-pinecone todavía no soporta Python 3.14). Embebe la consulta con el MISMO
    modelo de la ingesta y consulta el namespace configurado, con filtros de metadata.
  * BM25Retriever: índice léxico en memoria construido con el texto que ya está guardado en
    la metadata de Pinecone -> una sola fuente de verdad, sin base de datos adicional.
  * EnsembleRetriever: combina ambas listas con Reciprocal Rank Fusion ponderado
    (score = Σ peso / (rango + 60)). Los duplicados se colapsan por chunk_id, así que un
    chunk que aparece en ambas listas suma puntaje de las dos y sube en el ranking.
  * RAGSystem: encapsula todo y devuelve los top-5 chunks para una consulta.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from functools import lru_cache
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Okapi

import config

MODES = ("hibrido", "vectorial", "bm25")

# Stopwords mínimas en español: palabras frecuentes que no aportan a la búsqueda léxica
STOPWORDS = set(
    """a al algo como con cual cuales cuando de del donde el ella en entre es esa ese esta
    este esto hay la las lo los mas me mi muy no o para pero por porque que se si sin sobre
    su sus tambien te tiene un una uno unos y ya yo son ser hace hacer puedo puede cuanto
    cuantos cual""".split()
)
# Identificadores técnicos: palabras, snake_case y rutas con punto (functools.lru_cache)
_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def tokenize(text: str) -> list[str]:
    """Tokenizador para BM25 pensado para documentación técnica.

    * minúsculas y sin tildes ("caché" == "cache"),
    * conserva identificadores completos (lru_cache, asyncio.gather) y además agrega sus
      partes separadas por punto, para que "gather" también encuentre "asyncio.gather",
    * descarta stopwords y tokens de un solo carácter.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(_strip_accents(text.lower())):
        token = token.strip("._")
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append(token)
        if "." in token:
            tokens.extend(part for part in token.split(".") if len(part) > 1)
    return tokens


def bm25_text(doc: Document) -> str:
    """BM25 indexa también título, sección y etiquetas (igual que el embedding usa un header)."""
    m = doc.metadata
    tags = " ".join(m.get("tags") or [])
    return f"{m.get('title', '')} {m.get('section', '')} {tags}\n{doc.page_content}"


def build_bm25_retriever(documents: list[Document], k: int) -> BM25Retriever:
    if not documents:
        raise ValueError("BM25 necesita al menos un documento")
    vectorizer = BM25Okapi([tokenize(bm25_text(d)) for d in documents])
    return BM25Retriever(vectorizer=vectorizer, docs=documents, k=k, preprocess_func=tokenize)


def match_to_document(match: Any) -> Document:
    """Convierte un match de Pinecone en Document: el texto sale de la metadata."""
    metadata = dict(match.metadata or {})
    text = metadata.pop("text", "")
    metadata["score"] = float(match.score) if match.score is not None else None
    return Document(page_content=text, metadata=metadata)


class PineconeRetriever(BaseRetriever):
    """Búsqueda por similitud vectorial en un namespace de Pinecone (SDK nativo)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    embeddings: Any
    namespace: str = config.PINECONE_NAMESPACE
    k: int = config.CANDIDATES_PER_RETRIEVER
    filter: dict | None = None

    _query_cache: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    def embed_query(self, query: str) -> list[float]:
        # Cache simple: la evaluación repite la misma consulta en varios modos
        if query not in self._query_cache:
            self._query_cache[query] = self.embeddings.embed_query(query)
        return self._query_cache[query]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        response = self.index.query(
            vector=self.embed_query(query),
            top_k=self.k,
            namespace=self.namespace,
            filter=self.filter,
            include_metadata=True,
        )
        return [match_to_document(m) for m in response.matches]


def load_corpus(index, namespace: str = config.PINECONE_NAMESPACE) -> list[Document]:
    """Descarga todos los chunks del namespace (ids con list + metadata con fetch)."""
    ids: list[str] = []
    for page in index.list(namespace=namespace):
        ids.extend(item.id for item in page.vectors)

    documents: list[Document] = []
    for start in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[start:start + 100], namespace=namespace).vectors
        for vector in fetched.values():
            metadata = dict(vector.metadata or {})
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))
    documents.sort(key=lambda d: d.metadata.get("chunk_id", ""))
    return documents


class RAGSystem:
    """Recuperador híbrido que encapsula un EnsembleRetriever (BM25 + Pinecone).

    >>> rag = RAGSystem.from_pinecone()
    >>> docs = rag.search("¿Qué hace asyncio.gather?")   # top-5 fusionados
    """

    def __init__(
        self,
        documents: list[Document],
        vector_retriever: PineconeRetriever,
        k: int = config.TOP_K,
        candidates: int = config.CANDIDATES_PER_RETRIEVER,
        weights: tuple[float, float] = (config.BM25_WEIGHT, config.VECTOR_WEIGHT),
    ):
        if not documents:
            raise ValueError(
                f"El namespace '{config.PINECONE_NAMESPACE}' está vacío. Corré primero: "
                "python ingest.py"
            )
        self.documents = documents
        self.k = k
        self.candidates = max(candidates, k)
        self.weights = list(weights)
        self.vector_retriever = vector_retriever.model_copy(update={"k": self.candidates})
        self.bm25_retriever = build_bm25_retriever(documents, self.candidates)
        self.ensemble = self._build_ensemble(self.bm25_retriever, self.vector_retriever)
        self._by_category: dict[str, EnsembleRetriever] = {}

    @classmethod
    def from_pinecone(cls, **kwargs) -> "RAGSystem":
        index = config.get_index()
        documents = load_corpus(index)
        models = {d.metadata.get("embedding_id") for d in documents} - {None}
        if models and models != {config.EMBEDDING_ID}:
            raise RuntimeError(
                f"El namespace fue indexado con {sorted(models)} pero la configuración actual "
                f"usa '{config.EMBEDDING_ID}'. Reindexá con: python ingest.py --rebuild"
            )
        retriever = PineconeRetriever(index=index, embeddings=config.get_embeddings())
        return cls(documents, retriever, **kwargs)

    def _build_ensemble(self, bm25: BaseRetriever, vector: BaseRetriever) -> EnsembleRetriever:
        # id_key: deduplica por chunk_id en vez de comparar el texto completo
        return EnsembleRetriever(retrievers=[bm25, vector], weights=self.weights, id_key="chunk_id")

    def _ensemble_for(self, category: str | None) -> EnsembleRetriever:
        """Ensemble restringido a una categoría (filtro de metadata en ambos recuperadores)."""
        if category is None:
            return self.ensemble
        if category not in self._by_category:
            subset = [d for d in self.documents if d.metadata.get("category") == category]
            if not subset:
                raise ValueError(f"No hay documentos con category='{category}'")
            vector = self.vector_retriever.model_copy(
                update={"filter": {"category": {"$eq": category}}}
            )
            self._by_category[category] = self._build_ensemble(
                build_bm25_retriever(subset, self.candidates), vector
            )
        return self._by_category[category]

    def search(self, query: str, k: int | None = None, category: str | None = None) -> list[Document]:
        """Top-k documentos combinando resultados léxicos (BM25) y semánticos (Pinecone)."""
        k = k or self.k
        fused = self._ensemble_for(category).invoke(query)[:k]
        return [self._tag(doc, rank, "hibrido") for rank, doc in enumerate(fused, start=1)]

    def search_vector(self, query: str, k: int | None = None) -> list[Document]:
        """Sólo búsqueda semántica (para comparar en la evaluación)."""
        docs = self.vector_retriever.invoke(query)[: k or self.k]
        return [self._tag(doc, rank, "vectorial") for rank, doc in enumerate(docs, start=1)]

    def search_bm25(self, query: str, k: int | None = None) -> list[Document]:
        """Sólo búsqueda léxica (para comparar en la evaluación)."""
        docs = self.bm25_retriever.invoke(query)[: k or self.k]
        return [self._tag(doc, rank, "bm25") for rank, doc in enumerate(docs, start=1)]

    def retrieve(self, query: str, mode: str = "hibrido", k: int | None = None) -> list[Document]:
        if mode not in MODES:
            raise ValueError(f"modo debe ser uno de {MODES}")
        return {"hibrido": self.search, "vectorial": self.search_vector, "bm25": self.search_bm25}[
            mode
        ](query, k=k)

    @staticmethod
    def _tag(doc: Document, rank: int, mode: str) -> Document:
        return Document(page_content=doc.page_content, metadata={**doc.metadata, "rank": rank, "mode": mode})


@lru_cache(maxsize=1)
def get_rag_system() -> RAGSystem:
    return RAGSystem.from_pinecone()


def format_result(doc: Document, width: int = 160) -> str:
    m = doc.metadata
    where = f"{m.get('source')}" + (f" pág. {m['page']}" if "page" in m else "")
    score = f" score={m['score']:.3f}" if isinstance(m.get("score"), float) else ""
    snippet = " ".join(doc.page_content.split())[:width]
    return (
        f"#{m.get('rank')} [{m.get('doc_id')}] {m.get('title')} > {m.get('section')}\n"
        f"    fuente: {where} | categoría: {m.get('category')} | tags: {', '.join(m.get('tags') or [])}"
        f"{score}\n    {snippet}..."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta al recuperador híbrido")
    parser.add_argument("consulta", nargs="+")
    parser.add_argument("--modo", choices=MODES, default="hibrido")
    parser.add_argument("--categoria", default=None, help="filtra por metadata 'category'")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    query = " ".join(args.consulta)
    rag = get_rag_system()
    if args.categoria:
        docs = rag.search(query, k=args.k, category=args.categoria)
    else:
        docs = rag.retrieve(query, mode=args.modo, k=args.k)
    print(f"Consulta: {query}  (modo={args.modo}, k={args.k})\n")
    for doc in docs:
        print(format_result(doc), end="\n\n")


if __name__ == "__main__":
    main()
