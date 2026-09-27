"""Tests que corren sin API keys: un Pinecone falso en memoria y embeddings deterministas.

    python -m pytest -q

Cubren el flujo completo (setup -> ingesta -> RAGSystem -> métricas) sin tocar la red.
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import evaluate
import ingest
import setup_index
from rag_system import RAGSystem, tokenize


# ------------------------------ Dobles de prueba ------------------------------ #
class HashingEmbeddings:
    """Bolsa de palabras con hashing: determinista y con algo de 'semántica' léxica."""

    def __init__(self, dim=config.EMBEDDING_DIMENSION):
        self.dim = dim
        self.calls = 0

    def _embed(self, text):
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeIndex:
    """Subconjunto del cliente de datos de Pinecone usado por el proyecto."""

    def __init__(self):
        self.data: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def upsert(self, vectors, namespace="", **_):
        ns = self.data.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = (v["values"], v["metadata"])

    def _matches_filter(self, metadata, flt):
        return all(metadata.get(k) == cond["$eq"] for k, cond in (flt or {}).items())

    def query(self, vector, top_k, namespace="", filter=None, include_metadata=False, **_):
        scored = [
            SimpleNamespace(id=i, score=sum(a * b for a, b in zip(vector, vals)), metadata=dict(meta))
            for i, (vals, meta) in self.data.get(namespace, {}).items()
            if self._matches_filter(meta, filter)
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return SimpleNamespace(matches=scored[:top_k])

    def list(self, prefix=None, namespace="", **_):
        ids = sorted(i for i in self.data.get(namespace, {}) if i.startswith(prefix or ""))
        for start in range(0, len(ids), 100):
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids[start:start + 100]])

    def fetch(self, ids, namespace=""):
        ns = self.data.get(namespace, {})
        return SimpleNamespace(
            vectors={i: SimpleNamespace(id=i, metadata=dict(ns[i][1])) for i in ids if i in ns}
        )

    def delete(self, ids=None, delete_all=False, namespace="", **_):
        if delete_all:
            self.data.pop(namespace, None)
        for i in ids or []:
            self.data.get(namespace, {}).pop(i, None)

    def describe_index_stats(self):
        return SimpleNamespace(
            total_vector_count=sum(len(v) for v in self.data.values()),
            namespaces={ns: SimpleNamespace(vector_count=len(v)) for ns, v in self.data.items()},
        )


class FakePinecone:
    def __init__(self, exists=True, dimension=config.EMBEDDING_DIMENSION, metric="cosine"):
        self.exists, self.dimension, self.metric = exists, dimension, metric
        self.created_with = None
        self.index = FakeIndex()

    def has_index(self, name):
        return self.exists

    def create_index(self, **kwargs):
        self.created_with = kwargs
        self.exists = True
        self.dimension = kwargs["dimension"]

    def describe_index(self, name):
        return SimpleNamespace(
            dimension=self.dimension, metric=self.metric, status=SimpleNamespace(ready=True)
        )

    def Index(self, name):
        return self.index


@pytest.fixture
def fake_cloud(monkeypatch):
    pc, emb = FakePinecone(), HashingEmbeddings()
    monkeypatch.setattr(config, "get_pinecone_client", lambda: pc)
    monkeypatch.setattr(config, "get_embeddings", lambda: emb)
    return pc, emb


@pytest.fixture
def rag(fake_cloud):
    ingest.ingest()
    return RAGSystem.from_pinecone()


# -------------------------------- Infraestructura ----------------------------- #
def test_setup_crea_indice_serverless_si_no_existe():
    pc = FakePinecone(exists=False)
    assert setup_index.ensure_index(pc) is True
    args = pc.created_with
    assert args["dimension"] == 1536 and args["metric"] == "cosine"
    assert args["spec"].cloud == config.PINECONE_CLOUD
    assert args["spec"].region == config.PINECONE_REGION


def test_setup_no_recrea_un_indice_existente():
    pc = FakePinecone(exists=True)
    assert setup_index.ensure_index(pc) is False
    assert pc.created_with is None


def test_setup_detecta_mismatch_de_dimensiones():
    with pytest.raises(setup_index.IndexConfigError, match="dimensión 768"):
        setup_index.ensure_index(FakePinecone(exists=True, dimension=768))


def test_env_example_no_contiene_claves_reales():
    text = (config.BASE_DIR / ".env.example").read_text(encoding="utf-8")
    for var in ("PINECONE_API_KEY", "OPENAI_API_KEY", "INDEX_NAME"):
        assert var in text
    for line in text.splitlines():
        if line.split("=")[0] in ("PINECONE_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
            assert line.endswith("="), f"{line.split('=')[0]} debe quedar vacía en el ejemplo"


# ----------------------------------- Ingesta ---------------------------------- #
def test_carga_los_tres_formatos():
    docs = ingest.load_documents()
    assert {d.source_type for d in docs} == {"markdown", "json", "pdf"}
    pdf_pages = [d.page for d in docs if d.source_type == "pdf"]
    assert pdf_pages == [1, 2, 3]
    assert all(d.category and d.tags for d in docs)


def test_chunks_con_tamano_razonable_y_metadata_valida():
    for doc in ingest.load_documents():
        for chunk in ingest.chunk_document(doc):
            m = chunk.metadata
            assert m["n_tokens"] <= config.MAX_MERGED_TOKENS
            assert chunk.page_content.strip()
            # Pinecone no acepta None y sólo admite str, número, bool o lista de str
            for key, value in m.items():
                assert value is not None, key
                assert isinstance(value, (str, int, float, bool, list)), key
            assert m["chunk_id"].startswith(f"{m['doc_id']}#")
            assert ("page" in m) == (doc.source_type == "pdf")


def test_se_fusionan_los_chunks_demasiado_chicos():
    docs = {d.doc_id: d for d in ingest.load_documents()}
    chunks = ingest.chunk_document(docs["functools"])
    # La intro de functools (~70 tokens) quedaba sola; ahora va unida a la sección siguiente
    assert all(c.metadata["n_tokens"] >= config.MIN_CHUNK_TOKENS for c in chunks)
    assert "lru_cache: memoización" in chunks[0].page_content


def test_registro_pinecone_guarda_texto_original_en_metadata():
    doc = ingest.load_documents()[0]
    chunk = ingest.chunk_document(doc)[0]
    record = ingest.to_pinecone_record(chunk, [0.0] * 3)
    assert record["metadata"]["text"] == chunk.page_content
    assert record["metadata"]["source"] == doc.source
    # el embedding usa un encabezado de contexto con título y sección
    assert ingest.embedding_text(chunk).startswith(f"{doc.title} | ")


def test_ingesta_sube_todo_al_namespace_y_es_idempotente(fake_cloud):
    pc, emb = fake_cloud
    ingest.ingest()
    total = sum(len(ingest.chunk_document(d)) for d in ingest.load_documents())
    stored = pc.index.data[config.PINECONE_NAMESPACE]
    assert len(stored) == total
    assert "" not in pc.index.data  # nada en el namespace por defecto

    calls = emb.calls
    ingest.ingest()  # segunda corrida: nada cambió, no se vuelve a embeber
    assert emb.calls == calls


def test_ingesta_reemplaza_chunks_viejos_de_un_documento_modificado(fake_cloud):
    pc, _ = fake_cloud
    ingest.ingest()
    ns = pc.index.data[config.PINECONE_NAMESPACE]
    ns["asyncio#c99"] = ([0.0], {"content_hash": "viejo", "embedding_id": config.EMBEDDING_ID})
    for vid in [v for v in ns if v.startswith("asyncio#")]:
        ns[vid][1]["content_hash"] = "viejo"
    ingest.ingest()
    assert "asyncio#c99" not in pc.index.data[config.PINECONE_NAMESPACE]


# ------------------------------ Recuperador híbrido --------------------------- #
def test_tokenizador_normaliza_tildes_y_conserva_identificadores():
    tokens = tokenize("¿Cómo uso asyncio.gather y la caché de lru_cache?")
    assert "asyncio.gather" in tokens and "gather" in tokens and "asyncio" in tokens
    assert "lru_cache" in tokens and "cache" in tokens  # "caché" sin tilde
    assert "como" not in tokens and "de" not in tokens  # stopwords


def test_rag_system_devuelve_top5_sin_duplicados(rag):
    docs = rag.search("¿Qué diferencia hay entre ProcessPoolExecutor y ThreadPoolExecutor?")
    assert len(docs) == 5
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(set(ids)) == 5
    assert [d.metadata["rank"] for d in docs] == [1, 2, 3, 4, 5]
    assert docs[0].metadata["doc_id"] == "concurrent-futures"
    assert all(d.page_content for d in docs)  # el texto viene de la metadata de Pinecone


def test_ensemble_combina_ambos_recuperadores(rag):
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    assert isinstance(rag.ensemble, EnsembleRetriever)
    kinds = {type(r).__name__ for r in rag.ensemble.retrievers}
    assert kinds == {BM25Retriever.__name__, "PineconeRetriever"}
    assert rag.ensemble.id_key == "chunk_id"


def test_filtro_por_categoria(rag):
    docs = rag.search("¿Cómo agrupo elementos?", category="programacion-funcional")
    assert docs and all(d.metadata["category"] == "programacion-funcional" for d in docs)


def test_rechaza_namespace_con_otro_modelo_de_embeddings(fake_cloud):
    pc, _ = fake_cloud
    ingest.ingest()
    for _, meta in pc.index.data[config.PINECONE_NAMESPACE].values():
        meta["embedding_id"] = "otro:modelo:768"
    with pytest.raises(RuntimeError, match="--rebuild"):
        RAGSystem.from_pinecone()


# ---------------------------------- Métricas ---------------------------------- #
def test_metricas_basicas():
    retrieved = ["a", "b", "a", "c", "d"]
    assert evaluate.recall_at_k(retrieved, {"a"}, 5) == 1.0
    assert evaluate.recall_at_k(retrieved, {"z"}, 5) == 0.0
    assert evaluate.recall_at_k(retrieved, {"a", "z"}, 5) == 0.5
    assert evaluate.precision_at_k(retrieved, {"a"}, 5) == 0.4
    assert evaluate.reciprocal_rank(retrieved, {"c"}) == pytest.approx(1 / 4)
    assert evaluate.recall_at_k(retrieved, {"d"}, 3) == 0.0  # fuera del top-3


def test_golden_set_apunta_a_documentos_existentes():
    golden = evaluate.load_golden_set()
    assert len(golden) >= 5
    doc_ids = {d.doc_id for d in ingest.load_documents()}
    for item in golden:
        assert evaluate.relevant_ids(item) <= doc_ids, item


def test_evaluacion_completa_de_punta_a_punta(rag, capsys):
    golden = evaluate.load_golden_set()
    results = evaluate.run_evaluation(rag, golden, k=5)
    assert set(results) == {"hibrido", "vectorial", "bm25"}
    for mode, res in results.items():
        assert len(res) == len(golden)
        for r in res:
            assert 0 <= r.precision <= r.precision_max <= 1
            assert r.recall in (0.0, 1.0) and len(r.recuperados) == 5
    evaluate.print_report(results, 5)
    out = capsys.readouterr().out
    assert "Recall@5" in out and "Precision@5" in out
