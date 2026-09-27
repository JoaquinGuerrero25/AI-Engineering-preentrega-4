"""Pipeline de ingesta: lee /data (Markdown, JSON y PDF), fragmenta y sube a Pinecone.

Uso:
    python ingest.py            # sube sólo documentos nuevos o modificados
    python ingest.py --rebuild  # vacía el namespace y reindexa todo
    python ingest.py --dry-run  # muestra los chunks sin llamar a ninguna API

Formatos soportados:
  * .md   -> frontmatter YAML con id, title, category y tags; el cuerpo es el documento.
  * .json -> lista de entradas {id, title, category, tags, content}; cada una es un documento.
  * .pdf  -> una unidad por página (se conserva el número de página en la metadata).

Esquema de cada vector en Pinecone:
  id        "<doc_id>#c<n>"  (PDF: "<doc_id>#p<pag>-c<n>")  -> permite listar/borrar por prefijo
  values    embedding de "Título | Sección" + texto del chunk (contextual chunk header)
  metadata  text (texto ORIGINAL del chunk), doc_id, chunk_id, source, source_type, title,
            section, category, tags, page (sólo PDF), chunk_index, n_tokens,
            content_hash, embedding_id
  Guardar el texto en la metadata evita una base relacional extra: la respuesta de Pinecone
  ya trae todo lo necesario para construir el contexto y para armar el índice BM25.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SUPPORTED_EXTENSIONS = {".md", ".json", ".pdf"}

# Del separador más "fuerte" (cambio de sección) al más débil (carácter).
SEPARATORS = ["\n## ", "\n### ", "\n```\n", "\n\n", "\n", ". ", " ", ""]

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)


@dataclass
class SourceDocument:
    """Unidad lógica antes del chunking (un .md, una entrada JSON o una página de PDF)."""

    doc_id: str
    title: str
    source: str
    source_type: str
    category: str
    text: str
    tags: list[str] = field(default_factory=list)
    page: int | None = None
    content_hash: str = ""


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "doc"


def _clean_tags(tags) -> list[str]:
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return [str(t).strip().lower() for t in (tags or []) if str(t).strip()]


# ------------------------------- Carga ------------------------------------ #
def load_markdown(path: Path) -> list[SourceDocument]:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    match = _FRONTMATTER_RE.match(raw)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        body = raw[match.end():]
    return [
        SourceDocument(
            doc_id=str(meta.get("id") or _slug(path.stem)),
            title=str(meta.get("title") or path.stem),
            source=path.name,
            source_type="markdown",
            category=str(meta.get("category") or "general"),
            tags=_clean_tags(meta.get("tags")),
            text=body.strip(),
            content_hash=sha256(raw),
        )
    ]


def load_json(path: Path) -> list[SourceDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["documents"] if isinstance(data, dict) else data
    docs = []
    for entry in entries:
        content = str(entry["content"]).strip()
        docs.append(
            SourceDocument(
                doc_id=str(entry.get("id") or _slug(entry["title"])),
                title=str(entry["title"]),
                source=path.name,
                source_type="json",
                category=str(entry.get("category") or "general"),
                tags=_clean_tags(entry.get("tags")),
                text=content,
                content_hash=sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False)),
            )
        )
    return docs


def load_pdf(path: Path) -> list[SourceDocument]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    info = reader.metadata or {}
    title = str(info.get("/Title") or path.stem)
    # Categoría y etiquetas se leen de las propiedades del PDF (Subject / Keywords)
    category = str(info.get("/Subject") or "general")
    tags = _clean_tags(info.get("/Keywords"))
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            SourceDocument(
                doc_id=_slug(path.stem),
                title=title,
                source=path.name,
                source_type="pdf",
                category=category,
                tags=tags,
                text=text,
                page=number,
                content_hash=file_hash,
            )
        )
    return docs


LOADERS = {".md": load_markdown, ".json": load_json, ".pdf": load_pdf}


def load_documents(data_dir: Path = config.DATA_DIR) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(LOADERS[path.suffix.lower()](path))
    if not docs:
        raise FileNotFoundError(f"No hay documentos {sorted(SUPPORTED_EXTENSIONS)} en {data_dir}")
    return docs


# ------------------------------ Chunking ----------------------------------- #
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.TOKEN_ENCODING,
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=SEPARATORS,
        add_start_index=True,  # posición del chunk en el texto original
    )


def _section_at(text: str, position: int, default: str) -> str:
    """Último encabezado Markdown que aparece antes de `position` (o dentro del chunk)."""
    section = default
    for match in _HEADING_RE.finditer(text):
        if match.start() > position:
            break
        section = match.group(1).strip()
    return section


def _locate(text: str, content: str, cursor: int, probe: int = 80) -> list[int] | None:
    """Posición [inicio, fin) de un fragmento en el texto original.

    No se usa start_index del splitter porque vale -1 cuando el splitter reescribe los
    separadores del fragmento; se busca por prefijo y sufijo a partir de `cursor`.
    """
    start = text.find(content[:probe], cursor)
    if start < 0:
        return None
    tail = content[-probe:]
    end = text.find(tail, start + max(len(content) - 2 * probe, 0))
    return None if end < 0 else [start, end + len(tail)]


def merge_small_pieces(text: str, pieces: list[Document]) -> list[Document]:
    """Fusiona fragmentos menores a MIN_CHUNK_TOKENS con su vecino.

    El splitter corta primero por secciones, así que una introducción corta o el final
    de un documento pueden quedar como chunks de 50-100 tokens sin contexto semántico.
    Se unen con el chunk adyacente (siguiente si es el primero, anterior si no) usando
    las posiciones originales, así el solapamiento no se duplica. El resultado puede
    superar CHUNK_SIZE_TOKENS, pero nunca MAX_MERGED_TOKENS.
    """
    spans: list[list[int]] = []
    cursor = 0
    for piece in pieces:
        span = _locate(text, piece.page_content, cursor)
        if span is None:  # no se pudo ubicar en el original: se deja el chunking tal cual
            return pieces
        spans.append(span)
        cursor = span[0] + 1
    i = 0
    while len(spans) > 1 and i < len(spans):
        start, end = spans[i]
        if count_tokens(text[start:end]) >= config.MIN_CHUNK_TOKENS:
            i += 1
            continue
        j = i + 1 if i == 0 else i - 1
        lo, hi = min(start, spans[j][0]), max(end, spans[j][1])
        if count_tokens(text[lo:hi]) > config.MAX_MERGED_TOKENS:
            i += 1
            continue
        spans[min(i, j)] = [lo, hi]
        del spans[max(i, j)]
        i = max(min(i, j), 0)
    return [
        Document(page_content=text[s:e].strip(), metadata={"start_index": s}) for s, e in spans
    ]


def chunk_document(doc: SourceDocument, splitter=None) -> list[Document]:
    splitter = splitter or get_splitter()
    pieces = merge_small_pieces(doc.text, splitter.create_documents([doc.text]))
    chunks = []
    for i, piece in enumerate(pieces):
        start = piece.metadata.get("start_index", 0)
        # Si el chunk arranca justo en un encabezado, ése es su sección
        heading = _HEADING_RE.match(piece.page_content.lstrip())
        section = heading.group(1).strip() if heading else _section_at(doc.text, start, doc.title)
        chunk_id = f"{doc.doc_id}#p{doc.page}-c{i}" if doc.page else f"{doc.doc_id}#c{i}"
        metadata = {
            "chunk_id": chunk_id,
            "doc_id": doc.doc_id,
            "source": doc.source,
            "source_type": doc.source_type,
            "title": doc.title,
            "section": section,
            "category": doc.category,
            "tags": doc.tags,
            "chunk_index": i,
            "n_tokens": count_tokens(piece.page_content),
            "content_hash": doc.content_hash,
            "embedding_id": config.EMBEDDING_ID,
        }
        if doc.page is not None:
            metadata["page"] = doc.page  # Pinecone no acepta None en la metadata
        chunks.append(Document(page_content=piece.page_content, metadata=metadata))
    return chunks


def embedding_text(chunk: Document) -> str:
    """Texto que se embebe: encabezado de contexto + chunk (el título no se pierde al cortar)."""
    m = chunk.metadata
    header = f"{m['title']} | {m['section']}"
    if "page" in m:
        header += f" | página {m['page']}"
    return f"{header}\n\n{chunk.page_content}"


def to_pinecone_record(chunk: Document, vector: list[float]) -> dict:
    return {
        "id": chunk.metadata["chunk_id"],
        "values": vector,
        "metadata": {**chunk.metadata, "text": chunk.page_content},
    }


# ------------------------------- Pinecone ---------------------------------- #
def list_ids(index, prefix: str, namespace: str) -> list[str]:
    ids: list[str] = []
    for page in index.list(prefix=prefix, namespace=namespace):
        ids.extend(item.id for item in page.vectors)
    return ids


def is_up_to_date(index, doc_id: str, content_hash: str, namespace: str) -> tuple[bool, list[str]]:
    """¿Ya está este documento en Pinecone con el mismo contenido y modelo de embeddings?"""
    existing = list_ids(index, prefix=f"{doc_id}#", namespace=namespace)
    if not existing:
        return False, existing
    stored = index.fetch(ids=existing[:1], namespace=namespace).vectors
    meta = {}
    if stored:
        meta = next(iter(stored.values())).metadata or {}
    same = meta.get("content_hash") == content_hash and meta.get("embedding_id") == config.EMBEDDING_ID
    return same, existing


def clear_namespace(index, namespace: str) -> None:
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:  # el namespace todavía no existe: nada que borrar
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise


def wait_for_count(index, namespace: str, expected: int, timeout_s: float = 60) -> int:
    """Pinecone Serverless es eventualmente consistente: esperamos a ver los vectores."""
    deadline = time.monotonic() + timeout_s
    count = -1
    while time.monotonic() < deadline:
        stats = index.describe_index_stats()
        ns = (getattr(stats, "namespaces", None) or {}).get(namespace)
        count = getattr(ns, "vector_count", 0) if ns else 0
        if count >= expected:
            break
        time.sleep(2)
    return count


def ingest(rebuild: bool = False) -> None:
    from setup_index import ensure_index

    docs = load_documents()
    pc = config.get_pinecone_client()
    ensure_index(pc)  # crea el índice si falta y valida dimensión/métrica
    index = config.get_index(pc)
    embeddings = config.get_embeddings()
    namespace = config.PINECONE_NAMESPACE

    if rebuild:
        print(f"--rebuild: vaciando el namespace '{namespace}'...")
        clear_namespace(index, namespace)

    # Agrupar por doc_id: un PDF aporta varias páginas bajo el mismo documento
    by_doc: dict[str, list[SourceDocument]] = {}
    for doc in docs:
        by_doc.setdefault(doc.doc_id, []).append(doc)

    splitter = get_splitter()
    total_chunks = uploaded = skipped = 0
    for doc_id, units in by_doc.items():
        chunks = [c for unit in units for c in chunk_document(unit, splitter)]
        total_chunks += len(chunks)
        if not rebuild:
            up_to_date, existing = is_up_to_date(index, doc_id, units[0].content_hash, namespace)
            if up_to_date and len(existing) == len(chunks):
                skipped += 1
                print(f"  = {doc_id:<28} sin cambios ({len(chunks)} chunks)")
                continue
            if existing:  # el documento cambió: borrar sus chunks viejos (pueden ser más)
                index.delete(ids=existing, namespace=namespace)

        vectors = embeddings.embed_documents([embedding_text(c) for c in chunks])
        if any(len(v) != config.EMBEDDING_DIMENSION for v in vectors):
            raise ValueError(
                f"El modelo devolvió vectores de {len(vectors[0])} dims y el índice espera "
                f"{config.EMBEDDING_DIMENSION}."
            )
        records = [to_pinecone_record(c, v) for c, v in zip(chunks, vectors)]
        index.upsert(vectors=records, namespace=namespace, batch_size=100, show_progress=False)
        uploaded += 1
        print(f"  + {doc_id:<28} {len(chunks)} chunks subidos ({units[0].source})")

    count = wait_for_count(index, namespace, total_chunks)
    print(
        f"\nListo: {len(by_doc)} documentos -> {total_chunks} chunks | "
        f"{uploaded} subidos, {skipped} sin cambios | "
        f"namespace '{namespace}' con {count} vectores en el índice '{config.INDEX_NAME}'."
    )


def dry_run() -> None:
    docs = load_documents()
    splitter = get_splitter()
    total = 0
    for doc in docs:
        chunks = chunk_document(doc, splitter)
        total += len(chunks)
        where = f"pág. {doc.page}" if doc.page else doc.source
        tokens = [c.metadata["n_tokens"] for c in chunks]
        print(f"{doc.doc_id:<28} [{doc.category}] {where:<32} {len(chunks)} chunks, tokens={tokens}")
    print(f"\nTotal: {len(docs)} unidades -> {total} chunks "
          f"(chunk_size={config.CHUNK_SIZE_TOKENS}, overlap={config.CHUNK_OVERLAP_TOKENS} tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos en Pinecone")
    parser.add_argument("--rebuild", action="store_true", help="vacía el namespace y reindexa todo")
    parser.add_argument("--dry-run", action="store_true", help="sólo muestra el chunking, sin APIs")
    args = parser.parse_args()
    dry_run() if args.dry_run else ingest(rebuild=args.rebuild)
