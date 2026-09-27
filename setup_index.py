"""Inicialización de la infraestructura: crea el índice Serverless de Pinecone si no existe.

Uso:
    python setup_index.py

Qué hace:
  1. Verifica si INDEX_NAME existe en el proyecto de Pinecone.
  2. Si no existe, lo crea en modo Serverless (cloud/región configurables) con la
     dimensión del modelo de embeddings (1536) y métrica coseno, y espera a que esté listo.
  3. Si ya existe, valida que su dimensión y métrica coincidan con la configuración.
     Así se evita el error clásico de subir vectores de 1536 dims a un índice de 768.
"""

from __future__ import annotations

import time

import config


class IndexConfigError(RuntimeError):
    """El índice existe pero no es compatible con el modelo de embeddings configurado."""


def _describe(pc, name: str) -> tuple[int | None, str | None, bool]:
    desc = pc.describe_index(name)
    # dimension/metric son atributos derivados; en índices con varios campos no existen
    dimension = getattr(desc, "dimension", None)
    metric = getattr(desc, "metric", None)
    ready = bool(getattr(getattr(desc, "status", None), "ready", False))
    return dimension, (str(metric).lower() if metric else None), ready


def validate_index(pc, name: str = config.INDEX_NAME) -> None:
    dimension, metric, _ = _describe(pc, name)
    if dimension is not None and dimension != config.EMBEDDING_DIMENSION:
        raise IndexConfigError(
            f"El índice '{name}' tiene dimensión {dimension}, pero el modelo "
            f"'{config.EMBEDDING_MODEL}' está configurado en {config.EMBEDDING_DIMENSION}. "
            "Usá otro INDEX_NAME o borrá el índice desde la consola de Pinecone."
        )
    if metric is not None and metric != config.PINECONE_METRIC:
        raise IndexConfigError(
            f"El índice '{name}' usa la métrica '{metric}' y se esperaba "
            f"'{config.PINECONE_METRIC}'."
        )


def wait_until_ready(pc, name: str, timeout_s: float = 120, poll_s: float = 2) -> None:
    deadline = time.monotonic() + timeout_s
    while not _describe(pc, name)[2]:
        if time.monotonic() > deadline:
            raise TimeoutError(f"El índice '{name}' no quedó listo en {timeout_s} s")
        time.sleep(poll_s)


def ensure_index(pc=None, name: str = config.INDEX_NAME) -> bool:
    """Crea el índice si no existe. Devuelve True si lo creó, False si ya existía."""
    from pinecone import ServerlessSpec

    pc = pc or config.get_pinecone_client()
    if pc.has_index(name):
        validate_index(pc, name)
        return False

    pc.create_index(
        name=name,
        dimension=config.EMBEDDING_DIMENSION,
        metric=config.PINECONE_METRIC,
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    wait_until_ready(pc, name)
    return True


def main() -> None:
    pc = config.get_pinecone_client()
    created = ensure_index(pc)
    estado = "creado" if created else "ya existía (configuración verificada)"
    print(f"Índice '{config.INDEX_NAME}': {estado}")
    print(
        f"  Serverless {config.PINECONE_CLOUD}/{config.PINECONE_REGION} | "
        f"dimensión {config.EMBEDDING_DIMENSION} | métrica {config.PINECONE_METRIC}"
    )
    print(f"  Modelo de embeddings: {config.EMBEDDING_MODEL} ({config.EMBEDDING_PROVIDER})")

    stats = config.get_index(pc).describe_index_stats()
    namespaces = getattr(stats, "namespaces", None) or {}
    print(f"  Vectores totales: {getattr(stats, 'total_vector_count', 0)}")
    for ns, info in namespaces.items():
        print(f"    namespace '{ns}': {getattr(info, 'vector_count', '?')} vectores")


if __name__ == "__main__":
    main()
