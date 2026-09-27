"""Evaluación cuantitativa del recuperador con un Golden Set: Recall@k y Precision@k.

Uso:
    python evaluate.py                 # k=5, compara modos bm25 / vectorial / híbrido
    python evaluate.py -k 3
    python evaluate.py --guardar resultados/metricas.json

Definiciones (por pregunta, luego se promedia):
  * Relevantes: los chunks del documento esperado (`documento_id_esperado`) y, si se
    indican, de `documentos_relevantes_extra`.
  * Recall@k    = documentos relevantes que aparecen en el top-k / documentos relevantes.
                  Con un único documento esperado vale 1 si aparece y 0 si no
                  ("¿está el documento correcto entre los k recuperados?").
  * Precision@k = chunks del top-k que pertenecen a un documento relevante / k
                  ("¿qué porcentaje de los k recuperados son realmente útiles?").
  * P@k máx.    = techo alcanzable: si el documento esperado sólo tiene 2 chunks, la
                  precisión nunca puede superar 2/k. Se informa para interpretar P@k.
  * MRR         = 1 / posición del primer chunk relevante (métrica extra de ranking).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document

import config
from rag_system import MODES, RAGSystem

MODE_LABELS = {"bm25": "BM25 (léxico)", "vectorial": "Pinecone (semántico)", "hibrido": "Híbrido (Ensemble)"}


@dataclass
class QueryResult:
    pregunta: str
    esperado: str
    recuperados: list[str]
    recall: float
    precision: float
    precision_max: float
    reciprocal_rank: float


def load_golden_set(path: Path = config.GOLDEN_SET_PATH) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        if not {"pregunta", "documento_id_esperado"} <= item.keys():
            raise ValueError(f"Entrada inválida en el golden set: {item}")
    return items


def relevant_ids(item: dict) -> set[str]:
    return {item["documento_id_esperado"], *item.get("documentos_relevantes_extra", [])}


def recall_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(retrieved_doc_ids[:k])) / len(relevant)


def precision_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    return sum(doc_id in relevant for doc_id in retrieved_doc_ids[:k]) / k


def reciprocal_rank(retrieved_doc_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def evaluate_query(item: dict, docs: list[Document], k: int, chunks_per_doc: dict[str, int]) -> QueryResult:
    relevant = relevant_ids(item)
    retrieved = [d.metadata["doc_id"] for d in docs[:k]]
    available = sum(chunks_per_doc.get(doc_id, 0) for doc_id in relevant)
    return QueryResult(
        pregunta=item["pregunta"],
        esperado=item["documento_id_esperado"],
        recuperados=retrieved,
        recall=recall_at_k(retrieved, relevant, k),
        precision=precision_at_k(retrieved, relevant, k),
        precision_max=min(available, k) / k,
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
    )


def run_evaluation(rag: RAGSystem, golden_set: list[dict], k: int, modes=MODES) -> dict[str, list[QueryResult]]:
    chunks_per_doc: dict[str, int] = {}
    for doc in rag.documents:
        chunks_per_doc[doc.metadata["doc_id"]] = chunks_per_doc.get(doc.metadata["doc_id"], 0) + 1
    return {
        mode: [evaluate_query(item, rag.retrieve(item["pregunta"], mode=mode, k=k), k, chunks_per_doc)
               for item in golden_set]
        for mode in modes
    }


def summarize(results: list[QueryResult]) -> dict[str, float]:
    return {
        "recall": mean(r.recall for r in results),
        "precision": mean(r.precision for r in results),
        "precision_max": mean(r.precision_max for r in results),
        "mrr": mean(r.reciprocal_rank for r in results),
    }


def print_report(all_results: dict[str, list[QueryResult]], k: int) -> None:
    line = "=" * 92
    print(line)
    print(f" EVALUACIÓN DEL RECUPERADOR  |  índice '{config.INDEX_NAME}'  |  "
          f"namespace '{config.PINECONE_NAMESPACE}'  |  k={k}")
    print(line)

    detail_mode = "hibrido" if "hibrido" in all_results else next(iter(all_results))
    print(f"\nDetalle por pregunta — {MODE_LABELS[detail_mode]}\n")
    for i, r in enumerate(all_results[detail_mode], start=1):
        estado = "OK " if r.recall == 1 else ("PARCIAL" if r.recall > 0 else "FALLO")
        print(f" {i}. {r.pregunta}")
        print(f"    esperado: {r.esperado:<24} [{estado}] Recall@{k}={r.recall:.2f}  "
              f"Precision@{k}={r.precision:.2f} (máx {r.precision_max:.2f})  RR={r.reciprocal_rank:.2f}")
        print(f"    top-{k}: {', '.join(r.recuperados)}\n")

    print(f"Resumen (promedio sobre {len(all_results[detail_mode])} preguntas)\n")
    header = f" {'Modo':<24}{'Recall@' + str(k):>12}{'Precision@' + str(k):>15}{'P@' + str(k) + ' máx.':>11}{'MRR':>8}"
    print(header)
    print(" " + "-" * (len(header) - 1))
    for mode, results in all_results.items():
        s = summarize(results)
        print(f" {MODE_LABELS[mode]:<24}{s['recall']:>12.2f}{s['precision']:>15.2f}"
              f"{s['precision_max']:>11.2f}{s['mrr']:>8.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k y Precision@k sobre el golden set")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_SET_PATH)
    parser.add_argument("--modos", nargs="+", choices=MODES, default=["bm25", "vectorial", "hibrido"])
    parser.add_argument("--guardar", type=Path, default=None, help="guarda los resultados en JSON")
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden)
    rag = RAGSystem.from_pinecone(k=args.k)
    results = run_evaluation(rag, golden_set, args.k, args.modos)
    print_report(results, args.k)

    if args.guardar:
        args.guardar.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k": args.k,
            "index": config.INDEX_NAME,
            "namespace": config.PINECONE_NAMESPACE,
            "embedding": config.EMBEDDING_ID,
            "resumen": {mode: summarize(res) for mode, res in results.items()},
            "detalle": {mode: [asdict(r) for r in res] for mode, res in results.items()},
        }
        args.guardar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Resultados guardados en {args.guardar}")


if __name__ == "__main__":
    main()
