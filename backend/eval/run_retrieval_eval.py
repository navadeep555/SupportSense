"""Measure retrieval quality on the labelled set in eval/retrieval_eval_set.json.

Run from backend/ (after `python -m app.rag.ingest`):
    python -m eval.run_retrieval_eval

Metrics (answerable queries only, i.e. relevant_doc_ids is not empty):
  Recall@k : of the relevant documents, what fraction appear in the top-k chunks?
             Averaged over queries. 1.0 = every relevant doc was retrieved.
  MRR@k    : 1 / (rank of the first relevant chunk), 0 if none in top-k.
             Rewards putting the right document near the top.

Evidence check (all queries):
  For each threshold we count answerable queries correctly ACCEPTED and
  unanswerable ones correctly REJECTED, which shows the trade-off between
  answering too little and answering things the docs don't cover.
"""
import json
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.rag.retriever import load_retriever

EVAL_FILE = Path(__file__).with_name("retrieval_eval_set.json")
MODES = ("dense", "keyword", "hybrid")


def recall_at_k(retrieved_docs: list[str], relevant: set[str]) -> float:
    return len(relevant & set(retrieved_docs)) / len(relevant)


def reciprocal_rank(retrieved_docs: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def main() -> None:
    settings = get_settings()
    retriever = load_retriever(settings)
    k = settings.retrieval_top_k
    queries = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["queries"]
    answerable = [q for q in queries if q["relevant_doc_ids"]]
    unanswerable = [q for q in queries if not q["relevant_doc_ids"]]
    print(f"{len(answerable)} answerable + {len(unanswerable)} unanswerable queries, k={k}\n")

    print(f"{'mode':8s} {'Recall@'+str(k):>9s} {'MRR@'+str(k):>7s} {'ms/query':>9s}  misses")
    for mode in MODES:
        recalls, rrs, misses, started = [], [], [], time.perf_counter()
        for q in answerable:
            result = retriever.retrieve(q["query"], top_k=k, mode=mode)
            docs = [rc.chunk.doc_id for rc in result.chunks]
            relevant = set(q["relevant_doc_ids"])
            recalls.append(recall_at_k(docs, relevant))
            rrs.append(reciprocal_rank(docs, relevant))
            if recalls[-1] < 1.0:
                misses.append(q["id"])
        ms = (time.perf_counter() - started) * 1000 / len(answerable)
        print(f"{mode:8s} {statistics.mean(recalls):9.3f} {statistics.mean(rrs):7.3f} {ms:9.1f}  "
              f"{', '.join(misses) or '-'}")

    # Best cosine similarity per query (hybrid mode, the mode the app uses).
    best = {q["id"]: retriever.retrieve(q["query"], top_k=k).best_similarity for q in queries}
    ans_scores = sorted(best[q["id"]] for q in answerable)
    una_scores = sorted(best[q["id"]] for q in unanswerable)
    print("\nBest-chunk cosine similarity (hybrid):")
    print(f"  answerable   min {ans_scores[0]:.3f}  median {statistics.median(ans_scores):.3f}  max {ans_scores[-1]:.3f}")
    print(f"  unanswerable min {una_scores[0]:.3f}  median {statistics.median(una_scores):.3f}  max {una_scores[-1]:.3f}")
    for q in unanswerable:
        print(f"    {q['id']} {best[q['id']]:.3f}  {q['query']}")

    print("\nThreshold sweep (accept = evidence_sufficient):")
    print(f"  {'threshold':>9s} {'answerable accepted':>20s} {'unanswerable rejected':>22s} {'accuracy':>9s}")
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        accepted = sum(s >= t for s in ans_scores)
        rejected = sum(s < t for s in una_scores)
        acc = (accepted + rejected) / len(queries)
        marker = "  <- configured" if abs(t - settings.min_evidence_similarity) < 1e-9 else ""
        print(f"  {t:9.2f} {accepted:>12d}/{len(answerable):<7d} {rejected:>13d}/{len(unanswerable):<8d} {acc:9.3f}{marker}")


if __name__ == "__main__":
    main()
