"""End-to-end answer evaluation: real retrieval + real LLM + two faithfulness judges.

Run from backend/ with Ollama running (takes several minutes on a laptop):
    python -m eval.run_answer_eval

Per question it runs the same generate_draft() the API uses, then measures:
  answerable questions   answer rate, key-fact recall, forbidden (outdated) facts,
                         citation hit/precision against the labelled documents
  unanswerable questions correct hand-off rate (no confident made-up answer)
  every drafted answer   sentence faithfulness by the NLI judge (independent model)
                         and the LLM judge (same model as the generator), plus
                         the sentences where the two judges disagree
  cost                   LLM latency and tokens
Results: eval/results/answer_eval.json (summary + every answer, for inspection).
"""
import json
from collections import Counter
import statistics
import sys
import time

from app.config import BACKEND_DIR, get_settings
from app.llm.factory import create_llm_client
from app.rag.retriever import load_retriever
from app.services.answer_service import generate_draft
from eval.faithfulness import LLMJudge, NLIJudge

EVAL_DIR = BACKEND_DIR / "eval"
RESULT_PATH = EVAL_DIR / "results" / "answer_eval.json"
ANSWERED = {"grounded", "needs_attention"}


def has(text: str, spellings: list[str]) -> bool:
    lowered = text.lower()
    return any(s.lower() in lowered for s in spellings)


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def main() -> None:
    settings = get_settings()
    llm = create_llm_client(settings)
    if llm.provider == "mock" and "--allow-mock" not in sys.argv:
        sys.exit("LLM_PROVIDER is 'mock': this evaluation is meant for the real model. "
                 "Set LLM_PROVIDER=ollama (or pass --allow-mock to test the script).")
    if not llm.is_available():
        sys.exit(f"{llm.model} is not available. Start Ollama and run: ollama pull {llm.model}")

    queries = {q["id"]: q["query"] for q in
               json.loads((EVAL_DIR / "retrieval_eval_set.json").read_text(encoding="utf-8"))["queries"]}
    items = json.loads((EVAL_DIR / "answer_eval_set.json").read_text(encoding="utf-8"))["items"]
    retriever = load_retriever(settings)
    nli, llm_judge = NLIJudge(), LLMJudge(llm)

    rows = []
    for n, item in enumerate(items, start=1):
        query = queries[item["id"]]
        started = time.perf_counter()
        draft = generate_draft(query, retriever, llm, top_k=settings.retrieval_top_k,
                               max_retries=settings.llm_max_retries)
        row = {"id": item["id"], "query": query, "expect": item["expect"], "status": draft.status,
               "answer": draft.answer, "warnings": draft.warnings, "llm": draft.llm,
               "seconds": round(time.perf_counter() - started, 2),
               "cited_docs": sorted({c.doc_id for c in draft.citations}),
               "best_similarity": draft.retrieval.best_similarity}

        if item["expect"] == "answer" and draft.status in ANSWERED:
            facts = [has(draft.answer, f) for f in item["key_facts"]]
            row["facts_found"] = f"{sum(facts)}/{len(facts)}"
            row["fact_recall"] = sum(facts) / len(facts) if facts else 1.0
            row["forbidden_found"] = [f for f in item["forbidden"] if f.lower() in draft.answer.lower()]
            relevant = set(item["relevant_doc_ids"])
            row["citation_hit"] = bool(relevant & set(row["cited_docs"]))
            row["citation_precision"] = ratio(len(relevant & set(row["cited_docs"])), len(row["cited_docs"]))

        if draft.status in ANSWERED:
            sources = {c.source_id: c.text for c in draft.citations}
            checks = nli.check_answer(draft.answer, sources)
            sentences = []
            for check in checks:
                entry = {"sentence": check.sentence, "cited": check.cited_ids,
                         "nli_supported": check.supported, "nli_entailment": check.entailment}
                if check.supported is not None:
                    premise = "\n".join(sources[c] for c in dict.fromkeys(check.cited_ids) if c in sources)
                    entry["llm_supported"] = llm_judge.is_supported(premise, check.sentence)
                sentences.append(entry)
            row["sentences"] = sentences

        rows.append(row)
        print(f"[{n:>2}/{len(items)}] {item['id']} {draft.status:<22} {row['seconds']:>5}s "
              f"{row.get('facts_found', '')}")

    summary = summarize(rows)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({"model": llm.model, "summary": summary, "items": rows}, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {RESULT_PATH.relative_to(BACKEND_DIR)}")


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["expect"] == "answer"]
    answered = [r for r in answerable if r["status"] in ANSWERED]
    handoff_expected = [r for r in rows if r["expect"] == "handoff"]
    sentences = [s for r in rows for s in r.get("sentences", [])]
    cited = [s for s in sentences if s["nli_supported"] is not None]
    disagreements = [{"id": r["id"], "sentence": s["sentence"], "nli": s["nli_supported"], "llm": s["llm_supported"]}
                     for r in rows for s in r.get("sentences", [])
                     if s["nli_supported"] is not None and s["nli_supported"] != s["llm_supported"]]
    both_unsupported = [{"id": r["id"], "sentence": s["sentence"]} for r in rows for s in r.get("sentences", [])
                        if s["nli_supported"] is False and s.get("llm_supported") is False]
    latencies = [r["llm"]["latency_ms"] for r in rows if r["llm"] and r["llm"].get("latency_ms")]

    return {
        "answerable": {
            "questions": len(answerable),
            "answered": len(answered),
            "answer_rate": ratio(len(answered), len(answerable)),
            "status_counts": dict(Counter(r["status"] for r in answerable)),
            "mean_key_fact_recall": ratio(sum(r["fact_recall"] for r in answered), len(answered)),
            "all_key_facts_present": ratio(sum(r["fact_recall"] == 1.0 for r in answered), len(answered)),
            "forbidden_fact_answers": [r["id"] for r in answered if r["forbidden_found"]],
            "citation_hit_rate": ratio(sum(r["citation_hit"] for r in answered), len(answered)),
            "mean_citation_precision": ratio(sum(r["citation_precision"] or 0 for r in answered), len(answered)),
            "missed_facts": {r["id"]: r["facts_found"] for r in answered if r["fact_recall"] < 1.0},
        },
        "unanswerable": {
            "questions": len(handoff_expected),
            "correct_handoffs": sum(r["status"] == "insufficient_evidence" for r in handoff_expected),
            "handoff_rate": ratio(sum(r["status"] == "insufficient_evidence" for r in handoff_expected),
                                  len(handoff_expected)),
            "answered_anyway": [r["id"] for r in handoff_expected if r["status"] in ANSWERED],
            "u06_either_status": next((r["status"] for r in rows if r["id"] == "u06"), None),
        },
        "faithfulness": {
            "sentences": len(sentences),
            "sentences_with_citation": len(cited),
            "uncited_sentences": len(sentences) - len(cited),
            "nli_supported_rate": ratio(sum(s["nli_supported"] for s in cited), len(cited)),
            "llm_supported_rate": ratio(sum(s["llm_supported"] for s in cited), len(cited)),
            "judges_agree_rate": ratio(len(cited) - len(disagreements), len(cited)),
            "both_judges_unsupported": both_unsupported,
            "disagreements": disagreements,
        },
        "latency": {
            "llm_calls": len(latencies),
            "mean_ms": round(statistics.mean(latencies)) if latencies else None,
            "median_ms": round(statistics.median(latencies)) if latencies else None,
            "max_ms": max(latencies) if latencies else None,
        },
    }


if __name__ == "__main__":
    main()
