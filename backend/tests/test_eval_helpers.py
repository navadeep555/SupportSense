"""The evaluation code is code too: test the parts that decide pass/fail."""
from eval.faithfulness import _confusion, split_sentences
from eval.run_answer_eval import has, summarize


def test_split_sentences_keeps_citations_per_sentence():
    parts = split_sentences("Sign out and back in [S1]. Wait 30 minutes [S1][S2]! Thanks.")
    assert parts == [("Sign out and back in.", ["S1"]), ("Wait 30 minutes!", ["S1", "S2"]), ("Thanks.", [])]


def test_split_sentences_does_not_break_decimals_or_menu_paths():
    parts = split_sentences("Premium costs 12.99 USD [S1]. Go to Settings > Plan [S2].")
    assert [p[0] for p in parts] == ["Premium costs 12.99 USD.", "Go to Settings > Plan."]


def test_key_fact_matching_is_case_insensitive_and_accepts_alternatives():
    assert has("Refunds within 14 Days.", ["14 days", "14-day"])
    assert not has("Refunds within 7 days.", ["14 days"])


def test_confusion_counts():
    result = _confusion([True, True, False, False], [True, False, False, True])
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["accuracy"] == 0.5


def test_summary_rates_and_disagreements():
    answered = {"id": "a01", "expect": "answer", "status": "grounded", "fact_recall": 0.5, "facts_found": "1/2",
                "forbidden_found": ["7 days"], "citation_hit": True, "citation_precision": 1.0,
                "llm": {"latency_ms": 4000},
                "sentences": [{"sentence": "A.", "nli_supported": True, "llm_supported": False},
                              {"sentence": "B.", "nli_supported": None}]}
    handoff = {"id": "u01", "expect": "handoff", "status": "grounded", "llm": {"latency_ms": 2000}}
    s = summarize([answered, handoff])
    assert s["answerable"]["missed_facts"] == {"a01": "1/2"}
    assert s["answerable"]["forbidden_fact_answers"] == ["a01"]
    assert s["unanswerable"]["answered_anyway"] == ["u01"]
    assert s["faithfulness"]["uncited_sentences"] == 1
    assert s["faithfulness"]["disagreements"] == [{"id": "a01", "sentence": "A.", "nli": True, "llm": False}]
    assert s["latency"]["median_ms"] == 3000
