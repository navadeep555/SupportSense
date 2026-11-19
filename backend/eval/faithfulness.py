"""Faithfulness judge: is each sentence of a draft actually supported by the source it cites?

Citation validation (app/services/citations.py) only proves that [S1] points at
a chunk we really sent. It cannot tell whether S1 SAYS what the sentence claims.
This judge checks that with a Natural Language Inference (NLI) model:

    premise    = the cited chunk text
    hypothesis = the sentence (with [S#] markers removed)
    output     = P(entailment), P(neutral), P(contradiction)

A sentence counts as supported when "entailment" is the most likely label.
The NLI model (cross-encoder/nli-deberta-v3-small) is a different model from
the generator (qwen2.5), so it is not grading its own homework.

Validate the judge before trusting it:
    python -m eval.faithfulness      (NLI + LLM judge; needs Ollama running)
"""
import json
import re
from dataclasses import dataclass

from app.config import BACKEND_DIR

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
JUDGE_SET = BACKEND_DIR / "eval" / "faithfulness_judge_set.json"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
CITATION = re.compile(r"\[(S\d+)\]")


@dataclass
class SentenceCheck:
    sentence: str
    cited_ids: list[str]
    supported: bool | None      # None = sentence has no citation, nothing to check against
    entailment: float | None


def split_sentences(answer: str) -> list[tuple[str, list[str]]]:
    """'Do A [S1]. Do B [S2].' -> [('Do A.', ['S1']), ('Do B.', ['S2'])]"""
    result = []
    for raw in SENTENCE_END.split(answer.strip()):
        cited = CITATION.findall(raw)
        clean = re.sub(r"\s+([.,!?])", r"\1", CITATION.sub("", raw)).strip()
        if clean:
            result.append((clean, cited))
    return result


class NLIJudge:
    def __init__(self, model_name: str = NLI_MODEL):
        from sentence_transformers import CrossEncoder  # heavy import, only when used

        self.model = CrossEncoder(model_name, device="cpu")
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail, self.contra, self.neutral = labels["entailment"], labels["contradiction"], labels["neutral"]

    def _raw(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        probs = self.model.predict(pairs, apply_softmax=True, batch_size=16)
        return [{"entailment": float(p[self.entail]), "neutral": float(p[self.neutral]),
                 "contradiction": float(p[self.contra])} for p in probs]

    def scores(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Score each (premise, hypothesis), comparing the hypothesis with WINDOWS of the premise.

        NLI models are trained on short premises; a whole 100-word chunk makes
        them answer "neutral" even for exact paraphrases (validation: 5/12
        supported sentences accepted). So each premise is cut into single
        sentences and adjacent pairs, and the window with the highest
        entailment is used (the SummaC "zero-shot" idea).
        """
        windows, owner = [], []
        for i, (premise, hypothesis) in enumerate(pairs):
            sentences = [s for s in SENTENCE_END.split(premise.strip()) if s]
            spans = sentences + [f"{a} {b}" for a, b in zip(sentences, sentences[1:])]
            for span in spans or [premise]:
                windows.append((span, hypothesis))
                owner.append(i)
        best: dict[int, dict[str, float]] = {}
        for i, score in zip(owner, self._raw(windows)):
            if i not in best or score["entailment"] > best[i]["entailment"]:
                best[i] = score
        return [best[i] for i in range(len(pairs))]

    def supported(self, score: dict[str, float]) -> bool:
        return max(score, key=score.get) == "entailment"

    def check_answer(self, answer: str, sources: dict[str, str]) -> list[SentenceCheck]:
        """sources maps citation id ("S1") -> chunk text for the draft being checked."""
        sentences = split_sentences(answer)
        to_score, index = [], []
        for i, (sentence, cited) in enumerate(sentences):
            premise = "\n".join(sources[c] for c in dict.fromkeys(cited) if c in sources)
            if premise:
                to_score.append((premise, sentence))
                index.append(i)
        scored = dict(zip(index, self.scores(to_score))) if to_score else {}
        return [SentenceCheck(s, cited, self.supported(scored[i]) if i in scored else None,
                              round(scored[i]["entailment"], 3) if i in scored else None)
                for i, (s, cited) in enumerate(sentences)]


LLM_JUDGE_SYSTEM = (
    "You check whether a SOURCE text supports a CLAIM. Answer supported=true only if every fact in the "
    "claim is stated in or directly follows from the source. Advice that matches steps in the source "
    "counts as supported. Anything added, changed or applied to a different case is unsupported.")
LLM_JUDGE_SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"}}, "required": ["supported"]}


class LLMJudge:
    """Second opinion from the local LLM. Caveats: it is the SAME model that wrote
    the drafts (self-preference bias), and its prompt was written after seeing the
    judge-set failures, so its validation score is optimistic."""

    def __init__(self, llm):
        self.llm = llm

    def is_supported(self, premise: str, sentence: str) -> bool:
        reply = self.llm.generate(LLM_JUDGE_SYSTEM, f"SOURCE:\n{premise}\n\nCLAIM:\n{sentence}", LLM_JUDGE_SCHEMA)
        return bool(json.loads(reply.content)["supported"])


def _confusion(predicted: list[bool], labels: list[bool]) -> dict:
    tp = sum(p and l for p, l in zip(predicted, labels))
    tn = sum(not p and not l for p, l in zip(predicted, labels))
    fp = sum(p and not l for p, l in zip(predicted, labels))
    fn = sum(not p and l for p, l in zip(predicted, labels))
    return {"pairs": len(labels), "accuracy": round((tp + tn) / len(labels), 3),
            "catches_unsupported": round(tn / (tn + fp), 3) if tn + fp else None,
            "accepts_supported": round(tp / (tp + fn), 3) if tp + fn else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def validate_judges(llm=None) -> dict:
    """Accuracy of each judge on hand-labelled pairs; printed and returned."""
    chunks = {c["chunk_id"]: c["text"] for c in
              json.loads((BACKEND_DIR / "data" / "index" / "chunks.json").read_text(encoding="utf-8"))}
    pairs = json.loads(JUDGE_SET.read_text(encoding="utf-8"))["pairs"]
    labels = [p["supported"] for p in pairs]
    judge = NLIJudge()
    scores = judge.scores([(chunks[p["chunk_id"]], p["sentence"]) for p in pairs])
    nli = [judge.supported(s) for s in scores]
    for pair, predicted, score in zip(pairs, nli, scores):
        if predicted != pair["supported"]:
            print(f"  NLI wrong {pair['id']} (entail {score['entailment']:.2f}, label {pair['supported']}): "
                  f"{pair['sentence']}")
    results = {"nli": _confusion(nli, labels)}
    if llm is not None:
        llm_judge = LLMJudge(llm)
        results["llm"] = _confusion([llm_judge.is_supported(chunks[p["chunk_id"]], p["sentence"])
                                     for p in pairs], labels)
    for name, r in results.items():
        print(f"{name.upper()} judge on {r['pairs']} pairs: accuracy {r['accuracy']}, "
              f"catches {r['tn']}/{r['tn'] + r['fp']} unsupported, accepts {r['tp']}/{r['tp'] + r['fn']} supported")
    return results


if __name__ == "__main__":
    from app.config import get_settings
    from app.llm.ollama_client import OllamaClient

    s = get_settings()
    out = BACKEND_DIR / "eval" / "results" / "judge_validation.json"
    results = validate_judges(OllamaClient(s.ollama_base_url, s.ollama_model, s.llm_timeout_seconds))
    out.write_text(json.dumps({"nli_model": NLI_MODEL, "llm_model": s.ollama_model, **results}, indent=2),
                   encoding="utf-8")
    print(f"Saved -> {out.relative_to(BACKEND_DIR)}")
