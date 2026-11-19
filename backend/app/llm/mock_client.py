"""Mock provider: a fake LLM with no model at all.

Used (1) while Ollama is not installed, so the whole app still works end to
end, and (2) in tests, which must be fast, free and give the same output
every run. Its "answer" is just the opening sentences of source S1, clearly
labelled, so nobody mistakes it for a real generated reply.
"""
import json
import re
import time

from app.llm.base import LLMResult

# Matches the <source id="S1" ...>text</source> blocks built in app/llm/prompts.py
SOURCE_RE = re.compile(r'<source id="(S\d+)"[^>]*>\s*(.*?)\s*</source>', re.DOTALL)


class MockLLMClient:
    provider = "mock"
    model = "mock-extractive"

    def generate(self, system: str, user: str, json_schema: dict) -> LLMResult:
        started = time.perf_counter()
        sources = SOURCE_RE.findall(user)
        if sources:
            source_id, text = sources[0]
            first_sentences = " ".join(re.split(r"(?<=[.!?])\s+", text)[:2])
            reply = {
                "evidence_sufficient": True,
                "answer": f"[MOCK - no LLM call] From our documentation: {first_sentences} [{source_id}]",
                "cited_source_ids": [source_id],
                "missing_information": "",
            }
        else:
            reply = {"evidence_sufficient": False, "answer": "[MOCK] No sources were provided.",
                     "cited_source_ids": [], "missing_information": "No sources."}
        return LLMResult(
            content=json.dumps(reply), provider=self.provider, model=self.model,
            input_tokens=None, output_tokens=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def is_available(self) -> bool:
        return True
