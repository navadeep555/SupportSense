"""Test doubles: a tiny, deterministic embedder so tests don't need the real model."""
import zlib

import numpy as np

from app.rag.chunker import Chunk
from app.rag.keyword_index import tokenize


class FakeEmbedder:
    """Bag-of-words "embedding": each word adds 1 to a bucket chosen by a hash.

    Texts sharing words get similar vectors, which is all retrieval logic needs
    for unit tests. crc32 is used (not hash()) because Python randomises hash()
    for strings on every run, which would make tests non-reproducible.
    """

    model_name = "fake-bag-of-words"
    dimension = 64

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, zlib.crc32(token.encode()) % self.dimension] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class FakeLLM:
    """Scripted LLM: returns the given replies in order (or raises given exceptions)."""

    provider = "fake"
    model = "fake-llm"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []  # (system, user) for every call, so tests can inspect prompts

    def generate(self, system, user, json_schema):
        from app.llm.base import LLMResult

        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(content=reply, provider=self.provider, model=self.model,
                         input_tokens=100, output_tokens=20, latency_ms=5)

    def is_available(self):
        return True


def make_chunk(doc_id: str, section: str, text: str, superseded_by: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{section}::0", doc_id=doc_id, title=doc_id, section=section,
        text=text, category="test", last_updated="2026-01-01",
        superseded_by=superseded_by, source_path=f"{doc_id}.md",
    )
