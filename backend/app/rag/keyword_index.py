"""Keyword search with BM25, the second half of hybrid retrieval.

Embeddings capture MEANING but are weak at EXACT tokens such as error codes
("SN-2003"), prices or plan names. BM25 scores a chunk higher when it contains
the query's words, especially rare words, so "SN-2003" strongly matches the
one chunk that mentions it.

BM25 is cheap to build, so it is rebuilt in memory from the chunks at startup
instead of being saved to disk.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.chunker import Chunk

# Words like "sn-2003" or "4k" stay as one token.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Very common words carry no topic information; dropping them reduces noise.
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it "
    "its me my no not of on or our so that the their them then there this to was "
    "we what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class KeywordIndex:
    def __init__(self, chunks: list[Chunk]):
        self._bm25 = BM25Okapi([tokenize(c.embedding_text()) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_position, bm25_score) with score > 0, best first."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        best = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in best if scores[i] > 0]
