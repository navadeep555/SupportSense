"""Stages 6-7 of RAG: retrieve relevant chunks and decide if evidence is sufficient.

Flow for one query (mode="hybrid"):
  1. Embed the query with the same model used at ingestion.
  2. Dense search: FAISS returns the closest chunks by meaning.
  3. Keyword search: BM25 returns chunks sharing the query's words.
  4. Fuse both ranked lists with Reciprocal Rank Fusion (RRF) -> top_k chunks.
  5. Give every final chunk its cosine similarity to the query.
  6. Evidence check: is the best cosine similarity >= the threshold?
  7. Flag outdated documents (superseded_by) so the LLM step can prefer current ones.

No LLM is involved anywhere in this file.
"""
from dataclasses import dataclass, field
from typing import Literal

from app.rag.chunker import Chunk
from app.rag.embedder import TextEmbedder
from app.rag.keyword_index import KeywordIndex
from app.rag.vector_store import VectorStore

Mode = Literal["dense", "keyword", "hybrid"]
RRF_K = 60  # standard constant from the original RRF paper (Cormack et al., 2009)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rank: int                 # 1 = best
    similarity: float         # cosine similarity to the query (from embeddings)
    keyword_score: float      # BM25 score (0 if no shared words)
    fused_score: float        # RRF score used for ordering


@dataclass
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    evidence_sufficient: bool
    best_similarity: float
    threshold: float
    reason: str
    conflicts: list[dict] = field(default_factory=list)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked lists of chunk positions into one list.

    Each list gives a chunk 1 / (k + rank). Scores are summed across lists, so
    a chunk ranked well by BOTH searches wins. RRF only uses ranks, which matters
    because cosine scores (0..1) and BM25 scores (0..~20) are on different scales
    and cannot be added directly.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            scores[position] = scores.get(position, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def find_conflicts(chunks: list[RetrievedChunk]) -> list[dict]:
    """Report retrieved chunks that come from a document marked as superseded."""
    retrieved_docs = {rc.chunk.doc_id for rc in chunks}
    conflicts, seen = [], set()
    for rc in chunks:
        c = rc.chunk
        if c.superseded_by and c.doc_id not in seen:
            seen.add(c.doc_id)
            conflicts.append({
                "outdated_doc_id": c.doc_id,
                "current_doc_id": c.superseded_by,
                "current_doc_retrieved": c.superseded_by in retrieved_docs,
            })
    return conflicts


class Retriever:
    def __init__(self, store: VectorStore, embedder: TextEmbedder, min_similarity: float,
                 candidate_pool: int = 20):
        self.store = store
        self.embedder = embedder
        self.keyword_index = KeywordIndex(store.chunks)
        self.min_similarity = min_similarity
        self.candidate_pool = candidate_pool

    def retrieve(self, query: str, top_k: int = 5, mode: Mode = "hybrid") -> RetrievalResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self.embedder.embed([query])[0]
        pool = max(self.candidate_pool, top_k)

        dense_hits = self.store.search(query_vector, pool) if mode != "keyword" else []
        keyword_hits = self.keyword_index.search(query, pool) if mode != "dense" else []
        keyword_scores = dict(keyword_hits)

        if mode == "dense":
            ordered = [(p, s) for p, s in dense_hits]
        elif mode == "keyword":
            ordered = keyword_hits
        else:
            ordered = reciprocal_rank_fusion([[p for p, _ in dense_hits], [p for p, _ in keyword_hits]])

        results = [
            RetrievedChunk(
                chunk=self.store.chunks[position],
                rank=rank,
                similarity=round(self.store.similarity(query_vector, position), 4),
                keyword_score=round(keyword_scores.get(position, 0.0), 4),
                fused_score=round(score, 6),
            )
            for rank, (position, score) in enumerate(ordered[:top_k], start=1)
        ]

        best = max((r.similarity for r in results), default=0.0)
        sufficient = best >= self.min_similarity
        if not results:
            reason = "No chunks matched the query."
        elif sufficient:
            reason = f"Best chunk similarity {best:.2f} >= threshold {self.min_similarity:.2f}."
        else:
            reason = (f"Best chunk similarity {best:.2f} < threshold {self.min_similarity:.2f}: "
                      "the knowledge base probably does not cover this question.")

        return RetrievalResult(
            query=query, mode=mode, chunks=results, evidence_sufficient=sufficient,
            best_similarity=round(best, 4), threshold=self.min_similarity, reason=reason,
            conflicts=find_conflicts(results),
        )


def load_retriever(settings) -> Retriever:
    """Load the saved index + embedding model. Called once at server startup."""
    from app.rag.embedder import SentenceTransformerEmbedder

    store = VectorStore.load(settings.index_dir)
    built_with = store.manifest.get("embedding_model")
    if built_with != settings.embedding_model:
        # Vectors from different models live in different "spaces"; comparing
        # them gives meaningless scores, so refuse instead of silently degrading.
        raise RuntimeError(
            f"index was built with {built_with!r} but EMBEDDING_MODEL is "
            f"{settings.embedding_model!r}. Re-run: python -m app.rag.ingest")
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return Retriever(store, embedder, settings.min_evidence_similarity,
                     settings.retrieval_candidate_pool)
