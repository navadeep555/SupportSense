"""POST /api/retrieve: run retrieval only (no LLM) and show the evidence found.

Useful for debugging and for explaining RAG: you can see exactly which chunks
a question retrieves, their scores, and whether the evidence check passed.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.rag.retriever import RetrievalResult, Retriever
from app.schemas import EvidenceChunk, RetrieveRequest, RetrieveResponse

router = APIRouter(tags=["retrieval"])


def get_retriever(request: Request) -> Retriever:
    """FastAPI dependency: hand the route the retriever loaded at startup."""
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge index not loaded. Run `python -m app.rag.ingest` and restart the server.",
        )
    return retriever


def to_response(result: RetrievalResult) -> RetrieveResponse:
    return RetrieveResponse(
        query=result.query,
        mode=result.mode,
        evidence_sufficient=result.evidence_sufficient,
        best_similarity=result.best_similarity,
        threshold=result.threshold,
        reason=result.reason,
        conflicts=result.conflicts,
        chunks=[
            EvidenceChunk(
                rank=rc.rank, chunk_id=rc.chunk.chunk_id, doc_id=rc.chunk.doc_id,
                title=rc.chunk.title, section=rc.chunk.section, text=rc.chunk.text,
                source_path=rc.chunk.source_path, last_updated=rc.chunk.last_updated,
                superseded_by=rc.chunk.superseded_by, similarity=rc.similarity,
                keyword_score=rc.keyword_score, fused_score=rc.fused_score,
            )
            for rc in result.chunks
        ],
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(body: RetrieveRequest, retriever: Retriever = Depends(get_retriever)) -> RetrieveResponse:
    return to_response(retriever.retrieve(body.query, top_k=body.top_k))
