"""POST /api/answer: full RAG for one question (retrieve -> generate -> validate).

Phase 5 will call the same service for stored tickets; this endpoint lets us
test the pipeline before the database exists.
"""
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.retrieval import get_retriever
from app.config import get_settings
from app.llm.base import LLMClient
from app.rag.retriever import Retriever
from app.schemas import AnswerRequest, AnswerResponse, RetrievalSummary
from app.services.answer_service import generate_draft

router = APIRouter(tags=["answers"])


def get_llm(request: Request) -> LLMClient:
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM client not configured.")
    return llm


@router.post("/answer", response_model=AnswerResponse)
def answer(body: AnswerRequest, retriever: Retriever = Depends(get_retriever),
           llm: LLMClient = Depends(get_llm)) -> AnswerResponse:
    draft = generate_draft(body.query, retriever, llm, top_k=body.top_k,
                           max_retries=get_settings().llm_max_retries)
    r = draft.retrieval
    # A failed LLM call is still a valid API response (status "generation_failed")
    # so the UI can show the evidence and let an agent write the reply by hand.
    return AnswerResponse(
        status=draft.status,
        answer=draft.answer,
        citations=[asdict(c) for c in draft.citations],
        warnings=draft.warnings,
        llm_called=draft.llm_called,
        llm=draft.llm,
        sources_sent=draft.sources_sent,
        pii_masked=draft.pii_masked,
        retrieval=RetrievalSummary(evidence_sufficient=r.evidence_sufficient,
                                   best_similarity=r.best_similarity, threshold=r.threshold,
                                   reason=r.reason, conflicts=r.conflicts),
    )
