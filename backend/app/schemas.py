"""Pydantic models: the exact shape of API requests and responses.

FastAPI uses these to (1) reject bad input with HTTP 422 before our code runs,
(2) convert our Python objects into JSON, and (3) document the API at /docs.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------- Retrieval (Phase 3) ----------
class RetrieveRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000, examples=["I paid but premium is not unlocked"])
    top_k: int = Field(default=5, ge=1, le=10)


class EvidenceChunk(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_path: str
    last_updated: str
    superseded_by: str | None
    similarity: float
    keyword_score: float
    fused_score: float


class Conflict(BaseModel):
    outdated_doc_id: str
    current_doc_id: str
    current_doc_retrieved: bool


class RetrievalSummary(BaseModel):
    evidence_sufficient: bool
    best_similarity: float
    threshold: float
    reason: str
    conflicts: list[Conflict]


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    evidence_sufficient: bool
    best_similarity: float
    threshold: float
    reason: str
    conflicts: list[Conflict]
    chunks: list[EvidenceChunk]


# ---------- Answer generation (Phase 4) ----------
class AnswerRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000,
                       examples=["My subscription payment was successful, but I still cannot access the premium features."])
    top_k: int = Field(default=5, ge=1, le=10)


class CitationOut(BaseModel):
    source_id: str
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    similarity: float
    outdated: bool


class LLMInfo(BaseModel):
    provider: str
    model: str
    attempts: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class AnswerResponse(BaseModel):
    status: str
    answer: str
    citations: list[CitationOut]
    warnings: list[str]
    llm_called: bool
    llm: LLMInfo | None
    sources_sent: list[str]
    retrieval: RetrievalSummary
    pii_masked: dict[str, int] = {}


# ---------- Tickets, drafts and reviews (Phase 5) ----------
class TicketCreate(BaseModel):
    subject: str = Field(min_length=3, max_length=200, examples=["Premium not working after payment"])
    body: str = Field(min_length=3, max_length=4000,
                      examples=["I paid yesterday but still can't use Premium. Call me on +91 98765 43210."])
    # A simple shape check; a real system would verify the address by sending a link.
    customer_email: str | None = Field(default=None, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class TicketOut(BaseModel):
    # from_attributes: build this model straight from a SQLAlchemy object (ticket.subject, ...)
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    body: str
    subject_masked: str
    body_masked: str
    pii_found: dict[str, int]
    customer_email: str | None
    status: str
    category: str | None
    category_confidence: float | None
    category_low_confidence: bool
    escalated: bool
    escalation_score: float | None
    escalation_reasons: list[str]
    created_at: datetime
    updated_at: datetime


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticket_id: int
    status: str
    answer: str
    citations: list[CitationOut]
    warnings: list[str]
    sources_sent: list[str]
    pii_masked: dict[str, int]
    evidence_sufficient: bool
    best_similarity: float
    llm_called: bool
    llm_provider: str | None
    llm_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    created_at: datetime
    review_decision: str | None
    final_reply: str | None
    edit_ratio: float | None
    review_note: str | None
    reviewed_at: datetime | None


class TicketDetail(TicketOut):
    drafts: list[DraftOut]


class TicketList(BaseModel):
    total: int
    items: list[TicketOut]


class DraftRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=10)


class ReviewRequest(BaseModel):
    decision: Literal["approved", "edited", "rejected"]
    final_reply: str | None = Field(default=None, min_length=1, max_length=8000)
    note: str | None = Field(default=None, min_length=3, max_length=1000)

    @model_validator(mode="after")
    def check_required_fields(self) -> "ReviewRequest":
        # Rules that involve two fields can't be expressed with Field(); this runs
        # after the single-field checks and turns a failure into HTTP 422.
        if self.decision == "edited" and not self.final_reply:
            raise ValueError("final_reply is required when decision is 'edited'.")
        if self.decision == "rejected" and not self.note:
            raise ValueError("note (why the draft was rejected) is required when decision is 'rejected'.")
        return self


# ---------- Reports (Phase 5) ----------
class DraftMetrics(BaseModel):
    total_drafts: int
    by_status: dict[str, int]          # grounded / needs_attention / ...
    reviewed: int
    by_decision: dict[str, int]        # approved / edited / rejected
    approval_rate: float | None        # approved unchanged / reviewed
    acceptance_rate: float | None      # (approved + edited) / reviewed
    avg_edit_ratio_when_edited: float | None
    avg_llm_latency_ms: float | None


class KnowledgeGap(BaseModel):
    ticket_id: int
    draft_id: int
    reason: str            # no_matching_docs / model_found_docs_insufficient / agent_rejected
    detail: str
    best_similarity: float
    subject_masked: str
    body_masked: str
    created_at: datetime


class KnowledgeGapReport(BaseModel):
    total: int
    items: list[KnowledgeGap]
