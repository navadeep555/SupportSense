"""Pydantic models: the exact shape of API requests and responses.

FastAPI uses these to (1) reject bad input with HTTP 422 before our code runs,
(2) convert our Python objects into JSON, and (3) document the API at /docs.
"""
from pydantic import BaseModel, Field


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
