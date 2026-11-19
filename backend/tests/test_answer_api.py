import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMUnavailableError
from app.llm.mock_client import MockLLMClient
from app.main import app
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder, FakeLLM, make_chunk


@pytest.fixture
def client():
    chunks = [make_chunk("login", "reset", "reset your password with the forgot password link")]
    embedder = FakeEmbedder()
    store = VectorStore.build(embedder.embed([c.embedding_text() for c in chunks]), chunks)
    app.state.retriever = Retriever(store, embedder, min_similarity=0.3)
    app.state.llm = MockLLMClient()
    yield TestClient(app)
    app.state.retriever = None
    app.state.llm = None


def test_answer_returns_draft_with_citations(client):
    response = client.post("/api/answer", json={"query": "I forgot my password, how do I reset it?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "grounded"
    assert body["citations"][0]["chunk_id"] == "login::reset::0"
    assert body["llm"]["provider"] == "mock"
    assert body["retrieval"]["evidence_sufficient"] is True


def test_llm_outage_still_returns_200_with_evidence(client):
    app.state.llm = FakeLLM(LLMUnavailableError("Cannot reach Ollama"))
    body = client.post("/api/answer", json={"query": "forgot my password reset"}).json()
    assert body["status"] == "generation_failed"
    assert body["sources_sent"] == ["login::reset::0"]


def test_bad_input_returns_422(client):
    assert client.post("/api/answer", json={"query": "hi"}).status_code == 422


def test_health_reports_llm(client):
    body = client.get("/api/health").json()
    assert body["llm"] == {"provider": "mock", "model": "mock-extractive", "available": True}
