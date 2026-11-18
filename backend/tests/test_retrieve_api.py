"""API tests for POST /api/retrieve using a small in-memory retriever (no real model)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder, make_chunk


@pytest.fixture
def client():
    chunks = [make_chunk("login", "reset", "reset your password with the forgot password link")]
    embedder = FakeEmbedder()
    store = VectorStore.build(embedder.embed([c.embedding_text() for c in chunks]), chunks)
    # Not using `with TestClient(app)`, so the startup hook (which loads the real
    # model) does not run; we put a fake retriever on app.state ourselves.
    app.state.retriever = Retriever(store, embedder, min_similarity=0.3)
    yield TestClient(app)
    app.state.retriever = None


def test_retrieve_returns_ranked_evidence(client):
    response = client.post("/api/retrieve", json={"query": "forgot my password", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"][0]["doc_id"] == "login"
    assert body["evidence_sufficient"] is True
    assert {"chunk_id", "title", "section", "similarity"} <= body["chunks"][0].keys()


@pytest.mark.parametrize("payload", [
    {"query": ""},                      # too short
    {"query": "ok"},                    # 2 characters, below min_length=3
    {"query": "password", "top_k": 0},  # top_k below 1
    {"query": "password", "top_k": 99}, # top_k above 10
    {"top_k": 3},                       # query missing
])
def test_invalid_requests_return_422(client, payload):
    assert client.post("/api/retrieve", json=payload).status_code == 422


def test_returns_503_when_index_not_loaded():
    app.state.retriever = None
    response = TestClient(app).post("/api/retrieve", json={"query": "forgot password"})
    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]
