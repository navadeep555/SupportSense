"""End-to-end tests of the ticket workflow through HTTP, on an in-memory SQLite database."""
import json

import pytest
from fastapi.testclient import TestClient

from app.db.database import create_db_engine, init_db
from app.main import app
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder, FakeLLM, make_chunk

CHUNKS = [make_chunk("plans", "premium", "after payment premium unlocks within 5 minutes sign out and sign in")]


def reply(answer, cited=("S1",), sufficient=True):
    return json.dumps({"evidence_sufficient": sufficient, "answer": answer,
                       "cited_source_ids": list(cited), "missing_information": ""})


@pytest.fixture
def client():
    embedder = FakeEmbedder()
    store = VectorStore.build(embedder.embed([c.embedding_text() for c in CHUNKS]), CHUNKS)
    app.state.retriever = Retriever(store, embedder, min_similarity=0.3)
    app.state.session_factory = init_db(create_db_engine("sqlite://"))  # fresh, empty DB per test
    app.state.llm = FakeLLM(*[reply("Premium unlocks within 5 minutes [S1]. Sign out and in [S1].")] * 5)
    yield TestClient(app)
    app.state.retriever = app.state.session_factory = app.state.llm = None


def new_ticket(client, body="I paid but premium is not unlocked after payment", **extra):
    response = client.post("/api/tickets", json={"subject": "Premium payment", "body": body, **extra})
    assert response.status_code == 201
    return response.json()


def test_create_ticket_masks_pii_but_keeps_original(client):
    t = new_ticket(client, body="Premium missing. Call +91 98765 43210 or mail ravi@example.com",
                   customer_email="ravi@example.com")
    assert t["status"] == "open" and t["id"] == 1
    assert t["body_masked"] == "Premium missing. Call [PHONE] or mail [EMAIL]"
    assert "98765 43210" in t["body"]  # the agent still sees the original
    assert t["pii_found"] == {"phone": 1, "email": 1}


def test_llm_never_sees_raw_pii(client):
    t = new_ticket(client, body="premium not unlocked after payment, call me on 9876543210")
    client.post(f"/api/tickets/{t['id']}/drafts")
    _, user_prompt = app.state.llm.calls[0]
    assert "9876543210" not in user_prompt and "[PHONE]" in user_prompt


def test_validation_errors(client):
    assert client.post("/api/tickets", json={"subject": "Hi", "body": "x"}).status_code == 422
    bad_email = {"subject": "Help me", "body": "something", "customer_email": "not-an-email"}
    assert client.post("/api/tickets", json=bad_email).status_code == 422


def test_list_filter_and_pagination(client):
    for _ in range(3):
        new_ticket(client)
    client.post("/api/tickets/1/drafts")
    page = client.get("/api/tickets", params={"limit": 2}).json()
    assert page["total"] == 3 and [t["id"] for t in page["items"]] == [3, 2]  # newest first
    assert client.get("/api/tickets", params={"status": "drafted"}).json()["total"] == 1
    assert client.get("/api/tickets", params={"status": "bogus"}).status_code == 422


def test_unknown_ids_return_404(client):
    assert client.get("/api/tickets/99").status_code == 404
    assert client.post("/api/tickets/99/drafts").status_code == 404
    assert client.post("/api/drafts/99/review", json={"decision": "approved"}).status_code == 404


def test_draft_is_stored_and_ticket_becomes_drafted(client):
    t = new_ticket(client)
    draft = client.post(f"/api/tickets/{t['id']}/drafts").json()
    assert draft["status"] == "grounded"
    assert draft["citations"][0]["doc_id"] == "plans"
    assert draft["llm_provider"] == "fake" and draft["input_tokens"] == 100
    detail = client.get(f"/api/tickets/{t['id']}").json()
    assert detail["status"] == "drafted" and detail["drafts"][0]["id"] == draft["id"]


def test_approve_strips_markers_and_resolves(client):
    t = new_ticket(client)
    d = client.post(f"/api/tickets/{t['id']}/drafts").json()
    reviewed = client.post(f"/api/drafts/{d['id']}/review", json={"decision": "approved"}).json()
    assert reviewed["final_reply"] == "Premium unlocks within 5 minutes. Sign out and in."
    assert reviewed["edit_ratio"] == 0.0
    assert client.get(f"/api/tickets/{t['id']}").json()["status"] == "resolved"
    # Resolved: no second review, no new draft.
    assert client.post(f"/api/drafts/{d['id']}/review", json={"decision": "approved"}).status_code == 409
    assert client.post(f"/api/tickets/{t['id']}/drafts").status_code == 409


def test_edit_records_edit_ratio(client):
    t = new_ticket(client)
    d = client.post(f"/api/tickets/{t['id']}/drafts").json()
    final = "Hi! Premium unlocks within 5 minutes. Please sign out and in."
    reviewed = client.post(f"/api/drafts/{d['id']}/review",
                           json={"decision": "edited", "final_reply": final}).json()
    assert reviewed["final_reply"] == final
    assert 0 < reviewed["edit_ratio"] < 0.5


def test_review_body_rules(client):
    t = new_ticket(client)
    d = client.post(f"/api/tickets/{t['id']}/drafts").json()
    url = f"/api/drafts/{d['id']}/review"
    assert client.post(url, json={"decision": "edited"}).status_code == 422      # needs final_reply
    assert client.post(url, json={"decision": "rejected"}).status_code == 422    # needs note
    assert client.post(url, json={"decision": "maybe"}).status_code == 422


def test_reject_reopens_and_only_latest_draft_is_reviewable(client):
    t = new_ticket(client)
    first = client.post(f"/api/tickets/{t['id']}/drafts").json()
    client.post(f"/api/drafts/{first['id']}/review", json={"decision": "rejected", "note": "wrong policy"})
    assert client.get(f"/api/tickets/{t['id']}").json()["status"] == "open"

    second = client.post(f"/api/tickets/{t['id']}/drafts").json()
    third = client.post(f"/api/tickets/{t['id']}/drafts").json()
    response = client.post(f"/api/drafts/{second['id']}/review", json={"decision": "approved"})
    assert response.status_code == 409 and str(third["id"]) in response.json()["detail"]


def test_failed_generation_cannot_be_approved(client):
    from app.llm.base import LLMUnavailableError
    app.state.llm = FakeLLM(LLMUnavailableError("Ollama down"))
    t = new_ticket(client)
    d = client.post(f"/api/tickets/{t['id']}/drafts").json()
    assert d["status"] == "generation_failed"
    assert client.post(f"/api/drafts/{d['id']}/review", json={"decision": "approved"}).status_code == 409


def test_draft_metrics(client):
    empty = client.get("/api/reports/draft-metrics").json()
    assert empty["total_drafts"] == 0 and empty["approval_rate"] is None

    for decision in ("approved", "edited", "rejected"):
        t = new_ticket(client)
        d = client.post(f"/api/tickets/{t['id']}/drafts").json()
        payload = {"decision": decision, "final_reply": "Premium unlocks in 5 minutes.", "note": "not helpful"}
        client.post(f"/api/drafts/{d['id']}/review", json=payload)

    m = client.get("/api/reports/draft-metrics").json()
    assert m["total_drafts"] == 3 and m["reviewed"] == 3
    assert m["by_decision"] == {"approved": 1, "edited": 1, "rejected": 1}
    assert m["approval_rate"] == 0.333 and m["acceptance_rate"] == 0.667
    assert m["avg_edit_ratio_when_edited"] > 0
    assert m["avg_llm_latency_ms"] == 5.0


def test_knowledge_gap_report(client):
    covered = new_ticket(client)
    gap = new_ticket(client, body="Do you sell gift cards for birthdays? mail me at a@b.com")
    rejected = new_ticket(client)
    client.post(f"/api/tickets/{covered['id']}/drafts")
    client.post(f"/api/tickets/{gap['id']}/drafts")              # weak evidence -> no LLM call
    d = client.post(f"/api/tickets/{rejected['id']}/drafts").json()
    client.post(f"/api/drafts/{d['id']}/review", json={"decision": "rejected", "note": "outdated steps"})

    report = client.get("/api/reports/knowledge-gaps").json()
    assert report["total"] == 2
    by_ticket = {item["ticket_id"]: item for item in report["items"]}
    assert by_ticket[gap["id"]]["reason"] == "no_matching_docs"
    assert "[EMAIL]" in by_ticket[gap["id"]]["body_masked"]       # reports never show raw PII
    assert by_ticket[rejected["id"]]["reason"] == "agent_rejected"
    assert by_ticket[rejected["id"]]["detail"] == "outdated steps"
    assert report["items"][0]["ticket_id"] == gap["id"]           # least covered topic first
