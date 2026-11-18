import pytest

from app.rag.retriever import Retriever, reciprocal_rank_fusion
from app.rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder, make_chunk

CHUNKS = [
    make_chunk("login", "locked", "account locked after five wrong password attempts wait thirty minutes"),
    make_chunk("playback", "sn-2003", "error SN-2003 means protected content drm verification failed"),
    make_chunk("billing", "duplicate", "charged twice duplicate payment refund within ten business days"),
    make_chunk("refund-old", "window", "refund within 7 days of purchase", superseded_by="refund-new"),
    make_chunk("refund-new", "window", "refund within 14 days of purchase"),
]


@pytest.fixture
def retriever():
    embedder = FakeEmbedder()
    vectors = embedder.embed([c.embedding_text() for c in CHUNKS])
    store = VectorStore.build(vectors, CHUNKS)
    return Retriever(store, embedder, min_similarity=0.3, candidate_pool=5)


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    order = [p for p, _ in fused]
    assert order[0] == 1                     # rank 1 + rank 2
    assert order.index(3) < order.index(2)   # 3 appears in both lists, 2 in one
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_relevant_chunk_is_ranked_first(retriever):
    result = retriever.retrieve("my account is locked after wrong password", top_k=3)
    assert result.chunks[0].chunk.doc_id == "login"
    assert result.evidence_sufficient


def test_keyword_mode_finds_exact_error_code(retriever):
    result = retriever.retrieve("SN-2003", top_k=1, mode="keyword")
    assert result.chunks[0].chunk.doc_id == "playback"


def test_unrelated_question_is_marked_insufficient(retriever):
    result = retriever.retrieve("do you sell gift cards for birthdays", top_k=3)
    assert not result.evidence_sufficient
    assert "does not cover" in result.reason


def test_outdated_document_is_flagged_as_conflict(retriever):
    result = retriever.retrieve("refund within days of purchase", top_k=5)
    assert {"outdated_doc_id": "refund-old", "current_doc_id": "refund-new",
            "current_doc_retrieved": True} in result.conflicts


def test_scores_are_attached_to_every_chunk(retriever):
    result = retriever.retrieve("charged twice refund", top_k=3)
    assert [c.rank for c in result.chunks] == [1, 2, 3]
    assert all(-1.0 <= c.similarity <= 1.0 for c in result.chunks)


@pytest.mark.parametrize("query,top_k", [("   ", 3), ("refund", 0)])
def test_invalid_input_is_rejected(retriever, query, top_k):
    with pytest.raises(ValueError):
        retriever.retrieve(query, top_k=top_k)


def test_vector_store_round_trips_to_disk(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore.build(embedder.embed([c.embedding_text() for c in CHUNKS]), CHUNKS,
                              {"embedding_model": embedder.model_name})
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.chunks == CHUNKS
    assert loaded.index.ntotal == len(CHUNKS)
    assert loaded.manifest["embedding_model"] == "fake-bag-of-words"
