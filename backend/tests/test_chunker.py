import pytest

from app.rag.chunker import chunk_sections, split_words
from app.rag.loader import Section


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert split_words(words(10), chunk_size=50, overlap=10) == [words(10)]


def test_long_text_is_split_with_overlap():
    pieces = split_words(words(25), chunk_size=10, overlap=3)
    # step = 10 - 3 = 7, so windows start at 0, 7, 14, 21.
    # The window at 14 covers w14..w23, so a 4th window (w21..w24) is needed.
    assert pieces[0].split() == [f"w{i}" for i in range(0, 10)]
    assert pieces[1].split()[:3] == ["w7", "w8", "w9"]  # 3-word overlap with chunk 0
    assert pieces[-1].split() == ["w21", "w22", "w23", "w24"]  # nothing lost at the end
    assert len(pieces) == 4


def test_every_word_is_covered():
    pieces = split_words(words(103), chunk_size=20, overlap=5)
    covered = {w for p in pieces for w in p.split()}
    assert covered == set(words(103).split())


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 12), (0, 0), (10, -1)])
def test_invalid_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_words("a b c", chunk_size=size, overlap=overlap)


def test_chunks_keep_metadata_and_get_stable_ids():
    section = Section(doc_id="billing", title="Billing", section="Failed Payments!",
                      text=words(25), category="billing_payment", last_updated="2026-01-01",
                      superseded_by=None, source_path="billing.md")
    chunks = chunk_sections([section], chunk_size=10, overlap=3)
    assert [c.chunk_id for c in chunks] == [f"billing::failed-payments::{i}" for i in range(4)]
    assert all(c.doc_id == "billing" and c.section == "Failed Payments!" for c in chunks)
    assert chunks[0].embedding_text().startswith("Billing > Failed Payments!: w0")
