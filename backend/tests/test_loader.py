import pytest

from app.config import get_settings
from app.rag.loader import load_documents, parse_front_matter, split_sections

DOC = """---
doc_id: demo
title: Demo Doc
category: billing_payment
last_updated: 2026-01-01
---

# Demo Doc

Intro text before any section.

## First section
Alpha beta.

## Second section
Gamma delta.
"""


def test_parse_front_matter_reads_metadata_and_body():
    metadata, body = parse_front_matter(DOC)
    assert metadata["doc_id"] == "demo"
    assert metadata["last_updated"] == "2026-01-01"
    assert "## First section" in body


def test_parse_front_matter_handles_windows_line_endings():
    metadata, _ = parse_front_matter(DOC.replace("\n", "\r\n"))
    assert metadata["title"] == "Demo Doc"


def test_missing_required_field_is_rejected():
    with pytest.raises(ValueError, match="missing front-matter fields"):
        parse_front_matter("---\ndoc_id: x\n---\nbody")


def test_document_without_front_matter_is_rejected():
    with pytest.raises(ValueError):
        parse_front_matter("# Just a title\nNo metadata here.")


def test_split_sections_keeps_intro_and_skips_title():
    _, body = parse_front_matter(DOC)
    sections = split_sections(body)
    assert [h for h, _ in sections] == ["Overview", "First section", "Second section"]
    assert sections[1][1] == "Alpha beta."


def test_duplicate_doc_ids_are_rejected(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "b.md").write_text(DOC, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_documents(tmp_path)


def test_real_knowledge_base_is_valid():
    """The shipped StreamNest documents parse, and superseded_by links resolve."""
    sections = load_documents(get_settings().knowledge_base_dir)
    doc_ids = {s.doc_id for s in sections}
    assert len(doc_ids) == 10
    outdated = {s.doc_id: s.superseded_by for s in sections if s.superseded_by}
    assert outdated == {"refund-policy-2024": "refund-policy-2026"}
