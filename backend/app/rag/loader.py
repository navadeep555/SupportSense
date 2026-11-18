"""Stage 1-2 of RAG: load knowledge documents and extract text + metadata.

Each Markdown file looks like:

    ---
    doc_id: refund-policy-2026
    title: Refund Policy (current)
    category: billing_payment
    last_updated: 2026-01-15
    superseded_by:            (optional)
    ---
    # Title
    ## Section heading
    Section text...

Output: one `Section` per '## ' heading, carrying the document's metadata.
"""
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("doc_id", "title", "category", "last_updated")


@dataclass(frozen=True)
class Section:
    doc_id: str
    title: str
    section: str
    text: str
    category: str
    last_updated: str
    superseded_by: str | None
    source_path: str


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split '---\\nkey: value\\n---\\nbody' into ({key: value}, body)."""
    raw = raw.replace("\r\n", "\n")  # Windows line endings -> '\n'
    if not raw.startswith("---\n"):
        raise ValueError("document must start with a '---' front-matter block")
    end = raw.find("\n---", 4)
    if end == -1:
        raise ValueError("front-matter block is not closed with '---'")

    metadata: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"invalid front-matter line: {line!r}")
        metadata[key.strip()] = value.strip()

    missing = [f for f in REQUIRED_FIELDS if not metadata.get(f)]
    if missing:
        raise ValueError(f"missing front-matter fields: {missing}")

    body = raw[end + len("\n---"):]
    return metadata, body


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(section_heading, section_text)] using '## ' headings.

    Text before the first '## ' heading (other than the '# ' title) is kept
    under the heading "Overview" so nothing is silently dropped.
    """
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append((heading, text))

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # document title; already in metadata
        else:
            lines.append(line)
    flush()
    return sections


def load_document(path: Path) -> list[Section]:
    metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
    return [
        Section(
            doc_id=metadata["doc_id"],
            title=metadata["title"],
            section=heading,
            text=text,
            category=metadata["category"],
            last_updated=metadata["last_updated"],
            superseded_by=metadata.get("superseded_by") or None,
            source_path=path.name,
        )
        for heading, text in split_sections(body)
    ]


def load_documents(kb_dir: Path) -> list[Section]:
    """Load every *.md file in kb_dir (sorted, so results are reproducible)."""
    paths = sorted(Path(kb_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md documents found in {kb_dir}")

    sections: list[Section] = []
    seen: dict[str, str] = {}
    for path in paths:
        doc_sections = load_document(path)
        if not doc_sections:
            raise ValueError(f"{path.name} has no text")
        doc_id = doc_sections[0].doc_id
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc_id!r} in {path.name} and {seen[doc_id]}")
        seen[doc_id] = path.name
        sections.extend(doc_sections)

    # A superseded_by pointer must name a document that actually exists.
    for s in sections:
        if s.superseded_by and s.superseded_by not in seen:
            raise ValueError(f"{s.doc_id} is superseded_by unknown doc {s.superseded_by!r}")
    return sections
