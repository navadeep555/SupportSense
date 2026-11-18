"""Stage 3 of RAG: split sections into overlapping, word-based chunks.

Why chunk at all? The embedding model reads at most ~256 tokens (~180 words);
longer text is silently cut off. Smaller chunks also make retrieval precise:
a match points at one paragraph, not a whole document.

Why overlap? If an answer straddles a chunk boundary, the overlap keeps the
end of one chunk and the start of the next together in at least one chunk.
"""
import re
from dataclasses import asdict, dataclass

from app.rag.loader import Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # stable, human-readable ID, e.g. "billing-payments::failed-or-declined-payments::0"
    doc_id: str
    title: str
    section: str
    text: str
    category: str
    last_updated: str
    superseded_by: str | None
    source_path: str

    def embedding_text(self) -> str:
        """Text that gets embedded/indexed: title + section give the chunk context.

        A chunk saying "The lock lifts automatically" means little alone;
        prefixed with its heading it clearly concerns locked accounts.
        """
        return f"{self.title} > {self.section}: {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window over words: windows of `chunk_size`, moving by `chunk_size - overlap`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break  # this window already reached the last word
    return pieces


def chunk_sections(sections: list[Section], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for s in sections:
        base_id = f"{s.doc_id}::{slugify(s.section)}"
        for i, piece in enumerate(split_words(s.text, chunk_size, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{base_id}::{i}",
                doc_id=s.doc_id,
                title=s.title,
                section=s.section,
                text=piece,
                category=s.category,
                last_updated=s.last_updated,
                superseded_by=s.superseded_by,
                source_path=s.source_path,
            ))

    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk IDs: two sections in one document share a heading")
    return chunks
