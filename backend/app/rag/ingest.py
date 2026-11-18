"""OFFLINE ingestion: documents -> sections -> chunks -> embeddings -> FAISS index.

Run from the backend/ folder whenever the knowledge base changes:
    python -m app.rag.ingest
"""
import time
from collections import Counter
from datetime import datetime, timezone

from app.config import get_settings
from app.rag.chunker import chunk_sections
from app.rag.embedder import SentenceTransformerEmbedder, TextEmbedder
from app.rag.loader import load_documents
from app.rag.vector_store import VectorStore


def run_ingestion(settings, embedder: TextEmbedder | None = None) -> dict:
    started = time.perf_counter()

    sections = load_documents(settings.knowledge_base_dir)                   # stages 1-2
    chunks = chunk_sections(sections, settings.chunk_size_words,
                            settings.chunk_overlap_words)                    # stage 3
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    vectors = embedder.embed([c.embedding_text() for c in chunks])           # stage 4

    manifest = {
        "embedding_model": embedder.model_name,
        "dimension": int(vectors.shape[1]),
        "documents": len({c.doc_id for c in chunks}),
        "sections": len(sections),
        "chunks": len(chunks),
        "chunk_size_words": settings.chunk_size_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store = VectorStore.build(vectors, chunks, manifest)                     # stage 5
    store.save(settings.index_dir)

    manifest["seconds"] = round(time.perf_counter() - started, 2)
    manifest["chunks_per_document"] = dict(Counter(c.doc_id for c in chunks))
    return manifest


def main() -> None:
    settings = get_settings()
    print(f"Knowledge base: {settings.knowledge_base_dir}")
    print(f"Embedding model: {settings.embedding_model}")
    summary = run_ingestion(settings)
    print(f"Indexed {summary['chunks']} chunks from {summary['documents']} documents "
          f"({summary['sections']} sections), dimension {summary['dimension']}, "
          f"in {summary['seconds']}s")
    for doc_id, n in sorted(summary["chunks_per_document"].items()):
        print(f"  {doc_id:32s} {n} chunks")
    print(f"Saved to: {settings.index_dir}")


if __name__ == "__main__":
    main()
