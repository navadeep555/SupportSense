"""Stage 5 of RAG: store vectors in FAISS and keep a mapping back to the chunks.

FAISS only stores numbers. It answers "which stored vectors are closest to
this query vector?" with their POSITIONS (0, 1, 2, ...). We keep `chunks`
in exactly the same order, so position i in FAISS <-> self.chunks[i].

On disk (data/index/):
    faiss.index   - the vectors
    chunks.json   - the chunk text + metadata, same order as the vectors
    manifest.json - which embedding model/settings built this index, and when
"""
import json
from pathlib import Path

import faiss
import numpy as np

from app.rag.chunker import Chunk

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], manifest: dict | None = None):
        if index.ntotal != len(chunks):
            raise ValueError(f"index has {index.ntotal} vectors but there are {len(chunks)} chunks")
        self.index = index
        self.chunks = chunks
        self.manifest = manifest or {}

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk], manifest: dict | None = None) -> "VectorStore":
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("vectors must have shape (number_of_chunks, dimension)")
        # IndexFlatIP = exact search by Inner Product (dot product). Because our
        # vectors are normalised, inner product == cosine similarity.
        # "Flat" = compares against every vector; fine for thousands of chunks.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(index, chunks, manifest)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_position, cosine_similarity), best first."""
        k = min(k, len(self.chunks))
        scores, positions = self.index.search(query_vector.reshape(1, -1).astype("float32"), k)
        return [(int(p), float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]

    def similarity(self, query_vector: np.ndarray, position: int) -> float:
        """Cosine similarity between the query and one stored chunk."""
        stored = self.index.reconstruct(position)  # get the stored vector back
        return float(np.dot(query_vector.reshape(-1), stored))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=2), encoding="utf-8")
        (directory / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        if not (directory / INDEX_FILE).exists():
            raise FileNotFoundError(
                f"no FAISS index in {directory}. Build it with: python -m app.rag.ingest")
        index = faiss.read_index(str(directory / INDEX_FILE))
        chunks = [Chunk(**d) for d in json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))]
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        return cls(index, chunks, manifest)
