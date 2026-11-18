"""Stage 4 of RAG: turn text into vectors (embeddings) with Sentence Transformers.

An embedding model is a neural network trained so that texts with similar
meaning produce vectors pointing in similar directions. "I can't log in" and
"account locked after failed sign-in" share few words but get close vectors.

The SAME model must embed both the documents (offline) and the ticket
(online); vectors from two different models are not comparable.
"""
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    """Anything with .dimension and .embed() works (lets tests use a fake)."""

    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        # Imported here, not at module top, so importing this file stays fast
        # (sentence-transformers loads PyTorch, which takes a few seconds).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dimension = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dimension) float32 array of unit-length vectors.

        normalize_embeddings=True scales each vector to length 1, so the dot
        product of two vectors equals their cosine similarity (range -1..1).
        """
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
