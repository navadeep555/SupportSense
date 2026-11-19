"""Semantic escalation detector: Logistic Regression on sentence embeddings.

Why embeddings: keyword rules scored recall 0.0 on a holdout written in new
words ("attorney", "intruder", "two identical debits"). The MiniLM embedder
(already used for RAG) maps sentences with similar MEANING to nearby vectors,
so "Expect to hear from my attorney" lands near "I'm contacting my lawyer".

The threshold is not 0.5: it is chosen during training (cross-validation on
development data) as the highest value that still catches >= 90% of
escalations, because a missed escalation costs more than a false alarm.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.ml.artifacts import load_bundle, save_bundle

TRAIN_COMMAND = "python -m app.ml.train_escalation"


class EscalationModel:
    def __init__(self, model: LogisticRegression, threshold: float, embedding_model: str,
                 metadata: dict | None = None):
        self.model = model
        self.threshold = threshold
        self.embedding_model = embedding_model  # vectors must come from the SAME embedder
        self.metadata = metadata or {}

    def score(self, vector: np.ndarray) -> float:
        """Probability that the ticket needs escalation (vector: one normalized embedding)."""
        return float(self.model.predict_proba(vector.reshape(1, -1))[0, 1])

    def save(self, path: Path) -> None:
        save_bundle(path, model=self.model, threshold=self.threshold,
                    embedding_model=self.embedding_model, metadata=self.metadata)

    @classmethod
    def load(cls, path: Path) -> "EscalationModel":
        b = load_bundle(path, TRAIN_COMMAND)
        return cls(b["model"], b["threshold"], b["embedding_model"], b["metadata"])
