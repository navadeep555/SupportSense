"""Ticket category classifier: TF-IDF features + Logistic Regression (scikit-learn).

Why this model and not an LLM or a neural network?
  - 250 labelled tickets is tiny; linear models on TF-IDF are the strong,
    standard baseline for short-text classification at this size.
  - Trains in about a second on a CPU, predicts in about a millisecond.
  - Explainable: every word/character n-gram has a weight per category.
  - Logistic Regression gives probabilities, so we can show confidence.

Features (combined with FeatureUnion):
  word 1-2 grams        "refund", "charged twice", "free trial"
  char 3-5 grams        survive typos: "pasword" still shares "pas", "swor"... with "password"
"""
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from app.ml.artifacts import load_bundle, save_bundle

CATEGORIES = ["account_access", "billing_payment", "subscription", "technical_issue", "general_inquiry"]
# Below this, the agent should double-check the category (5 classes: random guessing = 0.2).
LOW_CONFIDENCE = 0.45


def word_features() -> TfidfVectorizer:
    # sublinear_tf: a word repeated 5 times is not 5x as important.
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1)


def char_features() -> TfidfVectorizer:
    # char_wb = character n-grams inside word boundaries (padded with spaces).
    return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)


def build_pipeline(kind: str = "logreg_word_char") -> Pipeline:
    """The candidate models compared in training. All produce probabilities."""
    if kind == "naive_bayes_word":
        return Pipeline([("features", word_features()), ("model", MultinomialNB(alpha=0.5))])
    if kind == "logreg_word":
        return Pipeline([("features", word_features()),
                         ("model", LogisticRegression(C=10, max_iter=2000))])
    if kind == "logreg_word_char":
        features = FeatureUnion([("word", word_features()), ("char", char_features())])
        return Pipeline([("features", features), ("model", LogisticRegression(C=10, max_iter=2000))])
    raise ValueError(f"Unknown model kind: {kind}")


@dataclass
class Prediction:
    category: str
    confidence: float                 # probability of the chosen category
    scores: dict[str, float]          # probability of every category
    low_confidence: bool


class TicketClassifier:
    def __init__(self, pipeline: Pipeline, metadata: dict | None = None):
        self.pipeline = pipeline
        self.metadata = metadata or {}

    def predict(self, text: str) -> Prediction:
        probabilities = self.pipeline.predict_proba([text])[0]
        labels = self.pipeline.classes_
        scores = {label: round(float(p), 4) for label, p in zip(labels, probabilities)}
        best = max(scores, key=scores.get)
        return Prediction(best, scores[best], scores, scores[best] < LOW_CONFIDENCE)

    def save(self, path: Path) -> None:
        save_bundle(path, pipeline=self.pipeline, metadata=self.metadata)

    @classmethod
    def load(cls, path: Path) -> "TicketClassifier":
        bundle = load_bundle(path, "python -m app.ml.train")
        return cls(bundle["pipeline"], bundle["metadata"])
