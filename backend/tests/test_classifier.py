import csv
from collections import Counter

import pytest

from app.config import get_settings
from app.ml import artifacts
from app.ml.classifier import CATEGORIES, TicketClassifier, build_pipeline
from app.ml.train import load_dataset


def test_dataset_is_balanced_unique_and_valid():
    with get_settings().tickets_dataset.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 250
    assert Counter(r["category"] for r in rows) == {c: 50 for c in CATEGORIES}
    assert len({r["text"].lower() for r in rows}) == 250          # no duplicate tickets
    assert {r["escalate"] for r in rows} == {"0", "1"}


@pytest.fixture(scope="module")
def classifier():
    texts, labels = load_dataset(get_settings().tickets_dataset)
    return TicketClassifier(build_pipeline().fit(texts, labels))


@pytest.mark.parametrize("text, expected", [
    ("I was charged twice on my credit card", "billing_payment"),
    ("forgot my password, reset link not working", "account_access"),
    ("video keeps buffering on my smart tv", "technical_issue"),
    ("how do I cancel my plan", "subscription"),
])
def test_obvious_tickets(classifier, text, expected):
    prediction = classifier.predict(text)
    assert prediction.category == expected
    assert abs(sum(prediction.scores.values()) - 1.0) < 1e-3     # probabilities sum to 1


def test_char_ngrams_survive_typos(classifier):
    assert classifier.predict("pasword resett lnk not wrking").category == "account_access"


def test_save_load_roundtrip(classifier, tmp_path):
    path = tmp_path / "model.joblib"
    classifier.save(path)
    loaded = TicketClassifier.load(path)
    text = "refund for my last payment"
    assert loaded.predict(text) == classifier.predict(text)


def test_missing_or_mismatched_model_is_refused(classifier, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="python -m app.ml.train"):
        TicketClassifier.load(tmp_path / "missing.joblib")
    path = tmp_path / "model.joblib"
    classifier.save(path)
    monkeypatch.setattr(artifacts.sklearn, "__version__", "0.0.1")
    with pytest.raises(RuntimeError, match="Retrain"):
        TicketClassifier.load(path)


def test_unknown_pipeline_kind():
    with pytest.raises(ValueError):
        build_pipeline("deep_magic")
