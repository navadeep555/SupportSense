import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.ml.classifier import Prediction
from app.ml.escalation_model import EscalationModel
from app.services.escalation import detect_escalation
from app.services.triage import Triager
from tests.fakes import FakeEmbedder


@pytest.mark.parametrize("text, reason", [
    ("I think my account was hacked", "account_security"),
    ("Somebody bought Premium on my account", "account_security"),
    ("I was billed again after cancelling", "financial_dispute"),
    ("I will file a chargeback", "financial_dispute"),
    ("this is fraud", "financial_dispute"),
    ("my lawyer will be in touch", "legal_or_regulatory"),
    ("I'll report this to the police", "legal_or_regulatory"),
    ("this is the third time I'm writing", "repeated_contact"),
    ("emailed support four times, zero reply", "repeated_contact"),
])
def test_rules_flag_with_reason(text, reason):
    result = detect_escalation(text)
    assert result.escalate
    assert any(r.startswith(reason) for r in result.reasons)


@pytest.mark.parametrize("text", [
    "I have an issue with subtitles",                      # "issue" must not match "sue"
    "This is my second time using the app",                # not a repeated complaint
    "No reply needed, the new app looks great",
    "My kids keep hacking away at the remote",
    "Can I watch the police drama on Basic?",
    "Is there a fraud warning feature for my card?",
])
def test_rules_ignore_context_traps(text):
    assert not detect_escalation(text).escalate


class FakeClassifier:
    def predict(self, text):
        return Prediction("billing_payment", 0.3, {"billing_payment": 0.3}, low_confidence=True)


def semantic_model(embedder, threshold=0.5):
    """Tiny real LogisticRegression trained on fake embeddings: 'lawyer'-like text -> escalate."""
    texts = ["lawyer legal action court", "attorney legal court", "how do I change plan", "subtitles missing"]
    model = LogisticRegression(C=10).fit(embedder.embed(texts), [1, 1, 0, 0])
    return EscalationModel(model, threshold, embedder.model_name)


def test_triage_combines_classifier_rules_and_semantic_model():
    embedder = FakeEmbedder()
    triager = Triager(FakeClassifier(), semantic_model(embedder), embedder)
    t = triager.triage("expect my attorney in court")   # no keyword rule matches these words
    assert not detect_escalation("expect my attorney in court").escalate
    assert t.category == "billing_payment" and t.category_low_confidence
    assert t.escalated and t.escalation_score >= 0.5
    assert t.escalation_reasons[0].startswith("semantic:")
    calm = triager.triage("subtitles missing on my plan")
    assert not calm.escalated and calm.escalation_reasons == []


def test_semantic_model_disabled_if_embedder_differs():
    embedder = FakeEmbedder()
    model = semantic_model(embedder)
    model.embedding_model = "some-other-model"
    triager = Triager(None, model, embedder)
    assert triager.escalation_model is None                     # vectors would be meaningless
    t = triager.triage("my lawyer will call")                    # rules still work
    assert t.escalated and t.escalation_score is None and t.category is None


def test_escalation_model_save_load(tmp_path):
    embedder = FakeEmbedder()
    model = semantic_model(embedder, threshold=0.4)
    model.save(tmp_path / "esc.joblib")
    loaded = EscalationModel.load(tmp_path / "esc.joblib")
    vector = embedder.embed(["lawyer court"])[0]
    assert loaded.threshold == 0.4
    assert np.isclose(loaded.score(vector), model.score(vector))
