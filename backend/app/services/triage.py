"""Triage = what kind of ticket is this, and does it need a senior agent first?

Runs once when a ticket is created, on the MASKED text (PII is irrelevant to
both questions, and keeping it out of every model is the simpler rule).

Each part degrades gracefully if its model file is missing:
  no classifier        -> category stays None (the API still works)
  no escalation model  -> keyword rules only (lower recall, see eval/results)
"""
from dataclasses import dataclass, field

from app.ml.classifier import TicketClassifier
from app.ml.escalation_model import EscalationModel
from app.rag.embedder import TextEmbedder
from app.services.escalation import detect_escalation


@dataclass
class Triage:
    category: str | None = None
    category_confidence: float | None = None
    category_low_confidence: bool = False
    escalated: bool = False
    escalation_score: float | None = None          # semantic model probability
    escalation_reasons: list[str] = field(default_factory=list)


class Triager:
    def __init__(self, classifier: TicketClassifier | None = None,
                 escalation_model: EscalationModel | None = None,
                 embedder: TextEmbedder | None = None):
        self.classifier = classifier
        # The semantic model is only usable with the embedder it was trained on.
        usable = (escalation_model is not None and embedder is not None
                  and escalation_model.embedding_model == embedder.model_name)
        self.escalation_model = escalation_model if usable else None
        self.embedder = embedder if usable else None

    def triage(self, text: str) -> Triage:
        result = Triage()
        if self.classifier is not None:
            prediction = self.classifier.predict(text)
            result.category = prediction.category
            result.category_confidence = prediction.confidence
            result.category_low_confidence = prediction.low_confidence

        # Rules: precise and explainable, but only catch phrasings someone listed.
        rules = detect_escalation(text)
        result.escalation_reasons = list(rules.reasons)
        result.escalated = rules.escalate

        # Semantic model: catches paraphrases ("attorney", "intruder").
        if self.escalation_model is not None:
            vector = self.embedder.embed([text])[0]
            score = self.escalation_model.score(vector)
            result.escalation_score = round(score, 4)
            if score >= self.escalation_model.threshold:
                result.escalated = True
                if not rules.escalate:
                    result.escalation_reasons.append(
                        f"semantic: similar to past escalations (score {score:.2f})")
        return result
