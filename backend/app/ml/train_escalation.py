"""Train the semantic escalation model and compare it with the keyword rules.

Run from backend/ (needs the embedding model, downloaded in Phase 3):
    python -m app.ml.train_escalation

Data, in the order it was used (see eval/README notes in this docstring):
  data/tickets/synthetic_tickets.csv  250 tickets: rules v1 written from these   -> development
  eval/escalation_round1.csv          24 tickets: rules v1 got P 0.44 / R 0.33,
                                      rules v2 revised from the failures          -> development
  eval/escalation_holdout.csv         24 tickets written after rules v2 were frozen:
                                      rules v2 got P 0.00 / R 0.00, which led to this
                                      semantic model. Never used for fitting or
                                      threshold choice; reported once.
Caveat: one person wrote all of it, and the holdout has only 24 tickets.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.config import BACKEND_DIR, get_settings
from app.ml.escalation_model import EscalationModel
from app.rag.embedder import SentenceTransformerEmbedder
from app.services.escalation import detect_escalation

SEED = 42
TARGET_RECALL = 0.9          # catch at least 9 of 10 escalations (chosen before training)
C_VALUES = [1.0, 10.0]       # regularisation strengths compared by cross-validation
ROUND1 = BACKEND_DIR / "eval" / "escalation_round1.csv"
HOLDOUT = BACKEND_DIR / "eval" / "escalation_holdout.csv"
REPORT_PATH = BACKEND_DIR / "eval" / "results" / "escalation_metrics.json"


def load_rows(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], np.array([int(r["escalate"]) for r in rows])


def precision_recall(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(((predicted == 1) & (truth == 1)).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    fn = int(((predicted == 0) & (truth == 1)).sum())
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0, "tp": tp, "fp": fp, "fn": fn}


def pick_threshold(probabilities: np.ndarray, truth: np.ndarray) -> tuple[float, dict]:
    """Highest threshold whose recall is still >= TARGET_RECALL (higher threshold = fewer false alarms)."""
    best = (0.05, precision_recall((probabilities >= 0.05).astype(int), truth))
    for threshold in np.arange(0.05, 0.96, 0.05):
        stats = precision_recall((probabilities >= threshold).astype(int), truth)
        if stats["recall"] >= TARGET_RECALL:
            best = (round(float(threshold), 2), stats)
    return best


def main() -> None:
    settings = get_settings()
    dev_a, y_a = load_rows(settings.tickets_dataset)
    dev_b, y_b = load_rows(ROUND1)
    hold, y_hold = load_rows(HOLDOUT)
    dev_texts, y_dev = dev_a + dev_b, np.concatenate([y_a, y_b])
    print(f"Development: {len(dev_texts)} tickets ({y_dev.sum()} escalations); holdout: {len(hold)}")

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    x_dev = embedder.embed(dev_texts)

    # ---- choose C and threshold with cross-validation on development data only ----
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    choices = []
    for c in C_VALUES:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        cv_prob = cross_val_predict(model, x_dev, y_dev, cv=folds, method="predict_proba")[:, 1]
        threshold, stats = pick_threshold(cv_prob, y_dev)
        print(f"C={c:<5} CV threshold {threshold}: precision {stats['precision']} recall {stats['recall']}")
        choices.append((stats["precision"], c, threshold, stats))
    _, best_c, threshold, cv_stats = max(choices)
    print(f"Chosen C={best_c}, threshold={threshold}\n")

    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(x_dev, y_dev)
    escalation = EscalationModel(model, threshold, settings.embedding_model, metadata={
        "C": best_c, "trained_on": len(dev_texts), "cv": cv_stats})
    escalation.save(settings.escalation_model_path)

    # ---- compare rules / semantic / combined on every set ----
    report = {"chosen": {"C": best_c, "threshold": threshold, "cv_on_development": cv_stats}, "sets": {}}
    sets = [("synthetic (development)", dev_a, y_a), ("round1 (development)", dev_b, y_b),
            ("HOLDOUT", hold, y_hold)]
    for name, texts, truth in sets:
        rules = np.array([int(detect_escalation(t).escalate) for t in texts])
        semantic = (model.predict_proba(embedder.embed(texts))[:, 1] >= threshold).astype(int)
        results = {"rules": precision_recall(rules, truth),
                   "semantic": precision_recall(semantic, truth),
                   "rules_or_semantic": precision_recall(rules | semantic, truth)}
        report["sets"][name] = results
        print(f"{name}  ({truth.sum()} escalations / {len(truth)})")
        for method, r in results.items():
            print(f"  {method:<18} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
                  f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})")
    print("\nNote: semantic scores on the development sets are training scores (the model saw them).")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model -> {settings.escalation_model_path}")
    print(f"Saved metrics -> {REPORT_PATH.relative_to(BACKEND_DIR)}")


if __name__ == "__main__":
    main()
