"""Train, compare and evaluate the ticket classifier.

Run from backend/:
    python -m app.ml.train

Steps (the order matters, to keep the test score honest):
  1. Split 250 tickets 80/20, stratified (each category keeps its share).
  2. Compare candidate models with 5-fold cross-validation on the TRAIN part only.
  3. Evaluate the chosen model ONCE on the untouched TEST part -> reported metrics.
  4. Refit the chosen model on all 250 tickets and save it for the API
     (more data for the deployed model; the reported score stays the held-out one).
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.config import BACKEND_DIR, get_settings
from app.ml.classifier import CATEGORIES, LOW_CONFIDENCE, TicketClassifier, build_pipeline

CANDIDATES = ["naive_bayes_word", "logreg_word", "logreg_word_char"]
SEED = 42  # fixed so every run gives the same split and the same numbers
REPORT_PATH = BACKEND_DIR / "eval" / "results" / "classifier_metrics.json"


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def main() -> None:
    settings = get_settings()
    texts, labels = load_dataset(settings.tickets_dataset)
    print(f"Loaded {len(texts)} tickets from {settings.tickets_dataset.name}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED)
    print(f"Train {len(x_train)} / test {len(x_test)} (stratified, seed {SEED})\n")

    # ---- 2. model selection on the training set only ----
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("5-fold cross-validation on TRAIN (macro F1, mean +/- std):")
    baseline = cross_val_score(DummyClassifier(strategy="most_frequent"), x_train, y_train,
                               cv=folds, scoring="f1_macro")
    print(f"  {'baseline_most_frequent':<22} {baseline.mean():.3f} +/- {baseline.std():.3f}")
    cv_results = {}
    for kind in CANDIDATES:
        scores = cross_val_score(build_pipeline(kind), x_train, y_train, cv=folds, scoring="f1_macro")
        cv_results[kind] = {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}
        print(f"  {kind:<22} {scores.mean():.3f} +/- {scores.std():.3f}")
    chosen = max(cv_results, key=lambda k: cv_results[k]["mean"])
    print(f"Chosen: {chosen}\n")

    # ---- 3. one evaluation on the held-out test set ----
    model = build_pipeline(chosen).fit(x_train, y_train)
    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)
    macro_f1 = f1_score(y_test, predicted, average="macro")
    print(f"HELD-OUT TEST: accuracy {accuracy:.3f}, macro F1 {macro_f1:.3f}\n")
    print(classification_report(y_test, predicted, labels=CATEGORIES, digits=3))

    matrix = confusion_matrix(y_test, predicted, labels=CATEGORIES)
    short = [c.split("_")[0][:7] for c in CATEGORIES]
    print("Confusion matrix (rows = true, columns = predicted):")
    print(" " * 9 + " ".join(f"{s:>7}" for s in short))
    for name, row in zip(short, matrix):
        print(f"{name:>8} " + " ".join(f"{n:>7}" for n in row))

    # Is confidence useful? Compare accuracy on confident vs low-confidence predictions.
    confidence = model.predict_proba(x_test).max(axis=1)
    correct = np.array(predicted) == np.array(y_test)
    sure = confidence >= LOW_CONFIDENCE
    selective = {
        "threshold": LOW_CONFIDENCE,
        "confident_share": round(float(sure.mean()), 4),
        "accuracy_when_confident": round(float(correct[sure].mean()), 4) if sure.any() else None,
        "accuracy_when_low_confidence": round(float(correct[~sure].mean()), 4) if (~sure).any() else None,
    }
    print(f"\nConfidence >= {LOW_CONFIDENCE}: {sure.sum()}/{len(sure)} tickets, "
          f"accuracy {selective['accuracy_when_confident']}; "
          f"below: {(~sure).sum()} tickets, accuracy {selective['accuracy_when_low_confidence']}")

    mistakes = [{"text": t, "true": y, "predicted": p, "confidence": round(float(c), 3)}
                for t, y, p, c in zip(x_test, y_test, predicted, confidence) if y != p]
    print(f"\nMisclassified test tickets ({len(mistakes)}):")
    for m in mistakes:
        print(f"  [{m['true']} -> {m['predicted']} @ {m['confidence']}] {m['text']}")

    # ---- 4. refit on everything and save ----
    final = TicketClassifier(build_pipeline(chosen).fit(texts, labels), metadata={
        "model": chosen, "trained_on": len(texts),
        "test_accuracy": round(accuracy, 4), "test_macro_f1": round(macro_f1, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    final.save(settings.classifier_path)
    print(f"\nSaved model -> {settings.classifier_path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": {"total": len(texts), "train": len(x_train), "test": len(x_test), "seed": SEED},
        "baseline_cv_macro_f1": round(float(baseline.mean()), 4),
        "cv_macro_f1": cv_results, "chosen_model": chosen,
        "test": {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
                 "per_class": classification_report(y_test, predicted, labels=CATEGORIES,
                                                    output_dict=True, zero_division=0),
                 "confusion_matrix": {"labels": CATEGORIES, "rows_true": np.asarray(matrix).tolist()},
                 "selective": selective, "misclassified": mistakes},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {REPORT_PATH.relative_to(BACKEND_DIR)}")


if __name__ == "__main__":
    main()
