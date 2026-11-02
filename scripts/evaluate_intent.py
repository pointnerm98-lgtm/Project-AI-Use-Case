"""Evaluate the trained intent classifier on the held-out test set.

Run from the project root (after training):

    python scripts/evaluate_intent.py

Produces, for the presentation:

* a classification report on the console,
* ``reports/confusion_matrix.png``       - where the model confuses intents,
* ``reports/metrics_baseline.json``      - all metrics as machine-readable data,
* ``reports/misclassified_baseline.csv`` - every wrong prediction with its
  probability, used to document the iterative optimisation,
* a confidence-threshold analysis that answers "below which probability should
  the bot say it did not understand?".
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# Use a non-interactive backend: the script must run on a machine without a
# display and only ever writes the plot to a file.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from peakcast.intent_classifier import (
    INTENT_LABELS,
    MODEL_PATH,
    REPORTS_DIR,
    TEST_CSV,
    load_dataset,
    load_model,
)

CONFUSION_MATRIX_PNG: Path = REPORTS_DIR / "confusion_matrix.png"
METRICS_JSON: Path = REPORTS_DIR / "metrics_baseline.json"
MISCLASSIFIED_CSV: Path = REPORTS_DIR / "misclassified_baseline.csv"

# Candidate thresholds for the "I did not understand" fallback.
CANDIDATE_THRESHOLDS: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70)


def collect_predictions(
    pipeline: Any, texts: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Predict intents and keep the full probability distribution.

    Args:
        pipeline: The fitted pipeline.
        texts: Test sentences.

    Returns:
        A tuple ``(predicted_labels, confidences, probability_matrix)`` where
        ``confidences`` holds the probability of the predicted intent and
        ``probability_matrix`` has one row per sentence and one column per class
        in ``pipeline.classes_`` order.
    """
    probability_matrix = pipeline.predict_proba(texts)
    best_indices = probability_matrix.argmax(axis=1)
    predicted = [str(pipeline.classes_[i]) for i in best_indices]
    confidences = probability_matrix[np.arange(len(texts)), best_indices]
    return predicted, confidences, probability_matrix


def save_confusion_matrix(true_labels: list[str], predicted: list[str]) -> None:
    """Draw the confusion matrix and write it to reports/.

    Rows are the true intents, columns the predicted ones, so every off-diagonal
    cell reads as "true intent X was predicted as Y".

    Args:
        true_labels: Gold intents of the test set.
        predicted: Predicted intents.
    """
    matrix = confusion_matrix(true_labels, predicted, labels=INTENT_LABELS)

    figure, axes = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=INTENT_LABELS)
    display.plot(ax=axes, cmap="Blues", colorbar=False, values_format="d")

    axes.set_title("PeakCast baseline intent classifier\nconfusion matrix on the test set")
    axes.set_xlabel("predicted intent")
    axes.set_ylabel("true intent")
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()

    CONFUSION_MATRIX_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CONFUSION_MATRIX_PNG, dpi=150)
    plt.close(figure)


def save_misclassified(
    texts: list[str],
    true_labels: list[str],
    predicted: list[str],
    probability_matrix: np.ndarray,
    class_order: list[str],
) -> pd.DataFrame:
    """Write every wrong prediction to a CSV file.

    Also records the probability the model gave to the *correct* intent. A
    near-miss (true intent second with a similar score) needs different data
    than a confident error, so this column guides the next iteration.

    Args:
        texts: Test sentences.
        true_labels: Gold intents.
        predicted: Predicted intents.
        probability_matrix: Full probability output.
        class_order: Class labels in the column order of the matrix.

    Returns:
        The DataFrame that was written.
    """
    column_of = {label: index for index, label in enumerate(class_order)}
    rows: list[dict[str, Any]] = []

    for row_index, (text, true_label, predicted_label) in enumerate(
        zip(texts, true_labels, predicted)
    ):
        if true_label == predicted_label:
            continue
        probabilities = probability_matrix[row_index]
        rows.append(
            {
                "text": text,
                "true_intent": true_label,
                "predicted_intent": predicted_label,
                "predicted_probability": round(
                    float(probabilities[column_of[predicted_label]]), 4
                ),
                "true_intent_probability": round(
                    float(probabilities[column_of[true_label]]), 4
                ),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "text",
            "true_intent",
            "predicted_intent",
            "predicted_probability",
            "true_intent_probability",
        ],
    )
    MISCLASSIFIED_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MISCLASSIFIED_CSV, index=False, encoding="utf-8")
    return frame


def analyse_thresholds(
    true_labels: list[str], predicted: list[str], confidences: np.ndarray
) -> list[dict[str, Any]]:
    """Evaluate candidate confidence thresholds for the fallback answer.

    A threshold trades two error types against each other:

    * **caught errors** - wrong predictions below the threshold. The bot asks
      back instead of answering wrongly. This is the benefit.
    * **lost answers** - correct predictions below the threshold. The bot asks
      back although it was right. This is the cost, and it is the more annoying
      failure for a user standing at a trailhead.

    Args:
        true_labels: Gold intents.
        predicted: Predicted intents.
        confidences: Probability of each predicted intent.

    Returns:
        One summary dict per candidate threshold.
    """
    correct = np.array(
        [t == p for t, p in zip(true_labels, predicted)], dtype=bool
    )
    total_errors = int((~correct).sum())

    summaries: list[dict[str, Any]] = []
    for threshold in CANDIDATE_THRESHOLDS:
        below = confidences < threshold
        caught_errors = int((below & ~correct).sum())
        lost_answers = int((below & correct).sum())
        remaining_errors = total_errors - caught_errors
        summaries.append(
            {
                "threshold": threshold,
                "caught_errors": caught_errors,
                "lost_answers": lost_answers,
                "remaining_errors": remaining_errors,
                # Share of the wrong answers that the threshold intercepts.
                "error_catch_rate": round(caught_errors / total_errors, 3)
                if total_errors
                else 0.0,
                # Share of correct answers sacrificed to achieve that.
                "correct_answer_loss_rate": round(
                    lost_answers / int(correct.sum()), 3
                )
                if correct.any()
                else 0.0,
            }
        )
    return summaries


def main() -> None:
    """Evaluate the saved model and write all report artefacts."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("PeakCast - evaluating the baseline intent classifier")
    print("=" * 55)

    pipeline = load_model(MODEL_PATH)
    texts, true_labels = load_dataset(TEST_CSV)
    print(f"Test sentences : {len(texts)}")

    predicted, confidences, probability_matrix = collect_predictions(pipeline, texts)
    class_order = [str(c) for c in pipeline.classes_]

    accuracy = accuracy_score(true_labels, predicted)
    print(f"\nTest accuracy  : {accuracy:.4f}  "
          f"({int(accuracy * len(texts))}/{len(texts)} correct)")

    print("\n--- classification report ---")
    report_text = classification_report(
        true_labels, predicted, labels=INTENT_LABELS, zero_division=0
    )
    print(report_text)

    # Same report as nested data, for the JSON file and the slides.
    report_dict = classification_report(
        true_labels,
        predicted,
        labels=INTENT_LABELS,
        zero_division=0,
        output_dict=True,
    )

    save_confusion_matrix(true_labels, predicted)
    print(f"Confusion matrix -> {CONFUSION_MATRIX_PNG.name}")

    misclassified = save_misclassified(
        texts, true_labels, predicted, probability_matrix, class_order
    )
    print(f"Misclassified    -> {MISCLASSIFIED_CSV.name} ({len(misclassified)} rows)")

    print("\n--- misclassified test sentences ---")
    if misclassified.empty:
        print("none")
    else:
        for row in misclassified.itertuples(index=False):
            print(
                f"  {row.true_intent:<18} -> {row.predicted_intent:<18}"
                f" p={row.predicted_probability:.2f}"
                f" (true intent got {row.true_intent_probability:.2f})"
                f"  | {row.text}"
            )

    print("\n--- confidence distribution ---")
    correct_mask = np.array([t == p for t, p in zip(true_labels, predicted)])
    print(f"  correct predictions : mean p={confidences[correct_mask].mean():.3f}  "
          f"min p={confidences[correct_mask].min():.3f}")
    if (~correct_mask).any():
        print(f"  wrong predictions   : mean p={confidences[~correct_mask].mean():.3f}  "
              f"max p={confidences[~correct_mask].max():.3f}")

    print("\n--- confidence threshold analysis ---")
    thresholds = analyse_thresholds(true_labels, predicted, confidences)
    print(f"  {'thresh':>7} {'caught errors':>14} {'lost answers':>13}"
          f" {'errors left':>12}")
    for entry in thresholds:
        print(f"  {entry['threshold']:>7.2f} {entry['caught_errors']:>14}"
              f" {entry['lost_answers']:>13} {entry['remaining_errors']:>12}")

    # --- machine-readable metrics ------------------------------------------
    metrics: dict[str, Any] = {
        "model": "baseline TF-IDF (word 1-2 grams) + LogisticRegression",
        "evaluated_on": str(date.today()),
        "n_test_sentences": len(texts),
        "n_test_per_intent": len(texts) // len(INTENT_LABELS),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "per_intent": {
            label: {
                "precision": round(float(report_dict[label]["precision"]), 4),
                "recall": round(float(report_dict[label]["recall"]), 4),
                "f1": round(float(report_dict[label]["f1-score"]), 4),
                "support": int(report_dict[label]["support"]),
            }
            for label in INTENT_LABELS
        },
        "confidence": {
            "mean_correct": round(float(confidences[correct_mask].mean()), 4),
            "mean_wrong": round(float(confidences[~correct_mask].mean()), 4)
            if (~correct_mask).any()
            else None,
            "max_wrong": round(float(confidences[~correct_mask].max()), 4)
            if (~correct_mask).any()
            else None,
        },
        "threshold_analysis": thresholds,
        "n_misclassified": int(len(misclassified)),
        # Reminder for anyone reading the numbers later, including the slides.
        "caveat": (
            "Only 10 test sentences per intent: one error shifts that intent's "
            "recall by 10 percentage points. Read per-intent values together "
            "with the cross-validation mean from train_intent.py."
        ),
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetrics          -> {METRICS_JSON.name}")


if __name__ == "__main__":
    main()
