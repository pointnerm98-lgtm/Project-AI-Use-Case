"""Evaluate a trained intent classifier on the held-out test set.

Run from the project root (after training):

    python scripts/evaluate_intent.py                    # baseline (step 3)
    python scripts/evaluate_intent.py --variant iter1    # iteration 1 (step 4)

Both variants go through exactly the same code path, so their metrics are
directly comparable; only the model file and the output names differ.

Produces, for the presentation:

* a classification report on the console,
* a confusion matrix PNG         - where the model confuses intents,
* a metrics JSON                 - all metrics as machine-readable data,
* a misclassified CSV            - every wrong prediction with its probability,
  used to document the iterative optimisation.

The confidence threshold itself is **not** derived here. It is a tuned
parameter and is therefore taken from out-of-fold cross-validation in
``scripts/optimize_intent.py``; this script only reports how the already-chosen
threshold behaves on the test set.
"""

from __future__ import annotations

import argparse
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
    CONFIDENCE_THRESHOLD,
    INTENT_LABELS,
    ITER1_MODEL_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TEST_CSV,
    analyse_thresholds,
    load_dataset,
    load_model,
)


class Variant:
    """Which model to evaluate and where its report files go.

    Grouping these together keeps the baseline artefacts from step 3 untouched
    when the improved model is evaluated.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model_path: Path,
        confusion_png: str,
        metrics_json: str,
        misclassified_csv: str,
    ) -> None:
        self.name = name
        self.description = description
        self.model_path = model_path
        self.confusion_png = REPORTS_DIR / confusion_png
        self.metrics_json = REPORTS_DIR / metrics_json
        self.misclassified_csv = REPORTS_DIR / misclassified_csv


VARIANTS: dict[str, Variant] = {
    "baseline": Variant(
        name="baseline",
        description="baseline TF-IDF (word 1-2 grams) + LogisticRegression",
        model_path=MODEL_PATH,
        confusion_png="confusion_matrix.png",
        metrics_json="metrics_baseline.json",
        misclassified_csv="misclassified_baseline.csv",
    ),
    "iter1": Variant(
        name="iter1",
        description="iteration 1: word 1-2 grams + char_wb n-grams + LogisticRegression",
        model_path=ITER1_MODEL_PATH,
        confusion_png="confusion_matrix_improved.png",
        metrics_json="metrics_improved.json",
        misclassified_csv="misclassified_improved.csv",
    ),
}


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


def save_confusion_matrix(
    true_labels: list[str], predicted: list[str], variant: Variant
) -> None:
    """Draw the confusion matrix and write it to reports/.

    Rows are the true intents, columns the predicted ones, so every off-diagonal
    cell reads as "true intent X was predicted as Y".

    Args:
        true_labels: Gold intents of the test set.
        predicted: Predicted intents.
        variant: Which model is being evaluated; decides the output file.
    """
    matrix = confusion_matrix(true_labels, predicted, labels=INTENT_LABELS)

    figure, axes = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=INTENT_LABELS)
    display.plot(ax=axes, cmap="Blues", colorbar=False, values_format="d")

    axes.set_title(
        f"PeakCast intent classifier ({variant.name})\n"
        "confusion matrix on the test set"
    )
    axes.set_xlabel("predicted intent")
    axes.set_ylabel("true intent")
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()

    variant.confusion_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(variant.confusion_png, dpi=150)
    plt.close(figure)


def save_misclassified(
    texts: list[str],
    true_labels: list[str],
    predicted: list[str],
    probability_matrix: np.ndarray,
    class_order: list[str],
    variant: Variant,
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
    variant.misclassified_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(variant.misclassified_csv, index=False, encoding="utf-8")
    return frame


def parse_arguments() -> Variant:
    """Read the command line and return the variant to evaluate.

    Returns:
        The selected ``Variant``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a PeakCast intent classifier on the test set."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="baseline",
        help="Which trained model to evaluate (default: baseline).",
    )
    return VARIANTS[parser.parse_args().variant]


def main() -> None:
    """Evaluate the saved model and write all report artefacts."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    variant = parse_arguments()

    print(f"PeakCast - evaluating the intent classifier ({variant.name})")
    print("=" * 58)
    print(f"Model          : {variant.description}")

    pipeline = load_model(variant.model_path)
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

    save_confusion_matrix(true_labels, predicted, variant)
    print(f"Confusion matrix -> {variant.confusion_png.name}")

    misclassified = save_misclassified(
        texts, true_labels, predicted, probability_matrix, class_order, variant
    )
    print(f"Misclassified    -> {variant.misclassified_csv.name}"
          f" ({len(misclassified)} rows)")

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

    # How the threshold chosen on cross-validation behaves on the test set.
    # Shown for transparency only - the value itself is not picked here.
    print(f"\n--- threshold analysis on the test set"
          f" (CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD:.2f}) ---")
    thresholds = analyse_thresholds(true_labels, predicted, confidences)
    print(f"  {'thresh':>7} {'caught errors':>14} {'lost answers':>13}"
          f" {'errors left':>12}")
    for entry in thresholds:
        marker = "  <- chosen" if entry["threshold"] == CONFIDENCE_THRESHOLD else ""
        print(f"  {entry['threshold']:>7.2f} {entry['caught_errors']:>14}"
              f" {entry['lost_answers']:>13} {entry['remaining_errors']:>12}{marker}")

    # --- machine-readable metrics ------------------------------------------
    metrics: dict[str, Any] = {
        "model": variant.description,
        "variant": variant.name,
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
        "confidence_threshold_in_use": CONFIDENCE_THRESHOLD,
        "threshold_analysis": thresholds,
        "n_misclassified": int(len(misclassified)),
        # Reminder for anyone reading the numbers later, including the slides.
        "caveat": (
            "Only 10 test sentences per intent: one error shifts that intent's "
            "recall by 10 percentage points. Read per-intent values together "
            "with the cross-validation mean from train_intent.py / "
            "optimize_intent.py."
        ),
    }

    variant.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    variant.metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetrics          -> {variant.metrics_json.name}")


if __name__ == "__main__":
    main()
