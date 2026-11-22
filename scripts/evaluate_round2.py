"""Compare baseline, iteration 1 and iteration 2 on both exam sets (step 11).

Run from the project root:

    python scripts/evaluate_round2.py

What it does
------------
All three models are **rebuilt in memory** from their training data, rather
than loaded from ``models/``: the model files are git-ignored build artefacts,
so rebuilding is the only way to be sure each model was trained on exactly the
data its numbers claim. Training is deterministic (fixed seed, lbfgs), and the
script checks that the rebuilt baseline and iteration 1 reproduce the step 3/4
test scores before it reports anything new.

    baseline     word TF-IDF + LR             data/intents_train.csv
    iteration 1  word + char n-grams + LR     data/intents_train.csv
    iteration 2  same pipeline as iteration 1 data/intents_train.csv
                                              + data/intents_train_round2.csv

Each is scored on

    exam 1  data/intents_test.csv          the original held-out set (step 2)
    exam 2  data/intents_test_round2.csv   written after the user test, BEFORE
                                           the training additions, in a
                                           different style; never used for
                                           tuning

and by 5-fold stratified cross-validation on its own training data (same
splitter and seed as steps 3 and 4).

Outputs (new files only - the earlier metrics files are left untouched):

    reports/metrics_round2.json             all numbers
    reports/misclassified_round2.csv        every exam error of every model
    reports/confusion_matrix_iter2_exam2.png
    reports/comparison_round2.png
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from peakcast.intent_classifier import (
    CONFIDENCE_THRESHOLD,
    INTENT_LABELS,
    ITER1_C,
    ITER1_CHAR_NGRAM_RANGE,
    ITER2_TRAIN_CSVS,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_CSV,
    TEST_ROUND2_CSV,
    TRAIN_CSV,
    analyse_thresholds,
    build_pipeline,
    build_pipeline_iter1,
    load_dataset,
    load_datasets,
)

N_FOLDS: int = 5

METRICS_JSON: Path = REPORTS_DIR / "metrics_round2.json"
MISCLASSIFIED_CSV: Path = REPORTS_DIR / "misclassified_round2.csv"
CONFUSION_PNG: Path = REPORTS_DIR / "confusion_matrix_iter2_exam2.png"
COMPARISON_PNG: Path = REPORTS_DIR / "comparison_round2.png"

# The published step 3/4 test accuracies, which the rebuilt models must match.
EXPECTED_EXAM1 = {"baseline": 0.775, "iteration 1": 0.8125}


def make_models() -> dict[str, tuple[object, tuple[Path, ...]]]:
    """Return the three untrained pipelines with their training files."""
    return {
        "baseline": (build_pipeline(), (TRAIN_CSV,)),
        "iteration 1": (
            build_pipeline_iter1(C=ITER1_C, char_ngram_range=ITER1_CHAR_NGRAM_RANGE),
            (TRAIN_CSV,),
        ),
        "iteration 2": (
            build_pipeline_iter1(C=ITER1_C, char_ngram_range=ITER1_CHAR_NGRAM_RANGE),
            ITER2_TRAIN_CSVS,
        ),
    }


def score(pipeline, texts: list[str], labels: list[str]) -> dict:
    """Accuracy, macro F1 and the predictions on one exam set."""
    probabilities = pipeline.predict_proba(texts)
    predicted = pipeline.classes_[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    return {
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "macro_f1": round(
            float(f1_score(labels, predicted, average="macro", labels=INTENT_LABELS)), 4
        ),
        "n": len(labels),
        "n_wrong": int((np.asarray(labels) != predicted).sum()),
        "_predicted": predicted,
        "_confidence": confidence,
    }


def cross_validate(pipeline, texts: list[str], labels: list[str]) -> dict:
    """5-fold stratified CV with out-of-fold threshold analysis."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    text_array = np.asarray(texts, dtype=object)
    label_array = np.asarray(labels)
    for train_index, test_index in splitter.split(text_array, label_array):
        from sklearn.base import clone

        model = clone(pipeline)
        model.fit(text_array[train_index].tolist(), label_array[train_index])
        fold_scores.append(
            accuracy_score(label_array[test_index], model.predict(text_array[test_index].tolist()))
        )
    probabilities = cross_val_predict(
        pipeline, texts, labels, cv=splitter, method="predict_proba"
    )
    classes = np.array(sorted(set(labels)))
    predicted = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    return {
        "cv_mean_accuracy": round(float(np.mean(fold_scores)), 4),
        "cv_std_accuracy": round(float(np.std(fold_scores)), 4),
        "cv_fold_scores": [round(float(s), 4) for s in fold_scores],
        "n_training_sentences": len(texts),
        "threshold_analysis": analyse_thresholds(labels, predicted, confidence),
    }


def main() -> None:
    exam1 = load_dataset(TEST_CSV)
    exam2 = load_dataset(TEST_ROUND2_CSV)

    results: dict[str, dict] = {}
    errors: list[dict[str, str]] = []
    fitted = {}
    for name, (pipeline, train_csvs) in make_models().items():
        if not all(path.exists() for path in train_csvs):
            print(f"{name}: training file missing, skipped")
            continue
        texts, labels = load_datasets(train_csvs)
        cv = cross_validate(pipeline, texts, labels)
        pipeline.fit(texts, labels)
        fitted[name] = pipeline
        entry = {"training_files": [p.name for p in train_csvs], **cv}
        for exam_name, (exam_texts, exam_labels) in (("exam1", exam1), ("exam2", exam2)):
            result = score(pipeline, exam_texts, exam_labels)
            for text, gold, pred, conf in zip(
                exam_texts, exam_labels, result["_predicted"], result["_confidence"]
            ):
                if gold != pred:
                    errors.append(
                        {
                            "model": name,
                            "exam": exam_name,
                            "text": text,
                            "true": gold,
                            "predicted": str(pred),
                            "confidence": f"{conf:.2f}",
                        }
                    )
            entry[exam_name] = {k: v for k, v in result.items() if not k.startswith("_")}
        results[name] = entry

    # Reproduction check against the published step 3/4 numbers.
    for name, expected in EXPECTED_EXAM1.items():
        got = results[name]["exam1"]["accuracy"]
        status = "reproduced" if abs(got - expected) < 1e-9 else "MISMATCH"
        print(f"{name}: exam-1 accuracy {got} (published {expected}) - {status}")
        results[name]["exam1"]["reproduces_published"] = status == "reproduced"

    print()
    print(f"{'model':<12} {'CV mean':>8} {'±':>6} {'exam 1':>8} {'exam 2':>8} {'F1 ex2':>8}")
    for name, entry in results.items():
        print(
            f"{name:<12} {entry['cv_mean_accuracy']:>8.3f} {entry['cv_std_accuracy']:>6.3f} "
            f"{entry['exam1']['accuracy']:>8.3f} {entry['exam2']['accuracy']:>8.3f} "
            f"{entry['exam2']['macro_f1']:>8.3f}"
        )

    summary = {
        "evaluated_on": date.today().isoformat(),
        "confidence_threshold_in_use": CONFIDENCE_THRESHOLD,
        "exam1": TEST_CSV.name,
        "exam2": TEST_ROUND2_CSV.name,
        "note": (
            "CV is on each model's own training data, so iteration 2's CV is on a "
            "larger set and not strictly comparable to the earlier CV scores. "
            "Exam 2 has 5 sentences per intent: one error moves an intent's "
            "recall by 20 points."
        ),
        "models": results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with MISCLASSIFIED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "exam", "text", "true", "predicted", "confidence"]
        )
        writer.writeheader()
        writer.writerows(errors)

    if "iteration 2" in fitted:
        texts, labels = exam2
        predicted = fitted["iteration 2"].predict(texts)
        matrix = confusion_matrix(labels, predicted, labels=INTENT_LABELS)
        figure, axis = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(matrix, display_labels=INTENT_LABELS).plot(
            ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45
        )
        axis.set_title("Iteration 2 on exam set 2 (round 2)")
        figure.tight_layout()
        figure.savefig(CONFUSION_PNG, dpi=150)
        plt.close(figure)

        names = list(results)
        x = np.arange(len(names))
        width = 0.26
        figure, axis = plt.subplots(figsize=(8, 4.5))
        series = [
            ("CV (own training data)", [results[n]["cv_mean_accuracy"] for n in names]),
            ("Exam 1 (original)", [results[n]["exam1"]["accuracy"] for n in names]),
            ("Exam 2 (round 2)", [results[n]["exam2"]["accuracy"] for n in names]),
        ]
        for i, (label, values) in enumerate(series):
            bars = axis.bar(x + (i - 1) * width, values, width, label=label)
            axis.bar_label(bars, fmt="%.2f", fontsize=8)
        axis.set_xticks(x, names)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("accuracy")
        axis.set_title("Intent classifier: baseline vs iteration 1 vs iteration 2")
        axis.legend(loc="lower right", fontsize=8)
        figure.tight_layout()
        figure.savefig(COMPARISON_PNG, dpi=150)
        plt.close(figure)

    print(f"\nMetrics -> {METRICS_JSON.name}, errors -> {MISCLASSIFIED_CSV.name}")


if __name__ == "__main__":
    main()
