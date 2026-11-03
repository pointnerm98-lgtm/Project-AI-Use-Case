"""Train the baseline intent classifier and save it to models/.

Run from the project root:

    python scripts/train_intent.py

What it does:

1. Loads ``data/intents_train.csv``.
2. Measures 5-fold cross-validation accuracy on the training set. This is the
   honest estimate of how well the model generalises *within* the training data
   and is reported alongside the test-set score, because with only 10 test
   sentences per intent a single test error moves the numbers a lot.
3. Fits the pipeline on the full training set and writes it to
   ``models/intent_model.joblib``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project root importable so that "peakcast" is found when this file
# is executed directly as a script (the project is not pip-installed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from peakcast.intent_classifier import (
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TRAIN_CSV,
    build_pipeline,
    load_dataset,
    train_and_save,
)

# Number of cross-validation folds. With 40 sentences per intent this leaves
# 8 sentences per intent in each validation fold.
N_FOLDS: int = 5

# Cross-validation summary, read by scripts/compare_iterations.py so the
# comparison chart does not have to retrain anything.
CV_JSON: Path = REPORTS_DIR / "cv_baseline.json"


def run_cross_validation(texts: list[str], labels: list[str]) -> np.ndarray:
    """Run stratified k-fold cross-validation on the training data.

    ``StratifiedKFold`` keeps the intent distribution identical in every fold.
    ``shuffle=True`` is essential here: the CSV is grouped by intent for
    readability, and without shuffling the folds would be cut along those
    blocks.

    Args:
        texts: Training sentences.
        labels: Matching intent labels.

    Returns:
        Array with one accuracy score per fold.
    """
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    # A fresh pipeline is built per fold by cross_val_score, so no information
    # leaks from the validation fold into the vectorizer vocabulary.
    return cross_val_score(
        build_pipeline(),
        texts,
        labels,
        cv=splitter,
        scoring="accuracy",
    )


def main() -> None:
    """Train, cross-validate and save the baseline intent model."""
    # Windows consoles default to cp1252 and would crash on Austrian place
    # names such as "Sölden" or "Großglockner".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("PeakCast - training the baseline intent classifier")
    print("=" * 55)

    texts, labels = load_dataset(TRAIN_CSV)
    print(f"Training sentences : {len(texts)}")
    print(f"Intents            : {len(set(labels))}")

    print(f"\n{N_FOLDS}-fold cross-validation on the training set ...")
    scores = run_cross_validation(texts, labels)
    for fold_index, score in enumerate(scores, start=1):
        print(f"  fold {fold_index}: {score:.3f}")
    print(f"\n  mean accuracy : {scores.mean():.4f}")
    print(f"  std deviation : {scores.std():.4f}")

    print("\nFitting the final model on all training data ...")
    pipeline = train_and_save(MODEL_PATH)

    # Vocabulary size is worth reporting: it shows how many distinct features
    # the small dataset produces and explains the model's memory footprint.
    vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"  TF-IDF features : {vocabulary_size}")
    print(f"  saved to        : {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}")

    # Persist the cross-validation result for the before/after comparison.
    CV_JSON.parent.mkdir(parents=True, exist_ok=True)
    CV_JSON.write_text(
        json.dumps(
            {
                "model": "baseline: word 1-2 gram TF-IDF + LogisticRegression",
                "cv_mean_accuracy": round(float(scores.mean()), 4),
                "cv_std_accuracy": round(float(scores.std()), 4),
                "cv_fold_scores": [round(float(s), 4) for s in scores],
                "n_features": int(vocabulary_size),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  CV summary      -> {CV_JSON.name}")
    print("\nDone. Next step: python scripts/evaluate_intent.py")


if __name__ == "__main__":
    main()
