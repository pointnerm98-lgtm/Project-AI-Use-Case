"""Iteration 1: grid-search the intent classifier and re-derive the threshold.

Run from the project root:

    python scripts/optimize_intent.py

What it does:

1. Runs ``GridSearchCV`` over word + character n-gram TF-IDF features and the
   regularisation strength ``C``, scored by 5-fold stratified cross-validation
   on the **training set only**.
2. Writes every grid combination to ``reports/grid_search_iter1.csv``.
3. Refits the best combination on the full training set and saves it to
   ``models/intent_model_iter1.joblib``.
4. Re-derives the confidence threshold from **out-of-fold** cross-validation
   predictions of the chosen model.

The test set is never touched here. Every choice is made on cross-validation,
so the later test-set evaluation stays an honest estimate of generalisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from peakcast.intent_classifier import (
    ITER1_MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TRAIN_CSV,
    analyse_thresholds,
    build_pipeline_iter1,
    load_dataset,
    train_and_save,
)

GRID_SEARCH_CSV: Path = REPORTS_DIR / "grid_search_iter1.csv"
CV_JSON: Path = REPORTS_DIR / "cv_iter1.json"

N_FOLDS: int = 5

# Search space. Kept deliberately small: with 320 sentences a large grid would
# mostly fit noise, and every extra combination makes the "best" result more
# likely to be a lucky split rather than a real improvement.
PARAM_GRID: dict[str, list[Any]] = {
    # Inverse regularisation strength. The baseline's problem was that the
    # default C=1.0 over-regularises, so the grid spans both directions.
    "clf__C": [0.3, 1.0, 3.0, 10.0, 30.0],
    # How long the character n-grams are. Shorter ranges generalise more,
    # longer ranges capture more word-like structure.
    "feat__char__ngram_range": [(2, 4), (3, 5), (2, 5)],
}


def run_grid_search(
    texts: list[str], labels: list[str], splitter: StratifiedKFold
) -> GridSearchCV:
    """Search the parameter grid with cross-validated accuracy.

    Args:
        texts: Training sentences.
        labels: Matching intent labels.
        splitter: The cross-validation splitter to score with.

    Returns:
        The fitted ``GridSearchCV`` object, refit on the full training set.
    """
    search = GridSearchCV(
        estimator=build_pipeline_iter1(),
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=splitter,
        refit=True,     # refit the winner on all training data
        n_jobs=-1,      # the grid is small; use all cores
        return_train_score=False,
    )
    search.fit(texts, labels)
    return search


def save_grid_results(search: GridSearchCV) -> pd.DataFrame:
    """Write all grid combinations to CSV, best first.

    Args:
        search: The fitted search object.

    Returns:
        The DataFrame that was written.
    """
    results = pd.DataFrame(search.cv_results_)

    # Keep the columns that are readable on a slide, drop per-fold timings.
    columns = [
        "param_clf__C",
        "param_feat__char__ngram_range",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    table = results[columns].copy()
    table = table.rename(
        columns={
            "param_clf__C": "C",
            "param_feat__char__ngram_range": "char_ngram_range",
            "mean_test_score": "mean_cv_accuracy",
            "std_test_score": "std_cv_accuracy",
            "rank_test_score": "rank",
        }
    )
    table["mean_cv_accuracy"] = table["mean_cv_accuracy"].round(4)
    table["std_cv_accuracy"] = table["std_cv_accuracy"].round(4)
    table = table.sort_values("rank")

    GRID_SEARCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(GRID_SEARCH_CSV, index=False, encoding="utf-8")
    return table


def derive_threshold(
    texts: list[str],
    labels: list[str],
    best_params: dict[str, Any],
    splitter: StratifiedKFold,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Re-derive the confidence threshold from out-of-fold predictions.

    ``cross_val_predict`` gives every training sentence a prediction made by a
    model that did **not** see it during fitting, which is the closest honest
    stand-in for unseen input without spending the test set.

    Args:
        texts: Training sentences.
        labels: Matching intent labels.
        best_params: Winning parameters from the grid search.
        splitter: The same splitter used for the search.

    Returns:
        A tuple ``(threshold_table, confidences, correct_mask)``.
    """
    pipeline = build_pipeline_iter1(
        C=best_params["clf__C"],
        char_ngram_range=best_params["feat__char__ngram_range"],
    )
    probabilities = cross_val_predict(
        pipeline, texts, labels, cv=splitter, method="predict_proba"
    )

    # cross_val_predict orders columns by the sorted class labels.
    classes = np.unique(np.asarray(labels))
    predicted = classes[probabilities.argmax(axis=1)]
    confidences = probabilities.max(axis=1)
    correct = predicted == np.asarray(labels)

    return analyse_thresholds(labels, predicted, confidences), confidences, correct


def main() -> None:
    """Run the grid search, save the model and report the threshold analysis."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("PeakCast - iteration 1: grid search on the training set")
    print("=" * 58)

    texts, labels = load_dataset(TRAIN_CSV)
    print(f"Training sentences : {len(texts)}")

    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    n_combinations = len(PARAM_GRID["clf__C"]) * len(
        PARAM_GRID["feat__char__ngram_range"]
    )
    print(f"Grid combinations  : {n_combinations}"
          f"  ({n_combinations * N_FOLDS} fits)\n")

    search = run_grid_search(texts, labels, splitter)
    table = save_grid_results(search)

    print("--- grid search results (best first) ---")
    print(table.to_string(index=False))

    best_params = search.best_params_
    print(f"\nBest parameters   : C={best_params['clf__C']}, "
          f"char_ngram_range={best_params['feat__char__ngram_range']}")
    print(f"Best CV accuracy  : {search.best_score_:.4f}")
    print(f"Grid results      -> {GRID_SEARCH_CSV.name}")

    # Save the winning estimator. GridSearchCV already refit it on all
    # training data, but training through the shared helper keeps a single
    # code path for how models are produced and stored.
    pipeline = train_and_save(
        ITER1_MODEL_PATH,
        build_pipeline_iter1(
            C=best_params["clf__C"],
            char_ngram_range=best_params["feat__char__ngram_range"],
        ),
    )
    n_features = len(pipeline.named_steps["feat"].get_feature_names_out())
    print(f"Features          : {n_features}")
    print(f"Model             -> {ITER1_MODEL_PATH.name}")

    print("\n--- confidence threshold, from out-of-fold CV predictions ---")
    thresholds, confidences, correct = derive_threshold(
        texts, labels, best_params, splitter
    )
    print(f"  out-of-fold accuracy : {correct.mean():.4f}")
    print(f"  mean p correct       : {confidences[correct].mean():.3f}")
    print(f"  mean p wrong         : {confidences[~correct].mean():.3f}")
    print()
    print(f"  {'thresh':>7} {'errors caught':>14} {'answers lost':>13}"
          f" {'catch %':>9} {'loss %':>8}")
    for entry in thresholds:
        print(f"  {entry['threshold']:>7.2f} {entry['caught_errors']:>14}"
              f" {entry['lost_answers']:>13}"
              f" {entry['error_catch_rate']:>8.1%}"
              f" {entry['correct_answer_loss_rate']:>8.1%}")

    # Persist the cross-validation summary so the comparison chart can read it
    # without retraining anything.
    cv_summary = {
        "model": "iter1: word 1-2 grams + char_wb n-grams + LogisticRegression",
        "best_params": {
            "C": best_params["clf__C"],
            "char_ngram_range": list(best_params["feat__char__ngram_range"]),
        },
        "cv_mean_accuracy": round(float(search.best_score_), 4),
        "cv_std_accuracy": round(
            float(search.cv_results_["std_test_score"][search.best_index_]), 4
        ),
        "oof_accuracy": round(float(correct.mean()), 4),
        "oof_mean_confidence_correct": round(float(confidences[correct].mean()), 4),
        "oof_mean_confidence_wrong": round(float(confidences[~correct].mean()), 4),
        "n_features": int(n_features),
        "threshold_analysis": thresholds,
    }
    CV_JSON.write_text(
        json.dumps(cv_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCV summary        -> {CV_JSON.name}")
    print("\nNext: python scripts/evaluate_intent.py --variant iter1")


if __name__ == "__main__":
    main()
