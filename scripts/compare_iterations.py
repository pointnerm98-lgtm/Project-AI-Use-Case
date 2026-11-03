"""Compare the baseline against iteration 1 and spot-check specific sentences.

Run from the project root (after both models are trained and evaluated):

    python scripts/compare_iterations.py

Produces:

* ``reports/comparison_baseline_vs_iter1.png`` - the before/after chart for the
  presentation,
* a console spot-check of the avalanche safety sentence and of the baseline
  errors that contained words the baseline had never seen.

Reads the metric JSON files rather than retraining, so the numbers shown here
are exactly the ones in the reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from peakcast.intent_classifier import (
    CONFIDENCE_THRESHOLD,
    ITER1_MODEL_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    load_model,
    predict_intent,
)

COMPARISON_PNG: Path = REPORTS_DIR / "comparison_baseline_vs_iter1.png"

# --- Colours -----------------------------------------------------------------
# Categorical slots 1 and 2 of the reference palette, validated for colour-vision
# deficiency against the light chart surface (worst-pair CVD delta E 24.7,
# normal vision 33.6 - both well clear of the required floors).
COLOR_BASELINE: str = "#2a78d6"   # blue
COLOR_ITER1: str = "#eb6834"      # orange
SURFACE: str = "#fcfcfb"
INK_PRIMARY: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"

# Sentences that must be re-checked after every model change.
# The avalanche case is a safety matter, not an accuracy matter.
AVALANCHE_PROBE: str = "avalanche risk after the fresh snow"


def load_json(path: Path) -> dict[str, Any]:
    """Read a report JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed contents.

    Raises:
        FileNotFoundError: With a hint on which script produces the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the training and evaluation scripts first:\n"
            "  python scripts/train_intent.py\n"
            "  python scripts/evaluate_intent.py\n"
            "  python scripts/optimize_intent.py\n"
            "  python scripts/evaluate_intent.py --variant iter1"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics() -> list[tuple[str, float, float]]:
    """Gather the four headline metrics for both models.

    Returns:
        A list of ``(metric_label, baseline_value, iter1_value)`` tuples.
    """
    cv_baseline = load_json(REPORTS_DIR / "cv_baseline.json")
    cv_iter1 = load_json(REPORTS_DIR / "cv_iter1.json")
    test_baseline = load_json(REPORTS_DIR / "metrics_baseline.json")
    test_iter1 = load_json(REPORTS_DIR / "metrics_improved.json")

    return [
        (
            "Cross-validation\naccuracy (train)",
            cv_baseline["cv_mean_accuracy"],
            cv_iter1["cv_mean_accuracy"],
        ),
        ("Test\naccuracy", test_baseline["accuracy"], test_iter1["accuracy"]),
        ("Macro F1\n(test)", test_baseline["macro_f1"], test_iter1["macro_f1"]),
        (
            "Mean confidence\nof correct answers",
            test_baseline["confidence"]["mean_correct"],
            test_iter1["confidence"]["mean_correct"],
        ),
    ]


def plot_comparison(metrics: list[tuple[str, float, float]]) -> None:
    """Draw the before/after grouped bar chart.

    All four metrics share the same 0-1 scale, so one axis is enough; a second
    y-axis would make the two groups visually incomparable.

    Args:
        metrics: Output of :func:`collect_metrics`.
    """
    labels = [m[0] for m in metrics]
    baseline_values = [m[1] for m in metrics]
    iter1_values = [m[2] for m in metrics]

    positions = range(len(metrics))
    bar_width = 0.36
    gap = 0.02  # small surface gap so the paired bars stay visually separate

    figure, axes = plt.subplots(figsize=(10, 5.6))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    baseline_bars = axes.bar(
        [p - bar_width / 2 - gap / 2 for p in positions],
        baseline_values,
        bar_width,
        label="Baseline (step 3): word TF-IDF",
        color=COLOR_BASELINE,
    )
    iter1_bars = axes.bar(
        [p + bar_width / 2 + gap / 2 for p in positions],
        iter1_values,
        bar_width,
        label="Iteration 1 (step 4): word + char n-grams",
        color=COLOR_ITER1,
    )

    # Direct value labels: with only eight bars every value can be shown, so the
    # reader never has to measure against the axis.
    for bars in (baseline_bars, iter1_bars):
        for bar in bars:
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK_PRIMARY,  # text keeps ink colour, not the series colour
            )

    # The change is the actual message of the chart, so state it in words.
    # Placed above each pair: below the axis it would collide with the
    # two-line metric labels.
    for position, (_, before, after) in zip(positions, metrics):
        delta = (after - before) * 100
        axes.text(
            position,
            max(before, after) + 0.075,
            f"{delta:+.1f} pts",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=INK_PRIMARY,
        )

    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=11, color=INK_SECONDARY)
    axes.set_ylim(0, 1.0)
    axes.set_ylabel("score", fontsize=11, color=INK_SECONDARY)
    axes.tick_params(axis="y", colors=INK_SECONDARY)
    axes.tick_params(axis="x", length=0)

    axes.set_title(
        "PeakCast intent classifier: baseline vs. iteration 1",
        fontsize=14,
        color=INK_PRIMARY,
        pad=14,
    )

    # Recessive grid and axes: the bars carry the information, not the frame.
    axes.yaxis.grid(True, color="#e4e3df", linewidth=1)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color("#d9d8d3")

    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor=INK_SECONDARY,
    )

    figure.tight_layout()
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(COMPARISON_PNG, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def find_unseen_word_errors(baseline_pipeline: Any) -> pd.DataFrame:
    """Find baseline errors containing words absent from its vocabulary.

    This checks the actual mechanism rather than a hand-written list of typos:
    the baseline uses word features only, so a token it never saw in training
    contributes nothing at all to the prediction.

    Args:
        baseline_pipeline: The fitted baseline pipeline.

    Returns:
        The misclassified rows that contain at least one unseen word, with an
        extra ``unseen_words`` column.
    """
    vectorizer = baseline_pipeline.named_steps["tfidf"]
    vocabulary = vectorizer.vocabulary_

    # Use the vectorizer's own preprocessor and tokenizer rather than a naive
    # split, so the comparison matches what the model actually saw: lower-cased,
    # accent-stripped, and without the one-character tokens that the default
    # token pattern discards anyway.
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()

    errors = pd.read_csv(
        REPORTS_DIR / "misclassified_baseline.csv", encoding="utf-8"
    )

    unseen_per_row = [
        [token for token in tokenize(preprocess(str(text)))
         if token not in vocabulary]
        for text in errors["text"]
    ]
    errors["unseen_words"] = [" ".join(words) for words in unseen_per_row]
    return errors[[bool(words) for words in unseen_per_row]]


def spot_check(sentences: list[str]) -> None:
    """Print baseline and iteration-1 predictions side by side.

    Args:
        sentences: The sentences to classify with both models.
    """
    baseline = load_model(MODEL_PATH)
    iter1 = load_model(ITER1_MODEL_PATH)

    print(f"  {'sentence':<46} {'baseline':<28} {'iteration 1':<28}")
    print("  " + "-" * 102)
    for sentence in sentences:
        base_intent, base_probability = predict_intent(baseline, sentence)
        new_intent, new_probability = predict_intent(iter1, sentence)
        # Mark predictions the fallback threshold would reject.
        new_flag = "*" if new_probability < CONFIDENCE_THRESHOLD else " "
        print(
            f"  {sentence[:44]:<46}"
            f" {base_intent + f' {base_probability:.2f}':<28}"
            f" {new_intent + f' {new_probability:.2f}{new_flag}':<28}"
        )


def main() -> None:
    """Build the comparison chart and run the spot-checks."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("PeakCast - baseline vs. iteration 1")
    print("=" * 58)

    metrics = collect_metrics()
    print(f"  {'metric':<34} {'baseline':>10} {'iter1':>10} {'change':>10}")
    for label, before, after in metrics:
        flat_label = label.replace("\n", " ")
        print(f"  {flat_label:<34} {before:>10.4f} {after:>10.4f}"
              f" {(after - before) * 100:>+9.1f}p")

    plot_comparison(metrics)
    print(f"\nChart -> {COMPARISON_PNG.name}")

    baseline_pipeline = load_model(MODEL_PATH)
    unseen_errors = find_unseen_word_errors(baseline_pipeline)

    print(f"\n--- baseline errors caused by words it had never seen "
          f"({len(unseen_errors)} of 18) ---")
    for row in unseen_errors.itertuples(index=False):
        print(f"  \"{row.text}\"  (unseen: {row.unseen_words})")

    print(f"\n--- spot-check: safety probe ---")
    spot_check([AVALANCHE_PROBE])

    print(f"\n--- spot-check: the unseen-word errors ---")
    spot_check(list(unseen_errors["text"]))
    print(f"\n  * = below the fallback threshold of {CONFIDENCE_THRESHOLD:.2f},"
          f" so the bot would ask back instead of answering")


if __name__ == "__main__":
    main()
