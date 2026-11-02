"""Interactive command-line test for the intent classifier.

Run from the project root (after training):

    python scripts/try_intent.py

Type a sentence and the script shows the predicted intent, its probability and
the runner-up intents. Useful for spotting weaknesses by hand that the
aggregated test metrics hide, and for demonstrating the NLU component live.

Leave with ``quit``, ``exit`` or Ctrl+C.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.pipeline import Pipeline

from peakcast.intent_classifier import (
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    load_model,
)

# How many runner-up intents to display besides the winner.
TOP_K: int = 3

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q", ""})


def show_prediction(pipeline: Pipeline, text: str) -> None:
    """Print the ranked intents for one sentence.

    Args:
        pipeline: The fitted pipeline.
        text: The sentence to classify.
    """
    probabilities = pipeline.predict_proba([text])[0]

    # Sort descending by probability and keep the top entries.
    ranking = np.argsort(probabilities)[::-1][:TOP_K]
    best_index = int(ranking[0])
    best_intent = str(pipeline.classes_[best_index])
    best_probability = float(probabilities[best_index])

    print(f"  -> {best_intent}  (p = {best_probability:.3f})")

    # Make the fallback rule visible: this is what the dialogue manager will do.
    if best_probability < CONFIDENCE_THRESHOLD:
        print(f"     below threshold {CONFIDENCE_THRESHOLD:.2f}"
              f" -> bot would answer: \"Sorry, I did not understand that.\"")

    print("     runner-ups:", ", ".join(
        f"{pipeline.classes_[i]} {probabilities[i]:.3f}" for i in ranking[1:]
    ))


def main() -> None:
    """Load the model and run the interactive prediction loop."""
    # Austrian place names contain umlauts; Windows consoles default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        pipeline = load_model(MODEL_PATH)
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    print("PeakCast - interactive intent test")
    print(f"Fallback threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print("Type a sentence, or 'quit' to leave.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C / Ctrl+D should end the loop cleanly, not show a traceback.
            print("\nbye")
            break

        if text.lower() in EXIT_COMMANDS:
            print("bye")
            break

        show_prediction(pipeline, text)
        print()


if __name__ == "__main__":
    main()
