"""Intent classifier for PeakCast (NLU component, baseline model).

This module holds everything the three scripts in ``scripts/`` have in common:
project paths, dataset loading, the scikit-learn pipeline definition and
save/load/predict helpers.

Keeping it here instead of inside the scripts has two reasons:

1. The pipeline is defined **once**. Training and evaluation can therefore never
   drift apart, which would silently invalidate the reported metrics.
2. The dialogue manager and the Streamlit app will need exactly these functions
   later (including ``train_and_save`` for the "train automatically if the model
   file is missing" requirement noted in DEVLOG.md).

Baseline design: TF-IDF features + logistic regression, as fixed in the concept.
Deliberately kept simple so that later variants (character n-grams, sentence
embeddings) have an honest reference point to improve on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Project paths ---------------------------------------------------------
# Resolved relative to this file, so the scripts work no matter which
# directory they are called from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

TRAIN_CSV: Path = DATA_DIR / "intents_train.csv"
TEST_CSV: Path = DATA_DIR / "intents_test.csv"
MODEL_PATH: Path = MODELS_DIR / "intent_model.joblib"

# --- Reproducibility -------------------------------------------------------
# One seed for everything that can shuffle or iterate randomly, so every run
# produces identical numbers. Required for the portfolio submission.
RANDOM_STATE: int = 42

# --- Fallback threshold ----------------------------------------------------
# If the predicted intent's probability stays below this value, the dialogue
# manager should answer "I did not understand that" and ask back instead of
# guessing.
#
# PROVISIONAL for the baseline model. Derived from out-of-fold cross-validation
# predictions (never from the test set, because a threshold is a tuned
# parameter). The baseline's probabilities are compressed - its most confident
# correct prediction reaches only ~0.35 - so no threshold separates right from
# wrong answers cleanly: 0.25 intercepts 85% of the errors but also rejects 42%
# of the *correct* answers. See DEVLOG.md, step 3.
#
# TO-DO: re-derive this value once the feature set is fixed (adding character
# n-grams raises cross-validated accuracy from 0.68 to 0.79 and roughly doubles
# the confidence of correct predictions).
CONFIDENCE_THRESHOLD: float = 0.25

# The eight intents of the bot, in the order used for reports and plots.
# Fixed explicitly (rather than taken from the data) so that the confusion
# matrix axes stay comparable across runs and model variants.
INTENT_LABELS: list[str] = [
    "ask_temperature",
    "ask_sky",
    "ask_wind",
    "ask_precipitation",
    "ask_overview",
    "greet",
    "goodbye",
    "out_of_scope",
]


def load_dataset(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load a ``text,intent`` CSV file into parallel lists.

    Args:
        csv_path: Path to the CSV file (UTF-8, columns ``text`` and ``intent``).

    Returns:
        A tuple ``(texts, labels)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the expected columns are missing or rows are empty.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path, encoding="utf-8")

    missing = {"text", "intent"} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing column(s): {sorted(missing)}")

    # Guard against empty cells; a blank sentence would train on no features.
    if frame[["text", "intent"]].isna().any().any():
        raise ValueError(f"{csv_path.name} contains empty text or intent cells")

    texts = frame["text"].astype(str).tolist()
    labels = frame["intent"].astype(str).tolist()

    unknown = set(labels) - set(INTENT_LABELS)
    if unknown:
        raise ValueError(f"{csv_path.name} contains unknown intent(s): {sorted(unknown)}")

    return texts, labels


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the baseline intent-classification pipeline.

    Two stages:

    * ``TfidfVectorizer`` turns a sentence into weighted word counts. Word
      uni- and bigrams are used so that short phrases like "how cold" or
      "fresh snow" become single features, which carry far more intent
      information than the individual words.
    * ``LogisticRegression`` assigns one weight per feature and intent. It is
      chosen over e.g. a tree ensemble because it handles the very wide, very
      sparse TF-IDF matrix well, and because its coefficients can be inspected
      to explain *why* a sentence was classified a certain way.

    Args:
        random_state: Seed passed to the classifier for reproducible results.

    Returns:
        An untrained scikit-learn ``Pipeline``.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,        # "Rain" and "rain" are the same feature
        ngram_range=(1, 2),    # single words plus two-word phrases
        sublinear_tf=True,     # dampen repeated words; sentences are short
        min_df=1,              # keep rare words: the dataset is small on purpose
        strip_accents="unicode",  # "Sölden" and "Soelden" collapse to one feature
    )

    classifier = LogisticRegression(
        max_iter=1000,            # enough iterations to converge reliably
        random_state=random_state,
        # No class_weight needed: the dataset is balanced at 40 sentences per intent.
    )

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_and_save(model_path: Path = MODEL_PATH) -> Pipeline:
    """Train the baseline pipeline on the training CSV and persist it.

    Used by ``scripts/train_intent.py`` and intended for the later app startup
    check ("train the model if the file is missing").

    Args:
        model_path: Destination for the serialised pipeline.

    Returns:
        The fitted pipeline.
    """
    import joblib  # imported here to keep the module importable without joblib

    texts, labels = load_dataset(TRAIN_CSV)
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load a previously trained pipeline from disk.

    Args:
        model_path: Path to the ``.joblib`` file.

    Returns:
        The fitted pipeline.

    Raises:
        FileNotFoundError: If no model file exists, with a hint on how to create it.
    """
    import joblib

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}.\n"
            "Run it first:  python scripts/train_intent.py"
        )
    return joblib.load(model_path)


def predict_intent(pipeline: Pipeline, text: str) -> tuple[str, float]:
    """Predict the intent of a single sentence with its confidence.

    Args:
        pipeline: A fitted pipeline.
        text: The user message.

    Returns:
        A tuple ``(intent, probability)`` where ``probability`` is the model's
        confidence in the returned intent, between 0.0 and 1.0.
    """
    probabilities = pipeline.predict_proba([text])[0]
    best_index = int(probabilities.argmax())
    return str(pipeline.classes_[best_index]), float(probabilities[best_index])
