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
Deliberately kept simple so that later variants (character n-grams, and optionally
sentence embeddings) have an honest reference point to improve on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# --- Project paths ---------------------------------------------------------
# Resolved relative to this file, so the scripts work no matter which
# directory they are called from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

TRAIN_CSV: Path = DATA_DIR / "intents_train.csv"
TEST_CSV: Path = DATA_DIR / "intents_test.csv"

# Baseline model (step 3). Kept as a frozen reference point so that every later
# iteration can be compared against the same numbers.
MODEL_PATH: Path = MODELS_DIR / "intent_model.joblib"

# Iteration 1 model (step 4): word + character n-grams, hyper-parameters chosen
# by grid search on the training set.
ITER1_MODEL_PATH: Path = MODELS_DIR / "intent_model_iter1.joblib"

# The model the bot itself should use. Pointing at a named constant means the
# dialogue manager does not have to know which iteration is currently best.
ACTIVE_MODEL_PATH: Path = ITER1_MODEL_PATH

# --- Hyper-parameters of the active model ---------------------------------
# The winners of the step 4 grid search (see reports/cv_iter1.json). They are
# recorded here as constants for one reason: ``ensure_model`` has to be able to
# rebuild the model on a fresh checkout, and re-running the 75-fit grid search
# to rediscover numbers that are already known would make the bot's first
# start needlessly slow. The grid search remains the *source* of these values -
# scripts/optimize_intent.py still derives them and would reveal a change.
ITER1_C: float = 10.0
ITER1_CHAR_NGRAM_RANGE: tuple[int, int] = (2, 4)

# --- Reproducibility -------------------------------------------------------
# One seed for everything that can shuffle or iterate randomly, so every run
# produces identical numbers. Required for the portfolio submission.
RANDOM_STATE: int = 42

# --- Fallback threshold ----------------------------------------------------
# If the predicted intent's probability stays below this value, the dialogue
# manager should answer "I did not understand that" and ask back instead of
# guessing.
#
# Value for the ITERATION-1 model (step 4), re-derived from out-of-fold
# cross-validation predictions of that model - never from the test set, because
# a threshold is a tuned parameter and tuning it on test data would make the
# reported test score optimistic.
#
# Chosen at the knee of the trade-off curve (see reports/cv_iter1.json):
#
#   threshold   errors caught   correct answers lost
#       0.30        20.6 %              4.7 %
#       0.35        42.9 %              7.8 %   <- chosen
#       0.40        58.7 %             15.6 %
#       0.45        77.8 %             19.8 %
#
# 0.35 is the last point where the trade is favourable: going to 0.40 catches
# 10 more errors but sacrifices 20 more correct answers, and 0.45 would make the
# bot ask back on roughly a third of all messages, which is unusable for someone
# standing at a trailhead.
#
# For reference, the step 3 baseline could not support any threshold at all: its
# most confident correct prediction reached only p = 0.35, so catching 85 % of
# errors there cost 42 % of the correct answers.
CONFIDENCE_THRESHOLD: float = 0.35

# Threshold values examined by ``analyse_thresholds``. Wide enough to cover
# both the compressed baseline probabilities and the better-separated ones of
# the iteration-1 model.
CANDIDATE_THRESHOLDS: tuple[float, ...] = (
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70,
)

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


def build_pipeline_iter1(
    random_state: int = RANDOM_STATE,
    C: float = 1.0,
    char_ngram_range: tuple[int, int] = (3, 5),
) -> Pipeline:
    """Create the iteration-1 pipeline: word **and** character n-grams.

    The baseline sees only whole words, so a misspelled sentence such as
    "will it be sunnyy at the Katschberg" shares no feature at all with the
    training data and the model has nothing to go on. Character n-grams fix
    exactly that: ``sunny`` and ``sunnyy`` still share the sub-strings ``sunn``,
    ``unny`` and so on.

    ``char_wb`` is used rather than plain ``char`` because it builds n-grams
    inside word boundaries, so it never creates a feature that spans two
    unrelated words.

    The two vectorizers run in parallel through a ``FeatureUnion`` and their
    outputs are concatenated into one feature matrix.

    Args:
        random_state: Seed for the classifier.
        C: Inverse regularisation strength. Larger values regularise less.
            The default is scikit-learn's, so a grid search decides the value.
        char_ngram_range: Length range of the character n-grams.

    Returns:
        An untrained scikit-learn ``Pipeline``.
    """
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        lowercase=True,
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )

    features = FeatureUnion(
        [("word", word_vectorizer), ("char", char_vectorizer)]
    )

    classifier = LogisticRegression(
        max_iter=2000,  # more features than the baseline, so allow more steps
        C=C,
        random_state=random_state,
    )

    return Pipeline([("feat", features), ("clf", classifier)])


def analyse_thresholds(
    true_labels: list[str] | np.ndarray,
    predicted: list[str] | np.ndarray,
    confidences: np.ndarray,
    candidates: tuple[float, ...] = CANDIDATE_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Evaluate candidate confidence thresholds for the fallback answer.

    A threshold trades two error types against each other:

    * **caught errors** - wrong predictions below the threshold. The bot asks
      back instead of answering wrongly. This is the benefit.
    * **lost answers** - correct predictions below the threshold. The bot asks
      back although it was right. This is the cost, and it is the more annoying
      failure for a user standing at a trailhead.

    Lives here next to ``CONFIDENCE_THRESHOLD`` because it is the function that
    derives that constant. It is applied to *out-of-fold cross-validation*
    predictions rather than test predictions: a threshold is a tuned parameter,
    and tuning it on the test set would make the reported test score optimistic.

    Args:
        true_labels: Gold intents.
        predicted: Predicted intents.
        confidences: Probability of each predicted intent.
        candidates: Threshold values to evaluate.

    Returns:
        One summary dict per candidate threshold.
    """
    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted)
    correct = true_array == predicted_array

    total_errors = int((~correct).sum())
    total_correct = int(correct.sum())

    summaries: list[dict[str, Any]] = []
    for threshold in candidates:
        below = confidences < threshold
        caught_errors = int((below & ~correct).sum())
        lost_answers = int((below & correct).sum())
        summaries.append(
            {
                "threshold": threshold,
                "caught_errors": caught_errors,
                "lost_answers": lost_answers,
                "remaining_errors": total_errors - caught_errors,
                # Share of the wrong answers that the threshold intercepts.
                "error_catch_rate": round(caught_errors / total_errors, 3)
                if total_errors
                else 0.0,
                # Share of correct answers sacrificed to achieve that.
                "correct_answer_loss_rate": round(lost_answers / total_correct, 3)
                if total_correct
                else 0.0,
            }
        )
    return summaries


def train_and_save(
    model_path: Path = MODEL_PATH, pipeline: Pipeline | None = None
) -> Pipeline:
    """Train the baseline pipeline on the training CSV and persist it.

    Used by ``scripts/train_intent.py`` and intended for the later app startup
    check ("train the model if the file is missing").

    Args:
        model_path: Destination for the serialised pipeline.
        pipeline: Pipeline to train. Defaults to the baseline pipeline; the
            optimisation script passes the grid-searched iteration-1 pipeline.

    Returns:
        The fitted pipeline.
    """
    import joblib  # imported here to keep the module importable without joblib

    texts, labels = load_dataset(TRAIN_CSV)
    if pipeline is None:
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


def ensure_model(
    model_path: Path = ACTIVE_MODEL_PATH, verbose: bool = True
) -> Pipeline:
    """Load the model, training it first if the file is not there.

    This settles the to-do written down in step 1: ``models/*.joblib`` is
    git-ignored, because it is a build artefact that the training script
    regenerates in seconds. The consequence is that a fresh checkout - or the
    submitted ZIP - has no model file, so the bot has to be able to produce one
    itself rather than failing with "run this script first". Training takes
    about a second, which is cheap enough to do on first start.

    Args:
        model_path: Which model to load. The iteration-1 path is rebuilt with
            the grid-searched hyper-parameters; any other path gets the
            baseline pipeline.
        verbose: Print a short line while training, so a user watching the
            terminal knows why the first start pauses.

    Returns:
        The fitted pipeline.
    """
    if model_path.exists():
        return load_model(model_path)

    if verbose:
        print(f"No intent model at {model_path.name} - training it now ...")

    if model_path == ITER1_MODEL_PATH:
        pipeline = build_pipeline_iter1(
            C=ITER1_C, char_ngram_range=ITER1_CHAR_NGRAM_RANGE
        )
    else:
        pipeline = build_pipeline()

    trained = train_and_save(model_path=model_path, pipeline=pipeline)

    if verbose:
        print(f"  trained and saved to {model_path.name}")
    return trained


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
