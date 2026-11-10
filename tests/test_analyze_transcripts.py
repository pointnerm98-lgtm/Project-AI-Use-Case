"""Tests for the user-test summary script.

The script turns a folder of transcripts into the numbers that go on a slide,
which makes a wrong count worse than a crash: a crash is noticed, a quietly
wrong "80 % positive" is not. So the counts are checked against a fixture
whose expected values were worked out by hand.

The three canned replies (ask-back, help, out-of-scope) are detected by exact
text. That is deliberate - the script imports them from ``dialogue.py`` rather
than retyping them, so rewording a reply cannot silently stop it being
counted - and there is a test for exactly that coupling.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from peakcast.dialogue import ASK_FOR_PLACE, HELP_REPLY, OUT_OF_SCOPE_REPLY
from peakcast.transcript import (
    FEEDBACK_DOWN,
    FEEDBACK_UP,
    ROLE_BOT,
    ROLE_USER,
    TurnRecord,
    build_transcript,
)

from analyze_transcripts import analyse, load_transcripts

EXPORTED_AT = datetime(2026, 11, 10, 14, 0, 0)


def exchange(
    index: int,
    question: str,
    answer: str,
    **answer_fields,
) -> list[TurnRecord]:
    """Build a question and its answer as two records.

    Args:
        index: Index of the question; the answer takes the next one.
        question: What the user asked.
        answer: What the bot replied.
        **answer_fields: Fields to set on the answer record.

    Returns:
        The two records.
    """
    moment = EXPORTED_AT + timedelta(seconds=index * 10)
    return [
        TurnRecord(
            index=index,
            time=moment.strftime("%H:%M:%S"),
            role=ROLE_USER,
            text=question,
        ),
        TurnRecord(
            index=index + 1,
            time=(moment + timedelta(seconds=1)).strftime("%H:%M:%S"),
            role=ROLE_BOT,
            text=answer,
            **answer_fields,
        ),
    ]


@pytest.fixture
def transcript_folder(tmp_path: Path) -> Path:
    """Write two transcripts whose expected counts are known by hand.

    Contents, by design:

    * 2 conversations, 6 exchanges, so 6 user turns and 6 bot turns;
    * intents: ask_wind twice, ask_temperature once, out_of_scope once
      (two answers carry no intent: the guard and the ask-back);
    * 1 answer below the 0.35 threshold (p = 0.31);
    * 1 ask-back, 1 help reply, 1 out-of-scope refusal, 1 guard turn;
    * 1 thumbs up and 2 thumbs down;
    * 1 place that was an assumption;
    * sources: 3 live, 1 offline.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The folder holding the files.
    """
    folder = tmp_path / "user_tests"
    folder.mkdir()

    first = [
        *exchange(
            1,
            "how windy on the Hochkoenig tomorrow",
            "Wind on the Hochkönig tomorrow is around 16 km/h.",
            intent="ask_wind",
            confidence=0.92,
            place="Hochkönig, Salzburg",
            day="2026-11-11",
            source="live",
            feedback=FEEDBACK_UP,
        ),
        *exchange(
            3,
            "and tomorrow?",
            "Hochkönig tomorrow: wind around 16 km/h.",
            intent="ask_wind",
            confidence=0.31,
            place="Hochkönig, Salzburg",
            day="2026-11-11",
            source="live",
        ),
        *exchange(
            5,
            "what is the avalanche risk there",
            "I can't answer avalanche questions - use https://lawinen.at.",
            place="Hochkönig, Salzburg",
            guard=True,
            feedback=FEEDBACK_DOWN,
        ),
    ]
    (folder / "tester1.txt").write_text(
        build_transcript(first, exported_at=EXPORTED_AT, mode="live forecasts"),
        encoding="utf-8",
    )

    second = [
        *exchange(
            1,
            "how cold is it tomorrow",
            ASK_FOR_PLACE,
        ),
        *exchange(
            3,
            "Innsbruk",
            "Tomorrow in Innsbruck it should be between 1 °C and 8 °C.",
            intent="ask_temperature",
            confidence=0.88,
            place="Innsbruck, Tirol",
            day="2026-11-11",
            source="offline",
            fuzzy=True,
        ),
        *exchange(
            5,
            "qwertz",
            HELP_REPLY,
        ),
        *exchange(
            7,
            "can you book me a hotel in Innsbruck",
            OUT_OF_SCOPE_REPLY,
            intent="out_of_scope",
            confidence=0.91,
            feedback=FEEDBACK_DOWN,
        ),
    ]
    (folder / "tester2.txt").write_text(
        build_transcript(second, exported_at=EXPORTED_AT, mode="offline demo data"),
        encoding="utf-8",
    )

    return folder


def test_every_transcript_in_the_folder_is_read(transcript_folder: Path) -> None:
    """All files, and their names kept for the report."""
    transcripts = load_transcripts(transcript_folder)

    assert len(transcripts) == 2
    assert {t.source_name for t in transcripts} == {"tester1.txt", "tester2.txt"}


def test_a_missing_folder_says_what_to_do(tmp_path: Path) -> None:
    """The error must be actionable, not a bare traceback."""
    with pytest.raises(FileNotFoundError, match="--folder"):
        load_transcripts(tmp_path / "does_not_exist")


def test_turn_and_conversation_counts(transcript_folder: Path) -> None:
    """The headline numbers."""
    summary = analyse(load_transcripts(transcript_folder))

    assert summary["conversations"] == 2
    assert summary["user_turns"] == 7
    assert summary["bot_turns"] == 7


def test_intents_are_counted(transcript_folder: Path) -> None:
    """Per-intent counts, which is the main thing the slide needs.

    Only answers that actually carry an intent count: the guard turn, the
    ask-back and the help reply ran no classifier or produced no intent, and
    counting them as some intent would inflate the totals.
    """
    summary = analyse(load_transcripts(transcript_folder))

    assert summary["intents"] == {
        "ask_wind": 2,
        "ask_temperature": 1,
        "out_of_scope": 1,
    }


def test_low_confidence_turns_are_listed_with_their_question(
    transcript_folder: Path,
) -> None:
    """Below the threshold the bot was guessing; the question matters most.

    An answer on its own is not useful in a report - what made the bot unsure
    is the question before it.
    """
    summary = analyse(load_transcripts(transcript_folder))

    assert len(summary["low_confidence"]) == 1
    row = summary["low_confidence"][0]
    assert row["question"] == "and tomorrow?"
    assert row["confidence"] == pytest.approx(0.31)
    assert row["file"] == "tester1.txt"


def test_ask_backs_fallbacks_and_refusals_are_counted(
    transcript_folder: Path,
) -> None:
    """The three canned replies are each counted once."""
    summary = analyse(load_transcripts(transcript_folder))

    assert len(summary["ask_backs"]) == 1
    assert summary["ask_backs"][0]["question"] == "how cold is it tomorrow"

    assert len(summary["fallbacks"]) == 1
    assert summary["fallbacks"][0]["question"] == "qwertz"

    assert len(summary["out_of_scope"]) == 1


def test_canned_replies_are_detected_by_the_real_strings() -> None:
    """The script must follow a reworded reply automatically.

    It imports ASK_FOR_PLACE, HELP_REPLY and OUT_OF_SCOPE_REPLY from
    dialogue.py rather than holding copies, so changing the wording there
    cannot silently stop those turns being counted. This test fails if someone
    replaces the import with a literal.
    """
    import analyze_transcripts

    source = Path(analyze_transcripts.__file__).read_text(encoding="utf-8")
    assert "from peakcast.dialogue import" in source
    assert "ASK_FOR_PLACE" in source
    for literal in (ASK_FOR_PLACE, HELP_REPLY, OUT_OF_SCOPE_REPLY):
        assert literal not in source, "the reply text must not be copied in"


def test_guard_turns_are_counted(transcript_folder: Path) -> None:
    """How often the safety guard fired is a result worth reporting."""
    summary = analyse(load_transcripts(transcript_folder))

    assert len(summary["guard_turns"]) == 1
    assert summary["guard_turns"][0]["question"] == (
        "what is the avalanche risk there"
    )


def test_thumbs_are_counted_and_the_share_is_of_rated_answers(
    transcript_folder: Path,
) -> None:
    """Satisfaction is a share of the *rated* answers, not of all answers.

    The control is optional, so most turns carry no rating. Dividing by every
    answer would quietly understate the result and make the figure meaningless.
    """
    summary = analyse(load_transcripts(transcript_folder))
    feedback = summary["feedback"]

    assert feedback["up"] == 1
    assert feedback["down"] == 2
    assert feedback["rated"] == 3
    assert feedback["satisfaction_of_rated"] == pytest.approx(1 / 3, abs=0.001)


def test_thumbs_down_rows_carry_the_question_and_the_intent(
    transcript_folder: Path,
) -> None:
    """The negatives are the rows to read first, so they carry context."""
    summary = analyse(load_transcripts(transcript_folder))

    questions = {row["question"] for row in summary["thumbs_down"]}
    assert questions == {
        "what is the avalanche risk there",
        "can you book me a hotel in Innsbruck",
    }


def test_assumed_places_are_flagged(transcript_folder: Path) -> None:
    """A fuzzy place match is an assumption worth auditing after a test."""
    summary = analyse(load_transcripts(transcript_folder))

    assert len(summary["assumed_places"]) == 1
    assert summary["assumed_places"][0]["place"] == "Innsbruck, Tirol"


def test_forecast_sources_are_counted(transcript_folder: Path) -> None:
    """How many answers used sample data belongs in the evaluation."""
    summary = analyse(load_transcripts(transcript_folder))

    assert summary["sources"] == {"live": 2, "offline": 1}


def test_an_empty_folder_summarises_to_zeroes(tmp_path: Path) -> None:
    """No transcripts must not raise, and must not invent a satisfaction rate."""
    folder = tmp_path / "empty"
    folder.mkdir()

    summary = analyse(load_transcripts(folder))

    assert summary["conversations"] == 0
    assert summary["bot_turns"] == 0
    assert summary["intents"] == {}
    assert summary["feedback"]["satisfaction_of_rated"] is None
