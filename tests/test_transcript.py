"""Tests for the conversation transcript: writing, parsing and summarising.

The transcript is the only thing in this project that has to be both
**readable by a person and parsable by a script**, and those two pull in
opposite directions: a format that is pleasant to read is easy to parse
sloppily. So the writer and the parser are tested together, with a round-trip
test as the anchor - anything ``build_transcript`` writes,
``parse_transcript`` must read back.

The mapping from a live conversation to records is tested here too, because it
is a pure function: ``records_from_history`` takes plain data and returns plain
data, which is why it lives in ``peakcast.transcript`` rather than in
``app.py``. A field going missing from an exported transcript would be
invisible in the app and is caught here.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.dialogue import BotTurn
from peakcast.slots import DaySlot, Place
from peakcast.transcript import (
    FEEDBACK_DOWN,
    FEEDBACK_UP,
    ROLE_BOT,
    ROLE_USER,
    TurnRecord,
    build_transcript,
    parse_transcript,
    records_from_history,
    suggested_filename,
)

EXPORTED_AT = datetime(2026, 11, 10, 14, 23, 5)
TODAY = date(2026, 11, 10)

PLACE = Place(
    name="Hochkönig",
    state="Salzburg",
    lat=47.42,
    lon=13.06,
    elevation=2941,
    population=0,
    feature_code="MT",
    display_name="Hochkönig",
)


def a_day(offset: int = 1, part: str | None = None) -> DaySlot:
    """Build a day slot for the tests.

    Args:
        offset: Days from ``TODAY``.
        part: Part-of-day flag.

    Returns:
        The day slot.
    """
    return DaySlot(
        target_date=TODAY + timedelta(days=offset),
        offset_days=offset,
        part_of_day=part,
    )


def a_history(feedback_free: bool = True) -> list[dict]:
    """Build a small conversation history in the app's shape.

    Four messages: a question, its answer, an avalanche question and the
    guard's referral. That covers both the ordinary case and the one where no
    intent was ever produced.

    Args:
        feedback_free: Unused; present so the helper reads as a builder.

    Returns:
        The history list.
    """
    base = datetime(2026, 11, 10, 14, 21, 0)

    answer = BotTurn(
        text="Wind on the Hochkönig tomorrow is around 16 km/h.",
        intent="ask_wind",
        confidence=0.9237,
        place=PLACE,
        day=a_day(1),
        source="live",
    )
    referral = BotTurn(
        text="I can't answer avalanche questions - use https://lawinen.at.",
        intent=None,
        confidence=0.0,
        place=PLACE,
        guard_triggered=True,
    )

    return [
        {
            "role": "user",
            "text": "how windy on the Hochkoenig tomorrow",
            "turn": None,
            "time": base,
        },
        {
            "role": "assistant",
            "text": answer.text,
            "turn": answer,
            "time": base + timedelta(seconds=1),
        },
        {
            "role": "user",
            "text": "what is the avalanche risk there",
            "turn": None,
            "time": base + timedelta(seconds=20),
        },
        {
            "role": "assistant",
            "text": referral.text,
            "turn": referral,
            "time": base + timedelta(seconds=21),
        },
    ]


# --- The mapping from a live conversation --------------------------------


def test_records_carry_the_nlu_details_of_each_answer() -> None:
    """Everything the report needs per turn must survive the export."""
    records = records_from_history(a_history())

    assert [record.role for record in records] == [
        ROLE_USER, ROLE_BOT, ROLE_USER, ROLE_BOT
    ]
    assert [record.index for record in records] == [1, 2, 3, 4]

    answer = records[1]
    assert answer.intent == "ask_wind"
    assert answer.confidence == pytest.approx(0.9237)
    assert answer.place == "Hochkönig, Salzburg"
    assert answer.day == "2026-11-11"
    assert answer.source == "live"
    assert answer.time == "14:21:01"


def test_user_turns_carry_no_nlu_details() -> None:
    """A user message has no intent; the fields must stay empty."""
    records = records_from_history(a_history())
    question = records[0]

    assert question.intent is None
    assert question.confidence is None
    assert question.place is None
    assert question.source is None


def test_a_guarded_turn_records_the_guard_and_no_confidence() -> None:
    """The safety guard never runs the classifier, so there is no confidence.

    Writing ``confidence: 0.00`` there would be worse than writing nothing: it
    reads as "the model was certain it was nothing", when in fact the model
    was never asked.
    """
    records = records_from_history(a_history())
    referral = records[3]

    assert referral.guard is True
    assert referral.intent is None
    assert referral.confidence is None


def test_thumbs_are_attached_to_the_right_turn() -> None:
    """A rating belongs to the answer it was given on, not the next one.

    The keys are history indices, so an off-by-one here would silently
    attribute every rating to the wrong answer - and the mistake would only
    show up as puzzling numbers in the final report.
    """
    records = records_from_history(a_history(), feedback={1: 1, 3: 0})

    assert records[1].feedback == FEEDBACK_UP
    assert records[3].feedback == FEEDBACK_DOWN
    assert records[0].feedback is None
    assert records[2].feedback is None


def test_an_unrated_conversation_carries_no_feedback() -> None:
    """The control is optional, so no rating must mean no field."""
    records = records_from_history(a_history())
    assert all(record.feedback is None for record in records)


# --- Writing -------------------------------------------------------------


def test_transcript_has_a_header_with_mode_and_turn_count() -> None:
    """A reader must be able to tell live data from sample data."""
    text = build_transcript(
        records_from_history(a_history()),
        exported_at=EXPORTED_AT,
        mode="offline demo data",
    )

    assert "PeakCast conversation transcript" in text
    assert "Exported: 2026-11-10 14:23:05" in text
    assert "Mode: offline demo data" in text
    assert "Turns: 4" in text


def test_transcript_says_that_nothing_was_stored() -> None:
    """The file itself must state it, not only the app.

    The file is what the tester actually sends on, so it should be possible to
    check what is being shared without taking anyone's word for it.
    """
    text = build_transcript(
        records_from_history(a_history()), exported_at=EXPORTED_AT, mode="live"
    )

    assert "Nothing was stored on a" in text
    assert "read it, edit it or delete it" in text


def test_only_the_fields_that_apply_are_written() -> None:
    """A greeting must not carry an empty "place:" line.

    Readability is not a side concern for this file: a tester who cannot skim
    it cannot meaningfully consent to sending it.
    """
    greeting = TurnRecord(
        index=1, time="14:21:00", role=ROLE_BOT, text="Hi!", intent="greet",
        confidence=0.99,
    )
    text = build_transcript([greeting], exported_at=EXPORTED_AT, mode="live")

    assert "intent: greet" in text
    assert "place:" not in text
    assert "source:" not in text
    assert "feedback:" not in text


def test_filename_is_timestamped() -> None:
    """Exporting twice must not overwrite the first file."""
    assert suggested_filename(EXPORTED_AT) == (
        "peakcast-conversation-20261110-142305.txt"
    )


# --- Parsing, and the round trip ----------------------------------------


def test_round_trip_preserves_every_field() -> None:
    """What the writer writes, the parser must read back.

    The anchor test for the format. Confidence is compared rounded to two
    decimals, which is what the file deliberately stores: the transcript is
    written for a person to read, and 0.9237 tells a reader nothing that 0.92
    does not.
    """
    original = records_from_history(a_history(), feedback={1: 1, 3: 0})
    text = build_transcript(
        original, exported_at=EXPORTED_AT, mode="offline demo data"
    )

    parsed = parse_transcript(text, source_name="tester1.txt")

    assert parsed.mode == "offline demo data"
    assert parsed.exported_at == "2026-11-10 14:23:05"
    assert parsed.source_name == "tester1.txt"
    assert len(parsed.turns) == len(original)

    for before, after in zip(original, parsed.turns):
        assert after.index == before.index
        assert after.time == before.time
        assert after.role == before.role
        assert after.text == before.text.strip()
        assert after.intent == before.intent
        assert after.place == before.place
        assert after.day == before.day
        assert after.part_of_day == before.part_of_day
        assert after.source == before.source
        assert after.guard == before.guard
        assert after.fuzzy == before.fuzzy
        assert after.feedback == before.feedback
        if before.confidence is None:
            assert after.confidence is None
        else:
            assert after.confidence == pytest.approx(before.confidence, abs=0.005)


def test_multi_line_answers_survive_the_round_trip() -> None:
    """An answer containing a blank line must not split into two turns."""
    record = TurnRecord(
        index=1,
        time="14:21:00",
        role=ROLE_BOT,
        text="First line.\n\nSecond line after a gap.",
        intent="ask_overview",
        confidence=0.8,
    )
    text = build_transcript([record], exported_at=EXPORTED_AT, mode="live")
    parsed = parse_transcript(text)

    assert len(parsed.turns) == 1
    assert "Second line after a gap." in parsed.turns[0].text


def test_parser_skips_an_edited_block_instead_of_giving_up() -> None:
    """Testers are told they may edit the file, so the parser must cope.

    A strict parser would mean one hand-trimmed file costing a whole user
    test. Unknown keys are ignored for the same reason: a transcript from a
    later version of the app should still be readable.
    """
    text = build_transcript(
        records_from_history(a_history()), exported_at=EXPORTED_AT, mode="live"
    )
    damaged = text.replace("    intent: ask_wind", "    something_new: 42")
    damaged = damaged.replace("Exported:", "Exportiert:")

    parsed = parse_transcript(damaged)

    assert len(parsed.turns) == 4, "the turns must still be found"
    assert parsed.turns[1].intent is None, "the removed field is simply absent"
    assert parsed.turns[1].place == "Hochkönig, Salzburg", "the rest survives"


def test_parsing_an_empty_file_gives_an_empty_transcript() -> None:
    """A tester may send a file with nothing in it; that must not raise."""
    parsed = parse_transcript("")
    assert parsed.turns == []


def test_bot_turns_helper_filters_the_user_messages() -> None:
    """The analysis only ever counts the bot's turns."""
    parsed = parse_transcript(
        build_transcript(
            records_from_history(a_history()), exported_at=EXPORTED_AT, mode="live"
        )
    )
    assert len(parsed.bot_turns) == 2
    assert all(turn.role == ROLE_BOT for turn in parsed.bot_turns)
