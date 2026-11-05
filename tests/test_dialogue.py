"""Multi-turn conversation tests for the dialogue manager.

These are the tests that could not be written before this step. Every earlier
suite checks one message at a time; the behaviour that makes PeakCast a bot -
remembering a place, continuing a question, asking back and resuming - only
exists across turns, so it can only be tested across turns.

Every test runs with ``allow_live=False`` and a fixed ``today``, so no network
is touched and "tomorrow" is the same date on every machine on every day.
"""

from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.dialogue import (
    ASK_FOR_PLACE,
    CLOSING_AFTER_LIMIT,
    FAREWELL,
    GREETING,
    HELP_REPLY,
    MAX_ANSWERS,
    OUT_OF_SCOPE_REPLY,
    Bot,
    DialogueState,
    is_bare_slot_followup,
    start_conversation,
)
from peakcast.responses import OFFLINE_NOTE, OFFLINE_NOTE_BRIEF
from peakcast.slots import extract_slots
from peakcast.weather import SOURCE_OFFLINE

# A fixed Thursday, so "tomorrow" is always 2026-11-06 and "on Saturday" is
# always two days out.
TODAY = date(2026, 11, 5)


@pytest.fixture(scope="module")
def bot() -> Bot:
    """Build the bot once for the whole module.

    Module-scoped because loading (or, on a fresh checkout, training) the
    intent model is by far the slowest thing in a turn, and nothing in these
    tests mutates the bot - the conversation state is a separate object.
    """
    return Bot(
        allow_live=False,
        rng=random.Random(1234),
        today=TODAY,
        verbose_startup=False,
    )


@pytest.fixture
def state() -> DialogueState:
    """Give each test a fresh conversation."""
    return start_conversation()


def conversation(bot: Bot, messages: list[str]) -> list[str]:
    """Run a whole conversation and return the bot's replies.

    Args:
        bot: The bot.
        messages: What the user says, in order.

    Returns:
        One reply per message.
    """
    state = start_conversation()
    return [bot.respond(state, message).text for message in messages]


# --- Follow-ups: the point of the step ------------------------------------


def test_followup_with_a_new_day_keeps_place_and_intent(
    bot: Bot, state: DialogueState
) -> None:
    """"And tomorrow?" must continue the previous question.

    The message contains no weather word at all, so the classifier cannot
    carry this on its own - it returns ask_wind at p = 0.31, below the 0.35
    threshold. The answer comes from the remembered intent plus the new day.
    """
    first = bot.respond(state, "how windy is it on the Hochkoenig today")
    assert first.intent == "ask_wind"
    assert first.day is not None and first.day.offset_days == 0

    second = bot.respond(state, "and tomorrow?")

    assert second.intent == "ask_wind", "the wind question must carry over"
    assert second.place is not None and second.place.shown == "Hochkönig"
    assert second.day is not None and second.day.target_date == date(2026, 11, 6)
    assert "tomorrow" in second.text.lower()


def test_followup_with_a_new_place_keeps_intent_and_day(
    bot: Bot, state: DialogueState
) -> None:
    """"What about Kaprun?" must keep the question and the day.

    This one is harder than it looks: the classifier calls it ``out_of_scope``
    with p = 0.88, because it contains no weather word. Only the bare-slot
    test tells it apart from a real out-of-scope request.
    """
    bot.respond(state, "how windy is it on the Hochkoenig tomorrow")
    second = bot.respond(state, "what about Kaprun?")

    assert second.intent == "ask_wind"
    assert second.place is not None and second.place.shown == "Kaprun"
    assert second.day is not None and second.day.target_date == date(2026, 11, 6)
    assert OUT_OF_SCOPE_REPLY not in second.text


def test_part_of_day_followup_stays_on_the_remembered_day(
    bot: Bot, state: DialogueState
) -> None:
    """"And in the afternoon?" must mean the day under discussion.

    Resolving the bare part of day to *today* would silently answer about the
    wrong day halfway through a conversation about tomorrow. The date and the
    offset must stay in step, because the live path keys off the date and the
    offline dataset keys off the offset.
    """
    bot.respond(state, "how windy is it on the Hochkoenig tomorrow")
    second = bot.respond(state, "and in the afternoon?")

    assert second.day is not None
    assert second.day.target_date == date(2026, 11, 6)
    assert second.day.part_of_day == "afternoon"
    assert second.day.offset_days == 1, "offset must still be relative to today"
    assert "tomorrow" in second.text.lower()


def test_explicit_today_overrides_the_remembered_day(
    bot: Bot, state: DialogueState
) -> None:
    """Inheriting the day must not override a day the user actually named."""
    bot.respond(state, "how windy is it on the Hochkoenig tomorrow")
    second = bot.respond(state, "and what about today")

    assert second.day is not None
    assert second.day.target_date == TODAY


def test_a_new_question_without_a_day_inherits_the_day(
    bot: Bot, state: DialogueState
) -> None:
    """The day under discussion persists until the user changes it.

    After asking about tomorrow, "will it snow there" means tomorrow. The
    answer always names the day it used, so an inheritance the user did not
    want is visible rather than silent.
    """
    bot.respond(state, "how windy is it on the Hochkoenig tomorrow")
    second = bot.respond(state, "will it snow there")

    assert second.intent == "ask_precipitation"
    assert second.day is not None and second.day.target_date == date(2026, 11, 6)


def test_place_is_remembered_for_a_question_that_names_none(
    bot: Bot, state: DialogueState
) -> None:
    """A weather question with no place uses the remembered one."""
    bot.respond(state, "temperature in Lienz tomorrow")
    second = bot.respond(state, "and is it windy")

    assert second.place is not None and second.place.shown == "Lienz"


# --- The ask-back ---------------------------------------------------------


def test_ask_back_for_a_place_then_resume(bot: Bot, state: DialogueState) -> None:
    """With no place and nothing remembered, the bot must ask - then resume.

    The resumed answer has to keep both the original question and the day the
    user already gave; throwing the day away and answering for today would
    make the user repeat themselves.
    """
    first = bot.respond(state, "how windy is it tomorrow")

    assert first.text == ASK_FOR_PLACE
    assert state.pending == "place"
    assert state.pending_intent == "ask_wind"

    second = bot.respond(state, "Soelden")

    assert state.pending is None, "the pending question must be cleared"
    assert second.intent == "ask_wind", "the original question must resume"
    assert second.place is not None and second.place.shown == "Sölden"
    assert second.day is not None and second.day.target_date == date(2026, 11, 6)


def test_ask_back_is_not_needed_once_a_place_is_known(
    bot: Bot, state: DialogueState
) -> None:
    """The bot must not ask for a place it was already told."""
    bot.respond(state, "temperature in Lienz today")
    second = bot.respond(state, "how windy is it")

    assert second.text != ASK_FOR_PLACE
    assert second.place is not None and second.place.shown == "Lienz"


# --- The safety guard, mid-conversation ----------------------------------


def test_avalanche_question_mid_conversation_is_intercepted(
    bot: Bot, state: DialogueState
) -> None:
    """The guard must fire inside a running conversation, not just at the start.

    It also must not consume or corrupt the conversation: the place under
    discussion is still there afterwards, so the next weather question works.
    """
    bot.respond(state, "how much snow on the Schmitten tomorrow")

    guarded = bot.respond(state, "and what is the avalanche risk there")

    assert guarded.guard_triggered is True
    assert guarded.intent is None, "the classifier must not have been consulted"
    assert "lawinen.at" in guarded.text
    assert "cm" not in guarded.text, "a referral must carry no forecast values"

    # The conversation survives the interruption.
    after = bot.respond(state, "and the wind")
    assert after.place is not None and after.place.shown == "Schmittenhöhe"


def test_avalanche_referral_uses_the_remembered_place_for_the_region(
    bot: Bot, state: DialogueState
) -> None:
    """A Tyrolean place under discussion must add the Tyrol bulletin.

    avalanche.report covers only Tyrol, South Tyrol and Trentino, so it is
    offered only when the place is in Tyrol - and the place can come from the
    conversation rather than from this message.
    """
    bot.respond(state, "how windy on the Wildspitze tomorrow")
    guarded = bot.respond(state, "what about the avalanche risk")

    assert guarded.guard_triggered is True
    assert "avalanche.report" in guarded.text


def test_avalanche_referral_outside_tyrol_omits_the_euregio_bulletin(
    bot: Bot, state: DialogueState
) -> None:
    """Outside Tyrol the user must get only the nationwide portal."""
    bot.respond(state, "how much snow on the Hochschwab tomorrow")
    guarded = bot.respond(state, "avalanche danger there?")

    assert guarded.guard_triggered is True
    assert "lawinen.at" in guarded.text
    assert "avalanche.report" not in guarded.text


# --- Low confidence, out of scope, social --------------------------------


def test_low_confidence_with_nothing_to_go_on_sends_help(
    bot: Bot, state: DialogueState
) -> None:
    """Gibberish must get one concrete example, not a guessed forecast."""
    turn = bot.respond(state, "hmm")

    assert turn.text == HELP_REPLY
    assert turn.place is None


def test_low_confidence_at_the_start_does_not_invent_a_place(
    bot: Bot, state: DialogueState
) -> None:
    """With no conversation to lean on, a vague message must not be answered."""
    turn = bot.respond(state, "qwertz")

    assert turn.text == HELP_REPLY


def test_genuine_out_of_scope_is_declined_even_with_a_place(
    bot: Bot, state: DialogueState
) -> None:
    """"Book me a hotel in Innsbruck" must be declined, not answered.

    The counterpart to the "what about Kaprun?" test: both are confident
    out_of_scope with a place in them, and the bare-slot test is the only
    thing separating them.
    """
    bot.respond(state, "how windy is it on the Hochkoenig tomorrow")
    turn = bot.respond(state, "can you book me a hotel in Innsbruck")

    assert turn.text == OUT_OF_SCOPE_REPLY


@pytest.mark.parametrize(
    "message",
    [
        "recommend a good restaurant in Salzburg",
        "how much does a lift ticket cost in Ischgl",
        "is the cable car open on the Kitzsteinhorn",
        "tell me a joke",
    ],
)
def test_out_of_scope_at_the_start_is_declined(
    bot: Bot, state: DialogueState, message: str
) -> None:
    """Out of scope needs no conversation to be recognised.

    The first three all name a real place, so they also check that a place in
    the message is not by itself enough to turn a refusal into an answer.
    """
    turn = bot.respond(state, message)

    assert turn.text == OUT_OF_SCOPE_REPLY


def test_known_limitation_quantity_questions_leak_to_a_forecast(
    bot: Bot, state: DialogueState
) -> None:
    """Pin a classifier error the dialogue manager cannot repair.

    "how many people live in Linz" is labelled ``ask_temperature`` at p = 0.48
    - above the 0.35 threshold - and "how high is the Dachstein" at p = 0.62.
    Both were already recorded as errors in step 3: the model latches onto the
    quantity phrasing "how many" / "how high".

    The dialogue manager has no way to second-guess a confident classifier
    here, and inventing a rule for these two sentences would be fitting the
    test set. The honest fix is more `out_of_scope` training data in that
    phrasing style, collected from real conversations rather than from the
    test set - which is the open point recorded in step 5.

    This test asserts the *current, wrong* behaviour on purpose, so that a
    later data iteration which fixes it makes this test fail and prompts an
    update rather than passing unnoticed.
    """
    turn = bot.respond(state, "how many people live in Linz")

    assert turn.text != OUT_OF_SCOPE_REPLY, (
        "if this now passes as out_of_scope, the classifier improved - "
        "delete this test and the step 3 note that goes with it"
    )
    assert turn.intent == "ask_temperature"


def test_greeting_is_answered_with_an_example(
    bot: Bot, state: DialogueState
) -> None:
    """A greeting must say what the bot can do."""
    turn = bot.respond(state, "hello")

    assert turn.text == GREETING
    assert state.finished is False


def test_goodbye_ends_the_conversation(bot: Bot, state: DialogueState) -> None:
    """A goodbye must close the session."""
    bot.respond(state, "temperature in Lienz today")
    turn = bot.respond(state, "thanks bye")

    assert turn.text == FAREWELL
    assert turn.finished is True
    assert state.finished is True


def test_a_greeting_with_a_question_is_answered_not_greeted(
    bot: Bot, state: DialogueState
) -> None:
    """"Hi, weather in Lienz?" must get the forecast, not just a hello."""
    turn = bot.respond(state, "hello whats the weather in Lienz today")

    assert turn.text != GREETING
    assert turn.place is not None and turn.place.shown == "Lienz"


# --- The conversation limit ----------------------------------------------


def test_conversation_ends_after_the_answer_limit(bot: Bot) -> None:
    """The tenth answer must carry the closing and end the session.

    The closing is appended to the tenth answer rather than sent as an
    eleventh message, so the user still gets the answer they asked for.
    """
    state = start_conversation()
    replies = []
    for _ in range(MAX_ANSWERS):
        turn = bot.respond(state, "temperature in Lienz tomorrow")
        replies.append(turn)

    assert all(not t.finished for t in replies[:-1]), "must not end early"
    assert replies[-1].finished is True
    assert CLOSING_AFTER_LIMIT in replies[-1].text
    # The answer itself is still there, not replaced by the closing.
    assert "°C" in replies[-1].text
    assert state.answers_given == MAX_ANSWERS


def test_every_reply_counts_towards_the_limit(bot: Bot) -> None:
    """Help replies, greetings and refusals must all count.

    Otherwise a conversation of nothing but unanswerable messages would never
    end, which is the case the limit most needs to cover.
    """
    state = start_conversation()
    for _ in range(MAX_ANSWERS):
        turn = bot.respond(state, "hmm")

    assert state.answers_given == MAX_ANSWERS
    assert turn.finished is True


def test_a_finished_conversation_stays_finished(
    bot: Bot, state: DialogueState
) -> None:
    """After a goodbye, further messages must not be answered."""
    bot.respond(state, "bye")
    turn = bot.respond(state, "temperature in Lienz tomorrow")

    assert turn.finished is True
    assert turn.place is None


# --- The offline note -----------------------------------------------------


def test_offline_answers_disclose_the_sample_data(
    bot: Bot, state: DialogueState
) -> None:
    """The first offline answer must carry the full disclosure.

    A fictional forecast may never read as a real one, which is the whole
    reason WeatherReport carries its source.
    """
    turn = bot.respond(state, "temperature in Lienz tomorrow")

    assert turn.source == SOURCE_OFFLINE
    assert OFFLINE_NOTE in turn.text


def test_later_offline_answers_use_the_short_note(
    bot: Bot, state: DialogueState
) -> None:
    """Repeats must stay disclosed, but briefly.

    Ten identical paragraphs would turn the disclosure into wallpaper a reader
    skips, which defeats its purpose - so the full note is used once and a
    short tag afterwards.
    """
    first = bot.respond(state, "temperature in Lienz tomorrow")
    second = bot.respond(state, "and the wind")

    assert OFFLINE_NOTE in first.text
    assert OFFLINE_NOTE not in second.text
    assert OFFLINE_NOTE_BRIEF in second.text


# --- Beyond the forecast horizon -----------------------------------------


def test_a_day_beyond_the_horizon_is_declined_not_invented(
    bot: Bot, state: DialogueState
) -> None:
    """Nine days out must be refused in words, not answered with demo data.

    The network being down justifies sample data; a question about next week
    has an honest answer, and giving numbers instead would be worse.
    """
    turn = bot.respond(state, "how cold in Lienz next Friday")

    assert "week ahead" in turn.text
    assert "°C" not in turn.text
    # And the wording must not read "about on Friday".
    assert "about on" not in turn.text


# --- The bare-slot test, directly ----------------------------------------


@pytest.mark.parametrize(
    "message, expected",
    [
        # nothing but a slot: these are follow-ups
        ("what about Kaprun?", True),
        ("and tomorrow?", True),
        ("and in the afternoon?", True),
        ("Kaprun", True),
        ("what about the Dachstein", True),
        ("and Lienz?", True),
        # real content left over: these are not
        ("can you book me a hotel in Innsbruck", False),
        ("how do I get to Zell am See by train", False),
        ("how much does a lift ticket cost in Ischgl", False),
        ("is the cable car open on the Kitzsteinhorn", False),
    ],
)
def test_bare_slot_followup_detection(message: str, expected: bool) -> None:
    """The follow-up test must separate a bare slot from a real request.

    Confidence cannot do this: "what about Kaprun?" and "can you book me a
    hotel in Innsbruck" are both confident out_of_scope with a place in them.
    What separates them is that only the first is *nothing but* a slot.
    """
    slots = extract_slots(message, reference=TODAY)
    assert is_bare_slot_followup(message, slots.place, slots.day) is expected


# --- Determinism ----------------------------------------------------------


def test_same_seed_gives_the_same_conversation() -> None:
    """A seeded bot must be reproducible.

    The phrasing variants are chosen at random, which is what keeps the bot
    from sounding like a form letter - but it also means the tests and the
    documented example conversations need a seed to be stable.
    """
    messages = [
        "hello",
        "how windy is it on the Hochkoenig tomorrow",
        "and tomorrow?",
        "thanks bye",
    ]

    def run() -> list[str]:
        return conversation(
            Bot(
                allow_live=False,
                rng=random.Random(99),
                today=TODAY,
                verbose_startup=False,
            ),
            messages,
        )

    assert run() == run()
