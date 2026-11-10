"""Dialogue manager: one turn of conversation, with memory.

This is the component that makes PeakCast a *bot* rather than a pipeline. It
owns the conversation state and the order the other components run in:

    safety guard -> intent classifier -> slot extraction
                 -> context handling -> weather service -> response generator

Two of those steps deserve their reasoning written down.

**Why the guard runs first.** Measured in steps 3 and 4, the intent classifier
labels "avalanche risk after the fresh snow" as ``ask_precipitation`` with
p = 0.62, well above the 0.35 fallback threshold - and the *better* model was
the more confidently wrong one. A guard placed after the classifier would only
ever see messages the model had already mislabelled, which is exactly the set
it cannot be trusted on. So it runs on the raw message, before anything else
looks at it.

**Why context handling is its own step.** A pipeline can answer "how windy is
it on the Hochkönig tomorrow". It cannot answer "and tomorrow?", because that
message contains no weather word, no place, and nothing for a classifier to
work with. Everything that makes a follow-up answerable lives in
``_apply_context`` - the remembered place, the remembered intent, and the
decision to trust slots over a low-confidence intent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from peakcast.intent_classifier import (
    ACTIVE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    ensure_model,
    predict_intent,
)
from peakcast.responses import (
    ASK_OVERVIEW,
    fuzzy_confirmation,
    generate_answer,
)
from peakcast.safety_guard import check_message
from peakcast.slots import (
    MAX_DAY_OFFSET,
    DaySlot,
    Place,
    extract_slots,
    today_in_vienna,
)
from peakcast.weather import ForecastOutOfRangeError, get_forecast

# --- Intents ---------------------------------------------------------------
# The five intents that need a forecast. Kept as a frozenset so "is this a
# weather question" is one membership test rather than a chain of comparisons.
WEATHER_INTENTS: frozenset[str] = frozenset(
    {
        "ask_temperature",
        "ask_sky",
        "ask_wind",
        "ask_precipitation",
        "ask_overview",
    }
)

INTENT_GREET: str = "greet"
INTENT_GOODBYE: str = "goodbye"
INTENT_OUT_OF_SCOPE: str = "out_of_scope"

# --- Pending questions -----------------------------------------------------
# What the bot is waiting for, when it has asked something back. Only one kind
# so far; named rather than a bare boolean so a second kind (which day?) can be
# added without changing the shape of the state.
PENDING_PLACE: str = "place"

# --- Recognising a bare follow-up ------------------------------------------
# A follow-up like "what about Kaprun?" is a problem for the classifier, and
# measurably so: it comes back `out_of_scope` with p = 0.88, because it
# contains no weather word at all. So does "can you book me a hotel in
# Innsbruck" (p = 0.91) - and that one really *is* out of scope. Both are
# confident `out_of_scope` with a place in them, so confidence cannot separate
# them.
#
# What separates them is that the follow-up is **nothing but a slot**: strip
# out the place and day it names and only filler remains. The hotel question
# still has "book" and "hotel" left over. So that is the test, and the filler
# list below is kept deliberately short - every word added to it is a word the
# bot stops treating as content.
FOLLOWUP_FILLER: frozenset[str] = frozenset(
    {
        "and", "what", "about", "how", "whats", "hows", "ok", "okay", "also",
        "then", "so", "the", "a", "an", "in", "at", "on", "for", "of", "to",
        "there", "please", "thanks", "but", "or", "well", "now", "just",
        "me", "tell", "say", "again", "instead", "rather",
    }
)

# --- Conversation limit ----------------------------------------------------
# The concept asks the bot to end after roughly ten answers. The reason is not
# technical: an information bot that never closes invites aimless chatting, and
# a bounded conversation is also a bounded demo.
MAX_ANSWERS: int = 10

# --- Canned replies --------------------------------------------------------
GREETING: str = (
    "Hi! I'm PeakCast. I can tell you the temperature, sky, wind or "
    "precipitation forecast for places in Austria - towns, peaks and huts. "
    "Try: \"How windy is it on the Hochkönig tomorrow?\""
)

FAREWELL: str = "Take care up there - have a good tour!"

CLOSING_AFTER_LIMIT: str = (
    "That's ten answers, so I'll stop here. Start me again any time - "
    "and check the official mountain forecast before anything serious."
)

# Shown when the classifier is unsure and there is nothing in the message to
# fall back on. One concrete example, not a list: a user who did not get
# through the first time needs a model sentence, not a manual.
HELP_REPLY: str = (
    "Sorry, I didn't catch that. I answer short weather questions for places "
    "in Austria - for example \"Will it snow in Sölden tomorrow?\". "
    "You can ask about temperature, sky, wind or rain and snow."
)

OUT_OF_SCOPE_REPLY: str = (
    "That's outside what I do - I only answer weather questions for places in "
    "Austria. Ask me about temperature, sky, wind or precipitation and I'm "
    "happy to help."
)

ASK_FOR_PLACE: str = "Which place do you mean?"

# Sent when the user asks beyond the forecast horizon.
#
# Both dates are spelled out on purpose. "next Friday" is genuinely ambiguous
# in English - the bot reads it as the Friday of the following week, which is a
# documented convention, not a fact - so naming the date it actually resolved
# lets the user see the reading and correct it. Naming the last forecastable
# date as well turns a refusal into something actionable: the user learns what
# they *can* ask for instead of just what they cannot.
OUT_OF_RANGE_REPLY: str = (
    "{asked} is too far ahead - I can forecast up to {last}."
)


@dataclass
class DialogueState:
    """Everything the bot remembers between turns.

    Attributes:
        last_place: The place most recently talked about, so "and tomorrow?"
            and "is it windy?" have somewhere to point.
        last_day: The day most recently talked about, so a part-of-day
            follow-up ("and in the afternoon?") stays on that day instead of
            silently jumping to today.
        last_weather_intent: The last weather question asked. This is what a
            follow-up without a weather word reuses.
        pending: What the bot asked back for, or ``None``.
        pending_intent: The intent that was waiting on the ask-back, so the
            answer to "Which place do you mean?" resumes the original
            question rather than guessing a new one.
        answers_given: How many replies the bot has produced, for the limit.
        offline_note_sent: Whether the full "this is sample data" note has
            already been given, so later answers can use the short form
            instead of repeating a paragraph ten times.
        finished: ``True`` once the conversation is over.
    """

    last_place: Place | None = None
    last_day: DaySlot | None = None
    last_weather_intent: str | None = None
    pending: str | None = None
    pending_intent: str | None = None
    answers_given: int = 0
    offline_note_sent: bool = False
    finished: bool = False


def is_bare_slot_followup(message: str, place: Place | None, day: DaySlot) -> bool:
    """Report whether a message is nothing but a new place and/or day.

    Used to tell a follow-up from a genuine out-of-scope request, which the
    classifier cannot do: both "what about Kaprun?" and "can you book me a
    hotel in Innsbruck" come back as confident ``out_of_scope`` with a place
    in them. The first is nothing but a place; the second still says "book"
    and "hotel".

    Args:
        message: The raw user message.
        place: The place that was extracted, if any.
        day: The day that was extracted.

    Returns:
        ``True`` if everything left after removing the matched slot text is
        filler.
    """
    text = message.lower()
    for matched in (place.matched_text if place else "", day.matched_text):
        if matched:
            text = text.replace(matched.lower(), " ")

    remaining = {
        "".join(ch for ch in token if ch.isalpha())
        for token in text.split()
    }
    remaining.discard("")
    return not (remaining - FOLLOWUP_FILLER)


@dataclass
class BotTurn:
    """One reply, plus what produced it.

    The extra fields are not shown to the user. They exist so the CLI can
    display the bot's reasoning for a screenshot and so the tests can assert
    on *why* an answer came out, not just on its wording.

    Attributes:
        text: What to show the user.
        intent: The intent that was acted on, after context handling.
        confidence: The classifier's probability for its own top intent.
        place: The place used, if any.
        day: The day used, if any.
        source: ``"live"``, ``"offline"`` or ``""`` when no forecast was
            fetched.
        guard_triggered: ``True`` if the safety guard answered this turn.
        finished: ``True`` if the conversation ended on this turn.
    """

    text: str
    intent: str | None = None
    confidence: float = 0.0
    place: Place | None = None
    day: DaySlot | None = None
    source: str = ""
    guard_triggered: bool = False
    finished: bool = False


class Bot:
    """The assembled chatbot: holds the model, answers one message at a time.

    The model is loaded once per instance rather than per message, because
    loading it is the slowest thing in a turn by a wide margin.
    """

    def __init__(
        self,
        model_path: Path = ACTIVE_MODEL_PATH,
        allow_live: bool = True,
        rng: random.Random | None = None,
        today: date | None = None,
        verbose_startup: bool = True,
    ) -> None:
        """Create the bot, training the intent model if it is missing.

        Args:
            model_path: Which intent model to use.
            allow_live: Whether to call the live weather API. ``False`` forces
                the offline demo dataset.
            rng: Random source for phrasing variants. Pass a seeded
                ``random.Random`` for deterministic output; the tests do.
            today: The date to treat as today. Defaults to the current date in
                Vienna, and is injectable so tests do not depend on the clock.
            verbose_startup: Whether ``ensure_model`` may print while training.
        """
        # Satisfies the step 1 to-do: models/*.joblib is git-ignored, so a
        # fresh checkout has no model and the bot has to make one itself.
        self.pipeline = ensure_model(model_path, verbose=verbose_startup)
        self.allow_live = allow_live
        self.rng = rng or random.Random()
        self._today = today

    @property
    def today(self) -> date:
        """Return the date that counts as today."""
        return self._today or today_in_vienna()

    # --- The turn ---------------------------------------------------------

    def respond(self, state: DialogueState, message: str) -> BotTurn:
        """Answer one message and update the conversation state in place.

        Args:
            state: The conversation state; mutated.
            message: The raw user message.

        Returns:
            The reply and the decisions behind it.
        """
        if state.finished:
            # Defensive: the CLI stops asking once finished, but a caller that
            # keeps going should get a clear answer rather than a silent one.
            return BotTurn(text=FAREWELL, finished=True)

        # --- 1. Safety guard, on the raw message, before anything else.
        turn = self._check_safety(state, message)
        if turn is not None:
            return turn

        # --- 2. Intent.
        intent, confidence = predict_intent(self.pipeline, message)
        confident = confidence >= CONFIDENCE_THRESHOLD

        # --- 3. Slots.
        slots = extract_slots(message, reference=self.today)

        # --- 4. Non-weather intents are answered without a forecast. A
        #        confident `greet`/`goodbye`/`out_of_scope` short-circuits
        #        here, but only when the message carries no place: "hello,
        #        weather in Lienz?" should be answered, not greeted.
        if confident and intent in {INTENT_GREET, INTENT_GOODBYE}:
            if slots.place is None:
                return self._social(state, intent, confidence)

        # `out_of_scope` is believed unless the message is nothing but a new
        # slot in an ongoing weather conversation. That exception is what makes
        # "what about Kaprun?" work: it is confident out_of_scope (p = 0.88)
        # because it contains no weather word, and only the bare-slot test
        # tells it apart from "can you book me a hotel in Innsbruck" (p = 0.91).
        if confident and intent == INTENT_OUT_OF_SCOPE:
            continuing = (
                state.last_weather_intent is not None or state.pending is not None
            )
            if not (
                continuing
                and is_bare_slot_followup(message, slots.place, slots.day)
            ):
                return self._finish_turn(
                    state, OUT_OF_SCOPE_REPLY, intent=intent, confidence=confidence
                )

        # --- 5. Context handling: decide what this message is really asking.
        resolved = self._apply_context(state, intent, confident, slots)
        if resolved is None:
            # Nothing usable: no confident intent, no place, no day, and no
            # conversation to lean on.
            return self._finish_turn(
                state, HELP_REPLY, intent=intent, confidence=confidence
            )

        weather_intent, place, day = resolved

        # --- 6. Still no place? Ask back and remember what we were asked.
        if place is None:
            state.pending = PENDING_PLACE
            state.pending_intent = weather_intent
            # Remember the day as well, not just the question. Without this,
            # "how windy is it tomorrow" -> "Which place?" -> "Sölden" would
            # answer for *today*: the day the user already gave would be
            # thrown away by the ask-back.
            state.last_day = day
            return self._finish_turn(
                state, ASK_FOR_PLACE, intent=weather_intent, confidence=confidence
            )

        # The ask-back, if there was one, is now satisfied.
        state.pending = None
        state.pending_intent = None

        # --- 7. Remember, then answer.
        state.last_place = place
        state.last_day = day
        state.last_weather_intent = weather_intent

        return self._answer_weather(
            state, weather_intent, place, day, message, confidence
        )

    # --- Steps ------------------------------------------------------------

    def _check_safety(
        self, state: DialogueState, message: str
    ) -> BotTurn | None:
        """Run the avalanche guard, refining the referral with the state.

        The guard decides on the raw message alone. A place is extracted only
        to pick the right bulletin (avalanche.report applies to Tyrol only),
        and if extraction finds nothing the user still gets the correct
        nationwide portal - the safety behaviour never depends on the NLU
        succeeding.

        Args:
            state: Conversation state.
            message: Raw user message.

        Returns:
            The referral turn, or ``None`` if the message is safe to classify.
        """
        if check_message(message) is None:
            return None

        slots = extract_slots(message, reference=self.today)
        # Fall back to the remembered place, so "and what about the avalanche
        # risk?" after talking about a Tyrolean peak still gets the Tyrol
        # bulletin.
        place = slots.place or state.last_place
        referral = check_message(message, place.state if place else None)

        turn = self._finish_turn(state, referral or "", place=place)
        return BotTurn(
            text=turn.text,
            intent=None,
            confidence=0.0,
            place=place,
            guard_triggered=True,
            finished=turn.finished,
        )

    def _social(
        self, state: DialogueState, intent: str, confidence: float
    ) -> BotTurn:
        """Answer a greeting or a goodbye.

        Args:
            state: Conversation state.
            intent: ``greet`` or ``goodbye``.
            confidence: The classifier's confidence, for the CLI.

        Returns:
            The reply. A goodbye ends the conversation.
        """
        if intent == INTENT_GOODBYE:
            state.finished = True
            return self._finish_turn(
                state, FAREWELL, intent=intent, confidence=confidence
            )
        return self._finish_turn(
            state, GREETING, intent=intent, confidence=confidence
        )

    def _apply_context(
        self,
        state: DialogueState,
        intent: str,
        confident: bool,
        slots,
    ) -> tuple[str, Place | None, DaySlot] | None:
        """Work out which weather question this message is really asking.

        This is the core of the step, and it handles four cases:

        1. **An answer to an ask-back.** The bot asked "Which place do you
           mean?" and this message names one. The *pending* intent resumes, so
           the user does not have to repeat the question.
        2. **A confident weather question.** Taken at face value; a missing
           place falls back to the remembered one.
        3. **A follow-up with no weather word.** "and tomorrow?", "what about
           Kaprun?", "and in the afternoon?" - these carry a place and/or a
           day but nothing a classifier can latch onto, so the intent comes
           back low-confidence or simply wrong. If the message has a slot and
           there is a remembered weather intent, that intent is reused. This
           is what makes the conversation feel like a conversation.
        4. **Nothing usable.** Returns ``None``, and the caller sends help.

        Args:
            state: Conversation state.
            intent: The classifier's top intent.
            confident: Whether it cleared the threshold.
            slots: The extracted slots.

        Returns:
            ``(weather_intent, place, day)``, or ``None`` if the message
            cannot be read as a weather question at all.
        """
        place = slots.place
        day = slots.day

        # What THIS message actually named, recorded before any inheritance.
        # It has to be captured here, because the day inheritance below
        # replaces an unspecified day with the remembered one - after which
        # `day.is_default` is False and every message looks as though it named
        # a day. That is what made a mid-conversation "hmm" come back with a
        # repeat of the previous forecast instead of the help reply: the
        # inherited day made it look like a follow-up. A follow-up is a
        # message that brings something *new*, so the test has to be on what
        # the user supplied.
        named_place = slots.place is not None
        named_day = not slots.day.is_default

        # --- Which day is this about?
        #
        # Two kinds of inheritance, both deliberate:
        #
        # 1. A phrase that named only a part of the day ("in the afternoon")
        #    inherits the day under discussion - see DaySlot.part_only.
        # 2. A message with **no day word at all** also inherits it. Within
        #    one conversation the day under discussion persists until the user
        #    changes it, which is how people actually talk: after "how windy
        #    on the Hochkönig tomorrow", "will it snow there" means tomorrow,
        #    not today.
        #
        # The risk in (2) is answering about Saturday when the user meant now.
        # It is accepted for two reasons: an explicit "today" or "right now"
        # still overrides it (those set is_default=False), and every answer
        # names the day it used, so a wrong inheritance is visible and
        # correctable rather than silent.
        # The inherited day is built by hand rather than by re-running
        # find_day against the remembered date. Doing the latter looks tidier
        # but is wrong: find_day would resolve "in the afternoon" to offset 0
        # *relative to that date*, leaving a DaySlot whose target_date is
        # tomorrow and whose offset_days is 0. The live path keys off the date
        # and the offline dataset keys off the offset, so the two would answer
        # for different days - which is exactly the bug this comment replaced.
        # DaySlot.offset_days is always relative to today, and it stays that
        # way here.
        if day.part_only and state.last_day is not None:
            day = DaySlot(
                target_date=state.last_day.target_date,
                offset_days=state.last_day.offset_days,
                part_of_day=day.part_of_day,
                matched_text=day.matched_text,
                in_range=state.last_day.in_range,
                part_only=True,
            )
        elif day.is_default and state.last_day is not None:
            day = state.last_day

        # --- Case 1: the reply to "Which place do you mean?"
        if state.pending == PENDING_PLACE and place is not None:
            # The day was already inherited above if this message named none,
            # so a bare place name keeps the day the question was asked about.
            resumed = state.pending_intent or state.last_weather_intent or ASK_OVERVIEW
            return resumed, place, day

        # --- Case 2: a confident weather question.
        if confident and intent in WEATHER_INTENTS:
            return intent, place or state.last_place, day

        # --- Case 3: a follow-up. The message brings a new slot but no
        #     usable intent, and there is a previous weather question to
        #     continue.
        if (named_place or named_day) and state.last_weather_intent is not None:
            return state.last_weather_intent, place or state.last_place, day

        # --- A weather intent that did not clear the threshold, but the
        #     message does carry a place: better to answer the overview than
        #     to say "I didn't catch that" to someone who named a mountain.
        if place is not None and intent in WEATHER_INTENTS:
            return intent, place, day
        if place is not None and state.last_weather_intent is None:
            return ASK_OVERVIEW, place, day

        # --- Case 4: nothing to go on.
        return None

    def _answer_weather(
        self,
        state: DialogueState,
        intent: str,
        place: Place,
        day: DaySlot,
        message: str,
        confidence: float,
    ) -> BotTurn:
        """Fetch the forecast and write the answer.

        Args:
            state: Conversation state.
            intent: The weather intent to answer.
            place: The resolved place.
            day: The resolved day.
            message: The user's message, for the snow-before-rain decision.
            confidence: Classifier confidence, for the CLI.

        Returns:
            The reply.
        """
        try:
            report = get_forecast(place, day, allow_live=self.allow_live)
        except ForecastOutOfRangeError:
            # Not a failure to recover from: the question has an honest
            # answer, and inventing numbers would be worse than saying so.
            from peakcast.responses import format_calendar_date

            last_day = self.today + timedelta(days=MAX_DAY_OFFSET)
            return self._finish_turn(
                state,
                OUT_OF_RANGE_REPLY.format(
                    asked=format_calendar_date(day.target_date),
                    last=format_calendar_date(last_day),
                ),
                intent=intent,
                confidence=confidence,
                place=place,
                day=day,
            )

        answer = generate_answer(
            intent,
            report,
            place,
            day,
            self.today,
            message=message,
            rng=self.rng,
            brief_offline_note=state.offline_note_sent,
        )
        if report.is_offline:
            state.offline_note_sent = True

        # A name that needed typo tolerance or was a shortened compound is an
        # assumption, so it is stated rather than answered silently.
        if place.fuzzy:
            answer = f"{fuzzy_confirmation(place)} {answer}"

        return self._finish_turn(
            state,
            answer,
            intent=intent,
            confidence=confidence,
            place=place,
            day=day,
            source=report.source,
        )

    def _finish_turn(
        self,
        state: DialogueState,
        text: str,
        intent: str | None = None,
        confidence: float = 0.0,
        place: Place | None = None,
        day: DaySlot | None = None,
        source: str = "",
    ) -> BotTurn:
        """Count the reply, apply the conversation limit and build the turn.

        Every reply goes through here, which is what makes the limit reliable:
        there is one place that increments the counter, so no path can answer
        without being counted.

        Args:
            state: Conversation state; mutated.
            text: The reply text.
            intent: Intent acted on.
            confidence: Classifier confidence.
            place: Place used.
            day: Day used.
            source: Forecast source.

        Returns:
            The turn.
        """
        state.answers_given += 1

        if state.answers_given >= MAX_ANSWERS and not state.finished:
            # The closing is appended to the tenth answer rather than sent as
            # an eleventh message: the user still gets the answer they asked
            # for, and the conversation ends in the same breath.
            text = f"{text} {CLOSING_AFTER_LIMIT}"
            state.finished = True

        return BotTurn(
            text=text,
            intent=intent,
            confidence=confidence,
            place=place,
            day=day,
            source=source,
            finished=state.finished,
        )


def start_conversation() -> DialogueState:
    """Create a fresh conversation state.

    Returns:
        An empty state.
    """
    return DialogueState()
