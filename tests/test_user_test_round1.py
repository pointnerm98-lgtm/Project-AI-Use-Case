"""Regression tests for user test round 1 (step 11).

Every message here was typed by a real tester on a phone against the deployed
app, on **Sunday 22 November 2026**, and went wrong. Each test uses the
tester's exact sentence and pins the behaviour they expected. The same
sentences, with expected and actual behaviour, are listed in
``data/user_test_round1.csv`` for the presentation.

The tests were written before the fixes and failed against the step 10 code;
see DEVLOG.md, step 11.

Every test runs offline with a fixed ``today``, so the weekday arithmetic is the
tester's: Sunday 22 November, with the forecast horizon ending on Saturday
28 November.
"""

from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.dialogue import Bot, start_conversation
from peakcast.safety_guard import AVALANCHE_REFERRAL, check_message, mentions_avalanche

# The day of the user test.
TEST_DAY = date(2026, 11, 22)


@pytest.fixture(scope="module")
def bot() -> Bot:
    """One offline bot for the module, pinned to the day of the user test."""
    return Bot(
        allow_live=False,
        rng=random.Random(7),
        today=TEST_DAY,
        verbose_startup=False,
    )


def run(bot: Bot, messages: list[str]):
    """Run a conversation and return the list of turns."""
    state = start_conversation()
    return [bot.respond(state, message) for message in messages]


# =============================================================================
# 1. Safety
# =============================================================================


def test_tester_avalanche_typo_is_caught(bot: Bot) -> None:
    """ "is avlanceh danger?" got the generic help reply instead of the referral."""
    turn = run(bot, ["is avlanceh danger?"])[0]
    assert turn.guard_triggered
    assert "lawinen.at" in turn.text


@pytest.mark.parametrize(
    "message",
    [
        "is avlanceh danger?",
        "avalance risk",
        "avalnche on the dachstein",
        "lavine am Hochkönig",
        "lawiene heute",
        "lawinengefahr morgen",
        "Lavinengefahr Dachstein",
        "avalanches tomorrow?",
        "avalanshe danger",
        "avlance",
    ],
)
def test_avalanche_typos_are_caught(message: str) -> None:
    """Typos of avalanche/Lawine within a small edit distance trigger the guard."""
    assert mentions_avalanche(message), f"guard missed: {message!r}"
    assert check_message(message) == AVALANCHE_REFERRAL


# =============================================================================
# 2. Days
# =============================================================================
from peakcast.slots import extract_slots, find_day, find_days  # noqa: E402

TOMORROW = date(2026, 11, 23)
TUESDAY = date(2026, 11, 24)


@pytest.mark.parametrize(
    "message",
    [
        "weather in innsbruck tomoorow",
        "weather in innsbruck tmorrow",
        "weather in innsbruck tomorow",
        "weather in innsbruck tommorow",
        "which wether is tmorrow in salzburg",
    ],
)
def test_tomorrow_typos_are_read_as_tomorrow(message: str) -> None:
    """A typo in "tomorrow" was ignored and the bot silently answered for today."""
    day = find_day(message, reference=TEST_DAY)
    assert day.target_date == TOMORROW, message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("wind on the hochkoenig satruday", date(2026, 11, 28)),
        ("wind on the hochkoenig on wendsday", date(2026, 11, 25)),
        ("wind on the hochkoenig thurday", date(2026, 11, 26)),
        ("wind on the hochkoenig on tuseday", TUESDAY),
    ],
)
def test_weekday_typos_are_read(message: str, expected: date) -> None:
    """Weekday names with one or two typos still resolve."""
    assert find_day(message, reference=TEST_DAY).target_date == expected


@pytest.mark.parametrize(
    "message",
    [
        "will it be sunnyy at the Katschberg",
        "is it sunny in Lienz",
        "how much snow on the summit",
        "temperature in Mondsee",
        "hows the weather",
    ],
)
def test_day_fuzzy_matching_does_not_invent_days(message: str) -> None:
    """Words close to a day word ("sunny"/"Sunday", "summit") stay non-days."""
    assert find_day(message, reference=TEST_DAY).is_default, message


def test_next_tuesday_is_the_upcoming_tuesday(bot: Bot) -> None:
    """On a Sunday, "next tuesday" was read as Tuesday 1 December (out of range)."""
    turns = run(bot, ["weather in innsbruck tomorrow", "what about next tuesday?"])
    assert turns[1].day.target_date == TUESDAY
    assert "Tuesday 24 November" in turns[1].text


def test_next_weekday_equals_on_weekday() -> None:
    """"next <weekday>" and "on <weekday>" are the same day (new convention)."""
    for name in ("monday", "tuesday", "friday", "saturday"):
        a = find_day(f"next {name}", reference=TEST_DAY)
        b = find_day(f"on {name}", reference=TEST_DAY)
        assert a.target_date == b.target_date
        assert 0 <= a.offset_days <= 6


def test_weekday_answer_names_the_date(bot: Bot) -> None:
    """A day taken from a weekday name is answered with weekday + date."""
    turn = run(bot, ["how windy in innsbruck on thursday"])[0]
    assert "Thursday 26 November" in turn.text


@pytest.mark.parametrize(
    "phrase",
    ["the 24th of november", "24 November", "November 24", "24.11.", "on the 24th"],
)
def test_explicit_dates_work_on_their_own(bot: Bot, phrase: str) -> None:
    """Explicit dates resolve without a weekday, and the answer names the date."""
    day = find_day(f"weather in innsbruck {phrase}", reference=TEST_DAY)
    assert day.target_date == TUESDAY, phrase
    turn = run(bot, [f"weather in innsbruck {phrase}"])[0]
    assert "Tuesday 24 November" in turn.text


def test_yesterday_is_answered_honestly(bot: Bot) -> None:
    """ "yesterday" got today's forecast without a word."""
    turn = run(bot, ["wht temperature yesterday in zell am see"])[0]
    assert "from today on" in turn.text
    assert "today" in turn.text.lower()
    assert turn.day.target_date == TEST_DAY
    assert "°C" in turn.text


@pytest.mark.parametrize("phrase", ["yesterday", "last week", "the day before yesterday", "20 November"])
def test_past_days_are_flagged(phrase: str) -> None:
    """Past days resolve to a negative offset instead of silently to today."""
    day = find_day(f"temperature in Lienz {phrase}", reference=TEST_DAY)
    assert day.offset_days < 0, phrase
    assert not day.is_default


def test_this_week_gets_a_range_summary(bot: Bot) -> None:
    """ "this week" got only today's numbers."""
    turn = run(bot, ["how much snow will there be this week in obertauern"])[0]
    assert "Today in Obertauern" not in turn.text
    assert "in total" in turn.text
    assert "28 November" in turn.text


@pytest.mark.parametrize("phrase", ["this week", "the next days", "the coming days", "the next few days"])
def test_range_phrases_span_the_forecast(phrase: str) -> None:
    """Range phrases cover today to the end of the forecast horizon."""
    day = find_day(f"snow in Obertauern {phrase}", reference=TEST_DAY)
    assert day.span_days == 7
    assert day.offset_days == 0


def test_this_weekend_answers_saturday_and_sunday() -> None:
    """ "this weekend" covers both days, not only Saturday."""
    thursday = date(2026, 11, 26)
    slots = extract_slots("snow in Obertauern this weekend", reference=thursday)
    dates = [slots.day.target_date] + [d.target_date for d in slots.extra_days]
    assert dates == [date(2026, 11, 28), date(2026, 11, 29)]


def test_this_weekend_on_a_thursday_names_both_days() -> None:
    """Both weekend days appear in the answer."""
    bot = Bot(allow_live=False, rng=random.Random(7), today=date(2026, 11, 26), verbose_startup=False)
    turn = run(bot, ["how much snow in obertauern this weekend"])[0]
    assert "Saturday 28 November" in turn.text
    assert "Sunday 29 November" in turn.text


def test_this_weekend_on_a_sunday_says_sunday_is_out_of_range(bot: Bot) -> None:
    """On the test Sunday, next Sunday lies beyond the horizon - say so."""
    turn = run(bot, ["how much snow in obertauern this weekend"])[0]
    assert "Saturday 28 November" in turn.text
    assert "Sunday 29 November" in turn.text
    assert "too far ahead" in turn.text


def test_several_days_in_one_message(bot: Bot) -> None:
    """ "and on thursday? and friday?" only answered Thursday."""
    turns = run(bot, ["wind in innsbruck tomorrow", "ok great and on thursday? and friday?"])
    assert "Thursday 26 November" in turns[1].text
    assert "Friday 27 November" in turns[1].text


def test_find_days_returns_each_named_day_max_three() -> None:
    """Up to three named days are extracted, in the order they were named."""
    days = find_days("and on thursday? and friday?", reference=TEST_DAY)
    assert [d.target_date for d in days] == [date(2026, 11, 26), date(2026, 11, 27)]
    days = find_days("monday tuesday wednesday thursday", reference=TEST_DAY)
    assert len(days) == 3


# =============================================================================
# 3. Places and corrections
# =============================================================================
from peakcast.slots import find_place  # noqa: E402


def test_st_dot_without_space_is_normalised() -> None:
    """ "st.johann im pongau" must find Sankt Johann im Pongau."""
    place = find_place("i meant st.johann im pongau")
    assert place is not None
    assert place.name == "Sankt Johann im Pongau"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("snow obertauren tomorrow", "Obertauern"),         # typo, lower case, no preposition
        ("krippenstein wind tomorrow", "Hoher Krippenstein"),  # partial name, lower case
        ("weather st. johann im pongau", "Sankt Johann im Pongau"),
        ("how cold is it innsbruk", "Innsbruck"),
    ],
)
def test_long_place_names_need_no_capital_or_preposition(message: str, expected: str) -> None:
    """Long names are found in lower case without "in"/"on" in front."""
    place = find_place(message)
    assert place is not None, message
    assert place.name == expected


@pytest.mark.parametrize(
    "message",
    [
        "is it going to rain tomorrow",
        "is it hard to hike up there today",
        "i want to see the summit tomorrow",
        "what about tomorrow",
        "how much snow will there be this week",
        "thanks for the information",
        "whats the temperature yesterday",
    ],
)
def test_ordinary_lowercase_words_are_not_places(message: str) -> None:
    """The relaxed evidence rule must not turn ordinary words into villages."""
    assert find_place(message) is None, message


def test_correction_reruns_previous_question(bot: Bot) -> None:
    """ "i meant st.johann im pongau" was answered as out of scope."""
    turns = run(bot, ["how much snow in st.johann today", "i meant st.johann im pongau"])
    assert turns[1].place is not None
    assert turns[1].place.name == "Sankt Johann im Pongau"
    assert turns[1].intent == "ask_precipitation"
    assert "snow" in turns[1].text


@pytest.mark.parametrize(
    "correction",
    ["no, Lienz", "sorry, I meant Lienz", "no i meant lienz", "i mean Lienz"],
)
def test_correction_phrasings(bot: Bot, correction: str) -> None:
    """Corrections with only a place rerun the previous question there."""
    turns = run(bot, ["how windy is it in Innsbruck tomorrow", correction])
    assert turns[1].place is not None and turns[1].place.name == "Lienz"
    assert turns[1].intent == "ask_wind"
    assert turns[1].day.target_date == TOMORROW


def test_correction_of_the_day(bot: Bot) -> None:
    """ "no, on friday" keeps the place and the question, changes the day."""
    turns = run(bot, ["how windy is it in Innsbruck tomorrow", "no, on friday"])
    assert turns[1].place.name == "Innsbruck"
    assert turns[1].intent == "ask_wind"
    assert turns[1].day.target_date == date(2026, 11, 27)


def test_no_without_a_slot_is_not_a_correction(bot: Bot) -> None:
    """A "no" that brings no new place or day is not rerun as a correction."""
    turns = run(bot, ["how windy is it in Innsbruck tomorrow", "no, can you book me a hotel"])
    assert turns[1].intent == "out_of_scope"


# =============================================================================
# 4. Answers
# =============================================================================
from peakcast.responses import (  # noqa: E402
    ACTIVITY_HIKING,
    ACTIVITY_PARAGLIDING,
    ACTIVITY_SKI_TOURING,
    build_hints,
    describe_precipitation,
    detect_activity,
)
from peakcast.weather import WeatherReport  # noqa: E402


def _report(**overrides) -> WeatherReport:
    values = dict(
        place_name="Obertauern",
        state="Salzburg",
        elevation=1738,
        target_date=TOMORROW,
        part_of_day=None,
        temperature_min=-3.0,
        temperature_max=2.0,
        weather_code=61,
        sky="cloudy",
        cloud_cover=90.0,
        wind_speed_max=20.0,
        wind_gusts_max=30.0,
        rain_sum=2.1,
        snowfall_sum=0.0,
        precipitation_probability_max=88,
        source="offline",
    )
    values.update(overrides)
    return WeatherReport(**values)


def test_snow_question_states_zero_snow() -> None:
    """A snow question with no snow says so, and gives the rain instead."""
    phrase = describe_precipitation(_report(), snow_first=True)
    assert phrase == "no fresh snow expected - 2.1 mm of rain instead (88 % chance)"


def test_snow_question_on_a_dry_day_states_zero_snow(bot: Bot) -> None:
    """ "Today in Obertauern: dry." did not answer the snow question."""
    turn = run(bot, ["how much snow in obertauern tomorrow"])[0]
    assert "no fresh snow" in turn.text.lower() or "of fresh snow" in turn.text


def test_snow_follow_up_still_states_snow(bot: Bot) -> None:
    """ "and on friday?" after a snow question is still about snow."""
    turns = run(bot, ["how much snow in obertauern tomorrow", "and on friday?"])
    assert "snow" in turns[1].text.lower()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ski touring on the hochkoenig tomorrow, how windy?", ACTIVITY_SKI_TOURING),
        ("backcountry skiing at the Kitzsteinhorn", ACTIVITY_SKI_TOURING),
        ("want to hike the Schafberg", ACTIVITY_HIKING),
        ("can I fly at the Gerlitzen tomorrow", ACTIVITY_PARAGLIDING),
        ("paragliding at the Emberger Alm", ACTIVITY_PARAGLIDING),
        ("how windy is it in Lienz", None),
    ],
)
def test_detect_activity(message: str, expected: str | None) -> None:
    """The activity is read from the message."""
    assert detect_activity(message) == expected


def test_ski_tourer_is_not_told_about_flying(bot: Bot) -> None:
    """A ski tourer got "not flying weather"."""
    turn = run(bot, ["wind in innsbruck tomorrow", "ski touring on thursday?"])[1]
    assert "flying" not in turn.text
    assert "launch" not in turn.text


def test_activity_is_remembered_across_turns(bot: Bot) -> None:
    """The activity named earlier picks the hint for later answers."""
    turns = run(bot, ["going ski touring in innsbruck tomorrow", "how windy is it on thursday"])
    assert "flying" not in turns[1].text
    assert "wind chill" in turns[1].text


def test_paraglider_still_gets_flying_hint() -> None:
    hints = build_hints(_report(wind_gusts_max=45.0), False, activity=ACTIVITY_PARAGLIDING)
    assert "not flying weather" in hints[0]


def test_neutral_gust_hint_without_activity() -> None:
    hints = build_hints(_report(wind_gusts_max=65.0, rain_sum=0.0), False)
    assert "flying" not in hints[0] and "launch" not in hints[0]
    assert "exposed ridges" in hints[0]


def test_remaining_answers_caption_is_pluralised() -> None:
    """ "1 answer(s) left" -> "1 answer left" / "2 answers left"."""
    from app import remaining_answers_caption

    assert remaining_answers_caption(1) == "1 answer left in this conversation."
    assert remaining_answers_caption(2) == "2 answers left in this conversation."


# =============================================================================
# 5. Intent model
# =============================================================================


@pytest.mark.parametrize(
    "message",
    ["how are you today?", "what's up?", "who are you", "how is it going"],
)
def test_small_talk_gets_a_friendly_reply(bot: Bot, message: str) -> None:
    """ "how are you today?" got "That's outside what I do"."""
    turn = run(bot, [message])[0]
    assert "outside what I do" not in turn.text
    assert "wind" in turn.text  # the hint at what PeakCast can do
    assert not turn.finished


def test_thank_you_gets_a_reply_and_keeps_the_conversation(bot: Bot) -> None:
    turns = run(bot, ["wind in innsbruck tomorrow", "thank you"])
    assert "welcome" in turns[1].text.lower()
    assert not turns[1].finished


def test_thanks_bye_still_ends(bot: Bot) -> None:
    turns = run(bot, ["thanks bye"])
    assert turns[0].finished


def test_which_weather_with_typos_is_an_overview(bot: Bot) -> None:
    """ "which wether is tmorrow in salzburg" only gave cloud cover."""
    turn = run(bot, ["which wether is tmorrow in salzburg"])[0]
    assert turn.intent == "ask_overview"
    assert turn.day.target_date == TOMORROW


# =============================================================================
# Regressions found by the slot sweep while fixing round 1
# =============================================================================


@pytest.mark.parametrize(
    "message",
    [
        "how much sun can I expect in Saalbach tomorrow",
        "will the sun break through in Admont tomorrow",
    ],
)
def test_sun_is_not_sunday(message: str) -> None:
    """ "sun" is a weather word; the short form of Sunday needs "on" in front."""
    assert find_day(message, reference=TEST_DAY).target_date == TOMORROW


def test_bare_sun_is_not_a_day() -> None:
    assert find_day("how much sun in Saalbach", reference=TEST_DAY).is_default
    assert find_day("snow on sun?", reference=TEST_DAY).target_date == date(2026, 11, 22)


@pytest.mark.parametrize(
    "message", ["whats your favourite mountain", "what is your favourite food"]
)
def test_favourite_is_not_favoriten(message: str) -> None:
    """The relaxed long-name rule must not read "favourite" as Favoriten."""
    assert find_place(message) is None
