"""Smoke tests for the Streamlit interface.

These are deliberately *smoke* tests, not a second test suite for the bot. The
bot's behaviour is already covered by 279 tests against ``dialogue.py`` and
below; repeating any of it through the UI would mean two places to update
whenever an answer changes.

What is left for the UI to get wrong is wiring, and that is what is checked
here: does the page start at all, does a question actually reach the bot, does
the data-source badge say the right thing, does the end of a conversation lock
the input, does reset clear it. All of those are invisible to every other test
in the project.

The tests run with the offline dataset wherever a forecast is involved, so
they need no network - except one, which deliberately checks the live badge
and is marked ``live``.

``AppTest`` re-executes ``app.py`` in-process, so a long default timeout is
set: the first run of the session may have to train the intent model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APP_FILE = str(PROJECT_ROOT / "app.py")

# Generous, because the very first run in a session can include training the
# intent model from scratch.
TIMEOUT_SECONDS: int = 240

# Sidebar toggle positions, named so a reordering of the sidebar does not turn
# into a silently wrong test.
TOGGLE_OFFLINE: int = 0
TOGGLE_SHOW_NLU: int = 1


def start_app(offline: bool = True, show_nlu: bool = False) -> AppTest:
    """Run the app and set the sidebar toggles.

    Args:
        offline: Use the built-in sample dataset instead of the live API, so
            the test needs no network.
        show_nlu: Turn the NLU details on.

    Returns:
        The running app.
    """
    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT_SECONDS).run()

    if offline:
        app.sidebar.toggle[TOGGLE_OFFLINE].set_value(True).run()
    if show_nlu:
        app.sidebar.toggle[TOGGLE_SHOW_NLU].set_value(True).run()

    return app


def captions_of(app: AppTest) -> list[str]:
    """Collect every caption inside a chat message.

    Args:
        app: The running app.

    Returns:
        The caption texts, in order.
    """
    return [
        caption.value
        for message in app.chat_message
        for caption in message.caption
    ]


def assistant_texts(app: AppTest) -> list[str]:
    """Collect the assistant's messages.

    Args:
        app: The running app.

    Returns:
        The markdown of each assistant message.
    """
    return [
        block.value
        for message in app.chat_message
        if message.name == "assistant"
        for block in message.markdown
    ]


# --- The page starts -------------------------------------------------------


def test_app_starts_without_an_exception() -> None:
    """The page must render on a cold start.

    This is the test that would have caught the whole UI being broken, and it
    also covers the first-start path: if no model file exists, the app has to
    train one rather than raise.
    """
    app = start_app(offline=True)

    assert not app.exception
    assert app.title[0].value == "⛰️ PeakCast"


def test_the_welcome_message_and_examples_are_shown_first() -> None:
    """A new visitor must see what the bot does and how to ask.

    The examples are buttons rather than text so that someone on a phone can
    start without typing a sentence with umlauts in it.
    """
    app = start_app(offline=True)

    welcome = assistant_texts(app)
    assert len(welcome) == 1
    assert "PeakCast" in welcome[0]

    labels = [button.label for button in app.button]
    assert any("Großglockner" in label for label in labels)
    assert any("snow" in label.lower() for label in labels)


# --- A question reaches the bot -------------------------------------------


def test_an_example_question_gets_an_answer() -> None:
    """Clicking an example must produce a real answer from the bot.

    The click cannot answer in place - Streamlit reruns the script - so the
    question is queued and handled on the next run. That indirection is
    exactly the kind of wiring a smoke test is for.
    """
    app = start_app(offline=True)
    app.button[0].click().run()

    assert not app.exception

    messages = app.chat_message
    assert len(messages) == 2
    assert messages[0].name == "user"
    assert messages[1].name == "assistant"

    answer = assistant_texts(app)[-1]
    assert "Großglockner" in answer
    assert "km/h" in answer, "a wind question must come back with wind values"


def test_a_typed_question_gets_an_answer() -> None:
    """The chat input must reach the bot too, not only the buttons."""
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    assert not app.exception
    answer = assistant_texts(app)[-1]
    assert "Lienz" in answer
    assert "°C" in answer


def test_a_follow_up_keeps_the_context() -> None:
    """The UI must hold one conversation, not a series of unrelated questions.

    The dialogue state lives in ``st.session_state``; a plain variable would
    be reset on every rerun, and each message would start a new conversation.
    This is the test that would catch that.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("how windy is it on the Hochkoenig today").run()
    app.chat_input[0].set_value("and tomorrow?").run()

    assert not app.exception
    answer = assistant_texts(app)[-1]
    assert "Hochkönig" in answer, "the place must carry over"
    assert "tomorrow" in answer.lower()


# --- The data-source badge ------------------------------------------------


def test_offline_mode_shows_the_sample_data_badge() -> None:
    """Sample data must be labelled as such, visibly.

    This is the UI half of a rule the whole project is built around: a
    fictional forecast must never be able to look real. The badge is a caption
    under the message rather than a sentence inside it, because it has to be
    hard to miss even when the answer is skimmed.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    badges = captions_of(app)
    assert any("sample data" in badge for badge in badges)
    assert not any("live forecast" in badge for badge in badges)


@pytest.mark.live
def test_live_mode_shows_the_live_badge() -> None:
    """A real forecast must not be labelled as sample data.

    Marked ``live`` because it is the one UI test that needs the network. It
    earns its place: the sidebar offers "offline demo mode" while the bot
    takes ``allow_live``, and passing one where the other was expected served
    sample data in live mode - a bug this test now pins.
    """
    app = start_app(offline=False)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    badges = captions_of(app)
    assert any("live forecast" in badge for badge in badges)
    assert not any("sample data" in badge for badge in badges)


# --- The NLU details toggle ----------------------------------------------


def test_nlu_details_are_hidden_by_default() -> None:
    """A hiker does not need the intent name; the report does."""
    app = start_app(offline=True, show_nlu=False)
    app.chat_input[0].set_value("how windy on the Grossglockner tomorrow").run()

    badges = captions_of(app)
    assert not any("intent" in badge for badge in badges)


def test_nlu_details_can_be_switched_on() -> None:
    """With the toggle on, intent, confidence and slots must be visible."""
    app = start_app(offline=True, show_nlu=True)
    app.chat_input[0].set_value("how windy on the Grossglockner tomorrow").run()

    badges = captions_of(app)
    details = [badge for badge in badges if "intent" in badge]
    assert details, "the NLU caption should be shown"
    assert "ask_wind" in details[0]
    assert "Großglockner" in details[0]


# --- The end of a conversation -------------------------------------------


def test_goodbye_locks_the_input_and_offers_a_restart() -> None:
    """A finished conversation must not accept more input.

    The input is rendered *disabled* rather than removed, so the end of the
    conversation reads as a state of the page rather than as something having
    broken.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()
    app.chat_input[0].set_value("thanks bye").run()

    assert not app.exception
    assert app.chat_input[0].disabled is True
    assert any("finished" in info.value for info in app.info)
    assert "reset_main" in [button.key for button in app.button]


def test_reset_clears_the_conversation() -> None:
    """Reset must give a genuinely fresh conversation.

    Both the transcript and the bot's own memory have to go: if only the
    transcript were cleared, the next question would still inherit the old
    place and answer count.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()
    app.chat_input[0].set_value("thanks bye").run()

    app.button(key="reset_main").click().run()

    assert not app.exception
    assert app.chat_input[0].disabled is False
    # Back to the welcome message alone, with the examples offered again.
    assert len(app.chat_message) == 1
    assert any("Großglockner" in button.label for button in app.button)


def test_sidebar_reset_also_works_mid_conversation() -> None:
    """The sidebar button must work before the conversation has ended."""
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()
    assert len(app.chat_message) == 2

    app.button(key="reset_sidebar").click().run()

    assert not app.exception
    assert len(app.chat_message) == 1


def test_the_answer_limit_ends_the_conversation_in_the_ui() -> None:
    """The ten-answer limit must reach the screen, not just the state.

    Checked through the UI because the limit is the one piece of bot state
    that changes what the *page* offers - a disabled input and a restart
    button - and that wiring is this module's job.
    """
    app = start_app(offline=True)

    for _ in range(12):
        if app.chat_input[0].disabled:
            break
        app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    assert not app.exception
    assert app.chat_input[0].disabled is True
    assert any("finished" in info.value for info in app.info)


# --- The user-test features ----------------------------------------------


def download_buttons(app: AppTest) -> list:
    """Collect the real download buttons in the sidebar.

    Before anything has been asked the app shows a *disabled plain button*
    instead, so the absence of a download button is itself the assertion in
    one of the tests below.

    Args:
        app: The running app.

    Returns:
        The download-button elements.
    """
    return list(app.sidebar.get("download_button") or [])


def test_download_is_offered_only_once_there_is_something_to_export() -> None:
    """An empty conversation must not offer an empty file.

    The control is still visible, as a disabled button, so a tester can see
    the feature exists before they have asked anything.
    """
    app = start_app(offline=True)

    assert not download_buttons(app), "nothing to download yet"
    placeholder = [
        button for button in app.sidebar.button
        if button.key == "download_placeholder"
    ]
    assert placeholder and placeholder[0].disabled is True

    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    assert not app.exception
    assert len(download_buttons(app)) == 1


def test_a_feedback_control_is_offered_on_each_answer() -> None:
    """The thumbs are per answer, and only on answers.

    A rating on the user's own message would make no sense, so the widget is
    rendered only where a ``BotTurn`` exists.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    assert len(app.get("feedback") or []) == 1

    app.chat_input[0].set_value("and the wind").run()

    assert len(app.get("feedback") or []) == 2


def test_a_thumbs_down_survives_the_next_message() -> None:
    """A rating must stay attached as the conversation continues.

    Streamlit keeps a widget value for as long as its key exists, so this is
    really a test that the keys are stable across reruns - if they were
    rebuilt from a changing index, the rating would move or vanish.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    app.session_state["feedback_1"] = 0      # thumbs down on the first answer
    app.run()
    app.chat_input[0].set_value("and the wind").run()

    assert not app.exception
    assert app.session_state["feedback_1"] == 0


def test_reset_clears_the_feedback_too() -> None:
    """A new conversation must not inherit the old one's ratings.

    Without this, answer 1 of the new conversation would carry answer 1 of the
    old one's thumbs-down - and that stale rating would end up in an exported
    transcript, which is exactly the kind of error that quietly corrupts a
    user-test summary.
    """
    app = start_app(offline=True)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()
    app.session_state["feedback_1"] = 0
    app.run()

    app.button(key="reset_sidebar").click().run()

    assert not app.exception
    assert app.session_state.get("feedback_1") is None


# --- The offline fallback, as the host would see it ----------------------


def test_a_failing_weather_api_still_answers_and_says_it_is_sample_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the host cannot reach Open-Meteo, the app must degrade visibly.

    This is the deployment case: a cloud host with no outbound access, or
    Open-Meteo down, must not produce an error page. The bot falls back to its
    demo dataset, and the badge has to say so - the fallback is only acceptable
    *because* it is labelled.

    Live mode is used deliberately (``offline=False``): the point is that a
    failure in the live path lands on the offline data, not that the offline
    toggle works.
    """
    import requests

    def always_fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", always_fail)

    app = start_app(offline=False)
    app.chat_input[0].set_value("temperature in Lienz tomorrow").run()

    assert not app.exception, "a dead API must not break the page"

    answer = assistant_texts(app)[-1]
    assert "°C" in answer, "it must still answer"

    badges = captions_of(app)
    assert any("sample data" in badge for badge in badges)
    assert not any("live forecast" in badge for badge in badges)


# --- The sidebar ---------------------------------------------------------


def test_sidebar_explains_the_user_test_and_that_nothing_is_stored() -> None:
    """Participants must be told what happens to their conversation.

    "Nothing is stored on the server" is a factual claim the project makes -
    there is no database, no log file and no analytics endpoint - and it is
    the basis on which a tester agrees to take part, so it is asserted rather
    than trusted.
    """
    app = start_app(offline=True)

    sidebar_text = " ".join(
        [block.value for block in app.sidebar.markdown]
        + [block.value for block in app.sidebar.caption]
    )

    assert "user test" in sidebar_text.lower()
    assert "Nothing is stored on the server" in sidebar_text


def test_sidebar_carries_the_disclaimer_and_attribution() -> None:
    """The things the project is required to say must actually be on screen.

    The disclaimer (not a replacement for official mountain weather or
    avalanche reports) and the data attribution for Open-Meteo and GeoNames
    are obligations, not decoration, so they are asserted rather than trusted.
    """
    app = start_app(offline=True)

    sidebar_text = " ".join(
        [block.value for block in app.sidebar.markdown]
        + [block.value for block in app.sidebar.caption]
        + [block.value for block in app.sidebar.warning]
    )

    assert "avalanche" in sidebar_text.lower()
    assert "lawinen.at" in sidebar_text
    assert "Open-Meteo" in sidebar_text
    assert "GeoNames" in sidebar_text
    assert "CC BY" in sidebar_text
