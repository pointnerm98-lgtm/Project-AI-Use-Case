"""PeakCast - Streamlit chat interface.

Run from the project root:

    streamlit run app.py

This file is deliberately **thin**. It owns the screen and nothing else: no
intent logic, no slot handling, no forecast decisions, no wording. Every reply
comes from ``peakcast.dialogue.Bot.respond``, which is the same code path the
terminal client and the tests use.

That separation is the point. If the UI could decide anything about an answer,
there would be two bots - one tested and one on screen - and the tested one
would not be the one the tutor sees. So the rule here is: this module may
*render* what a ``BotTurn`` contains, and may not derive anything from it that
``dialogue.py`` has not already decided. The same rule applies to the
transcript export: the formatting and the record mapping live in
``peakcast.transcript``, so they can be tested without starting an app.

Design notes
------------

* **The target group is on a phone.** Someone at a trailhead is not holding a
  laptop, so the layout is centred and single-column, the messages are short,
  and nothing needs horizontal space. The sidebar collapses on a narrow
  screen, which is why nothing essential lives only there.
* **The data source is always visible.** Every answer carries a badge saying
  whether the numbers are a live forecast or sample data. A fictional forecast
  must never be able to look real, and a caption under the message is harder
  to miss than a sentence inside it.
* **Nothing is stored on the server.** There is no database, no log file and
  no analytics endpoint in this project. The conversation lives in the
  browser's session, and the only way it leaves is the tester pressing
  "Download conversation" and sending the file themselves.
* **NLU details are off by default.** They are evidence for the report, not
  something a hiker needs, so they live behind a sidebar toggle.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from peakcast.dialogue import MAX_ANSWERS, Bot, BotTurn, start_conversation
from peakcast.intent_classifier import ACTIVE_MODEL_PATH
from peakcast.slots import VIENNA
from peakcast.transcript import (
    TurnRecord,
    build_transcript,
    records_from_history,
    suggested_filename,
)
from peakcast.weather import SOURCE_LIVE, SOURCE_OFFLINE

# --- Page setup ------------------------------------------------------------
# "centered" rather than "wide": the audience is on a phone, and a wide layout
# stretches chat bubbles into unreadable lines on a desktop anyway.
st.set_page_config(
    page_title="PeakCast",
    page_icon="⛰️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Text ------------------------------------------------------------------
# Two short lines and nothing more: the start screen has to fit on a phone
# without scrolling, and the suggestions below it show how to ask better than
# a paragraph describing it would.
WELCOME: str = (
    "Hi! I'm **PeakCast** - short weather answers for towns, peaks, huts and "
    "passes in **Austria**."
)

# Shown as compact chips while the conversation is empty. Three, each showing
# off one capability: summit wind (the paragliding case from the concept),
# snow at a ski resort over a range of days, and a temperature question with
# a part of day. They are kept short so each fits on one row of a
# 390 px phone. No follow-up example any more: the chips are only shown
# before the first message, where a follow-up has nothing to follow.
SUGGESTIONS: tuple[str, ...] = (
    "Wind on the Großglockner tomorrow?",
    "Snow in Sölden this weekend?",
    "How cold is Innsbruck tomorrow morning?",
)

# Widget key of the suggestion chips. Named because the click callback has to
# read and clear it, and the smoke tests target it.
SUGGESTIONS_KEY: str = "suggestions"

DISCLAIMER: str = (
    "PeakCast informs - it does **not** replace official mountain weather or "
    "avalanche reports. For avalanche conditions always use the official "
    "service of the federal state: [lawinen.at](https://lawinen.at)."
)

ABOUT: str = (
    "A weather chatbot for hikers, climbers, ski tourers and paragliders in "
    "Austria. Plain Python: scikit-learn for the intent classifier, a "
    "rule-based parser for place and day, and Open-Meteo for the forecast. "
    "No chatbot framework and no LLM API."
)

ATTRIBUTION: str = (
    "**Forecast:** [Open-Meteo](https://open-meteo.com), free for "
    "non-commercial use.\n\n"
    "**Places:** [GeoNames](https://www.geonames.org), licensed "
    "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
)

USER_TEST_NOTE: str = (
    "Taking part in a user test? Have your conversation, give the answers a "
    "thumbs up or down, then download the transcript and send me the file. "
    "Nothing is stored on the server."
)

DOWNLOAD_HELP: str = (
    "Saves this conversation as a text file in your browser, including what "
    "the bot recognised and any thumbs you gave. Nothing is stored on the "
    "server - you can read it before you send it."
)

CONVERSATION_OVER: str = (
    "This conversation is finished. Start a new one to keep asking."
)

# --- Source badges ---------------------------------------------------------
# Keyed by WeatherReport.source. The offline wording says "sample data" rather
# than anything softer, because the whole point of carrying the source through
# three modules is that invented numbers are never presentable as real ones.
SOURCE_BADGES: dict[str, str] = {
    SOURCE_LIVE: ":green[● live forecast]",
    SOURCE_OFFLINE: ":orange[▲ sample data - not a real forecast]",
}

# Prefix of the per-answer feedback widget keys. Named here because two
# functions need it: the one that reads the thumbs and the one that clears
# them on reset.
FEEDBACK_KEY_PREFIX: str = "feedback_"


@st.cache_resource(show_spinner=False)
def load_bot(allow_live: bool) -> Bot:
    """Build the bot once per process and reuse it.

    ``st.cache_resource`` is the right cache here rather than
    ``st.cache_data``: a ``Bot`` holds a fitted scikit-learn pipeline, which
    is a live object to be shared, not a value to be copied. Without this,
    Streamlit would rebuild it on every single interaction - a model load (or
    worse, a training run) per keystroke.

    Keyed on ``allow_live`` so flipping the offline toggle gives a bot with
    the other setting instead of silently keeping the old one.

    Args:
        allow_live: Whether the bot may call the live weather API.

    Returns:
        The bot.
    """
    return Bot(allow_live=allow_live, verbose_startup=False)


def ensure_session() -> None:
    """Create the conversation state on first run.

    Everything that has to survive a rerun lives in ``st.session_state``:
    Streamlit re-executes this whole script top to bottom on every
    interaction, so a plain module-level variable would be reset each time.

    ``history`` holds the rendered turns, each a dict with ``role``, ``text``,
    ``turn`` and ``time``. The bot's own ``DialogueState`` holds what the *bot*
    remembers (last place, last day, answer count); the two are deliberately
    separate, because the transcript on screen and the conversation memory are
    different things.

    The timestamp is taken when the message happens rather than when the
    transcript is exported, so a downloaded conversation shows how it actually
    unfolded.
    """
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = start_conversation()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "queued" not in st.session_state:
        st.session_state.queued = None


def reset_conversation() -> None:
    """Start a fresh conversation, keeping the loaded model.

    The feedback widgets are cleared too. Their values live in
    ``st.session_state`` under per-message keys, and Streamlit keeps a widget
    value for as long as its key is there - so without this, answer 1 of the
    new conversation would inherit the thumbs-down from answer 1 of the old
    one, and that stale rating would end up in an exported transcript.
    """
    stale = [
        key
        for key in st.session_state
        if str(key).startswith(FEEDBACK_KEY_PREFIX)
    ]
    for key in stale:
        del st.session_state[key]

    st.session_state.dialogue = start_conversation()
    st.session_state.history = []
    st.session_state.queued = None


def queue_suggestion() -> None:
    """Queue the clicked suggestion chip as the next user message.

    Runs as the chips' ``on_change`` callback, i.e. before the script reruns.
    The chip's value is cleared again straight away, so a chip never stays
    highlighted and the same suggestion can be picked again after a reset.
    The queued text is then handled exactly like a typed message.
    """
    choice = st.session_state.get(SUGGESTIONS_KEY)
    if choice:
        st.session_state.queued = choice
    st.session_state[SUGGESTIONS_KEY] = None


def render_suggestions() -> None:
    """Draw the example questions as compact chips.

    ``st.pills`` rather than a column of full-width buttons: chips wrap into
    one or two rows on a phone, where four stacked buttons used to push the
    page past the bottom of the screen.
    """
    st.pills(
        "Suggestions",
        SUGGESTIONS,
        selection_mode="single",
        key=SUGGESTIONS_KEY,
        on_change=queue_suggestion,
        label_visibility="collapsed",
    )


def collect_turn_records() -> list[TurnRecord]:
    """Read the session's history and thumbs into transcript records.

    A thin adapter, and intentionally so: everything beyond reading
    ``st.session_state`` lives in ``peakcast.transcript.records_from_history``,
    which is a pure function and therefore testable without starting an app.
    This keeps app.py to what only app.py can do.

    Returns:
        One record per message, in order.
    """
    feedback = {
        index: st.session_state[f"{FEEDBACK_KEY_PREFIX}{index}"]
        for index in range(len(st.session_state.history))
        if st.session_state.get(f"{FEEDBACK_KEY_PREFIX}{index}") is not None
    }
    return records_from_history(st.session_state.history, feedback)


def render_download(offline_mode: bool) -> None:
    """Draw the "Download conversation" button.

    The transcript is built on every rerun rather than when the button is
    clicked, because ``st.download_button`` needs its payload up front - it
    hands the browser a file directly, with no server round trip. That is also
    the property that matters here: **nothing is stored on the server**, and
    this button is the only way a conversation leaves the browser.

    Before anything has been asked there is nothing to export, so a disabled
    plain button is shown instead - visible, so a tester knows the feature is
    there, but not offering an empty file.

    Args:
        offline_mode: Whether the offline toggle is on, so the transcript
            header can record which data source the conversation used.
    """
    records = collect_turn_records()

    if not records:
        st.button(
            "Download conversation",
            key="download_placeholder",
            disabled=True,
            use_container_width=True,
            help="Ask something first - there is nothing to export yet.",
        )
        return

    exported_at = datetime.now(VIENNA)
    st.download_button(
        "Download conversation",
        key="download_transcript",
        data=build_transcript(
            records,
            exported_at=exported_at,
            mode="offline demo data" if offline_mode else "live forecasts",
        ),
        file_name=suggested_filename(exported_at),
        mime="text/plain",
        use_container_width=True,
        help=DOWNLOAD_HELP,
    )


def render_sidebar() -> tuple[bool, bool]:
    """Draw the sidebar and return the two toggle values.

    Returns:
        ``(offline_mode, show_nlu)``.
    """
    with st.sidebar:
        st.subheader("About PeakCast")
        st.caption(ABOUT)

        st.divider()
        st.subheader("Settings")

        offline_mode = st.toggle(
            "Offline demo mode",
            value=False,
            help=(
                "Answer from the built-in sample dataset instead of calling "
                "the weather API. Useful if the network fails during a "
                "presentation. Answers are then clearly marked as sample data."
            ),
        )
        show_nlu = st.toggle(
            "Show NLU details",
            value=False,
            help=(
                "Show the recognised intent, its confidence and the extracted "
                "place and day under each answer."
            ),
        )

        st.divider()
        st.subheader("User test")
        render_download(offline_mode)
        st.caption(USER_TEST_NOTE)

        st.divider()
        # Keyed so the smoke tests can target this button rather than relying
        # on position: there are two "Start new conversation" buttons, here
        # and at the end of a finished conversation.
        if st.button(
            "Start new conversation",
            key="reset_sidebar",
            use_container_width=True,
        ):
            reset_conversation()
            st.rerun()

        st.divider()
        st.subheader("Important")
        st.warning(DISCLAIMER)

        st.divider()
        st.subheader("Data sources")
        st.caption(ATTRIBUTION)

    return offline_mode, show_nlu


def render_nlu_details(turn: BotTurn) -> None:
    """Show what the NLU decided for one turn.

    Evidence for the report rather than something a user needs, which is why
    it is behind a toggle and rendered as a caption.

    Args:
        turn: The turn to describe.
    """
    if turn.guard_triggered:
        st.caption(
            "**Safety guard** - the message was intercepted before the "
            "classifier ran."
        )
        return

    parts: list[str] = []
    if turn.intent is not None:
        parts.append(f"intent `{turn.intent}` (p = {turn.confidence:.2f})")
    if turn.place is not None:
        place = f"place **{turn.place.label}**"
        if turn.place.fuzzy:
            place += " _(assumed)_"
        parts.append(place)
    if turn.day is not None:
        day = f"day **{turn.day.target_date.isoformat()}**"
        if turn.day.part_of_day:
            day += f" ({turn.day.part_of_day})"
        parts.append(day)

    if parts:
        st.caption(" · ".join(parts))


def render_turn(
    index: int,
    role: str,
    text: str,
    turn: BotTurn | None,
    show_nlu: bool,
) -> None:
    """Draw one chat message, with its badge, feedback control and details.

    Args:
        index: Position in the history, used as the feedback widget's key.
        role: ``"user"`` or ``"assistant"``.
        text: The message text.
        turn: The ``BotTurn`` behind an assistant message, else ``None``.
        show_nlu: Whether to show the NLU details.
    """
    avatar = "🧑" if role == "user" else "⛰️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

        if turn is not None and turn.source:
            st.caption(SOURCE_BADGES.get(turn.source, turn.source))
        if turn is not None and show_nlu:
            render_nlu_details(turn)

        if turn is not None:
            # Optional, and unobtrusive on purpose: a rating control that
            # nagged would change the behaviour it exists to measure. The
            # value is held by Streamlit under this key, read back when the
            # transcript is built, and sent nowhere.
            st.feedback("thumbs", key=f"{FEEDBACK_KEY_PREFIX}{index}")


def handle_message(bot: Bot, message: str) -> None:
    """Send one message to the bot and record both sides of the exchange.

    Args:
        bot: The bot.
        message: What the user said.
    """
    st.session_state.history.append(
        {
            "role": "user",
            "text": message,
            "turn": None,
            "time": datetime.now(VIENNA),
        }
    )

    turn = bot.respond(st.session_state.dialogue, message)

    st.session_state.history.append(
        {
            "role": "assistant",
            "text": turn.text,
            "turn": turn,
            "time": datetime.now(VIENNA),
        }
    )


def main() -> None:
    """Draw the page."""
    ensure_session()
    offline_mode, show_nlu = render_sidebar()

    # No subtitle caption: the welcome line says the same thing, and on a
    # phone every line above the fold counts.
    st.title("⛰️ PeakCast")

    # Note the inversion: the sidebar offers "offline demo mode", the Bot
    # takes "allow_live". Getting this the wrong way round silently served
    # sample data in live mode, which the badge made visible and the smoke
    # test now pins.
    allow_live = not offline_mode

    # On a fresh checkout there is no trained model: models/*.joblib is
    # git-ignored because it is a build artefact. The bot trains one itself,
    # which takes about a second - the spinner is there so the first start
    # does not look like a hang. On a fresh cloud deploy this is the first
    # thing that happens, so it matters more there than it does locally.
    if not ACTIVE_MODEL_PATH.exists():
        with st.spinner(
            "First start: training the intent classifier (about a second) ..."
        ):
            bot = load_bot(allow_live)
    else:
        bot = load_bot(allow_live)

    state = st.session_state.dialogue

    # --- The transcript
    if not st.session_state.history:
        with st.chat_message("assistant", avatar="⛰️"):
            st.markdown(WELCOME)

    for index, entry in enumerate(st.session_state.history):
        render_turn(
            index, entry["role"], entry["text"], entry["turn"], show_nlu
        )

    # --- A message queued by a suggestion chip.
    # The chip cannot answer in place: its callback runs before the script
    # reruns, and the bot needs the loaded model, which only exists inside
    # this run. So the callback stores the question and it is handled here,
    # before anything else is drawn below the transcript.
    if st.session_state.queued is not None:
        queued = st.session_state.queued
        st.session_state.queued = None
        handle_message(bot, queued)
        st.rerun()

    # --- Suggestions, only while the conversation is empty.
    # Drawn in the chat area, where the history will appear, rather than as a
    # block of their own: Streamlit cannot tell when the chat input gets
    # focus, so "show on focus" is not possible without a custom component.
    # Once the first message is sent, typed or clicked, they are gone for the
    # rest of the conversation; a reset empties the history and brings them
    # back.
    if not st.session_state.history:
        render_suggestions()

    # --- Input, or the end of the conversation
    if state.finished:
        st.info(CONVERSATION_OVER)
        if st.button(
            "Start new conversation", key="reset_main", type="primary"
        ):
            reset_conversation()
            st.rerun()
        # The input is rendered disabled rather than hidden, so the end of the
        # conversation reads as a state of the page rather than as something
        # breaking.
        st.chat_input("Conversation finished", disabled=True)
    else:
        remaining = MAX_ANSWERS - state.answers_given
        if 0 < remaining <= 3:
            st.caption(f"{remaining} answer(s) left in this conversation.")

        message = st.chat_input("Ask about the weather in Austria ...")
        if message:
            handle_message(bot, message)
            st.rerun()


if __name__ == "__main__":
    main()
