"""Conversation transcripts: write one out, and read one back.

Used for the user tests. A tester has a conversation in the app, presses
"Download conversation", and sends the resulting ``.txt`` file. Nothing is
stored on the server - there is no database, no logging and no analytics
endpoint in this project, and the download button is the only way a
conversation leaves the browser.

Why a plain text file rather than JSON
--------------------------------------
The tester has to be willing to open it. A ``.txt`` that reads like a
conversation can be checked at a glance before it is sent, which is the
difference between informed consent and a blob of data someone forwards
blindly. It is also what a participant can reasonably be asked to review for
anything they would rather not share.

The cost is that it has to be parsed back, and a format that is pleasant to
read is easy to parse sloppily. So the writer and the parser live in the same
module, next to each other, and there is a round-trip test: anything
``build_transcript`` writes, ``parse_transcript`` must read back unchanged.

Format
------
A header, then one block per turn::

    [2] 14:21:03 PEAKCAST
    Wind on the Hochkoenig tomorrow is around 16 km/h ...
        intent: ask_wind
        confidence: 0.92
        place: Hochkoenig, Salzburg
        day: 2026-11-11
        source: live
        feedback: up

The turn header is machine-readable (index, time, speaker), the message text
follows on its own lines, and the NLU details are indented ``key: value``
pairs. Only the fields that apply to a turn are written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

# --- Format constants ------------------------------------------------------
ROLE_USER: str = "YOU"
ROLE_BOT: str = "PEAKCAST"

# Feedback values, as written in the file.
FEEDBACK_UP: str = "up"
FEEDBACK_DOWN: str = "down"

HEADER_TITLE: str = "PeakCast conversation transcript"
RULE: str = "-" * 70

# Indentation of the metadata lines. Four spaces, so a metadata line can never
# be confused with message text: message text is written flush left, and a
# user message that happens to start with spaces is stripped.
METADATA_INDENT: str = "    "

# ``[12] 14:21:03 PEAKCAST``
_TURN_HEADER_RE = re.compile(
    r"^\[(?P<index>\d+)\]\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<role>\S+)\s*$"
)
_METADATA_RE = re.compile(r"^ {4}(?P<key>[a-z_]+):\s*(?P<value>.*)$")
_HEADER_FIELD_RE = re.compile(r"^(?P<key>[A-Za-z ]+):\s*(?P<value>.*)$")


@dataclass
class TurnRecord:
    """One turn of a conversation, as written to and read from a transcript.

    Attributes:
        index: 1-based position in the conversation.
        time: Wall-clock time the turn happened, ``HH:MM:SS``.
        role: ``ROLE_USER`` or ``ROLE_BOT``.
        text: What was said.
        intent: Recognised intent, for bot turns that ran the classifier.
        confidence: The classifier's probability for that intent.
        place: The resolved place, as ``"Name, State"``.
        day: The resolved date, ISO format.
        part_of_day: ``morning``/``afternoon``/``evening``/``night``.
        source: ``live`` or ``offline``, when a forecast was fetched.
        guard: ``True`` if the safety guard answered this turn.
        fuzzy: ``True`` if the place was an assumption rather than an exact
            match.
        feedback: ``up``, ``down`` or ``None`` - whatever the tester clicked.
    """

    index: int
    time: str
    role: str
    text: str
    intent: str | None = None
    confidence: float | None = None
    place: str | None = None
    day: str | None = None
    part_of_day: str | None = None
    source: str | None = None
    guard: bool = False
    fuzzy: bool = False
    feedback: str | None = None


@dataclass
class Transcript:
    """A parsed transcript.

    Attributes:
        exported_at: When the file was written, as it appears in the header.
        mode: ``"live forecasts"`` or ``"offline demo data"``.
        turns: The turns, in order.
        source_name: Where this transcript was read from, for reports.
    """

    exported_at: str = ""
    mode: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    source_name: str = ""

    @property
    def bot_turns(self) -> list[TurnRecord]:
        """Return only the bot's turns."""
        return [turn for turn in self.turns if turn.role == ROLE_BOT]


def _metadata_lines(turn: TurnRecord) -> list[str]:
    """Render the indented ``key: value`` block for one turn.

    Only fields that apply are written, so a greeting does not carry an empty
    "place:" line and the file stays readable.

    Args:
        turn: The turn.

    Returns:
        The metadata lines, already indented.
    """
    pairs: list[tuple[str, str]] = []

    if turn.guard:
        pairs.append(("guard", "triggered"))
    if turn.intent is not None:
        pairs.append(("intent", turn.intent))
    if turn.confidence is not None:
        pairs.append(("confidence", f"{turn.confidence:.2f}"))
    if turn.place:
        pairs.append(("place", turn.place))
    if turn.fuzzy:
        pairs.append(("place_assumed", "yes"))
    if turn.day:
        pairs.append(("day", turn.day))
    if turn.part_of_day:
        pairs.append(("part_of_day", turn.part_of_day))
    if turn.source:
        pairs.append(("source", turn.source))
    if turn.feedback:
        pairs.append(("feedback", turn.feedback))

    return [f"{METADATA_INDENT}{key}: {value}" for key, value in pairs]


def build_transcript(
    turns: list[TurnRecord],
    exported_at: datetime,
    mode: str,
) -> str:
    """Render a conversation as the downloadable text file.

    Args:
        turns: The turns, in order.
        exported_at: When the export happened.
        mode: A short description of the data source setting, as shown in the
            app ("live forecasts" or "offline demo data").

    Returns:
        The complete file contents.
    """
    lines = [
        HEADER_TITLE,
        f"Exported: {exported_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {mode}",
        f"Turns: {len(turns)}",
        "",
        # Said in the file itself, not only in the app, because the file is
        # what the tester actually sends on - and they should be able to see
        # what they are sending without taking anyone's word for it.
        "This file was created in your browser. Nothing was stored on a",
        "server. You can read it, edit it or delete it before sending it.",
        RULE,
        "",
    ]

    for turn in turns:
        lines.append(f"[{turn.index}] {turn.time} {turn.role}")
        lines.extend(turn.text.strip().splitlines() or [""])
        lines.extend(_metadata_lines(turn))
        lines.append("")

    return "\n".join(lines)


def _coerce(turn: TurnRecord, key: str, value: str) -> None:
    """Apply one parsed metadata pair to a turn.

    Unknown keys are ignored rather than raising: a transcript from a future
    version of the app should still be readable by this parser, minus the
    fields it does not know.

    Args:
        turn: The turn being built; mutated.
        key: The metadata key.
        value: Its raw text value.
    """
    if key == "intent":
        turn.intent = value
    elif key == "confidence":
        try:
            turn.confidence = float(value)
        except ValueError:
            turn.confidence = None
    elif key == "place":
        turn.place = value
    elif key == "place_assumed":
        turn.fuzzy = value.strip().lower() in {"yes", "true"}
    elif key == "day":
        turn.day = value
    elif key == "part_of_day":
        turn.part_of_day = value
    elif key == "source":
        turn.source = value
    elif key == "guard":
        turn.guard = True
    elif key == "feedback":
        turn.feedback = value


def parse_transcript(text: str, source_name: str = "") -> Transcript:
    """Read a transcript file back into structured turns.

    Written to be forgiving, because the files come back from testers who are
    told they may edit them: a malformed or hand-trimmed block is skipped
    rather than aborting the whole file, and unknown metadata keys are
    ignored. The alternative - a strict parser - would mean one edited file
    losing a whole user test.

    Args:
        text: The file contents.
        source_name: The file name, carried into the result for reports.

    Returns:
        The parsed transcript.
    """
    transcript = Transcript(source_name=source_name)
    current: TurnRecord | None = None
    text_lines: list[str] = []
    in_header = True

    def close_current() -> None:
        """Attach the accumulated message text and store the finished turn."""
        if current is None:
            return
        current.text = "\n".join(text_lines).strip()
        transcript.turns.append(current)

    for raw_line in text.splitlines():
        header_match = _TURN_HEADER_RE.match(raw_line)
        if header_match:
            close_current()
            in_header = False
            current = TurnRecord(
                index=int(header_match.group("index")),
                time=header_match.group("time"),
                role=header_match.group("role"),
                text="",
            )
            text_lines = []
            continue

        if current is None:
            # Still in the file header: pick up the two fields worth keeping.
            if in_header:
                field_match = _HEADER_FIELD_RE.match(raw_line)
                if field_match:
                    key = field_match.group("key").strip().lower()
                    if key == "exported":
                        transcript.exported_at = field_match.group("value").strip()
                    elif key == "mode":
                        transcript.mode = field_match.group("value").strip()
            continue

        metadata_match = _METADATA_RE.match(raw_line)
        if metadata_match:
            _coerce(
                current,
                metadata_match.group("key"),
                metadata_match.group("value").strip(),
            )
            continue

        if raw_line.strip() or text_lines:
            text_lines.append(raw_line)

    close_current()
    return transcript


# --- Building records from a live conversation ----------------------------
# ``st.feedback("thumbs")`` returns 1 for up and 0 for down. The mapping is
# kept here, next to the strings it produces, rather than in the UI.
THUMBS_TO_FEEDBACK: dict[int, str] = {1: FEEDBACK_UP, 0: FEEDBACK_DOWN}


def records_from_history(
    history: list[dict],
    feedback: dict[int, int] | None = None,
) -> list[TurnRecord]:
    """Convert the app's message history into transcript records.

    Deliberately a **pure function**: it takes plain data and returns plain
    data, and knows nothing about Streamlit. The UI's only job is to hand over
    the history and whatever the thumbs widgets currently hold, which is what
    makes this mapping testable without starting an app - and the mapping is
    where a field would quietly go missing from an exported transcript.

    Args:
        history: One dict per message, with keys ``role`` (``"user"`` or
            ``"assistant"``), ``text``, ``time`` (a ``datetime``) and ``turn``
            (a ``BotTurn`` for assistant messages, else ``None``).
        feedback: History index -> the raw thumbs value (1 up, 0 down).

    Returns:
        One record per message, numbered from 1.
    """
    feedback = feedback or {}
    records: list[TurnRecord] = []

    for index, entry in enumerate(history):
        turn = entry.get("turn")
        record = TurnRecord(
            index=index + 1,
            time=entry["time"].strftime("%H:%M:%S"),
            role=ROLE_BOT if entry["role"] == "assistant" else ROLE_USER,
            text=entry["text"],
        )

        if turn is not None:
            record.intent = turn.intent
            # Confidence without an intent would be a meaningless 0.00, which
            # is what the safety guard produces: it never runs the classifier.
            record.confidence = turn.confidence if turn.intent else None
            record.place = turn.place.label if turn.place else None
            record.fuzzy = bool(turn.place and turn.place.fuzzy)
            record.day = turn.day.target_date.isoformat() if turn.day else None
            record.part_of_day = turn.day.part_of_day if turn.day else None
            record.source = turn.source or None
            record.guard = turn.guard_triggered
            record.feedback = THUMBS_TO_FEEDBACK.get(feedback.get(index))

        records.append(record)

    return records


def suggested_filename(exported_at: datetime) -> str:
    """Build the download file name.

    Timestamped so a tester who exports twice does not overwrite the first
    file, and prefixed so the files group together in a download folder.

    Args:
        exported_at: When the export happened.

    Returns:
        A file name such as ``peakcast-conversation-20261110-142305.txt``.
    """
    return f"peakcast-conversation-{exported_at.strftime('%Y%m%d-%H%M%S')}.txt"
