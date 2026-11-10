"""Summarise a folder of downloaded user-test transcripts.

Run from the project root:

    python scripts/analyze_transcripts.py                       # reads data/user_tests/
    python scripts/analyze_transcripts.py --folder path/to/dir
    python scripts/analyze_transcripts.py --json reports/user_tests.json

Reads every ``*.txt`` in the folder, parses it with
``peakcast.transcript.parse_transcript`` and prints a summary: how many
conversations and turns, which intents came up, how many turns fell below the
confidence threshold, how many ask-backs and fallbacks happened, which answers
were given a thumbs down, and how often the safety guard and the offline
fallback fired.

What this is for
----------------
The user tests produce a pile of text files. The numbers worth putting on a
slide are counts, and counting by hand invites counting what one hoped to
find. This script makes the summary reproducible from the files themselves.

It deliberately reports the **unflattering** figures alongside the rest -
thumbs down, fallbacks, ask-backs - because those are the ones that say what
to fix next, and a summary that only showed intent counts would be a
presentation rather than an evaluation.

Every quoted sentence in the output is a real one from a transcript, so the
report can be read next to the files it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.dialogue import ASK_FOR_PLACE, HELP_REPLY, OUT_OF_SCOPE_REPLY
from peakcast.intent_classifier import CONFIDENCE_THRESHOLD, REPORTS_DIR
from peakcast.slots import DATA_DIR
from peakcast.transcript import (
    FEEDBACK_DOWN,
    FEEDBACK_UP,
    ROLE_BOT,
    ROLE_USER,
    Transcript,
    TurnRecord,
    parse_transcript,
)

# Where testers' files are expected to be collected. Git-ignored: these are
# other people's conversations, and they do not belong in the repository.
DEFAULT_FOLDER: Path = DATA_DIR / "user_tests"

# How many example sentences to print per category. Enough to be concrete,
# few enough to read.
EXAMPLES_SHOWN: int = 5


def load_transcripts(folder: Path) -> list[Transcript]:
    """Read and parse every transcript in a folder.

    Args:
        folder: Directory holding the ``.txt`` files.

    Returns:
        One parsed transcript per file, sorted by file name.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"No such folder: {folder}\n"
            "Create it and put the transcripts testers sent you in it, or "
            "pass --folder."
        )

    transcripts: list[Transcript] = []
    for path in sorted(folder.glob("*.txt")):
        transcripts.append(
            parse_transcript(path.read_text(encoding="utf-8"), source_name=path.name)
        )
    return transcripts


def question_before(transcript: Transcript, turn: TurnRecord) -> str:
    """Find the user message a bot turn was answering.

    A bot turn on its own is not much use in a report: "the bot asked which
    place" matters far less than *which question* made it ask. The user turn
    is the one immediately before it.

    Args:
        transcript: The conversation the turn belongs to.
        turn: The bot turn.

    Returns:
        The preceding user message, or ``""`` if there is none.
    """
    previous = [
        candidate
        for candidate in transcript.turns
        if candidate.index < turn.index and candidate.role == ROLE_USER
    ]
    return previous[-1].text if previous else ""


def analyse(transcripts: list[Transcript]) -> dict[str, Any]:
    """Compute the summary figures.

    Args:
        transcripts: Parsed transcripts.

    Returns:
        A dict of counts and example lists, also used for the JSON export.
    """
    intents: Counter[str] = Counter()
    places: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    feedback: Counter[str] = Counter()

    low_confidence: list[dict[str, Any]] = []
    ask_backs: list[dict[str, str]] = []
    fallbacks: list[dict[str, str]] = []
    out_of_scope: list[dict[str, str]] = []
    guard_turns: list[dict[str, str]] = []
    thumbs_down: list[dict[str, Any]] = []
    assumed_places: list[dict[str, str]] = []

    user_turn_count = 0
    bot_turn_count = 0

    for transcript in transcripts:
        for turn in transcript.turns:
            if turn.role == ROLE_USER:
                user_turn_count += 1
                continue

            bot_turn_count += 1
            context = {
                "file": transcript.source_name,
                "question": question_before(transcript, turn),
                "answer": turn.text,
            }

            if turn.guard:
                guard_turns.append(
                    {"file": context["file"], "question": context["question"]}
                )
            if turn.intent:
                intents[turn.intent] += 1
            if turn.place:
                places[turn.place] += 1
            if turn.source:
                sources[turn.source] += 1
            if turn.feedback:
                feedback[turn.feedback] += 1

            # Below the fallback threshold the bot was guessing. These are
            # the turns most worth reading by hand.
            if turn.confidence is not None and turn.confidence < CONFIDENCE_THRESHOLD:
                low_confidence.append({**context, "confidence": turn.confidence,
                                       "intent": turn.intent})

            # The three canned replies are matched by their exact text, which
            # is why they are imported from dialogue.py rather than retyped:
            # if one of them is reworded, this script follows automatically.
            if turn.text.strip() == ASK_FOR_PLACE:
                ask_backs.append(
                    {"file": context["file"], "question": context["question"]}
                )
            if turn.text.strip() == HELP_REPLY:
                fallbacks.append(
                    {"file": context["file"], "question": context["question"]}
                )
            if turn.text.strip() == OUT_OF_SCOPE_REPLY:
                out_of_scope.append(
                    {"file": context["file"], "question": context["question"]}
                )

            if turn.feedback == FEEDBACK_DOWN:
                thumbs_down.append({**context, "intent": turn.intent,
                                    "confidence": turn.confidence})
            if turn.fuzzy and turn.place:
                assumed_places.append(
                    {"question": context["question"], "place": turn.place}
                )

    rated = feedback[FEEDBACK_UP] + feedback[FEEDBACK_DOWN]

    return {
        "conversations": len(transcripts),
        "user_turns": user_turn_count,
        "bot_turns": bot_turn_count,
        "intents": dict(intents.most_common()),
        "places": dict(places.most_common()),
        "sources": dict(sources),
        "feedback": {
            "up": feedback[FEEDBACK_UP],
            "down": feedback[FEEDBACK_DOWN],
            "rated": rated,
            # Share of *rated* answers, not of all answers: the control is
            # optional, so most turns carry no rating and dividing by every
            # answer would understate the result.
            "satisfaction_of_rated": (
                round(feedback[FEEDBACK_UP] / rated, 3) if rated else None
            ),
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "low_confidence": low_confidence,
        "ask_backs": ask_backs,
        "fallbacks": fallbacks,
        "out_of_scope": out_of_scope,
        "guard_turns": guard_turns,
        "thumbs_down": thumbs_down,
        "assumed_places": assumed_places,
    }


def print_report(summary: dict[str, Any], folder: Path) -> None:
    """Print the summary to the console.

    Args:
        summary: Output of ``analyse``.
        folder: Where the transcripts were read from.
    """
    print()
    print("=" * 72)
    print(f"  PeakCast user-test summary - {folder}")
    print("=" * 72)
    print(f"  conversations        {summary['conversations']:>5}")
    print(f"  questions asked      {summary['user_turns']:>5}")
    print(f"  answers given        {summary['bot_turns']:>5}")

    print()
    print("  Intents recognised")
    if summary["intents"]:
        total = sum(summary["intents"].values())
        for intent, count in summary["intents"].items():
            share = count / total
            print(f"    {intent:<20} {count:>4}   {share:>6.1%}")
    else:
        print("    (none - no answer carried an intent)")

    print()
    print("  Forecast source")
    for source, count in summary["sources"].items() or {}.items():
        print(f"    {source:<20} {count:>4}")
    if not summary["sources"]:
        print("    (none)")

    feedback = summary["feedback"]
    print()
    print("  Feedback (optional, so most answers carry none)")
    print(f"    thumbs up            {feedback['up']:>4}")
    print(f"    thumbs down          {feedback['down']:>4}")
    if feedback["satisfaction_of_rated"] is not None:
        print(
            f"    of the {feedback['rated']} rated answers, "
            f"{feedback['satisfaction_of_rated']:.0%} were positive"
        )

    print()
    print("  Where the bot struggled")
    struggles = [
        (
            f"below the {summary['confidence_threshold']} threshold",
            len(summary["low_confidence"]),
        ),
        ("asked which place", len(summary["ask_backs"])),
        ("did not understand", len(summary["fallbacks"])),
        ("refused as out of scope", len(summary["out_of_scope"])),
        ("safety guard fired", len(summary["guard_turns"])),
        ("place was an assumption", len(summary["assumed_places"])),
    ]
    width = max(len(label) for label, _ in struggles)
    for label, count in struggles:
        print(f"    {label:<{width}}  {count:>4}")

    sections: list[tuple[str, list[dict[str, Any]], str]] = [
        ("Thumbs down", summary["thumbs_down"], "these are the ones to read first"),
        ("Not understood", summary["fallbacks"], "candidates for new training data"),
        ("Asked which place", summary["ask_backs"], "the place was missing or unknown"),
        (
            "Low confidence",
            summary["low_confidence"],
            "answered anyway, so worth checking they were right",
        ),
    ]

    for title, rows, note in sections:
        if not rows:
            continue
        print()
        print(f"  {title} ({len(rows)}) - {note}:")
        for row in rows[:EXAMPLES_SHOWN]:
            question = row.get("question") or "(no question found)"
            extra = ""
            if row.get("confidence") is not None:
                extra = f"   [{row.get('intent')} p={row['confidence']:.2f}]"
            print(f"    - {question!r}{extra}")
        if len(rows) > EXAMPLES_SHOWN:
            print(f"    ... and {len(rows) - EXAMPLES_SHOWN} more")

    print()
    print("=" * 72)


def main() -> None:
    """Read the transcripts and print (optionally write) the summary."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_FOLDER,
        help=f"folder of downloaded transcripts (default: {DEFAULT_FOLDER})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the summary as JSON to this path",
    )
    args = parser.parse_args()

    transcripts = load_transcripts(args.folder)
    if not transcripts:
        print(f"No *.txt transcripts found in {args.folder}.")
        return

    summary = analyse(transcripts)
    print_report(summary, args.folder)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  summary written to {args.json}")
        print()
    else:
        print(f"  (pass --json {REPORTS_DIR / 'user_tests.json'} to save it)")
        print()


if __name__ == "__main__":
    main()
