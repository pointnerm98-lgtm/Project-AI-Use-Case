"""Terminal chat with the full PeakCast bot.

Run from the project root:

    python scripts/chat_cli.py                 # live forecasts
    python scripts/chat_cli.py --offline       # fictional demo dataset only
    python scripts/chat_cli.py --debug         # show intent, slots and source
    python scripts/chat_cli.py --seed 7        # fixed phrasing, for a screenshot

This is the whole bot: safety guard, intent classifier, slot extraction,
dialogue memory, weather service and response generator. It exists to test
complete conversations - the follow-ups and the ask-backs - which no
single-message script can exercise.

On a fresh checkout the intent model does not exist (``models/*.joblib`` is
git-ignored, being a build artefact). The bot trains it on first start and
says so, which takes about a second.

Leave with ``quit``, ``exit`` or Ctrl+C.
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.dialogue import MAX_ANSWERS, Bot, BotTurn, DialogueState, start_conversation

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})

WRAP_WIDTH: int = 76

# Shown at startup. One of each behaviour worth demonstrating: a plain
# question, a follow-up on the day, a follow-up on the place, and the guard.
EXAMPLES: tuple[str, ...] = (
    "How windy is it on the Hochkönig tomorrow?",
    "And tomorrow?",
    "What about Kaprun?",
    "What is the avalanche risk there?",
)


def say(speaker: str, text: str) -> None:
    """Print one wrapped, labelled line of dialogue.

    Args:
        speaker: ``"You"`` or ``"PeakCast"``.
        text: What was said.
    """
    prefix = f"{speaker}: "
    indent = " " * len(prefix)
    wrapped = textwrap.fill(
        text, width=WRAP_WIDTH, initial_indent=prefix, subsequent_indent=indent
    )
    print(wrapped)


def show_debug(turn: BotTurn, state: DialogueState) -> None:
    """Print what the bot decided, for a screenshot or for debugging.

    Args:
        turn: The reply and its reasoning.
        state: The conversation state after the turn.
    """
    if turn.guard_triggered:
        details = "SAFETY GUARD - never reached the classifier"
    else:
        details = f"intent={turn.intent} p={turn.confidence:.2f}"
        if turn.place is not None:
            details += f" | place={turn.place.label}"
            if turn.place.fuzzy:
                details += " (assumed)"
        if turn.day is not None:
            details += f" | day={turn.day.target_date}"
            if turn.day.part_of_day:
                details += f" {turn.day.part_of_day}"
        if turn.source:
            details += f" | forecast={turn.source}"
    print(f"        [{details} | answers={state.answers_given}/{MAX_ANSWERS}]")


def main() -> None:
    """Run the chat loop."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never call the weather API; use the fictional demo dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show the intent, confidence, slots and forecast source per turn",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the phrasing variants, so a conversation is reproducible",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else None
    bot = Bot(allow_live=not args.offline, rng=rng)
    state = start_conversation()

    print()
    print("PeakCast - weather chatbot for alpine sports in Austria")
    print("=" * WRAP_WIDTH)
    mode = "offline demo data" if args.offline else "live forecasts"
    print(f"  mode: {mode}   |   the chat ends after {MAX_ANSWERS} answers "
          "or when you say goodbye")
    print("  try:")
    for example in EXAMPLES:
        print(f"    {example}")
    print("  leave with 'quit'")
    print("=" * WRAP_WIDTH)
    print()

    while not state.finished:
        # When the input is piped rather than typed, the terminal never shows
        # what was "said", so a captured transcript would be half a
        # conversation. Printing the line after reading it keeps scripted runs
        # readable - which is how the example conversations in the report are
        # produced - and avoids leaving a dangling "You: " prompt behind when
        # the piped input runs out.
        piped = not sys.stdin.isatty()

        try:
            message = input("" if piped else "You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if piped:
            print(f"You: {message}")

        if not message:
            continue
        if message.lower() in EXIT_COMMANDS:
            break

        turn = bot.respond(state, message)
        print()
        say("PeakCast", turn.text)
        if args.debug:
            show_debug(turn, state)
        print()

    if state.finished:
        print("(conversation closed)")
    else:
        print("bye")


if __name__ == "__main__":
    main()
