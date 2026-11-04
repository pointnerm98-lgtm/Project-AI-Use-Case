"""Interactive command-line test for the weather service.

Run from the project root:

    python scripts/try_weather.py              # live, with offline fallback
    python scripts/try_weather.py --offline    # force the demo dataset

Type a place and a day the way a user would - "Grossglockner tomorrow",
"Innsbruck this afternoon", "Zell am See on Friday" - and the script shows the
slots that were extracted and the resulting ``WeatherReport``.

It deliberately runs the **whole chain** rather than taking a place name
directly: the safety guard first, then place and day extraction, then the
forecast. That makes it useful for the presentation, because one screenshot
shows every component doing its job, including the guard refusing an avalanche
question and the source line saying whether the numbers are real.

Leave with ``quit``, ``exit`` or Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.safety_guard import check_message
from peakcast.slots import extract_slots
from peakcast.weather import (
    SOURCE_OFFLINE,
    ForecastOutOfRangeError,
    WeatherReport,
    get_forecast,
)

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q", ""})

# A few examples printed at startup, chosen to show one of each behaviour:
# a valley town, a high summit, a part of day, a curated alias, a day beyond
# the horizon, and the safety guard.
EXAMPLES: tuple[str, ...] = (
    "Innsbruck today",
    "Grossglockner tomorrow",
    "Zell am See this afternoon",
    "the Glockner on Friday",
    "Soelden next Friday",
    "avalanche risk on the Dachstein",
)

BAR_WIDTH: int = 68


def rule(character: str = "-") -> str:
    """Return a horizontal rule for the output.

    Args:
        character: Character to repeat.

    Returns:
        The rule string.
    """
    return character * BAR_WIDTH


def format_report(report: WeatherReport) -> str:
    """Lay the report out as an aligned block for the terminal.

    Written for a screenshot: the units are spelled out, the sky category sits
    next to the raw cloud cover it came from, and the source line is
    unmissable, because a fictional forecast must never read as a real one.

    Args:
        report: The report to render.

    Returns:
        The formatted block.
    """
    when = report.target_date.strftime("%a %d %b %Y")
    if report.part_of_day:
        when += f", {report.part_of_day}"

    place = report.place_name
    if report.state:
        place += f", {report.state}"
    if report.elevation is not None:
        place += f"  ({report.elevation} m)"

    lines = [
        rule("="),
        f"  {place}",
        f"  {when}",
        rule(),
        f"  temperature      {report.temperature_min:>6.1f} .. "
        f"{report.temperature_max:>5.1f} °C",
        f"  sky              {report.sky:>14}   (cloud cover "
        f"{report.cloud_cover:.0f} %, WMO {report.weather_code})",
        f"  wind             {report.wind_speed_max:>6.1f} km/h"
        f"   gusts {report.wind_gusts_max:.1f} km/h",
        f"  rain             {report.rain_sum:>6.1f} mm"
        f"      probability {report.precipitation_probability_max} %",
        f"  snowfall         {report.snowfall_sum:>6.1f} cm",
        rule(),
    ]

    if report.is_offline:
        lines.append("  SOURCE: offline demo dataset - these values are FICTIONAL")
        lines.append("          (no live forecast was available)")
    else:
        model = ""
        if report.model_elevation is not None:
            model = f"  (model elevation {report.model_elevation:.0f} m)"
        lines.append(f"  source: live forecast from Open-Meteo{model}")

    lines.append(rule("="))
    return "\n".join(lines)


def handle(message: str, allow_live: bool) -> None:
    """Run one message through guard, slots and weather, and print the result.

    Args:
        message: What the user typed.
        allow_live: Whether to attempt a live request.
    """
    # 1. The safety guard runs first, before anything else looks at the text.
    #    The state is passed in only to refine the referral, so the guard is
    #    asked twice here: once without a place (to decide) and, if it fires,
    #    once with the extracted state (to add the Tyrol bulletin).
    if check_message(message) is not None:
        slots = extract_slots(message)
        state = slots.place.state if slots.place else None
        print()
        print(rule("="))
        print("  SAFETY GUARD - the message never reached the classifier")
        print(rule())
        for line in _wrap(check_message(message, state)):
            print(f"  {line}")
        print(rule("="))
        return

    # 2. Slot extraction.
    slots = extract_slots(message)

    print()
    print(f"  slots:  place = {slots.place.label if slots.place else '(none)'}", end="")
    if slots.place and slots.place.fuzzy:
        print("  [assumed - would be confirmed back to the user]", end="")
    print()
    day_text = slots.day.matched_text or "(none, defaulting to today)"
    print(f"          day   = {slots.day.target_date} from {day_text!r}", end="")
    if slots.day.part_of_day:
        print(f", part = {slots.day.part_of_day}", end="")
    print()

    if slots.place is None:
        print()
        print("  No place recognised. The dialogue manager would reuse the last")
        print("  place from the conversation, or ask which place is meant.")
        return

    # 3. The forecast.
    try:
        report = get_forecast(slots.place, slots.day, allow_live=allow_live)
    except ForecastOutOfRangeError as error:
        print()
        print("  Outside the forecast range - the bot would say so rather than")
        print(f"  guess: {error}")
        return

    print()
    print(format_report(report))


def _wrap(text: str, width: int = BAR_WIDTH - 4) -> list[str]:
    """Wrap text to the output width.

    Args:
        text: Text to wrap.
        width: Maximum line length.

    Returns:
        The wrapped lines.
    """
    import textwrap

    return textwrap.wrap(text, width=width)


def main() -> None:
    """Read messages from the terminal until the user quits."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the network and always use the fictional demo dataset",
    )
    args = parser.parse_args()
    allow_live = not args.offline

    print()
    print("PeakCast - weather service test")
    print(rule("="))
    print(
        "  mode: "
        + ("OFFLINE (fictional demo data only)" if args.offline else "live, "
           "falling back to the demo dataset if the API is unreachable")
    )
    print("  try: " + " | ".join(EXAMPLES[:3]))
    print("       " + " | ".join(EXAMPLES[3:]))
    print("  leave with 'quit'")
    print(rule("="))

    while True:
        try:
            message = input("\nplace and day > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if message.lower() in EXIT_COMMANDS:
            break

        handle(message, allow_live=allow_live)

    print("bye")


if __name__ == "__main__":
    main()
