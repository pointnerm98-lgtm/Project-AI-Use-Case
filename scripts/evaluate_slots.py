"""Evaluate place and day extraction, in the style of the intent metrics.

Run from the project root:

    python scripts/evaluate_slots.py

What it does:

1. Scores ``data/slots_test.csv`` - a hand-written set with one expected place
   and one expected day per sentence - and reports accuracy **per slot type**,
   because the two slots fail for completely different reasons and a single
   combined number would hide that.
2. Writes ``reports/metrics_slots.json`` and lists every error.
3. Runs the extractor over ``data/intents_train.csv`` and
   ``data/intents_test.csv`` and writes ``reports/slot_extraction_check.csv``,
   so the 400 sentences written for the *intent* classifier double as a free
   false-positive sweep for the *slot* extractor.

The reference date
------------------
Day words are relative, so the expected offsets in ``slots_test.csv`` are
stored relative to one fixed reference date, ``REFERENCE_DATE`` below (a
Wednesday). Using the real "today" instead would make the expected values go
stale overnight and the reported accuracy unreproducible - the same reason the
project pins a random seed.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.intent_classifier import REPORTS_DIR, TEST_CSV, TRAIN_CSV
from peakcast.slots import DATA_DIR, extract_slots, find_day, find_place

SLOTS_TEST_CSV: Path = DATA_DIR / "slots_test.csv"
METRICS_JSON: Path = REPORTS_DIR / "metrics_slots.json"
CHECK_CSV: Path = REPORTS_DIR / "slot_extraction_check.csv"

# A Wednesday. Fixed so that "on Friday" is always +2 and "next Friday" always
# +9, whatever day the script is actually run on.
REFERENCE_DATE: date = date(2026, 11, 4)


def evaluate_test_set(csv_path: Path = SLOTS_TEST_CSV) -> dict[str, Any]:
    """Score the hand-written slot test set.

    Args:
        csv_path: The test CSV.

    Returns:
        A metrics dict, including the list of individual errors.
    """
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    place_correct = 0
    day_correct = 0
    part_correct = 0
    both_correct = 0
    place_errors: list[dict[str, str]] = []
    day_errors: list[dict[str, str]] = []

    for row in rows:
        sentence = row["sentence"]
        expected_place = row["expected_place"].strip()
        expected_offset = int(row["expected_day"])
        expected_part = row["expected_part"].strip() or None

        place = find_place(sentence)
        day = find_day(sentence, reference=REFERENCE_DATE)

        found_place = place.name if place else ""
        place_ok = found_place == expected_place
        day_ok = day.offset_days == expected_offset
        part_ok = day.part_of_day == expected_part

        place_correct += place_ok
        day_correct += day_ok
        part_correct += part_ok
        both_correct += place_ok and day_ok

        if not place_ok:
            place_errors.append(
                {
                    "sentence": sentence,
                    "expected": expected_place or "(none)",
                    "found": found_place or "(none)",
                    "fuzzy": str(place.fuzzy) if place else "",
                }
            )
        if not day_ok:
            day_errors.append(
                {
                    "sentence": sentence,
                    "expected_offset": str(expected_offset),
                    "found_offset": str(day.offset_days),
                    "matched": day.matched_text,
                }
            )

    total = len(rows)
    # Sentences that name no place at all: scored separately, because getting
    # these right means *not* matching anything, which is the skill the
    # ambiguous-name handling exists for.
    no_place_rows = [r for r in rows if not r["expected_place"].strip()]
    no_place_correct = sum(
        1 for r in no_place_rows if find_place(r["sentence"]) is None
    )

    return {
        "n_sentences": total,
        "place_accuracy": round(place_correct / total, 4),
        "day_accuracy": round(day_correct / total, 4),
        "part_of_day_accuracy": round(part_correct / total, 4),
        "both_slots_accuracy": round(both_correct / total, 4),
        "place_correct": place_correct,
        "day_correct": day_correct,
        "n_sentences_without_a_place": len(no_place_rows),
        "correctly_found_no_place": no_place_correct,
        "reference_date": REFERENCE_DATE.isoformat(),
        "place_errors": place_errors,
        "day_errors": day_errors,
    }


def sweep_intent_data() -> tuple[list[dict[str, str]], dict[str, int]]:
    """Run the extractor over the intent datasets to hunt false positives.

    The 400 sentences written for the intent classifier were never designed as
    slot tests, which is what makes them useful here: they contain plenty of
    ordinary English that a gazetteer matcher can trip over, and every place
    they do name is one a real user might ask about.

    Returns:
        A tuple ``(rows, summary)`` for the check CSV and the console.
    """
    rows: list[dict[str, str]] = []
    summary = {"sentences": 0, "with_place": 0, "fuzzy": 0, "with_day_word": 0}

    for source, csv_path in (("train", TRAIN_CSV), ("test", TEST_CSV)):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                sentence = record["text"]
                slots = extract_slots(sentence, reference=REFERENCE_DATE)
                summary["sentences"] += 1
                if slots.place is not None:
                    summary["with_place"] += 1
                    if slots.place.fuzzy:
                        summary["fuzzy"] += 1
                if not slots.day.is_default:
                    summary["with_day_word"] += 1

                rows.append(
                    {
                        "source": source,
                        "intent": record["intent"],
                        "sentence": sentence,
                        "place": slots.place.name if slots.place else "",
                        "state": slots.place.state if slots.place else "",
                        "feature_code": (
                            slots.place.feature_code if slots.place else ""
                        ),
                        "matched_text": (
                            slots.place.matched_text if slots.place else ""
                        ),
                        "fuzzy": str(slots.place.fuzzy) if slots.place else "",
                        "day_offset": str(slots.day.offset_days),
                        "day_matched": slots.day.matched_text,
                        "part_of_day": slots.day.part_of_day or "",
                        "day_is_default": str(slots.day.is_default),
                    }
                )

    return rows, summary


def main() -> None:
    """Score the slot test set and sweep the intent data."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. The hand-written slot test set
    metrics = evaluate_test_set()

    print("--- Slot extraction on data/slots_test.csv -----------------------")
    print(f"  sentences                 {metrics['n_sentences']:>6}")
    print(f"  reference date            {metrics['reference_date']:>12}")
    print()
    print(f"  place accuracy            {metrics['place_accuracy']:>6.3f}"
          f"   ({metrics['place_correct']}/{metrics['n_sentences']})")
    print(f"  day accuracy              {metrics['day_accuracy']:>6.3f}"
          f"   ({metrics['day_correct']}/{metrics['n_sentences']})")
    print(f"  part-of-day accuracy      {metrics['part_of_day_accuracy']:>6.3f}")
    print(f"  both slots correct        {metrics['both_slots_accuracy']:>6.3f}")
    print()
    print(f"  sentences naming no place {metrics['n_sentences_without_a_place']:>6}"
          f"   correctly left empty: {metrics['correctly_found_no_place']}")

    if metrics["place_errors"]:
        print()
        print(f"  place errors ({len(metrics['place_errors'])}):")
        for error in metrics["place_errors"]:
            print(f"    {error['sentence']}")
            print(f"       expected {error['expected']!r}, found {error['found']!r}")
    if metrics["day_errors"]:
        print()
        print(f"  day errors ({len(metrics['day_errors'])}):")
        for error in metrics["day_errors"]:
            print(f"    {error['sentence']}")
            print(
                f"       expected offset {error['expected_offset']}, "
                f"found {error['found_offset']} (matched {error['matched']!r})"
            )

    METRICS_JSON.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"  metrics written to {METRICS_JSON}")

    # --- 2. The sweep over the intent datasets
    rows, summary = sweep_intent_data()
    with CHECK_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("--- Sweep over the intent datasets -------------------------------")
    print(f"  sentences                 {summary['sentences']:>6}")
    print(f"  a place was found in      {summary['with_place']:>6}"
          f"   ({summary['with_place'] / summary['sentences']:.1%})")
    print(f"    of those, fuzzy matches {summary['fuzzy']:>6}")
    print(f"  a day word was found in   {summary['with_day_word']:>6}"
          f"   ({summary['with_day_word'] / summary['sentences']:.1%})")
    print(f"  written to {CHECK_CSV}")


if __name__ == "__main__":
    main()
