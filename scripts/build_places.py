"""Build the compact Austrian place list from the GeoNames country dump.

Run from the project root:

    python scripts/build_places.py              # download (cached) and build
    python scripts/build_places.py --refresh    # force a fresh download

What it does:

1. Downloads ``AT.zip`` from GeoNames into ``data/geonames_raw/`` (git-ignored).
2. Keeps only the feature codes a weather bot for alpine sports can be asked
   about (populated places and relevant terrain), see ``KEEP_FEATURE_CODES``.
3. Cleans the alternate-name field, resolves the federal state and elevation.
4. Writes ``data/places_at.csv`` - the compact, tracked file the bot actually
   loads - and prints a build report.

Data source: GeoNames (https://www.geonames.org), licensed CC BY 4.0.
The raw dump is **not** committed: it is 52 912 rows of which the bot needs a
fraction, and this script regenerates it on demand. The compact result *is*
committed, so the bot runs from the submitted files without a download.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

# Make the project root importable (see train_intent.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peakcast.intent_classifier import DATA_DIR
from peakcast.slots import AMBIGUOUS_ENGLISH_WORDS, PLACES_CSV, normalise_name

# --- Download --------------------------------------------------------------
GEONAMES_URL: str = "https://download.geonames.org/export/dump/AT.zip"
RAW_DIR: Path = DATA_DIR / "geonames_raw"
RAW_ZIP: Path = RAW_DIR / "AT.zip"
RAW_MEMBER: str = "AT.txt"

# GeoNames asks for a descriptive user agent on automated downloads.
USER_AGENT: str = "PeakCast/1.0 (IU student project; weather chatbot)"

# --- Column layout of the GeoNames dump ------------------------------------
# Documented in the readme.txt inside AT.zip. Named here so the parsing code
# reads as something other than a wall of magic indices.
COL_NAME: int = 1            # UTF-8 name
COL_ASCII: int = 2           # ASCII transliteration
COL_ALT_NAMES: int = 3       # comma-separated, many languages and scripts
COL_LAT: int = 4
COL_LON: int = 5
COL_FEATURE_CLASS: int = 6   # P, T, H, S, ...
COL_FEATURE_CODE: int = 7    # PPL, PK, MT, ...
COL_ADMIN1: int = 10         # federal state code
COL_POPULATION: int = 14
COL_ELEVATION: int = 15      # often empty
COL_DEM: int = 16            # digital elevation model, a usable fallback
N_COLUMNS: int = 19

# --- Which features to keep ------------------------------------------------
# Chosen by inspecting the dump's own code distribution against the place names
# that actually occur in data/intents_*.csv, not from the GeoNames code list in
# the abstract. Reasoning per group:
#
#   populated places  - the bot's bread and butter ("weather in Lienz")
#   terrain           - the alpine half of the use case: peaks, mountains,
#                       ranges, passes, saddles, ridges, hills, valleys,
#                       cirques and named ski slopes are all things a hiker,
#                       ski tourer or paraglider names as a destination
#   glaciers, lakes   - year-round destinations ("Moelltaler Gletscher",
#                       "Wolfgangsee")
#   mountain huts     - a hut is very often *the* destination of a tour, and
#                       several well-known paragliding launch sites are mapped
#                       only as huts (e.g. "Emberger Alpe")
KEEP_FEATURE_CODES: frozenset[str] = frozenset(
    {
        # --- P: populated places
        "PPLC",    # national capital
        "PPLA",    # seat of a first-order administrative division
        "PPLA2",
        "PPLA3",
        "PPLA4",
        "PPL",     # ordinary populated place
        "PPLL",    # populated locality
        "PPLX",    # section of a populated place (city districts)
        # --- T: terrain
        "PK",      # peak
        "MT",      # mountain
        "MTS",     # mountains (a range)
        "RGN",     # named region, e.g. the Raxalpe massif, ski regions
        "HLL",     # hill
        "PASS",    # pass
        "SDL",     # saddle
        "RDGE",    # ridge
        "VAL",     # valley
        "CRQ",     # cirque
        "SLP",     # slope (named ski runs)
        # --- H: water and ice
        "GLCR",    # glacier
        "LK",      # lake
        # --- S: man-made spots worth keeping
        "HUT",     # mountain hut
    }
)

# --- Deliberately excluded: the administrative divisions --------------------
# ADM1/ADM2/ADM3 look attractive because they carry the *municipality*
# population, but they are wrong for this bot in a way that matters, and the
# reason is worth recording:
#
#   Soelden         ADM3  pop 3 145  elevation 2 856 m   <- the whole municipality
#                   PPLA3 pop 2 205  elevation 1 368 m   <- the actual village
#   Zell am See     ADM2  pop 86 923 ("Politischer Bezirk Zell am See")
#                   PPLA3 pop  7 619 ("Zell am See", the town)
#
# Two separate problems. First, the ADM entry's coordinates and elevation
# describe an *area*, so a forecast fetched for them can be ~1 500 m above the
# village the user meant - a different temperature, a different snow line and
# a completely different answer. Second, because ambiguous names are resolved
# by population (see slots.py), the district would systematically outrank the
# town and win every lookup.
EXCLUDED_ADMIN_CODES: frozenset[str] = frozenset(
    {"ADM1", "ADM1H", "ADM2", "ADM2H", "ADM3", "ADM3H", "ADM4", "ADM4H", "ADMD", "PCLI"}
)

# --- Federal states --------------------------------------------------------
# GeoNames admin1 code -> German name of the Austrian federal state. The bot
# echoes this back ("Zell am See, Salzburg") so the user can tell which of
# several same-named places was used.
FEDERAL_STATES: dict[str, str] = {
    "01": "Burgenland",
    "02": "Kärnten",
    "03": "Niederösterreich",
    "04": "Oberösterreich",
    "05": "Salzburg",
    "06": "Steiermark",
    "07": "Tirol",
    "08": "Vorarlberg",
    "09": "Wien",
}

# --- Alternate-name cleaning ----------------------------------------------
# The raw field mixes genuinely useful spellings with things that would only
# add noise or false matches:
#
#   useful : "Soelden" (ASCII spelling), "Insbruk" (a plausible typo),
#            "Moelltaler Gletscher" (the name everyone actually uses for the
#            glacier GeoNames calls "Wurtenkees")
#   noise  : Cyrillic/Greek/CJK/Arabic forms, postal codes ("70623"),
#            Wikipedia URLs, single letters, and exonyms in languages this bot
#            does not serve ("Becs", "Dunaj", "Fiena" for Vienna)
#
# Candidates are kept only if Latin-script, non-numeric and of a sensible
# length, and are then **ranked by similarity to the primary name** before the
# cap is applied. The ranking matters more than it looks: the raw field is
# sorted alphabetically, so a blind "keep the first N" kept
#
#   Vienna -> Bec, Bech, Becs, Bec, Bec, Bienne
#
# and discarded "Wien" at position 40 - i.e. it threw away the one spelling an
# Austrian user is most likely to type. Similarity ranking keeps the spelling
# variants that actually matter ("Soelden", "Solden" and "Sölden" all score
# 1.00 against "Sölden" once normalised) and pushes foreign exonyms down.
MIN_ALT_NAME_LENGTH: int = 3
MAX_ALT_NAME_LENGTH: int = 60
MAX_ALT_NAMES_PER_PLACE: int = 10

# --- Names GeoNames does not supply ---------------------------------------
# Similarity ranking cannot rescue a local name that simply looks nothing like
# the primary one. "Wien" scores 0.60 against "Vienna" - below several
# Slavic transliterations - and there is no signal in the dump that
# distinguishes the German name from the Hungarian one, because the country
# file carries no language codes. Rather than download the much larger
# alternateNames table for a handful of cases, the needed spellings are listed
# explicitly here.
#
# Kept deliberately short and justified entry by entry. This is an addition of
# *spellings to existing gazetteer rows*, not of new places: no coordinates are
# invented, so an alias can never point the forecast somewhere unverified.
EXTRA_NAME_ALIASES: dict[tuple[str, str], list[str]] = {
    # GeoNames files the capital under its English exonym. "Wien" is what both
    # Austrians and the bot's own training sentences use.
    ("Vienna", "PPLC"): ["Wien"],
}

OUTPUT_COLUMNS: list[str] = [
    "name",
    "ascii_name",
    "alt_names",      # pipe-separated; "|" cannot occur in a GeoNames name
    "lat",
    "lon",
    "elevation",
    "population",
    "state",
    "feature_code",
]


def download_dump(refresh: bool = False) -> Path:
    """Fetch the GeoNames Austria dump, reusing the cached copy if present.

    Args:
        refresh: Download again even if the cached file exists.

    Returns:
        Path to the local ``AT.zip``.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_ZIP.exists() and not refresh:
        size_mb = RAW_ZIP.stat().st_size / 1_048_576
        print(f"Using cached dump: {RAW_ZIP}  ({size_mb:.1f} MB)")
        return RAW_ZIP

    print(f"Downloading {GEONAMES_URL} ...")
    request = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    RAW_ZIP.write_bytes(payload)
    print(f"  saved {len(payload) / 1_048_576:.1f} MB to {RAW_ZIP}")
    return RAW_ZIP


def is_latin_script(text: str) -> bool:
    """Report whether every letter in ``text`` is Latin script.

    Used to drop Cyrillic, Greek, Hebrew, Arabic and CJK alternate names, which
    an English-speaking user of this bot will never type.

    Args:
        text: Candidate name.

    Returns:
        ``True`` if the string contains at least one letter and all of its
        letters are Latin.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return all("LATIN" in unicodedata.name(ch, "") for ch in letters)


def name_similarity(candidate: str, name: str, ascii_name: str) -> float:
    """Score how much an alternate name looks like a spelling of the primary one.

    Both sides are put through the matcher's own normalisation first, so that
    "Soelden", "Solden" and "Sölden" score 1.00 against each other rather than
    being penalised for the very difference the bot has to see through.

    Args:
        candidate: The alternate name.
        name: Primary name.
        ascii_name: ASCII name.

    Returns:
        The best similarity ratio against either primary form, 0.0 to 1.0.
    """
    normalised_candidate = normalise_name(candidate)
    return max(
        difflib.SequenceMatcher(
            None, normalised_candidate, normalise_name(primary)
        ).ratio()
        for primary in (name, ascii_name)
    )


def clean_alt_names(
    raw_field: str, name: str, ascii_name: str, feature_code: str = ""
) -> list[str]:
    """Reduce the GeoNames alternate-name field to the usable spellings.

    Candidates are filtered (Latin script, no digits, sensible length), then
    sorted by ``name_similarity`` so that genuine spelling variants survive the
    cap and foreign exonyms do not. Any spellings listed in
    ``EXTRA_NAME_ALIASES`` for this place are prepended and never dropped.

    Args:
        raw_field: The raw comma-separated field.
        name: The primary name, excluded from the result.
        ascii_name: The ASCII name, excluded from the result.
        feature_code: Used to look the place up in ``EXTRA_NAME_ALIASES``.

    Returns:
        At most ``MAX_ALT_NAMES_PER_PLACE`` alternate names, plus any manual
        aliases.
    """
    already_known = {name.casefold(), ascii_name.casefold()}

    # Manual aliases first: they are the ones the ranking cannot find.
    kept: list[str] = []
    for alias in EXTRA_NAME_ALIASES.get((name, feature_code), []):
        if alias.casefold() not in already_known:
            already_known.add(alias.casefold())
            kept.append(alias)

    candidates: list[str] = []
    for candidate in raw_field.split(","):
        candidate = candidate.strip()

        if not (MIN_ALT_NAME_LENGTH <= len(candidate) <= MAX_ALT_NAME_LENGTH):
            continue
        # Postal codes and numeric ids are stored in this field too.
        if any(ch.isdigit() for ch in candidate):
            continue
        if candidate.startswith("http"):
            continue
        if not is_latin_script(candidate):
            continue
        if candidate.casefold() in already_known:
            continue

        already_known.add(candidate.casefold())
        candidates.append(candidate)

    candidates.sort(key=lambda c: -name_similarity(c, name, ascii_name))
    return kept + candidates[:MAX_ALT_NAMES_PER_PLACE]


def parse_dump(zip_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Read the dump and keep the rows relevant to the bot.

    Args:
        zip_path: Path to ``AT.zip``.

    Returns:
        A tuple ``(rows, stats)`` where ``rows`` are output-ready dicts and
        ``stats`` counts what happened, for the build report.
    """
    stats: dict[str, int] = {
        "total": 0,
        "kept": 0,
        "dropped_feature_code": 0,
        "state_unknown": 0,
        "dropped_duplicate": 0,
    }

    with zipfile.ZipFile(zip_path) as archive:
        raw_text = archive.read(RAW_MEMBER).decode("utf-8")

    rows: list[dict[str, object]] = []
    # Same name at the same rounded coordinate = the same real-world place
    # listed twice (GeoNames does this, e.g. Gerlitzen as both MT and PPL).
    seen: set[tuple[str, str, str]] = set()

    for line in raw_text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < N_COLUMNS:
            continue

        stats["total"] += 1

        feature_code = fields[COL_FEATURE_CODE]
        if feature_code not in KEEP_FEATURE_CODES:
            stats["dropped_feature_code"] += 1
            continue

        # GeoNames uses the admin1 code "00" (or an empty field) for features
        # that do not sit inside one single federal state. Rows like that are
        # kept with an empty state rather than dropped: the first version
        # dropped them and lost 277 rows including the Raxalpe, the Hohe
        # Tauern, the Totes Gebirge, the Raetikon and the Lechtaler Alpen -
        # i.e. several of the best-known mountain ranges in the country, and
        # one name ("the Rax") that appears in the bot's own training data.
        # The only thing lost with an empty state is the disambiguation hint
        # in the answer, and Place.label degrades to the bare name.
        state = FEDERAL_STATES.get(fields[COL_ADMIN1], "")
        if not state:
            stats["state_unknown"] += 1

        name = fields[COL_NAME].strip()
        ascii_name = fields[COL_ASCII].strip() or name

        # Round coordinates to ~100 m for the duplicate check only; the full
        # precision is kept in the output.
        key = (
            name.casefold(),
            f"{float(fields[COL_LAT]):.3f}",
            f"{float(fields[COL_LON]):.3f}",
        )
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)

        # Prefer the surveyed elevation; fall back to the DEM value, which is
        # present almost everywhere. Elevation matters for a mountain weather
        # bot, so an approximate value beats an empty one.
        elevation = fields[COL_ELEVATION].strip() or fields[COL_DEM].strip()

        rows.append(
            {
                "name": name,
                "ascii_name": ascii_name,
                "alt_names": "|".join(
                    clean_alt_names(
                        fields[COL_ALT_NAMES], name, ascii_name, feature_code
                    )
                ),
                "lat": fields[COL_LAT],
                "lon": fields[COL_LON],
                "elevation": elevation,
                "population": fields[COL_POPULATION] or "0",
                "state": state,
                "feature_code": feature_code,
            }
        )
        stats["kept"] += 1

    return rows, stats


def write_places_csv(rows: list[dict[str, object]], out_path: Path = PLACES_CSV) -> None:
    """Write the compact place list.

    Sorted by population descending, then by name: the most important places
    come first, which makes the file readable when opened by hand.

    Args:
        rows: Output-ready dicts.
        out_path: Destination CSV.
    """
    rows_sorted = sorted(
        rows, key=lambda r: (-int(str(r["population"])), str(r["name"]).casefold())
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_sorted)


def report(rows: list[dict[str, object]], stats: dict[str, int], out_path: Path) -> None:
    """Print a build report, including the ambiguous names found.

    The ambiguous-name section is the interesting one: it lists the place names
    that are also ordinary English words, i.e. the ones that would cause false
    positives if they were matched without context. The matcher's policy for
    them lives in ``peakcast.slots``.

    Args:
        rows: The rows that were written.
        stats: Counters from ``parse_dump``.
        out_path: The file that was written.
    """
    size_kb = out_path.stat().st_size / 1024
    print()
    print("--- Build report -------------------------------------------------")
    print(f"  rows in dump            {stats['total']:>7}")
    print(f"  dropped: feature code   {stats['dropped_feature_code']:>7}")
    print(f"  kept, state unknown     {stats['state_unknown']:>7}")
    print(f"  dropped: duplicate      {stats['dropped_duplicate']:>7}")
    print(f"  kept                    {stats['kept']:>7}")
    print(f"  written to              {out_path}  ({size_kb:.0f} KB)")

    by_code: dict[str, int] = {}
    for row in rows:
        code = str(row["feature_code"])
        by_code[code] = by_code.get(code, 0) + 1
    print()
    print("  kept by feature code:")
    for code, count in sorted(by_code.items(), key=lambda kv: -kv[1]):
        print(f"    {code:<6} {count:>6}")

    # Names that collide with ordinary English words.
    risky: dict[str, list[str]] = {}
    for row in rows:
        candidates = [str(row["name"]), str(row["ascii_name"])]
        candidates += [a for a in str(row["alt_names"]).split("|") if a]
        for candidate in candidates:
            if candidate.casefold() in AMBIGUOUS_ENGLISH_WORDS:
                risky.setdefault(candidate.casefold(), []).append(
                    f"{row['name']} ({row['feature_code']}, pop {row['population']})"
                )
    print()
    print(f"  ambiguous names (also English words): {len(risky)}")
    for word in sorted(risky):
        places = risky[word]
        print(f"    {word:<12} {len(places):>3} place(s)  e.g. {places[0]}")
    print("  -> these require a preposition or capitalisation to match;")
    print("     see peakcast/slots.py (AMBIGUOUS_ENGLISH_WORDS).")


def main() -> None:
    """Download, filter and write the compact place list."""
    # The report prints place names and state names containing umlauts. On a
    # Windows console the default code page cannot encode them, so force UTF-8
    # output rather than letting the script die on a print statement.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download the dump again instead of using the cached copy",
    )
    args = parser.parse_args()

    zip_path = download_dump(refresh=args.refresh)
    rows, stats = parse_dump(zip_path)
    write_places_csv(rows)
    report(rows, stats, PLACES_CSV)


if __name__ == "__main__":
    main()
