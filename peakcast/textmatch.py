"""Small, dependency-free typo tolerance shared by the rule-based components.

Both the avalanche guard (``peakcast.safety_guard``) and the day extractor
(``peakcast.slots``) have to recognise a *closed* vocabulary - "avalanche",
"Lawine", "tomorrow", "Saturday" - under the typos a user produces on a phone.
User test round 1 (DEVLOG step 11) supplied the real ones:

    avlanceh, avalnche, lavine, lawiene, tomoorow, tmorrow, satruday

The measure is the **optimal string alignment distance** (Levenshtein plus
adjacent transposition, each counting as one edit). Plain Levenshtein is not
enough: a swapped pair is the most common phone typo, and Levenshtein charges
it as two edits - "satruday" -> "saturday" would cost 2 instead of 1, and
"avlanceh" -> "avalanche" 3 instead of 2.

Kept in its own module, with no imports from the rest of the package, so the
guard stays a plain function with no dependencies - the property that makes it
auditable.
"""

from __future__ import annotations


def osa_distance(a: str, b: str) -> int:
    """Return the optimal string alignment distance between two strings.

    Insertions, deletions, substitutions and transpositions of two adjacent
    characters each cost one edit.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The number of edits.

    Examples:
        >>> osa_distance("satruday", "saturday")
        1
        >>> osa_distance("avlanceh", "avalanche")
        2
    """
    if a == b:
        return 0
    rows, cols = len(a) + 1, len(b) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,         # deletion
                table[i][j - 1] + 1,         # insertion
                table[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                table[i][j] = min(table[i][j], table[i - 2][j - 2] + 1)  # swap
    return table[-1][-1]
