"""How much did the agent change the AI draft before sending it?

edit_ratio = word-level Levenshtein distance / length of the longer text
  0.0  -> sent exactly as drafted
  0.2  -> about 1 word in 5 inserted, deleted or replaced
  1.0  -> completely rewritten

Word level (not character level) because "refund" -> "refunds" is one small
edit to a human, and it is ~25x cheaper: a 150-word reply is 150x150 cells
instead of ~900x900.
"""
from app.services.citations import INLINE_CITATION_RE


def strip_citation_markers(text: str) -> str:
    """[S1] markers are for the agent, not the customer; remove them before sending."""
    cleaned = INLINE_CITATION_RE.sub("", text)
    return " ".join(cleaned.split()).replace(" .", ".").replace(" ,", ",")


def word_edit_distance(a: list[str], b: list[str]) -> int:
    """Classic dynamic programming, keeping only one previous row (O(len(b)) memory).

    prev[j] = cost to turn the first i-1 words of `a` into the first j words of `b`.
    """
    prev = list(range(len(b) + 1))
    for i, word_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, word_b in enumerate(b, start=1):
            curr[j] = min(prev[j] + 1,                        # delete word_a
                          curr[j - 1] + 1,                    # insert word_b
                          prev[j - 1] + (word_a != word_b))   # keep or replace
        prev = curr
    return prev[-1]


def edit_ratio(draft: str, final: str) -> float:
    """Compare what the AI wrote with what was sent, ignoring citation markers and case."""
    a = strip_citation_markers(draft).lower().split()
    b = strip_citation_markers(final).lower().split()
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return round(word_edit_distance(a, b) / longest, 4)
