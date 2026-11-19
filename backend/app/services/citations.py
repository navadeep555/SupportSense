"""Citation validation: did the model cite only sources we actually gave it?

What this CAN prove: every [S#] in the answer refers to a chunk that was
retrieved and sent in the prompt, so the model did not invent a source.

What it CANNOT prove: that the cited chunk really supports the sentence.
A model can attach [S1] to a claim S1 never makes. Checking that needs a
human reviewer or a separate faithfulness evaluation (Phase 8).
"""
import re
from dataclasses import dataclass, field

INLINE_CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass
class CitationCheck:
    valid_ids: list[str]                 # cited AND retrieved, in order of first use
    invalid_ids: list[str]               # cited but never retrieved (invented)
    cleaned_answer: str                  # answer with invalid [S#] markers removed
    warnings: list[str] = field(default_factory=list)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))  # keeps first-seen order


def validate_citations(answer: str, declared_ids: list[str], allowed_ids: set[str]) -> CitationCheck:
    inline_ids = INLINE_CITATION_RE.findall(answer)
    all_ids = _unique(inline_ids + [d.strip() for d in declared_ids])

    valid = [i for i in all_ids if i in allowed_ids]
    invalid = [i for i in all_ids if i not in allowed_ids]

    # Remove invented markers from the text so the agent never sees a broken link.
    cleaned = INLINE_CITATION_RE.sub(lambda m: m.group(0) if m.group(1) in allowed_ids else "", answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).replace(" .", ".").strip()

    warnings = []
    if invalid:
        warnings.append(f"Removed citations to sources that were not retrieved: {', '.join(invalid)}.")
    if not valid:
        warnings.append("The answer cites no retrieved source, so it cannot be traced to the knowledge base.")
    elif not set(valid) & set(inline_ids):
        warnings.append("Sources are listed but no sentence is marked with a citation.")
    return CitationCheck(valid, invalid, cleaned, warnings)
