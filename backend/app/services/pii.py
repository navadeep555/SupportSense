"""PII masking: replace personal data with placeholders before text leaves our control.

Masked text is what we send to the LLM and show in analytics reports. With a
local Ollama model nothing leaves the machine, but the same code protects us
if we switch to a cloud LLM, and keeps personal data out of reports and logs.

What it catches (regular expressions, so it is fast and predictable):
  emails, phone numbers (10-12 digits), payment card numbers (13-19 digits that
  pass the Luhn checksum), and passwords typed after "password:" / "password is".
What it does NOT catch: names, street addresses, free-form IDs. Those need a
named-entity-recognition model; this is a documented limitation, not a guarantee.
"""
import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# A run of digits that may contain spaces, dashes, dots or brackets, e.g.
# "+91 98765 43210", "(555) 123-4567", "4111 1111 1111 1111".
# The look-arounds stop it from starting/ending inside a word, so the "2003"
# in error code "SN-2003" is never touched.
NUMBER_RUN_RE = re.compile(r"(?<![\w-])[+(]?\d[\d ().-]{7,}\d(?![\w-])")

PASSWORD_RE = re.compile(r"\b(password|passcode|pwd)(\s*(?:is|:|=)\s*)\S+", re.IGNORECASE)


@dataclass
class MaskResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)  # PII type -> how many were masked


def luhn_valid(digits: str) -> bool:
    """Luhn checksum used by every card network; random 16-digit numbers fail it ~90% of the time."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:   # double every second digit from the right
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_pii(text: str) -> MaskResult:
    found: dict[str, int] = {}

    def count(kind: str) -> None:
        found[kind] = found.get(kind, 0) + 1

    def replace_number(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            count("card")
            return "[CARD]"
        if 10 <= len(digits) <= 12:
            count("phone")
            return "[PHONE]"
        return match.group(0)  # dates, short codes, order amounts: leave alone

    def replace_password(match: re.Match) -> str:
        count("password")
        return f"{match.group(1)}{match.group(2)}[PASSWORD]"

    def replace_email(_match: re.Match) -> str:
        count("email")
        return "[EMAIL]"

    # Order matters: passwords first (a password could look like an email or number).
    text = PASSWORD_RE.sub(replace_password, text)
    text = EMAIL_RE.sub(replace_email, text)
    text = NUMBER_RUN_RE.sub(replace_number, text)
    return MaskResult(text, found)
