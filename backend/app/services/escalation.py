"""Escalation flag: should a senior agent look at this ticket first?

Rules, not machine learning, on purpose:
  - Every flag comes with a human-readable reason ("financial_dispute: 'charged twice'").
  - Missing a hacked account or a legal threat is costly, and with only ~24
    escalation examples a trained model would be unreliable.
  - Rules are easy to audit and extend; the trade-off is that they only catch
    phrasings someone thought of (measured in eval/run_escalation_eval.py).

Escalation does not block the AI draft; it raises priority and warns the agent.
"""
import re
from dataclasses import dataclass, field

# reason -> patterns. \b = word boundary, so "sue" does not match "issue".
# Revision 2: revision 1 scored 100% on the tickets it was written from but only
# precision 0.44 / recall 0.33 on new ones (eval/escalation_round1.csv), so the
# rules now cover more phrasings and require context around risky keywords.
_N = r"(?:twice|two times|three times|four times|\d+ times|again)"          # repetition words
_CONTACT = r"(?:writing|contacting|asking|reporting|emailing|messaging|calling|raising)"

RULES: dict[str, list[str]] = {
    "account_security": [
        r"\b(?:was|been|got|am|is|were) hacked\b|\bhacker",            # not "hacking away at the remote"
        r"\bcompromised\b",
        r"\bunauthori[sz]ed (?:access|login|log ?in|sign ?in|purchase|upgrade)",
        r"\b(?:that|which|who) (?:are|is|was|were) not me\b",
        r"\bi (?:didn'?t|did not) do (?:it|this|that)\b",
        r"\b(?:someone|somebody)\b.{0,40}\b(?:changed|using|watching|logged|bought|purchased|upgraded|reset|accessed|signed)\b",
        r"\bsomeone else'?s profile\b",
        r"\blogins? from (?:another|a different|an unknown)",
        r"\b(?:bought|purchased|upgraded|changed|logged in|signed in)\b.{0,40}\bwithout (?:my )?(?:permission|asking|consent|knowledge)",
    ],
    "financial_dispute": [
        rf"\b(?:charged|billed|took money|deducted)\b.{{0,40}}\b{_N}",   # "billed again after cancelling"
        r"\bcharged (?:me )?(?:double|after i cancel)",
        r"\bdouble charge",
        r"\b(?:\d+|two|three|multiple) payments to\b",
        r"\bcharge\b.*\b(?:don'?t|do not) recogni[sz]e\b",
        r"\b(?:payment|charge|transaction)s? i (?:never|didn'?t|did not) (?:make|made|authori[sz]e)",
        r"\bunauthori[sz]ed (?:charge|payment|transaction)",
        r"\bchargeback\b|\bdispute (?:this|the) (?:charge|transaction|payment)\b",
        r"\b(?:this is|that'?s|it'?s|is) fraud\b|\bfraudulent\b",     # an accusation, not "fraud warning feature"
        r"\bmoney (?:got |was |has been )?deducted\b",
    ],
    "legal_or_regulatory": [
        r"\blawyer\b", r"\blegal action\b", r"\bsue (?:you|streamnest)\b", r"\bconsumer (?:court|forum)\b",
        r"\bgdpr\b", r"\bdata protection (?:authority|officer)\b", r"\bregulator\b", r"\bombudsman\b",
        r"\b(?:report|complain)\w*\b.{0,30}\bpolice\b",               # not "police drama"
    ],
    "repeated_contact": [
        rf"\b(?:second|third|fourth|fifth|\d+(?:st|nd|rd|th)) time (?:that )?(?:i'?m |i am |i'?ve |i have )?{_CONTACT}",
        rf"\b(?:emailed|contacted|messaged|called|written to|asked|reported|wrote)\b.{{0,30}}\b{_N}",
        r"\bnobody (?:has )?(?:replied|responded|answers|answered|helped)\b",
        r"\b(?:no|zero) (?:response|reply)\b(?! (?:is )?needed)",       # not "no reply needed"
        r"\bwaiting (?:for )?(?:a|one|two|\d+) (?:week|weeks|days)\b",
    ],
}
COMPILED = {reason: [re.compile(p, re.IGNORECASE) for p in patterns] for reason, patterns in RULES.items()}


@dataclass
class EscalationResult:
    escalate: bool
    reasons: list[str] = field(default_factory=list)  # e.g. ["financial_dispute: 'charged twice'"]


def detect_escalation(text: str) -> EscalationResult:
    reasons = []
    for reason, patterns in COMPILED.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                reasons.append(f"{reason}: '{match.group(0)}'")
                break  # one example per reason is enough for the agent
    return EscalationResult(bool(reasons), reasons)
