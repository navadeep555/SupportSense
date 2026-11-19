# Manual review of the end-to-end answer evaluation (qwen2.5:7b, 2026-11-19)

Automatic metrics were checked by reading every sentence that either judge flagged, every
uncited sentence, and every answer with a missed key fact, against the knowledge-base text.
Verdicts: **OK** = true per the docs, **FILLER** = no factual claim ("contact support"),
**WRONG** = not in the docs or contradicts them.

## Sentences with a factual problem (6 of 80 drafted sentences)

| Item | Sentence | Verdict | Caught by |
|---|---|---|---|
| a24 | "Your 2018 Samsung TV should be able to run StreamNest, as it supports Samsung smart TVs from 2020 onwards." | **WRONG, misleading**: the docs support 2020+ TVs, so a 2018 TV is not supported | NLI, LLM (key-fact check passed: "2020" is present) |
| a10 | "To request a refund, go to your account settings and select the option to request a refund." | **WRONG, misleading**: docs say in-app chat or email | nobody (sentence had no citation) |
| a21 | "Try signing out and signing back in to refresh your session." (for SN-2003) | WRONG: not an SN-2003 step | NLI, LLM |
| a22 | "If the problem persists, try signing out and signing back in…" (VPN) | WRONG: not in the cited source | NLI, LLM |
| a24 | "If you encounter any issues, signing out and signing back in might help." | WRONG: not in the cited source | NLI, LLM |
| a30 | "…consider enabling two-step verification." | WRONG (mild): generic advice, not in sources | NLI only |

Pattern: 3 of 6 are the "sign out and back in" tip (from the Premium document) applied to
unrelated problems. Fixing this with a prompt change and re-running this same set would tune on
the test set, so it is recorded as a known issue instead.

**Answers containing at least one wrong sentence: 5 of 29 (17%). Answers that would mislead a
customer if sent unchecked: 2 of 29 (a24, a10).** This is why every draft goes to a human.

## True statements flagged by a judge (judge errors)

| Item | Sentence (shortened) | Flagged by |
|---|---|---|
| a03 | reset during the lock works once it lifts | NLI, LLM |
| a19 | HD needs 5 Mbps, 4K needs 25 Mbps | NLI, LLM |
| a22 | turn off the VPN and restart the app (true, but the cited chunk is the FAQ, not SN-4040) | NLI, LLM |
| a27 | cancel the paid subscription before deleting (true, but cited the cancellation guide, not the privacy doc) | NLI, LLM |
| a07, a10, a14, a18, a20 ×2, a26 | grace period, refund eligibility, prorating, 10 months, SN-1001 steps, Premium downloads | NLI only |

FILLER sentences flagged by both judges: a03, a07, a10 ("contact support").

## Judges vs. manual verdict (50 cited sentences)

| | Wrong sentences caught (of 5 cited) | True sentences wrongly flagged | Filler flagged |
|---|---|---|---|
| NLI (sentence windows) | **5/5** | 11 | 3 |
| LLM (qwen2.5:7b) | 4/5 (missed a30) | 4 | 3 |

NLI is the stricter judge (catches everything, many false alarms); the LLM judge is more precise
but missed one. Neither checks uncited sentences, which is where a10's invented step hid.

## Grader errors in the key-fact check
- a12 scored 0/1, but the answer is correct ("must be processed by Apple"): the accepted
  spellings were too narrow. Manual: 1/1.
- a22 scored 0/2: "Turning off" does not contain the substring "turn off". Manual: 1/2 (the
  answer does not explain that licensing differs by country).
- a24 scored 1/1 although the conclusion is wrong (see above).
- a06 was handed off ("model reported missing information") although the docs cover duplicate
  charges: a safe but unnecessary hand-off.

## Bottom line (manual)
- 30 answerable questions: 29 drafted, 1 unnecessary hand-off (a06).
- Of the 29 drafts, **24 (83%) contain only statements supported by the docs**; 5 contain at
  least one unsupported sentence; 2 of those (a10, a24) would mislead a customer.
- 7 of 7 out-of-scope questions were handed off; u06 (Bitcoin, either outcome allowed) was too.
