# Evaluation

Every number SupportSense reports comes from a script in this folder, with its data set
committed next to it. Run commands from `backend/`. Results are written to `eval/results/`.

| What is measured | Command | Data | Needs |
|---|---|---|---|
| Retrieval (Recall@k, MRR, evidence threshold) | `python -m eval.run_retrieval_eval` | `retrieval_eval_set.json` (30 answerable + 8 unanswerable) | embedding model |
| Ticket category classifier | `python -m app.ml.train` | `data/tickets/synthetic_tickets.csv` (250) | — |
| Escalation (rules vs semantic model) | `python -m app.ml.train_escalation` | synthetic set + `escalation_round1.csv` + `escalation_holdout.csv` | embedding model |
| Faithfulness judges (validated first) | `python -m eval.faithfulness` | `faithfulness_judge_set.json` (24 labelled pairs) | NLI model, Ollama |
| End-to-end answers | `python -m eval.run_answer_eval` | `answer_eval_set.json` (same 38 questions + key facts) | Ollama, NLI model |

The fast automated tests (no models, no Ollama) are separate: `run_tests.ps1` in the repo root.

## Results (qwen2.5:7b, all-MiniLM-L6-v2, CPU laptop)

### Retrieval (Phase 3), 38 questions
| Mode | Recall@5 | MRR@5 |
|---|---|---|
| Dense (FAISS) | 0.933 | 0.928 |
| Keyword (BM25) | 0.967 | 0.917 |
| **Hybrid (RRF)** | **0.983** | **0.942** |

Evidence threshold 0.35: answerable best-similarity 0.395–0.780, unanswerable 0.128–0.390, so two
unanswerable questions (Bitcoin 0.390, referral 0.375) pass the threshold and reach the LLM.

### Category classifier (Phase 6), 50 held-out tickets
- Cross-validation (macro F1): baseline 0.07, Naive Bayes 0.67, LogReg words 0.71, **LogReg words+chars 0.79**.
- Held-out test: **accuracy 0.74, macro F1 0.74** (50 tickets, so roughly ±0.12).
- Confidence ≥ 0.45 on 70% of tickets, which are 0.89 accurate; the rest (0.40 accurate) are flagged "check".

### Escalation (Phase 6)
| Method | Dev set (written from it) | Fresh holdout (24) |
|---|---|---|
| Keyword rules v2 | P 1.00 / R 1.00 | **P 0.00 / R 0.00** |
| Semantic model (embeddings + LogReg) | — | **P 0.79 / R 0.92** |
| Rules OR semantic (deployed) | — | P 0.73 / R 0.92 |

### Faithfulness judges (Phase 8), 24 labelled pairs
| Judge | Accuracy | Catches unsupported | Accepts supported |
|---|---|---|---|
| NLI, whole chunk as premise | 0.67 | 11/12 | 5/12 |
| **NLI, sentence windows (used)** | **0.92** | 11/12 | 11/12 |
| LLM judge (qwen2.5:7b) | 1.00 | 12/12 | 12/12 |

The LLM judge is the same model as the generator and its prompt was written after seeing these
pairs, so its score is optimistic; the NLI judge is the primary, independent check.

### End-to-end answers (Phase 8), 38 questions through the real pipeline
Automatic (`results/answer_eval.json`):
| Metric | Result |
|---|---|
| Answerable questions drafted (not handed off) | 29/30 (0.967) |
| Out-of-scope questions handed off | **7/7** (+ u06 Bitcoin, either allowed: handed off) |
| Mean key-fact recall / all facts present | 0.914 / 0.897 |
| Outdated (2024 refund) facts in answers | 0 |
| Answers citing a correct document / citation precision | 1.00 / 0.948 |
| Cited sentences supported: NLI judge / LLM judge | 0.62 / 0.78 (judges agree on 84%) |
| LLM latency mean / median / max | 4.2 s / 4.1 s / 7.7 s |

Manual review of every flagged and uncited sentence (`results/manual_review.md`), which
matters more than the automatic numbers above:
- **24 of 29 drafts (83%) contain only statements the docs support.** 5 contain an unsupported
  sentence; **2 would mislead a customer** (a24: says a 2018 TV is supported when the docs say
  2020+; a10: invents a refund menu that does not exist).
- 3 of the 6 bad sentences are one pattern: the "sign out and back in" tip applied to unrelated
  problems. Recorded as a known issue, not fixed by re-tuning on this set.
- The key-fact grader has blind spots: it passed a24 because "2020" appeared, and failed two
  correct phrasings (a12 "processed by Apple", a22 "turning off").
- Judges vs. manual: NLI caught 5/5 wrong cited sentences with 11 false alarms; the LLM judge
  caught 4/5 with 4 false alarms. **Neither checks uncited sentences** (30 of 80), which is where
  a10's invented step was.

Conclusion: the pipeline is reliable at *not answering* what the docs don't cover and at
avoiding outdated policy, but about 1 in 6 drafts still contains a claim an agent must catch.
Human review is required, not optional.

## Honest limitations
- One person wrote the documents, the questions, the tickets and the labels. Real customer text is
  messier, so treat every number as an upper bound on real-world quality.
- Sets are small (24–50 items per test set); differences of a few points are noise.
- Key-fact matching checks that specific facts appear, not that the whole answer is correct or
  well written. The faithfulness judges cover a different failure (claims the source does not make),
  and both judges make mistakes.
