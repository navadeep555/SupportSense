FICTIONAL DATA NOTICE
=====================
Every document in this folder describes "StreamNest", a made-up video-streaming
subscription service invented for the SupportSense learning project. Company name,
plans, prices, error codes and policies are all fictional.

Format of each .md file:
  - A front-matter block between two '---' lines with: doc_id, title, category,
    last_updated (YYYY-MM-DD) and optional superseded_by (doc_id of a newer doc).
  - '## Heading' lines start sections. Each section becomes one or more chunks.

refund-policy-2024.md is DELIBERATELY outdated (superseded by refund-policy-2026.md)
so we can test how the system handles conflicting evidence.
Topics deliberately NOT covered (to test "insufficient evidence"): gift cards,
student discounts, physical merchandise, careers, offline TV broadcast schedules.
(This file ends in .txt so the loader, which reads only *.md, ignores it.)
