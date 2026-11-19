# SupportSense frontend

React 19 + TypeScript + Vite single-page app for support agents:

- **Queue** (`/`): tickets, escalated first; filter by status, category, escalation (kept in the URL).
- **Ticket** (`/tickets/:id`): customer message, triage, AI drafts with clickable `[S1]` citations,
  approve / edit / reject.
- **New ticket** (`/new`): stands in for the customer channel in this demo.
- **Reports** (`/reports`): draft-quality metrics and knowledge gaps.

## Run (backend must be running on port 8000)

```bash
npm install
npm run dev        # http://localhost:5173 ; /api is proxied to http://127.0.0.1:8000
npm test           # Vitest + Testing Library
npm run build      # type-check + production build into dist/
```

## Design notes

- All backend calls go through `src/api/client.ts` with relative `/api/...` paths. The browser
  never holds a backend URL, key or secret.
- `src/api/types.ts` mirrors `backend/app/schemas.py`; keep them in sync.
- Model output and ticket text are rendered as React text, never as HTML, so injected markup
  cannot run (see `AnswerText.test.tsx`).
