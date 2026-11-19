import type { Draft, ReviewRequest } from '../api/types'
import { DRAFT_STATUS_HELP, formatDate, percent } from '../format'
import { AnswerText } from './AnswerText'
import { StatusBadge } from './Badges'
import { CitationList } from './CitationList'
import { ReviewForm } from './ReviewForm'

interface Props {
  draft: Draft
  reviewable: boolean // only the latest, unreviewed draft of an unresolved ticket
  onReview: (review: ReviewRequest) => Promise<void>
}

export function DraftPanel({ draft, reviewable, onReview }: Props) {
  const citedIds = new Set(draft.citations.map((c) => c.source_id))
  const anchor = `draft-${draft.id}`  // unique per draft, so two drafts on one page never share ids
  const info = draft.llm_called
    ? `${draft.llm_model} · ${((draft.latency_ms ?? 0) / 1000).toFixed(1)} s · ` +
      `${draft.input_tokens ?? '?'} in / ${draft.output_tokens ?? '?'} out tokens`
    : 'No AI call (not enough evidence in the documentation)'

  return (
    <section className="card">
      <header className="card-head">
        <h3>Draft #{draft.id}</h3>
        <StatusBadge status={draft.status} />
        <span className="muted">{formatDate(draft.created_at)}</span>
      </header>
      <p className="muted small">{DRAFT_STATUS_HELP[draft.status]}</p>

      {draft.warnings.length > 0 && (
        <ul className="warnings">
          {draft.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      {draft.answer ? <AnswerText text={draft.answer} citedIds={citedIds} anchorPrefix={anchor} /> : <p className="muted">No draft text.</p>}

      <h4>Sources</h4>
      <CitationList citations={draft.citations} anchorPrefix={anchor} />
      <p className="muted small">
        {info} · best evidence similarity {draft.best_similarity.toFixed(2)}
        {Object.keys(draft.pii_masked).length > 0 &&
          ` · hidden from AI: ${Object.entries(draft.pii_masked).map(([k, n]) => `${k} ×${n}`).join(', ')}`}
      </p>

      {draft.review_decision ? (
        <div className={`decision decision-${draft.review_decision}`}>
          <strong>Agent {draft.review_decision}</strong>
          {draft.edit_ratio !== null && ` · changed ${percent(draft.edit_ratio)} of words`}
          {draft.review_note && <p>Reason: {draft.review_note}</p>}
          {draft.final_reply && <p className="final">{draft.final_reply}</p>}
        </div>
      ) : (
        reviewable && <ReviewForm draft={draft} onSubmit={onReview} />
      )}
    </section>
  )
}
