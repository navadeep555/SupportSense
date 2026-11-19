// One ticket: the customer's message and triage on the left, AI drafts on the right.
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { ReviewRequest } from '../api/types'
import { useAsync } from '../api/useAsync'
import { CategoryBadge, EscalationBadge, StatusBadge } from '../components/Badges'
import { DraftPanel } from '../components/DraftPanel'
import { formatDate, humanize, percent } from '../format'

export function TicketPage() {
  const id = Number(useParams().id)
  const { data: ticket, error, loading, reload } = useAsync(() => api.getTicket(id), [id])
  const [generating, setGenerating] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  if (loading && !ticket) return <p className="muted">Loading…</p>
  if (error) return <p className="error" role="alert">{error}</p>
  if (!ticket) return null

  const drafts = [...ticket.drafts].reverse() // newest first
  const latest = drafts[0]
  const canDraft = ticket.status !== 'resolved' && !(latest && !latest.review_decision)

  async function generate() {
    setGenerating(true)
    setActionError(null)
    try {
      await api.createDraft(id)
      reload()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setGenerating(false)
    }
  }

  async function review(draftId: number, body: ReviewRequest) {
    await api.reviewDraft(draftId, body) // errors are shown inside the ReviewForm
    reload()
  }

  return (
    <div>
      <p><Link to="/">← Queue</Link></p>
      <div className="page-head">
        <h2>#{ticket.id} {ticket.subject}</h2>
        <StatusBadge status={ticket.status} />
        <EscalationBadge escalated={ticket.escalated} />
      </div>

      <div className="columns">
        <section className="card">
          <h3>Customer message</h3>
          <p className="muted small">
            {ticket.customer_email ?? 'no email given'} · {formatDate(ticket.created_at)}
          </p>
          <p className="message">{ticket.body}</p>
          {Object.keys(ticket.pii_found).length > 0 && (
            <p className="muted small">
              Personal data detected and hidden from the AI:{' '}
              {Object.entries(ticket.pii_found).map(([k, n]) => `${k} ×${n}`).join(', ')}
            </p>
          )}

          <h3>Triage</h3>
          <dl className="facts">
            <dt>Category</dt>
            <dd>
              <CategoryBadge category={ticket.category} confidence={ticket.category_confidence}
                             lowConfidence={ticket.category_low_confidence} />{' '}
              <span className="muted">({percent(ticket.category_confidence)} confident)</span>
            </dd>
            <dt>Escalation</dt>
            <dd>
              {ticket.escalated ? 'Yes — handle with priority' : 'No'}
              {ticket.escalation_score !== null && (
                <span className="muted"> (model score {ticket.escalation_score.toFixed(2)})</span>
              )}
            </dd>
          </dl>
          {ticket.escalation_reasons.length > 0 && (
            <ul className="warnings">
              {ticket.escalation_reasons.map((r) => (
                <li key={r}>{humanize(r)}</li>
              ))}
            </ul>
          )}
        </section>

        <div>
          {canDraft && (
            <div className="card">
              <button className="primary" disabled={generating} onClick={() => void generate()}>
                {generating ? 'Generating draft… (can take 5–15 s)' : latest ? 'Generate a new draft' : 'Generate AI draft'}
              </button>
              {actionError && <p className="error" role="alert">{actionError}</p>}
            </div>
          )}
          {drafts.length === 0 && !canDraft && <p className="muted">No drafts.</p>}
          {drafts.map((d) => (
            <DraftPanel
              key={d.id}
              draft={d}
              reviewable={d.id === latest.id && ticket.status !== 'resolved'}
              onReview={(body) => review(d.id, body)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
