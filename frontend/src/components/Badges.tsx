import type { Category, DraftStatus, TicketStatus } from '../api/types'
import { CATEGORY_LABELS, humanize } from '../format'

export function StatusBadge({ status }: { status: TicketStatus | DraftStatus }) {
  return <span className={`badge status-${status}`}>{humanize(status)}</span>
}

export function EscalationBadge({ escalated }: { escalated: boolean }) {
  return escalated ? <span className="badge escalated">Escalated</span> : null
}

export function CategoryBadge(props: {
  category: Category | null
  confidence: number | null
  lowConfidence: boolean
}) {
  if (!props.category) return <span className="muted">uncategorised</span>
  return (
    <span className="category" title={`Classifier confidence ${Math.round((props.confidence ?? 0) * 100)}%`}>
      {CATEGORY_LABELS[props.category]}
      {props.lowConfidence && <span className="badge check">check</span>}
    </span>
  )
}
