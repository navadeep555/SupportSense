// Draft-quality metrics and the knowledge-gap list (both computed by the backend).
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useAsync } from '../api/useAsync'
import { formatDate, humanize, percent } from '../format'

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="stat" title={hint}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function Counts({ title, counts }: { title: string; counts: Record<string, number> }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  return (
    <div className="card">
      <h4>{title}</h4>
      {total === 0 && <p className="muted">No data yet.</p>}
      {Object.entries(counts).map(([key, n]) => (
        <div key={key} className="bar-row">
          <span>{humanize(key)}</span>
          <span className="bar"><span style={{ width: `${(n / total) * 100}%` }} /></span>
          <span>{n}</span>
        </div>
      ))}
    </div>
  )
}

export function ReportsPage() {
  const metrics = useAsync(() => api.draftMetrics(), [])
  const gaps = useAsync(() => api.knowledgeGaps(), [])
  const m = metrics.data

  return (
    <div>
      <h2>Reports</h2>
      {metrics.error && <p className="error" role="alert">{metrics.error}</p>}
      {m && (
        <>
          <div className="stats">
            <Stat label="Drafts generated" value={String(m.total_drafts)} hint="All AI drafts, any status" />
            <Stat label="Reviewed by agents" value={String(m.reviewed)} hint="Approved + edited + rejected" />
            <Stat label="Sent unchanged" value={percent(m.approval_rate)} hint="approved / reviewed" />
            <Stat label="Used (as-is or edited)" value={percent(m.acceptance_rate)} hint="(approved + edited) / reviewed" />
            <Stat label="Words changed when edited" value={percent(m.avg_edit_ratio_when_edited)}
                  hint="Average word-level edit ratio of edited drafts" />
            <Stat label="Avg AI latency"
                  value={m.avg_llm_latency_ms === null ? '—' : `${(m.avg_llm_latency_ms / 1000).toFixed(1)} s`}
                  hint="Average time of LLM calls" />
          </div>
          <div className="columns">
            <Counts title="Drafts by status" counts={m.by_status} />
            <Counts title="Agent decisions" counts={m.by_decision} />
          </div>
        </>
      )}

      <h3>Knowledge gaps</h3>
      <p className="muted small">
        Questions the documentation could not answer, and drafts agents rejected. Least-covered first;
        personal data is masked.
      </p>
      {gaps.error && <p className="error" role="alert">{gaps.error}</p>}
      {gaps.data && gaps.data.items.length === 0 && <p className="muted">No gaps found yet.</p>}
      {gaps.data && gaps.data.items.length > 0 && (
        <table className="queue">
          <thead>
            <tr>
              <th>Ticket</th>
              <th>Reason</th>
              <th>Customer question (masked)</th>
              <th>Best match</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {gaps.data.items.map((g) => (
              <tr key={g.draft_id}>
                <td><Link to={`/tickets/${g.ticket_id}`}>#{g.ticket_id}</Link></td>
                <td>{humanize(g.reason)}<div className="muted small">{g.detail}</div></td>
                <td><strong>{g.subject_masked}</strong><div className="small">{g.body_masked}</div></td>
                <td>{g.best_similarity.toFixed(2)}</td>
                <td className="muted">{formatDate(g.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
