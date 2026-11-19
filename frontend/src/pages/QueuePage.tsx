// The agent's work queue: escalated tickets first (the backend sorts), with filters.
//
// Filters live in the URL (?status=open&escalated=true), not only in React
// state, so a filtered view survives a refresh and can be shared as a link.
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { CATEGORIES } from '../api/types'
import { useAsync } from '../api/useAsync'
import { CategoryBadge, EscalationBadge, StatusBadge } from '../components/Badges'
import { CATEGORY_LABELS, formatDate } from '../format'

const PAGE_SIZE = 20

export function QueuePage() {
  const [params, setParams] = useSearchParams()
  const status = params.get('status') ?? ''
  const category = params.get('category') ?? ''
  const escalatedOnly = params.get('escalated') === 'true'
  const page = Math.max(0, Number(params.get('page') ?? 0))

  const { data, error, loading } = useAsync(
    () =>
      api.listTickets({
        status: status || undefined,
        category: category || undefined,
        escalated: escalatedOnly ? true : undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    [status, category, escalatedOnly, page],
  )

  function setFilter(key: string, value: string) {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value)
    else next.delete(key)
    next.delete('page') // a new filter starts again at page 1
    setParams(next)
  }

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1

  return (
    <div>
      <div className="page-head">
        <h2>Ticket queue</h2>
        <Link className="button primary" to="/new">New ticket</Link>
      </div>

      <div className="filters">
        <label>
          Status
          <select value={status} onChange={(e) => setFilter('status', e.target.value)}>
            <option value="">All</option>
            <option value="open">Open</option>
            <option value="drafted">Drafted</option>
            <option value="resolved">Resolved</option>
          </select>
        </label>
        <label>
          Category
          <select value={category} onChange={(e) => setFilter('category', e.target.value)}>
            <option value="">All</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{CATEGORY_LABELS[c]}</option>
            ))}
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={escalatedOnly}
            onChange={(e) => setFilter('escalated', e.target.checked ? 'true' : '')}
          />
          Escalated only
        </label>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      {loading && !data && <p className="muted">Loading…</p>}
      {data && data.items.length === 0 && <p className="muted">No tickets match these filters.</p>}

      {data && data.items.length > 0 && (
        <table className="queue">
          <thead>
            <tr>
              <th>#</th>
              <th>Subject</th>
              <th>Category</th>
              <th>Status</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((t) => (
              <tr key={t.id} className={t.escalated ? 'row-escalated' : undefined}>
                <td>{t.id}</td>
                <td>
                  <Link to={`/tickets/${t.id}`}>{t.subject}</Link> <EscalationBadge escalated={t.escalated} />
                </td>
                <td>
                  <CategoryBadge category={t.category} confidence={t.category_confidence}
                                 lowConfidence={t.category_low_confidence} />
                </td>
                <td><StatusBadge status={t.status} /></td>
                <td className="muted">{formatDate(t.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {data && data.total > PAGE_SIZE && (
        <div className="row pager">
          <button disabled={page === 0} onClick={() => setFilter('page', String(page - 1))}>Previous</button>
          <span className="muted">Page {page + 1} of {pages} · {data.total} tickets</span>
          <button disabled={page + 1 >= pages} onClick={() => setFilter('page', String(page + 1))}>Next</button>
        </div>
      )}
    </div>
  )
}
