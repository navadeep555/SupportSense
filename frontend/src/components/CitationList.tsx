import type { Citation } from '../api/types'

/** The evidence behind a draft, so the agent can check each claim against its source. */
export function CitationList({ citations, anchorPrefix }: { citations: Citation[]; anchorPrefix: string }) {
  if (citations.length === 0) return <p className="muted">No sources cited.</p>
  return (
    <ol className="citations">
      {citations.map((c) => (
        <li key={c.source_id} id={`${anchorPrefix}-${c.source_id}`}>
          <div className="citation-head">
            <span className="cite">{c.source_id}</span>
            <strong>{c.title}</strong> › {c.section}
            {c.outdated && <span className="badge escalated">outdated policy</span>}
            <span className="muted"> · similarity {c.similarity.toFixed(2)}</span>
          </div>
          <blockquote>{c.text}</blockquote>
        </li>
      ))}
    </ol>
  )
}
