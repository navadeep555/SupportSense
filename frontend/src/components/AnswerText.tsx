// Renders an AI answer with its [S1] markers turned into links to the cited source.
//
// SECURITY: the answer comes from an LLM that read customer text, so it is
// untrusted. We split it into plain strings and let React render them as text
// (React escapes < > & automatically). We never use dangerouslySetInnerHTML:
// a ticket containing "<img src=x onerror=...>" must show up as characters,
// not run as code in the agent's browser (cross-site scripting / XSS).
import { Fragment } from 'react'

export function AnswerText(props: { text: string; citedIds: Set<string>; anchorPrefix: string }) {
  const { text, citedIds, anchorPrefix } = props
  // split() with a capture group keeps the separators: "a [S1] b" -> ["a ", "S1", " b"]
  const parts = text.split(/\[(S\d+)\]/)
  return (
    <p className="answer">
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          citedIds.has(part) ? (
            <a key={i} className="cite" href={`#${anchorPrefix}-${part}`} title={`Jump to source ${part}`}>
              {part}
            </a>
          ) : (
            <span key={i} className="cite invalid">{part}?</span>
          )
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </p>
  )
}
