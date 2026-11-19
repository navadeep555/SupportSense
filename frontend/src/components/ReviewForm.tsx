// The agent's decision on a draft: send it (as-is or edited) or reject it.
//
// The textarea starts with the draft WITHOUT [S1] markers (that is what the
// customer would receive). On send we compare: unchanged -> "approved",
// changed -> "edited" with the new text. The backend then measures how much
// was changed (edit_ratio), which feeds the draft-quality metrics.
import { useState } from 'react'
import type { Draft, ReviewRequest } from '../api/types'
import { stripCitations } from '../format'

interface Props {
  draft: Draft
  onSubmit: (review: ReviewRequest) => Promise<void>
}

export function ReviewForm({ draft, onSubmit }: Props) {
  const original = stripCitations(draft.answer)
  const [reply, setReply] = useState(original)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const edited = reply.trim() !== original.trim()

  async function submit(review: ReviewRequest) {
    setBusy(true)
    setError(null)
    try {
      await onSubmit(review)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  function send() {
    if (edited) void submit({ decision: 'edited', final_reply: reply.trim() })
    else void submit({ decision: 'approved' })
  }

  return (
    <div className="review">
      <label htmlFor="reply">Reply to customer {edited && <span className="badge check">edited</span>}</label>
      <textarea
        id="reply"
        rows={8}
        value={reply}
        onChange={(e) => setReply(e.target.value)}
        placeholder="Write the reply the customer will receive"
      />
      <div className="row">
        <button type="button" className="primary" disabled={busy || !reply.trim()} onClick={send}>
          {edited ? 'Send edited reply' : 'Approve & send'}
        </button>
        {edited && (
          <button type="button" disabled={busy} onClick={() => setReply(original)}>
            Undo edits
          </button>
        )}
      </div>

      <label htmlFor="note">Or reject this draft</label>
      <div className="row">
        <input
          id="note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Why is it unusable? (required, min 3 characters)"
        />
        <button
          type="button"
          className="danger"
          disabled={busy || note.trim().length < 3}
          onClick={() => void submit({ decision: 'rejected', note: note.trim() })}
        >
          Reject
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  )
}
