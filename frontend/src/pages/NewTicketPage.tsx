// A simple form standing in for the customer channel (email / chat) in this demo.
import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'

export function NewTicketPage() {
  const navigate = useNavigate()
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [email, setEmail] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(event: FormEvent) {
    event.preventDefault() // stop the browser's default full-page form submit
    setBusy(true)
    setError(null)
    try {
      const ticket = await api.createTicket({ subject, body, customer_email: email.trim() || null })
      navigate(`/tickets/${ticket.id}`)
    } catch (err) {
      // e.g. 422 "subject: String should have at least 3 characters"
      setError(err instanceof Error ? err.message : String(err))
      setBusy(false)
    }
  }

  return (
    <form className="card narrow" onSubmit={submit}>
      <h2>New ticket</h2>
      <label htmlFor="subject">Subject</label>
      <input id="subject" value={subject} onChange={(e) => setSubject(e.target.value)} maxLength={200} required />

      <label htmlFor="body">Message</label>
      <textarea id="body" rows={6} value={body} onChange={(e) => setBody(e.target.value)} maxLength={4000} required />

      <label htmlFor="email">Customer email (optional)</label>
      <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} />

      <p className="muted small">
        Phone numbers, emails, card numbers and passwords in the message are masked before any AI model sees them.
      </p>
      {error && <p className="error" role="alert">{error}</p>}
      <button className="primary" type="submit" disabled={busy}>
        {busy ? 'Creating…' : 'Create ticket'}
      </button>
    </form>
  )
}
