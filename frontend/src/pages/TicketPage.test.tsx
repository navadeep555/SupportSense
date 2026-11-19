import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Draft, TicketDetail } from '../api/types'
import { TicketPage } from './TicketPage'

const draft: Draft = {
  id: 5, ticket_id: 1, status: 'grounded', answer: 'Sign out and back in [S1].',
  citations: [{ source_id: 'S1', chunk_id: 'plans::premium::0', doc_id: 'plans', title: 'Plans',
    section: 'Premium not available', text: 'Sign out of the app and sign back in.', similarity: 0.66, outdated: false }],
  warnings: [], sources_sent: ['plans::premium::0'], pii_masked: { phone: 1 }, evidence_sufficient: true,
  best_similarity: 0.66, llm_called: true, llm_provider: 'ollama', llm_model: 'qwen2.5:7b',
  input_tokens: 900, output_tokens: 80, latency_ms: 4200, created_at: '2026-11-19T10:00:00Z',
  review_decision: null, final_reply: null, edit_ratio: null, review_note: null, reviewed_at: null,
}

function ticket(drafts: Draft[], status: TicketDetail['status'] = 'open'): TicketDetail {
  return {
    id: 1, subject: 'Premium missing', body: 'Paid but no Premium. Call 9876543210', subject_masked: 'Premium missing',
    body_masked: 'Paid but no Premium. Call [PHONE]', pii_found: { phone: 1 }, customer_email: null, status,
    category: 'subscription', category_confidence: 0.91, category_low_confidence: false, escalated: true,
    escalation_score: 0.7, escalation_reasons: ['financial_dispute: test'], created_at: '2026-11-19T10:00:00Z',
    updated_at: '2026-11-19T10:00:00Z', drafts,
  }
}

const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }))

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes><Route path="/tickets/:id" element={<TicketPage />} /></Routes>
    </MemoryRouter>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('TicketPage', () => {
  it('shows the message, triage, and a draft whose citation links to its source', async () => {
    vi.stubGlobal('fetch', vi.fn(() => json(ticket([draft], 'drafted'))))
    renderAt('/tickets/1')

    expect(await screen.findByText(/Paid but no Premium/)).toBeInTheDocument()
    expect(screen.getByText(/hidden from the AI: phone ×1/)).toBeInTheDocument()
    expect(screen.getByText('Yes — handle with priority')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'S1' })).toHaveAttribute('href', '#draft-5-S1')
    expect(document.getElementById('draft-5-S1')).toHaveTextContent('Sign out of the app and sign back in.')
    // An unreviewed draft exists, so no second draft can be requested yet.
    expect(screen.queryByRole('button', { name: /Generate/ })).toBeNull()
  })

  it('generates a draft, then reloads the ticket', async () => {
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => json(ticket([])))                  // first load: no drafts
      .mockImplementationOnce(() => json(draft, 201))                   // POST /drafts
      .mockImplementation(() => json(ticket([draft], 'drafted')))       // reload
    vi.stubGlobal('fetch', fetchMock)
    renderAt('/tickets/1')

    await userEvent.click(await screen.findByRole('button', { name: 'Generate AI draft' }))
    expect(await screen.findByText('Draft #5')).toBeInTheDocument()
    expect(fetchMock.mock.calls[1][0]).toBe('/api/tickets/1/drafts')
    expect(fetchMock.mock.calls[1][1]).toMatchObject({ method: 'POST' })
  })

  it('shows a readable error when generation fails', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockImplementationOnce(() => json(ticket([])))
      .mockImplementationOnce(() => json({ detail: 'Search index not loaded.' }, 503)))
    renderAt('/tickets/1')
    await userEvent.click(await screen.findByRole('button', { name: 'Generate AI draft' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Search index not loaded.')
  })

  it('shows the recorded decision instead of the review form once reviewed', async () => {
    const reviewed = { ...draft, review_decision: 'edited' as const, final_reply: 'Hi! Sign out and back in.',
      edit_ratio: 0.2, reviewed_at: '2026-11-19T10:05:00Z' }
    vi.stubGlobal('fetch', vi.fn(() => json(ticket([reviewed], 'resolved'))))
    renderAt('/tickets/1')
    expect(await screen.findByText('Agent edited')).toBeInTheDocument()
    expect(screen.getByText(/changed 20% of words/)).toBeInTheDocument()
    expect(screen.queryByLabelText(/Reply to customer/)).toBeNull()
  })
})
