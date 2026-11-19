import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Ticket } from '../api/types'
import { QueuePage } from './QueuePage'

function ticket(id: number, subject: string, escalated: boolean): Ticket {
  return {
    id, subject, body: '', subject_masked: subject, body_masked: '', pii_found: {}, customer_email: null,
    status: 'open', category: 'billing_payment', category_confidence: 0.3, category_low_confidence: true,
    escalated, escalation_score: null, escalation_reasons: [],
    created_at: '2026-11-19T10:00:00Z', updated_at: '2026-11-19T10:00:00Z',
  }
}

afterEach(() => vi.unstubAllGlobals())

describe('QueuePage', () => {
  it('shows tickets in the order the API returns, with escalation and low-confidence badges', async () => {
    const items = [ticket(2, 'Charged twice', true), ticket(1, 'Invoice please', false)]
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ total: 2, items }))))
    render(<MemoryRouter><QueuePage /></MemoryRouter>)

    const rows = await screen.findAllByRole('row')
    expect(within(rows[1]).getByText('Charged twice')).toBeInTheDocument()
    expect(within(rows[1]).getByText('Escalated')).toBeInTheDocument()
    expect(within(rows[2]).queryByText('Escalated')).toBeNull()
    expect(within(rows[2]).getByText('check')).toBeInTheDocument()   // low-confidence category
  })

  it('refetches with the filter when the agent picks "Escalated only"', async () => {
    const fetchMock = vi.fn().mockImplementation(
      () => Promise.resolve(new Response(JSON.stringify({ total: 0, items: [] }))))
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter><QueuePage /></MemoryRouter>)
    await screen.findByText('No tickets match these filters.')

    await userEvent.click(screen.getByLabelText('Escalated only'))
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(fetchMock.mock.calls[1][0]).toBe('/api/tickets?escalated=true&limit=20&offset=0')
  })

  it('shows an error when the API is down', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    render(<MemoryRouter><QueuePage /></MemoryRouter>)
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the backend')
  })
})
