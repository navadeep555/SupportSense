import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DraftMetrics, KnowledgeGapReport } from '../api/types'
import { ReportsPage } from './ReportsPage'

afterEach(() => vi.unstubAllGlobals())

function stubApi(metrics: DraftMetrics, gaps: KnowledgeGapReport) {
  vi.stubGlobal('fetch', vi.fn((url: string) =>
    Promise.resolve(new Response(JSON.stringify(url.includes('draft-metrics') ? metrics : gaps)))))
}

describe('ReportsPage', () => {
  it('shows "—" instead of 0% when nothing has been reviewed yet', async () => {
    stubApi({ total_drafts: 0, by_status: {}, reviewed: 0, by_decision: {}, approval_rate: null,
      acceptance_rate: null, avg_edit_ratio_when_edited: null, avg_llm_latency_ms: null }, { total: 0, items: [] })
    render(<MemoryRouter><ReportsPage /></MemoryRouter>)
    expect(await screen.findByText('Sent unchanged')).toBeInTheDocument()
    expect(screen.getAllByText('—').length).toBe(4)
    expect(screen.getByText('No gaps found yet.')).toBeInTheDocument()
  })

  it('shows rates and the masked knowledge-gap list', async () => {
    stubApi({ total_drafts: 4, by_status: { grounded: 3, insufficient_evidence: 1 }, reviewed: 3,
      by_decision: { approved: 1, edited: 1, rejected: 1 }, approval_rate: 0.333, acceptance_rate: 0.667,
      avg_edit_ratio_when_edited: 0.127, avg_llm_latency_ms: 4135 },
    { total: 1, items: [{ ticket_id: 3, draft_id: 3, reason: 'no_matching_docs', detail: 'similarity 0.16',
      best_similarity: 0.16, subject_masked: 'Gift cards', body_masked: 'mail me at [EMAIL]',
      created_at: '2026-11-19T10:00:00Z' }] })
    render(<MemoryRouter><ReportsPage /></MemoryRouter>)

    expect(await screen.findByText('33%')).toBeInTheDocument()
    expect(screen.getByText('67%')).toBeInTheDocument()
    expect(screen.getByText('4.1 s')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '#3' })).toHaveAttribute('href', '/tickets/3')
    expect(screen.getByText('mail me at [EMAIL]')).toBeInTheDocument()
  })
})
