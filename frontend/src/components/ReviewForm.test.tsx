import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { Draft } from '../api/types'
import { ReviewForm } from './ReviewForm'

function makeDraft(answer: string): Draft {
  return {
    id: 1, ticket_id: 1, status: 'grounded', answer, citations: [], warnings: [], sources_sent: [],
    pii_masked: {}, evidence_sufficient: true, best_similarity: 0.6, llm_called: true,
    llm_provider: 'ollama', llm_model: 'qwen2.5:7b', input_tokens: 900, output_tokens: 100,
    latency_ms: 4000, created_at: '2026-11-19T10:00:00Z', review_decision: null, final_reply: null,
    edit_ratio: null, review_note: null, reviewed_at: null,
  }
}

describe('ReviewForm', () => {
  it('prefills the reply without citation markers and approves it unchanged', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<ReviewForm draft={makeDraft('Sign out and back in [S1].')} onSubmit={onSubmit} />)

    expect(screen.getByLabelText(/Reply to customer/)).toHaveValue('Sign out and back in.')
    await userEvent.click(screen.getByRole('button', { name: 'Approve & send' }))
    expect(onSubmit).toHaveBeenCalledWith({ decision: 'approved' })
  })

  it('sends an edited reply when the agent changed the text', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<ReviewForm draft={makeDraft('Sign out [S1].')} onSubmit={onSubmit} />)

    const box = screen.getByLabelText(/Reply to customer/)
    await userEvent.clear(box)
    await userEvent.type(box, 'Hi! Please sign out.')
    await userEvent.click(screen.getByRole('button', { name: 'Send edited reply' }))
    expect(onSubmit).toHaveBeenCalledWith({ decision: 'edited', final_reply: 'Hi! Please sign out.' })
  })

  it('requires a reason before rejecting', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<ReviewForm draft={makeDraft('Sign out [S1].')} onSubmit={onSubmit} />)

    const reject = screen.getByRole('button', { name: 'Reject' })
    expect(reject).toBeDisabled()
    await userEvent.type(screen.getByLabelText(/reject this draft/i), 'wrong device steps')
    await userEvent.click(reject)
    expect(onSubmit).toHaveBeenCalledWith({ decision: 'rejected', note: 'wrong device steps' })
  })

  it('shows the server error if the review fails', async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error('Draft 1 was already reviewed (approved).'))
    render(<ReviewForm draft={makeDraft('Sign out [S1].')} onSubmit={onSubmit} />)
    await userEvent.click(screen.getByRole('button', { name: 'Approve & send' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('already reviewed')
  })

  it('cannot approve an empty draft (generation failed)', () => {
    render(<ReviewForm draft={makeDraft('')} onSubmit={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Approve & send' })).toBeDisabled()
  })
})
