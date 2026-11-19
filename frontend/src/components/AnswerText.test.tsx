import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AnswerText } from './AnswerText'

describe('AnswerText', () => {
  it('turns valid [S#] markers into links to the source', () => {
    render(<AnswerText text="Sign out [S1]. Wait [S2]." citedIds={new Set(['S1', 'S2'])} anchorPrefix="draft-7" />)
    expect(screen.getByRole('link', { name: 'S1' })).toHaveAttribute('href', '#draft-7-S1')
    expect(screen.getByRole('link', { name: 'S2' })).toHaveAttribute('href', '#draft-7-S2')
  })

  it('marks a citation that was not validated instead of linking it', () => {
    render(<AnswerText text="Free month [S9]." citedIds={new Set(['S1'])} anchorPrefix="d" />)
    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByText('S9?')).toHaveClass('invalid')
  })

  it('renders HTML in model output as plain text (no XSS)', () => {
    const evil = '<img src=x onerror="alert(1)"> hello [S1]'
    const { container } = render(<AnswerText text={evil} citedIds={new Set(['S1'])} anchorPrefix="d" />)
    expect(container.querySelector('img')).toBeNull()
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">')
  })
})
