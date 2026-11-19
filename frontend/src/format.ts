// Small display helpers shared by several pages.
import type { Category, DraftStatus } from './api/types'

export const CATEGORY_LABELS: Record<Category, string> = {
  account_access: 'Account access',
  billing_payment: 'Billing & payment',
  subscription: 'Subscription',
  technical_issue: 'Technical issue',
  general_inquiry: 'General inquiry',
}

// What each draft status means for the agent, in plain words.
export const DRAFT_STATUS_HELP: Record<DraftStatus, string> = {
  grounded: 'Every check passed. Still read it before sending.',
  needs_attention: 'A draft exists but a check raised a warning. Read the warnings first.',
  insufficient_evidence: 'The documentation does not cover this. A hand-off reply was prepared instead.',
  generation_failed: 'The AI model could not produce a draft. Write the reply yourself.',
}

export const INLINE_CITATION = /\[(S\d+)\]/g

/** Remove [S1] markers: they help the agent, but customers should never see them. */
export function stripCitations(text: string): string {
  return text.replace(INLINE_CITATION, '').replace(/\s+([.,])/g, '$1').replace(/\s{2,}/g, ' ').trim()
}

export function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function humanize(key: string): string {
  return key.replace(/_/g, ' ')
}
