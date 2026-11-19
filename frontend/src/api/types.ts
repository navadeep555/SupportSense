// TypeScript mirrors of the backend's Pydantic response models (backend/app/schemas.py).
// If the backend changes a field, the TypeScript compiler points at every place
// in the UI that uses it. Keep the two in sync by hand.

export const CATEGORIES = [
  'account_access',
  'billing_payment',
  'subscription',
  'technical_issue',
  'general_inquiry',
] as const
export type Category = (typeof CATEGORIES)[number]

export type TicketStatus = 'open' | 'drafted' | 'resolved'
export type DraftStatus = 'grounded' | 'needs_attention' | 'insufficient_evidence' | 'generation_failed'
export type ReviewDecision = 'approved' | 'edited' | 'rejected'

export interface Ticket {
  id: number
  subject: string
  body: string
  subject_masked: string
  body_masked: string
  pii_found: Record<string, number>
  customer_email: string | null
  status: TicketStatus
  category: Category | null
  category_confidence: number | null
  category_low_confidence: boolean
  escalated: boolean
  escalation_score: number | null
  escalation_reasons: string[]
  created_at: string // ISO date-time string from JSON
  updated_at: string
}

export interface Citation {
  source_id: string // "S1"
  chunk_id: string
  doc_id: string
  title: string
  section: string
  text: string
  similarity: number
  outdated: boolean
}

export interface Draft {
  id: number
  ticket_id: number
  status: DraftStatus
  answer: string
  citations: Citation[]
  warnings: string[]
  sources_sent: string[]
  pii_masked: Record<string, number>
  evidence_sufficient: boolean
  best_similarity: number
  llm_called: boolean
  llm_provider: string | null
  llm_model: string | null
  input_tokens: number | null
  output_tokens: number | null
  latency_ms: number | null
  created_at: string
  review_decision: ReviewDecision | null
  final_reply: string | null
  edit_ratio: number | null
  review_note: string | null
  reviewed_at: string | null
}

export interface TicketDetail extends Ticket {
  drafts: Draft[]
}

export interface TicketList {
  total: number
  items: Ticket[]
}

export interface TicketCreate {
  subject: string
  body: string
  customer_email?: string | null
}

export interface ReviewRequest {
  decision: ReviewDecision
  final_reply?: string
  note?: string
}

export interface DraftMetrics {
  total_drafts: number
  by_status: Record<string, number>
  reviewed: number
  by_decision: Record<string, number>
  approval_rate: number | null
  acceptance_rate: number | null
  avg_edit_ratio_when_edited: number | null
  avg_llm_latency_ms: number | null
}

export interface KnowledgeGap {
  ticket_id: number
  draft_id: number
  reason: 'no_matching_docs' | 'model_found_docs_insufficient' | 'agent_rejected'
  detail: string
  best_similarity: number
  subject_masked: string
  body_masked: string
  created_at: string
}

export interface KnowledgeGapReport {
  total: number
  items: KnowledgeGap[]
}

export interface Health {
  status: string
  rag_index_loaded: boolean
  database_ready: boolean
  classifier_loaded: boolean
  escalation_model_loaded: boolean
  llm: { provider: string | null; model: string | null; available: boolean }
}
