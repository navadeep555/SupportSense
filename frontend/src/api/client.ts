// The ONLY place that talks to the backend. Pages call api.xxx() and get typed data.
//
// Paths are relative ("/api/..."): in development Vite proxies them to FastAPI,
// in production one web server serves both. No backend URL, key or secret is
// ever in this code - anything shipped to the browser is readable by anyone.
import type {
  DraftMetrics,
  Draft,
  Health,
  KnowledgeGapReport,
  ReviewRequest,
  Ticket,
  TicketCreate,
  TicketDetail,
  TicketList,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** FastAPI errors: {"detail": "text"} or, for 422, {"detail": [{loc, msg}, ...]}. */
export function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((e: { loc?: unknown[]; msg?: string }) => {
        const field = (e.loc ?? []).slice(1).join('.') // drop "body"/"query" prefix
        return field ? `${field}: ${e.msg}` : String(e.msg)
      })
      .join('; ')
  }
  return 'Unexpected error'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    // fetch only throws when no HTTP response arrived at all.
    throw new ApiError(0, 'Cannot reach the backend. Is the API server running on port 8000?')
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      message = formatDetail((await response.json()).detail)
    } catch {
      /* body was not JSON: keep the status text */
    }
    throw new ApiError(response.status, message)
  }
  return (await response.json()) as T
}

export interface TicketFilters {
  status?: string
  category?: string
  escalated?: boolean
  limit?: number
  offset?: number
}

export const api = {
  health: () => request<Health>('/health'),

  listTickets: (filters: TicketFilters = {}) => {
    const params = new URLSearchParams()
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== '') params.set(key, String(value))
    }
    const query = params.toString()
    return request<TicketList>(`/tickets${query ? `?${query}` : ''}`)
  },

  getTicket: (id: number) => request<TicketDetail>(`/tickets/${id}`),

  createTicket: (body: TicketCreate) =>
    request<Ticket>('/tickets', { method: 'POST', body: JSON.stringify(body) }),

  createDraft: (ticketId: number) =>
    request<Draft>(`/tickets/${ticketId}/drafts`, { method: 'POST' }),

  reviewDraft: (draftId: number, body: ReviewRequest) =>
    request<Draft>(`/drafts/${draftId}/review`, { method: 'POST', body: JSON.stringify(body) }),

  draftMetrics: () => request<DraftMetrics>('/reports/draft-metrics'),

  knowledgeGaps: () => request<KnowledgeGapReport>('/reports/knowledge-gaps'),
}
