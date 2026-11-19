import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api, formatDetail } from './client'

afterEach(() => vi.unstubAllGlobals())

describe('formatDetail', () => {
  it('keeps plain messages and flattens 422 validation errors', () => {
    expect(formatDetail('Ticket 9 not found.')).toBe('Ticket 9 not found.')
    expect(formatDetail([{ loc: ['body', 'subject'], msg: 'String should have at least 3 characters' }]))
      .toBe('subject: String should have at least 3 characters')
  })
})

describe('api client', () => {
  it('builds relative /api URLs with only the filters that are set', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ total: 0, items: [] })))
    vi.stubGlobal('fetch', fetchMock)
    await api.listTickets({ status: 'open', category: undefined, escalated: true, limit: 20 })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/tickets?status=open&escalated=true&limit=20')
  })

  it('turns an HTTP error into ApiError with the backend message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Ticket 9 not found.' }), { status: 404 })))
    await expect(api.getTicket(9)).rejects.toMatchObject({ status: 404, message: 'Ticket 9 not found.' })
  })

  it('explains when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const error = await api.health().catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).message).toMatch(/Cannot reach the backend/)
  })
})
