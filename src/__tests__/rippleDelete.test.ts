import { describe, it, expect } from 'vitest'
import type { Cut } from '../types'

// Extract the ripple delete logic as a pure function for testing
function rippleDelete(cuts: Cut[], deletedIds: string[]): Cut[] {
  const deletedSet = new Set(deletedIds)
  const deletedCuts = cuts.filter(c => deletedSet.has(c.id))
  const remaining = cuts.filter(c => !deletedSet.has(c.id))

  const deletedByTrack = new Map<number, Cut[]>()
  for (const dc of deletedCuts) {
    const list = deletedByTrack.get(dc.trackId) ?? []
    list.push(dc)
    deletedByTrack.set(dc.trackId, list)
  }

  return remaining.map(cut => {
    const trackDeleted = deletedByTrack.get(cut.trackId)
    if (!trackDeleted) return cut

    const shift = trackDeleted
      .filter(dc => dc.start < cut.start)
      .reduce((sum, dc) => sum + (dc.end - dc.start), 0)

    if (shift <= 0) return cut

    return {
      ...cut,
      start: Math.max(0, cut.start - shift),
      end: Math.max(0.01, cut.end - shift),
    }
  })
}

const makeCut = (id: string, start: number, end: number, trackId = 0): Cut => ({
  id,
  start,
  end,
  sourceStart: start,
  sourceEnd: end,
  label: `Cut ${id}`,
  trackId,
})

describe('rippleDelete', () => {
  it('removes deleted cuts and shifts subsequent clips left', () => {
    const cuts = [
      makeCut('a', 0, 5),
      makeCut('b', 5, 10),
      makeCut('c', 10, 15),
    ]

    const result = rippleDelete(cuts, ['b'])

    expect(result).toHaveLength(2)
    expect(result[0]).toEqual(expect.objectContaining({ id: 'a', start: 0, end: 5 }))
    expect(result[1]).toEqual(expect.objectContaining({ id: 'c', start: 5, end: 10 }))
  })

  it('handles deleting the first clip', () => {
    const cuts = [
      makeCut('a', 0, 5),
      makeCut('b', 5, 10),
      makeCut('c', 10, 15),
    ]

    const result = rippleDelete(cuts, ['a'])

    expect(result).toHaveLength(2)
    expect(result[0]).toEqual(expect.objectContaining({ id: 'b', start: 0, end: 5 }))
    expect(result[1]).toEqual(expect.objectContaining({ id: 'c', start: 5, end: 10 }))
  })

  it('handles deleting the last clip (no shift needed)', () => {
    const cuts = [
      makeCut('a', 0, 5),
      makeCut('b', 5, 10),
    ]

    const result = rippleDelete(cuts, ['b'])

    expect(result).toHaveLength(1)
    expect(result[0]).toEqual(expect.objectContaining({ id: 'a', start: 0, end: 5 }))
  })

  it('handles multi-track: only shifts on affected track', () => {
    const cuts = [
      makeCut('a', 0, 5, 0),
      makeCut('b', 5, 10, 0),
      makeCut('x', 0, 10, 1),  // Different track
    ]

    const result = rippleDelete(cuts, ['a'])

    expect(result).toHaveLength(2)
    // Track 0: 'b' shifts left by 5
    expect(result[0]).toEqual(expect.objectContaining({ id: 'b', start: 0, end: 5 }))
    // Track 1: 'x' unaffected
    expect(result[1]).toEqual(expect.objectContaining({ id: 'x', start: 0, end: 10 }))
  })

  it('handles deleting multiple clips', () => {
    const cuts = [
      makeCut('a', 0, 3),
      makeCut('b', 3, 6),
      makeCut('c', 6, 9),
      makeCut('d', 9, 12),
    ]

    const result = rippleDelete(cuts, ['a', 'c'])

    expect(result).toHaveLength(2)
    // 'b' shifts left by 3 (duration of 'a')
    expect(result[0]).toEqual(expect.objectContaining({ id: 'b', start: 0, end: 3 }))
    // 'd' shifts left by 6 (duration of 'a' + 'c' = 3 + 3)
    expect(result[1]).toEqual(expect.objectContaining({ id: 'd', start: 3, end: 6 }))
  })

  it('returns empty array when all clips deleted', () => {
    const cuts = [makeCut('a', 0, 5)]
    const result = rippleDelete(cuts, ['a'])
    expect(result).toHaveLength(0)
  })

  it('handles gaps between clips', () => {
    const cuts = [
      makeCut('a', 0, 5),
      makeCut('b', 10, 15), // gap from 5-10
      makeCut('c', 20, 25), // gap from 15-20
    ]

    const result = rippleDelete(cuts, ['b'])

    expect(result).toHaveLength(2)
    expect(result[0]).toEqual(expect.objectContaining({ id: 'a', start: 0, end: 5 }))
    // 'c' shifts left by 5 (duration of 'b')
    expect(result[1]).toEqual(expect.objectContaining({ id: 'c', start: 15, end: 20 }))
  })
})
