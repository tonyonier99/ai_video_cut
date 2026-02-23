import { describe, it, expect } from 'vitest'
import { getSnapTime, getTrackTheme, formatTime, getTimelineCursor } from '../utils/timelineUtils'
import type { Cut } from '../types'

describe('getSnapTime', () => {
  const baseCuts: Cut[] = [
    { id: 'c1', start: 5, end: 10, track: 0, label: 'Cut 1' },
    { id: 'c2', start: 15, end: 20, track: 0, label: 'Cut 2' },
  ]

  it('snaps to currentTime when within threshold', () => {
    const result = getSnapTime(3.05, baseCuts, 3.0, 100)
    expect(result).toBe(3.0)
  })

  it('snaps to zero when close to origin', () => {
    const result = getSnapTime(0.05, baseCuts, 50.0, 100)
    expect(result).toBe(0)
  })

  it('snaps to cut start boundary', () => {
    const result = getSnapTime(5.05, baseCuts, 50.0, 100)
    expect(result).toBe(5)
  })

  it('snaps to cut end boundary', () => {
    const result = getSnapTime(10.05, baseCuts, 50.0, 100)
    expect(result).toBe(10)
  })

  it('returns null when no snap target is close', () => {
    const result = getSnapTime(12.5, baseCuts, 50.0, 100)
    expect(result).toBeNull()
  })

  it('ignores the cut specified by ignoreCutId', () => {
    const result = getSnapTime(5.05, baseCuts, 50.0, 100, 'c1')
    expect(result).not.toBe(5)
  })

  it('handles empty cuts array', () => {
    const result = getSnapTime(12.5, [], 50.0, 100)
    expect(result).toBeNull()
  })
})

describe('getTrackTheme', () => {
  it('returns video theme for video type', () => {
    const theme = getTrackTheme('video')
    expect(theme.primary).toBe('#3ea6ff')
  })

  it('returns audio theme for audio type', () => {
    const theme = getTrackTheme('audio')
    expect(theme.primary).toBe('#10b981')
  })

  it('returns text theme for text type', () => {
    const theme = getTrackTheme('text')
    expect(theme.primary).toBe('#6366f1')
  })

  it('returns default theme for unknown type', () => {
    const theme = getTrackTheme('unknown')
    expect(theme.primary).toBe('#3ea6ff')
  })
})

describe('formatTime', () => {
  it('formats zero correctly', () => {
    expect(formatTime(0)).toBe('00:00.00')
  })

  it('formats seconds with milliseconds', () => {
    expect(formatTime(5.5)).toBe('00:05.50')
  })

  it('formats minutes correctly', () => {
    expect(formatTime(65.25)).toBe('01:05.25')
  })

  it('formats large values correctly', () => {
    expect(formatTime(3661)).toBe('61:01.00')
    expect(formatTime(3661.5)).toBe('61:01.50')
  })
})

describe('getTimelineCursor', () => {
  it('returns ew-resize when scrubbing', () => {
    expect(getTimelineCursor(true, 'scrub', 'select')).toBe('ew-resize')
  })

  it('returns grabbing when panning', () => {
    expect(getTimelineCursor(true, 'pan', 'select')).toBe('grabbing')
  })

  it('returns move when moving', () => {
    expect(getTimelineCursor(true, 'move', 'select')).toBe('move')
  })

  it('returns ew-resize when trimming', () => {
    expect(getTimelineCursor(true, 'trim-start', 'select')).toBe('ew-resize')
    expect(getTimelineCursor(true, 'trim-end', 'select')).toBe('ew-resize')
  })

  it('returns grab for hand tool when not dragging', () => {
    expect(getTimelineCursor(false, null, 'hand')).toBe('grab')
  })

  it('returns crosshair for blade tool', () => {
    expect(getTimelineCursor(false, null, 'blade')).toBe('crosshair')
  })

  it('returns default for select tool', () => {
    expect(getTimelineCursor(false, null, 'select')).toBe('default')
  })
})
