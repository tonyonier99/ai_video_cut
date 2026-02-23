import { describe, it, expect } from 'vitest'
import { generateFCPXML } from '../utils/exportUtils'
import type { Cut } from '../types'

describe('generateFCPXML', () => {
  const testCuts: Cut[] = [
    { id: 'c1', start: 0, end: 10, track: 0, label: 'Intro' },
    { id: 'c2', start: 20, end: 35, track: 0, label: 'Main' },
  ]

  it('generates valid XML structure', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', '/path/to/test.mp4')
    expect(xml).toContain('<?xml version="1.0" encoding="UTF-8"?>')
    expect(xml).toContain('<fcpxml version="1.8">')
    expect(xml).toContain('</fcpxml>')
  })

  it('includes resource format and asset', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', '/path/to/test.mp4')
    expect(xml).toContain('format id="r1"')
    expect(xml).toContain('asset id="a1"')
    expect(xml).toContain('name="test.mp4"')
  })

  it('includes correct number of video clips', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', '/path/to/test.mp4')
    const videoTags = xml.match(/<video /g)
    expect(videoTags).toHaveLength(2)
  })

  it('uses cut labels in video names', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', '/path/to/test.mp4')
    expect(xml).toContain('name="Intro"')
    expect(xml).toContain('name="Main"')
  })

  it('calculates correct total duration in frames', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', '/path/to/test.mp4')
    // Total cut duration: (10-0) + (35-20) = 25 seconds * 30 fps = 750 frames
    expect(xml).toContain('duration="750/30s"')
  })

  it('calculates correct clip durations', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', '/path/to/test.mp4')
    // First clip: 10s * 30fps = 300 frames
    expect(xml).toContain('duration="300/30s"')
    // Second clip: 15s * 30fps = 450 frames
    expect(xml).toContain('duration="450/30s"')
  })

  it('handles empty cuts array', () => {
    const xml = generateFCPXML([], 60, 'test.mp4', '/path/to/test.mp4')
    expect(xml).toContain('<spine>')
    expect(xml).toContain('</spine>')
    const videoTags = xml.match(/<video /g)
    expect(videoTags).toBeNull()
  })

  it('handles null originalVideoPath', () => {
    const xml = generateFCPXML(testCuts, 60, 'test.mp4', null)
    expect(xml).toContain('localhost/path/to/test.mp4')
  })
})
