import { describe, it, expect } from 'vitest'
import { downsamplePeaks } from '../utils/waveformUtils'
import type { WaveformData } from '../utils/waveformUtils'

function makeWaveform(length: number, sampleRate: number, fillValue: number): WaveformData {
  const peaks = new Float32Array(length)
  for (let i = 0; i < length; i++) {
    peaks[i] = fillValue
  }
  return {
    peaks,
    duration: length / sampleRate,
    sampleRate,
  }
}

function makeRampWaveform(length: number, sampleRate: number): WaveformData {
  const peaks = new Float32Array(length)
  for (let i = 0; i < length; i++) {
    peaks[i] = i / length
  }
  return {
    peaks,
    duration: length / sampleRate,
    sampleRate,
  }
}

describe('downsamplePeaks', () => {
  it('returns the correct number of bars', () => {
    const wf = makeWaveform(44100, 44100, 0.5) // 1 second
    const bars = downsamplePeaks(wf, 0, 1, 10)
    expect(bars).toHaveLength(10)
  })

  it('returns uniform values for uniform input', () => {
    const wf = makeWaveform(44100, 44100, 0.8)
    const bars = downsamplePeaks(wf, 0, 1, 5)
    for (const bar of bars) {
      expect(bar.max).toBeCloseTo(0.8, 1)
      expect(bar.min).toBe(0) // peaks are absolute values, min should be 0
    }
  })

  it('slices based on sourceStart/sourceEnd', () => {
    const wf = makeRampWaveform(44100, 44100)
    // Only take the second half (0.5s to 1.0s)
    const bars = downsamplePeaks(wf, 0.5, 1.0, 2)
    expect(bars).toHaveLength(2)
    // The second half should have higher values than the first half
    const firstHalfBars = downsamplePeaks(wf, 0, 0.5, 2)
    expect(bars[0].max).toBeGreaterThan(firstHalfBars[0].max)
  })

  it('handles zero-length slice gracefully', () => {
    const wf = makeWaveform(44100, 44100, 0.5)
    const bars = downsamplePeaks(wf, 5, 5, 10) // sourceStart == sourceEnd
    expect(bars).toHaveLength(10)
    for (const bar of bars) {
      expect(bar.max).toBe(0)
    }
  })

  it('handles out-of-bounds sourceEnd', () => {
    const wf = makeWaveform(44100, 44100, 0.5) // 1 second
    const bars = downsamplePeaks(wf, 0, 2, 5) // sourceEnd > duration
    expect(bars).toHaveLength(5)
    // Should clamp to actual duration
    for (const bar of bars) {
      expect(bar.max).toBeCloseTo(0.5, 1)
    }
  })

  it('returns empty-like bars for zero targetBars', () => {
    const wf = makeWaveform(44100, 44100, 0.5)
    const bars = downsamplePeaks(wf, 0, 1, 0)
    expect(bars).toHaveLength(1)
  })

  it('returns single bar when targetBars is 1', () => {
    const wf = makeWaveform(44100, 44100, 0.7)
    const bars = downsamplePeaks(wf, 0, 1, 1)
    expect(bars).toHaveLength(1)
    expect(bars[0].max).toBeCloseTo(0.7, 1)
  })
})
