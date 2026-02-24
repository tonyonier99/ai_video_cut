import { describe, it, expect } from 'vitest';
import { generateEDL, secondsToTimecode } from '../utils/edlUtils';
import type { Cut, Asset } from '../types';

const makeCut = (overrides: Partial<Cut> & { id: string; start: number; end: number }): Cut => ({
  sourceStart: overrides.sourceStart ?? overrides.start,
  sourceEnd: overrides.sourceEnd ?? overrides.end,
  label: 'Test',
  trackId: 0,
  ...overrides,
});

const makeAsset = (overrides: Partial<Asset> & { id: string }): Asset => ({
  type: 'video',
  name: 'test.mp4',
  url: 'http://localhost/test.mp4',
  duration: 60,
  ...overrides,
});

describe('secondsToTimecode', () => {
  it('converts 0 seconds to 00:00:00:00', () => {
    expect(secondsToTimecode(0, 30)).toBe('00:00:00:00');
  });

  it('converts 1.5 seconds at 30fps', () => {
    expect(secondsToTimecode(1.5, 30)).toBe('00:00:01:15');
  });

  it('converts 3661 seconds (1h 1m 1s)', () => {
    expect(secondsToTimecode(3661, 30)).toBe('01:01:01:00');
  });

  it('handles fractional frames correctly at 24fps', () => {
    expect(secondsToTimecode(1.5, 24)).toBe('00:00:01:12');
  });

  it('handles negative input gracefully', () => {
    expect(secondsToTimecode(-5, 30)).toBe('00:00:00:00');
  });
});

describe('generateEDL', () => {
  const asset1 = makeAsset({ id: 'a1', name: 'interview.mp4' });
  const asset2 = makeAsset({ id: 'a2', name: 'bgmusic.wav', type: 'audio' });

  it('includes title and FCM header', () => {
    const edl = generateEDL([], [], 'My Project', 30);
    expect(edl).toContain('TITLE: My Project');
    expect(edl).toContain('FCM: NON-DROP FRAME');
  });

  it('generates correct event numbers', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 5, end: 10, assetId: 'a1' }),
    ];
    const edl = generateEDL(cuts, [asset1], 'Test', 30);
    expect(edl).toContain('001');
    expect(edl).toContain('002');
  });

  it('uses V for video and A for audio edit types', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 5, end: 10, assetId: 'a2', trackId: 99 }),
    ];
    const edl = generateEDL(cuts, [asset1, asset2], 'Test', 30);
    const lines = edl.split('\n');
    const videoLine = lines.find(l => l.startsWith('001'));
    const audioLine = lines.find(l => l.startsWith('002'));
    expect(videoLine).toContain('  V  C  ');
    expect(audioLine).toContain('  A  C  ');
  });

  it('truncates reel name to 8 characters', () => {
    const longAsset = makeAsset({ id: 'a1', name: 'verylongfilename.mp4' });
    const cuts = [makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' })];
    const edl = generateEDL(cuts, [longAsset], 'Test', 30);
    expect(edl).toContain('VERYLONG');
  });

  it('includes FROM CLIP NAME comments', () => {
    const cuts = [makeCut({ id: 'c1', start: 0, end: 5, label: 'Interview A', assetId: 'a1' })];
    const edl = generateEDL(cuts, [asset1], 'Test', 30);
    expect(edl).toContain('* FROM CLIP NAME: Interview A');
  });

  it('calculates correct source timecodes', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, sourceStart: 10, sourceEnd: 15, assetId: 'a1' }),
    ];
    const edl = generateEDL(cuts, [asset1], 'Test', 30);
    expect(edl).toContain('00:00:10:00 00:00:15:00');
  });

  it('calculates correct record timecodes for sequential clips', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 5, end: 12, assetId: 'a1' }),
    ];
    const edl = generateEDL(cuts, [asset1], 'Test', 30);
    expect(edl).toContain('00:00:00:00 00:00:05:00');
    expect(edl).toContain('00:00:05:00 00:00:12:00');
  });

  it('handles empty cuts', () => {
    const edl = generateEDL([], [asset1], 'Test', 30);
    expect(edl).toContain('TITLE: Test');
    const lines = edl.split('\n').filter(l => l.match(/^\d{3}/));
    expect(lines).toHaveLength(0);
  });
});
