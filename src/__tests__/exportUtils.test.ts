import { describe, it, expect } from 'vitest';
import { generateFCPXML } from '../utils/exportUtils';
import type { Cut, Asset, TrackConfig } from '../types';

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
  fps: 30,
  width: 1920,
  height: 1080,
  ...overrides,
});

const defaultTracks: TrackConfig[] = [
  { id: 0, type: 'video', name: 'V1', visible: true, locked: false },
  { id: 1, type: 'text', name: 'T1', visible: true, locked: false },
  { id: 99, type: 'audio', name: 'A1', visible: true, locked: false },
];

describe('generateFCPXML', () => {
  const asset1 = makeAsset({ id: 'a1', name: 'clip1.mp4' });
  const asset2 = makeAsset({ id: 'a2', name: 'clip2.mp4', type: 'audio' });

  it('generates valid XML structure', () => {
    const cuts = [makeCut({ id: 'c1', start: 0, end: 10, assetId: 'a1' })];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test Project', 30);
    expect(xml).toContain('<?xml version="1.0" encoding="UTF-8"?>');
    expect(xml).toContain('<fcpxml version="1.8">');
    expect(xml).toContain('</fcpxml>');
    expect(xml).toContain('<library>');
    expect(xml).toContain('<spine>');
  });

  it('creates per-asset resource elements', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 10, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 10, end: 20, assetId: 'a2', trackId: 99 }),
    ];
    const xml = generateFCPXML(cuts, [asset1, asset2], defaultTracks, 'Test', 30);
    expect(xml).toContain('asset id="a_a1"');
    expect(xml).toContain('asset id="a_a2"');
    expect(xml).toContain('name="clip1.mp4"');
    expect(xml).toContain('name="clip2.mp4"');
  });

  it('uses correct frame duration for different fps', () => {
    const cuts = [makeCut({ id: 'c1', start: 0, end: 10, assetId: 'a1' })];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 29.97);
    expect(xml).toContain('frameDuration="100/2997s"');
  });

  it('inserts gap elements between non-contiguous clips', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 10, end: 15, assetId: 'a1' }),
    ];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 30);
    expect(xml).toContain('<gap');
    expect(xml).toContain('duration="150/30s"');
  });

  it('does not insert gap for contiguous clips', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 5, end: 10, assetId: 'a1' }),
    ];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 30);
    expect(xml).not.toContain('<gap');
  });

  it('places audio clips on lane -1', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 10, assetId: 'a2', trackId: 99 }),
    ];
    const xml = generateFCPXML(cuts, [asset2], defaultTracks, 'Test', 30);
    expect(xml).toContain('lane="-1"');
  });

  it('places text clips as title elements on lane 1', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 5, label: 'Hello World', trackId: 1 }),
    ];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 30);
    expect(xml).toContain('<title');
    expect(xml).toContain('lane="1"');
    expect(xml).toContain('name="Hello World"');
  });

  it('uses sourceStart for start attribute on asset-clip', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 5, end: 15, sourceStart: 30, sourceEnd: 40, assetId: 'a1' }),
    ];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 30);
    expect(xml).toContain('start="900/30s"');
  });

  it('calculates correct sequence duration', () => {
    const cuts = [
      makeCut({ id: 'c1', start: 0, end: 10, assetId: 'a1' }),
      makeCut({ id: 'c2', start: 10, end: 25, assetId: 'a1' }),
    ];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 30);
    expect(xml).toContain('duration="750/30s"');
  });

  it('handles empty cuts array', () => {
    const xml = generateFCPXML([], [asset1], defaultTracks, 'Test', 30);
    expect(xml).toContain('<spine>');
    expect(xml).toContain('</spine>');
    expect(xml).not.toContain('<asset-clip');
  });

  it('escapes XML special characters in names', () => {
    const cuts = [makeCut({ id: 'c1', start: 0, end: 5, label: 'Test & <Clip>', assetId: 'a1' })];
    const xml = generateFCPXML(cuts, [asset1], defaultTracks, 'Test', 30);
    expect(xml).toContain('Test &amp; &lt;Clip&gt;');
  });

  it('uses format element with detected resolution and fps', () => {
    const asset4k = makeAsset({ id: 'a1', width: 3840, height: 2160 });
    const cuts = [makeCut({ id: 'c1', start: 0, end: 5, assetId: 'a1' })];
    const xml = generateFCPXML(cuts, [asset4k], defaultTracks, 'Test', 24);
    expect(xml).toContain('width="3840"');
    expect(xml).toContain('height="2160"');
    expect(xml).toContain('FFVideoFormat2160p24');
  });

  it('uses project name in output', () => {
    const xml = generateFCPXML([], [asset1], defaultTracks, 'My Cool Project', 30);
    expect(xml).toContain('name="My Cool Project"');
  });
});
