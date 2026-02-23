import { useRef, useState, useCallback } from 'react';
import type { WaveformData } from '../utils/waveformUtils';
import { extractWaveformData } from '../utils/waveformUtils';

interface WaveformCacheEntry {
  data: WaveformData | null;
  loading: boolean;
  error: string | null;
}

export function useWaveformCache() {
  const cacheRef = useRef<Map<string, WaveformCacheEntry>>(new Map());
  const [, setTick] = useState(0);

  const forceUpdate = useCallback(() => {
    setTick(t => t + 1);
  }, []);

  const getWaveform = useCallback((assetId: string): WaveformCacheEntry | undefined => {
    return cacheRef.current.get(assetId);
  }, []);

  const loadWaveform = useCallback(async (assetId: string, url: string) => {
    const existing = cacheRef.current.get(assetId);
    if (existing?.data || existing?.loading) return;

    cacheRef.current.set(assetId, { data: null, loading: true, error: null });
    forceUpdate();

    try {
      const data = await extractWaveformData(url);
      cacheRef.current.set(assetId, { data, loading: false, error: null });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load waveform';
      cacheRef.current.set(assetId, { data: null, loading: false, error: message });
    }
    forceUpdate();
  }, [forceUpdate]);

  return {
    getWaveform,
    loadWaveform,
  };
}
