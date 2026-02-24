import { useState, useCallback } from 'react';
import type { Cut } from '../types';

interface UseMarkPointsOptions {
  getCurrentTime: () => number;
  setCuts: React.Dispatch<React.SetStateAction<Cut[]>>;
  defaultTrackId: number;
  defaultAssetId?: string;
  onHistoryCommit?: (cuts: Cut[]) => void;
}

export function useMarkPoints({ getCurrentTime, setCuts, defaultTrackId, defaultAssetId, onHistoryCommit }: UseMarkPointsOptions) {
  const [inPoint, setInPoint] = useState<number | null>(null);
  const [outPoint, setOutPoint] = useState<number | null>(null);

  const markIn = useCallback(() => {
    const time = getCurrentTime();
    setInPoint(time);
    if (outPoint !== null && outPoint < time) {
      setOutPoint(null);
    }
  }, [getCurrentTime, outPoint]);

  const markOut = useCallback(() => {
    const time = getCurrentTime();
    setOutPoint(time);
    if (inPoint !== null && inPoint > time) {
      setInPoint(null);
    }
  }, [getCurrentTime, inPoint]);

  const clearMarks = useCallback(() => {
    setInPoint(null);
    setOutPoint(null);
  }, []);

  const createCutFromMarks = useCallback(() => {
    if (inPoint === null || outPoint === null) return;
    if (outPoint <= inPoint) return;

    const newCut: Cut = {
      id: Math.random().toString(36).substr(2, 9),
      start: inPoint,
      end: outPoint,
      sourceStart: inPoint,
      sourceEnd: outPoint,
      label: `Mark ${inPoint.toFixed(1)}-${outPoint.toFixed(1)}`,
      trackId: defaultTrackId,
      assetId: defaultAssetId,
    };

    setCuts(prev => {
      const next = [...prev, newCut];
      onHistoryCommit?.(next);
      return next;
    });

    setInPoint(null);
    setOutPoint(null);
  }, [inPoint, outPoint, setCuts, defaultTrackId, defaultAssetId, onHistoryCommit]);

  return {
    inPoint,
    outPoint,
    markIn,
    markOut,
    clearMarks,
    createCutFromMarks,
  };
}
