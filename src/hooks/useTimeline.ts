import { useState, useMemo, useCallback } from 'react';
import type { Cut } from '../types';
import { getSnapTime as getSnapTimeUtil } from '../utils/timelineUtils';
import type { SnapResult } from '../utils/timelineUtils';

interface UseTimelineOptions {
  duration: number;
  currentTime: number;
  onHistoryCommit?: (cuts: Cut[]) => void;
}

export function useTimeline({ duration, currentTime, onHistoryCommit }: UseTimelineOptions) {
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [selectedCutIds, setSelectedCutIds] = useState<string[]>([]);
  const [zoomLevel, setZoomLevel] = useState(10);
  const [isMagnetEnabled, setIsMagnetEnabled] = useState(true);

  const totalDuration = useMemo(() => {
    if (cuts.length === 0) return duration;
    return Math.max(duration, ...cuts.map(c => c.end));
  }, [cuts, duration]);

  const getSnapTime = useCallback((time: number, ignoreCutId?: string | null, gridInterval?: number): SnapResult => {
    return getSnapTimeUtil(time, cuts, currentTime, zoomLevel, ignoreCutId, gridInterval);
  }, [cuts, currentTime, zoomLevel]);

  const handleSplit = useCallback((splitTime?: number) => {
    const time = splitTime !== undefined ? splitTime : currentTime;
    const targetCut = cuts.find(c => time > c.start + 0.01 && time < c.end - 0.01);
    if (!targetCut) return;

    const durationFromStart = time - targetCut.start;
    const sourceSplitTime = (targetCut.sourceStart ?? targetCut.start) + durationFromStart;

    const newCutId = Math.random().toString(36).substr(2, 9);
    const firstHalf: Cut = {
      ...targetCut,
      end: time,
      sourceEnd: sourceSplitTime,
      sourceStart: targetCut.sourceStart ?? targetCut.start
    };
    const secondHalf: Cut = {
      id: newCutId,
      start: time,
      end: targetCut.end,
      sourceStart: sourceSplitTime,
      sourceEnd: targetCut.sourceEnd ?? targetCut.end,
      label: targetCut.label,
      trackId: targetCut.trackId || 0,
      assetId: targetCut.assetId
    };

    const newCuts = cuts.map(c => c.id === targetCut.id ? firstHalf : c);
    const index = newCuts.findIndex(c => c.id === targetCut.id);
    newCuts.splice(index + 1, 0, secondHalf);

    const result = [...newCuts];
    setCuts(result);
    setSelectedCutIds([newCutId]);
    onHistoryCommit?.(result);
  }, [cuts, currentTime, onHistoryCommit]);

  const handleDelete = useCallback(() => {
    if (selectedCutIds.length === 0) return;
    const result = cuts.filter(c => !selectedCutIds.includes(c.id));
    setCuts(result);
    setSelectedCutIds([]);
    onHistoryCommit?.(result);
  }, [selectedCutIds, cuts, onHistoryCommit]);

  const handleRippleDelete = useCallback(() => {
    if (selectedCutIds.length === 0) return;

    const deletedSet = new Set(selectedCutIds);
    const deletedCuts = cuts.filter(c => deletedSet.has(c.id));
    const remaining = cuts.filter(c => !deletedSet.has(c.id));

    // Group deleted cuts by track
    const deletedByTrack = new Map<number, Cut[]>();
    for (const dc of deletedCuts) {
      const list = deletedByTrack.get(dc.trackId) ?? [];
      list.push(dc);
      deletedByTrack.set(dc.trackId, list);
    }

    // For each remaining cut on affected tracks, shift left
    const result = remaining.map(cut => {
      const trackDeleted = deletedByTrack.get(cut.trackId);
      if (!trackDeleted) return cut;

      // Sum durations of deleted cuts that were positioned before this cut
      const shift = trackDeleted
        .filter(dc => dc.start < cut.start)
        .reduce((sum, dc) => sum + (dc.end - dc.start), 0);

      if (shift <= 0) return cut;

      return {
        ...cut,
        start: Math.max(0, cut.start - shift),
        end: Math.max(0.01, cut.end - shift),
      };
    });

    setCuts(result);
    setSelectedCutIds([]);
    onHistoryCommit?.(result);
  }, [selectedCutIds, cuts, onHistoryCommit]);

  const handleAlignCuts = useCallback(() => {
    if (cuts.length === 0) return;

    const sortedCuts = [...cuts].sort((a, b) => a.start - b.start);
    let timelineOffset = 0;

    const newCuts = sortedCuts.map(cut => {
      const dur = cut.end - cut.start;
      const nc = {
        ...cut,
        start: timelineOffset,
        end: timelineOffset + dur,
        sourceStart: cut.sourceStart ?? cut.start,
        sourceEnd: cut.sourceEnd ?? cut.end
      };
      timelineOffset += dur;
      return nc;
    });

    setCuts(newCuts);
    onHistoryCommit?.(newCuts);
  }, [cuts, onHistoryCommit]);

  const handleZoom = useCallback((zoomIn: boolean) => {
    setZoomLevel(prev => {
      return zoomIn ? Math.min(prev * 1.2, 300) : Math.max(prev / 1.2, 1);
    });
  }, []);

  return {
    cuts,
    setCuts,
    selectedCutIds,
    setSelectedCutIds,
    zoomLevel,
    setZoomLevel,
    isMagnetEnabled,
    setIsMagnetEnabled,
    totalDuration,
    getSnapTime,
    handleSplit,
    handleDelete,
    handleRippleDelete,
    handleAlignCuts,
    handleZoom,
  };
}
