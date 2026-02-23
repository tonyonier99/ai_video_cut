import { useState, useMemo, useCallback } from 'react';
import type { Cut } from '../types';
import { getSnapTime as getSnapTimeUtil } from '../utils/timelineUtils';

interface UseTimelineOptions {
  duration: number;
  currentTime: number;
}

export function useTimeline({ duration, currentTime }: UseTimelineOptions) {
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [selectedCutIds, setSelectedCutIds] = useState<string[]>([]);
  const [zoomLevel, setZoomLevel] = useState(10);
  const [isMagnetEnabled, setIsMagnetEnabled] = useState(true);

  const totalDuration = useMemo(() => {
    if (cuts.length === 0) return duration;
    return Math.max(duration, ...cuts.map(c => c.end));
  }, [cuts, duration]);

  const getSnapTime = useCallback((time: number, ignoreCutId?: string | null): number | null => {
    return getSnapTimeUtil(time, cuts, currentTime, zoomLevel, ignoreCutId);
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

    setCuts([...newCuts]);
    setSelectedCutIds([newCutId]);
  }, [cuts, currentTime]);

  const handleDelete = useCallback(() => {
    if (selectedCutIds.length === 0) return;
    setCuts(prev => prev.filter(c => !selectedCutIds.includes(c.id)));
    setSelectedCutIds([]);
  }, [selectedCutIds]);

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
    alert("間隔已成功移除！片段現在已緊密對齊。");
  }, [cuts]);

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
    handleAlignCuts,
    handleZoom,
  };
}
