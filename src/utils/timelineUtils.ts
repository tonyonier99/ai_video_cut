import type { Cut, TrackTheme } from '../types';

const SNAP_THRESHOLD_PX = 10;

export function getSnapTime(
  time: number,
  cuts: Cut[],
  currentTime: number,
  zoomLevel: number,
  ignoreCutId?: string | null
): number | null {
  if (Math.abs(time - currentTime) * zoomLevel < SNAP_THRESHOLD_PX) return currentTime;
  if (Math.abs(time) * zoomLevel < SNAP_THRESHOLD_PX) return 0;

  let closestTime: number | null = null;
  let minDiff = SNAP_THRESHOLD_PX / zoomLevel;

  for (const cut of cuts) {
    if (cut.id === ignoreCutId) continue;

    const diffStart = Math.abs(cut.start - time);
    if (diffStart < minDiff) {
      minDiff = diffStart;
      closestTime = cut.start;
    }

    const diffEnd = Math.abs(cut.end - time);
    if (diffEnd < minDiff) {
      minDiff = diffEnd;
      closestTime = cut.end;
    }
  }

  return closestTime ?? null;
}

export function getTrackTheme(type: string): TrackTheme {
  switch (type) {
    case 'video': return { primary: '#3ea6ff', secondary: 'rgba(62,166,255,0.1)', border: '#2563eb' };
    case 'audio': return { primary: '#10b981', secondary: 'rgba(16,185,129,0.1)', border: '#059669' };
    case 'text': return { primary: '#6366f1', secondary: 'rgba(99,102,241,0.1)', border: '#4f46e5' };
    default: return { primary: '#3ea6ff', secondary: 'rgba(62,166,255,0.1)', border: '#007aff' };
  }
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

export function getTimelineCursor(
  isDragging: boolean,
  dragType: string | null,
  activeTool: string
): string {
  if (isDragging) {
    if (dragType === 'scrub') return 'ew-resize';
    if (dragType === 'pan') return 'grabbing';
    if (dragType === 'move') return 'move';
    if (dragType === 'trim-start' || dragType === 'trim-end') return 'ew-resize';
    return 'default';
  }

  switch (activeTool) {
    case 'hand': return 'grab';
    case 'blade': return 'crosshair';
    case 'text': return 'text';
    case 'select': return 'default';
    default: return 'default';
  }
}
