import type { SnapType } from '../utils/timelineUtils';

interface SnapIndicatorProps {
  snapTime: number | null;
  snapType: SnapType;
  zoomLevel: number;
  containerHeight: number;
}

const SNAP_COLORS: Record<string, string> = {
  'cut-start': '#10b981',
  'cut-end': '#10b981',
  'grid': '#eab308',
  'playhead': '#3ea6ff',
  'zero': '#3ea6ff',
};

export function SnapIndicator({ snapTime, snapType, zoomLevel, containerHeight }: SnapIndicatorProps) {
  if (snapTime === null || !snapType) return null;

  const color = SNAP_COLORS[snapType] ?? '#10b981';
  const left = snapTime * zoomLevel;

  return (
    <div
      className="snap-indicator"
      style={{
        position: 'absolute',
        left,
        top: 0,
        width: 1,
        height: containerHeight,
        background: color,
        boxShadow: `0 0 6px ${color}`,
        pointerEvents: 'none',
        zIndex: 50,
        opacity: 0.8,
      }}
    />
  );
}
