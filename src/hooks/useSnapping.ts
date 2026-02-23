import { useState, useCallback } from 'react';
import type { SnapType } from '../utils/timelineUtils';

interface SnapState {
  snapTime: number | null;
  snapType: SnapType;
}

const INITIAL_STATE: SnapState = { snapTime: null, snapType: null };

export function useSnapping() {
  const [snapState, setSnapState] = useState<SnapState>(INITIAL_STATE);
  const [gridInterval, setGridInterval] = useState<number>(0); // 0 = off

  const showSnap = useCallback((time: number | null, type: SnapType) => {
    setSnapState({ snapTime: time, snapType: type });
  }, []);

  const clearSnap = useCallback(() => {
    setSnapState(INITIAL_STATE);
  }, []);

  return {
    snapTime: snapState.snapTime,
    snapType: snapState.snapType,
    gridInterval,
    setGridInterval,
    showSnap,
    clearSnap,
  };
}
