import { useState, useRef, useCallback } from 'react';
import type { Cut } from '../types';

const MAX_HISTORY = 50;

interface HistoryState {
  entries: Cut[][];
  index: number;
}

const INITIAL_STATE: HistoryState = { entries: [], index: -1 };

interface UseHistoryOptions {
  cuts: Cut[];
  setCuts: React.Dispatch<React.SetStateAction<Cut[]>>;
}

export function useHistory({ setCuts }: UseHistoryOptions) {
  const [state, setState] = useState<HistoryState>(INITIAL_STATE);
  const isUndoingRef = useRef(false);

  const canUndo = state.index > 0;
  const canRedo = state.index < state.entries.length - 1;
  const historyLength = state.entries.length;

  const initializeHistory = useCallback((initialCuts: Cut[]) => {
    setState({ entries: [initialCuts], index: 0 });
  }, []);

  const addToHistory = useCallback((newCuts: Cut[]) => {
    if (isUndoingRef.current) return;

    setState(prev => {
      // Truncate any future states beyond current index
      const truncated = prev.entries.slice(0, prev.index + 1);
      const updated = [...truncated, newCuts];
      const trimmed = updated.length > MAX_HISTORY
        ? updated.slice(updated.length - MAX_HISTORY)
        : updated;
      return {
        entries: trimmed,
        index: trimmed.length - 1,
      };
    });
  }, []);

  const handleUndo = useCallback(() => {
    isUndoingRef.current = true;
    setState(prev => {
      if (prev.index <= 0) {
        isUndoingRef.current = false;
        return prev;
      }
      setCuts(prev.entries[prev.index - 1]);
      return { ...prev, index: prev.index - 1 };
    });
    setTimeout(() => { isUndoingRef.current = false; }, 0);
  }, [setCuts]);

  const handleRedo = useCallback(() => {
    isUndoingRef.current = true;
    setState(prev => {
      if (prev.index >= prev.entries.length - 1) {
        isUndoingRef.current = false;
        return prev;
      }
      setCuts(prev.entries[prev.index + 1]);
      return { ...prev, index: prev.index + 1 };
    });
    setTimeout(() => { isUndoingRef.current = false; }, 0);
  }, [setCuts]);

  return {
    history: state.entries,
    historyIndex: state.index,
    historyLength,
    canUndo,
    canRedo,
    addToHistory,
    handleUndo,
    handleRedo,
    initializeHistory,
    isUndoingRef,
  };
}
