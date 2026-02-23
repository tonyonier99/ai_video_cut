import { useState, useRef, useEffect, useCallback } from 'react';
import type { Cut } from '../types';

const MAX_HISTORY = 20;

interface UseHistoryOptions {
  cuts: Cut[];
  setCuts: React.Dispatch<React.SetStateAction<Cut[]>>;
}

export function useHistory({ cuts, setCuts }: UseHistoryOptions) {
  const [history, setHistory] = useState<Cut[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const isUndoingRef = useRef(false);

  // Initialize history
  useEffect(() => {
    if (history.length === 0 && cuts.length > 0) {
      setHistory([cuts]);
      setHistoryIndex(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cuts.length]);

  const addToHistory = useCallback((newCuts: Cut[]) => {
    if (isUndoingRef.current) return;

    setHistory(prev => {
      const newHistory = prev.slice(0, historyIndex + 1);
      newHistory.push(newCuts);
      if (newHistory.length > MAX_HISTORY) {
        newHistory.shift();
      }
      return newHistory;
    });
    setHistoryIndex(prev => {
      const effectiveLength = Math.min(prev + 2, MAX_HISTORY);
      return effectiveLength - 1;
    });
  }, [historyIndex]);

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      isUndoingRef.current = true;
      const prevCuts = history[historyIndex - 1];
      setCuts(prevCuts);
      setHistoryIndex(historyIndex - 1);
      setTimeout(() => { isUndoingRef.current = false; }, 0);
    }
  }, [historyIndex, history, setCuts]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      isUndoingRef.current = true;
      const nextCuts = history[historyIndex + 1];
      setCuts(nextCuts);
      setHistoryIndex(historyIndex + 1);
      setTimeout(() => { isUndoingRef.current = false; }, 0);
    }
  }, [historyIndex, history, setCuts]);

  return {
    history,
    historyIndex,
    addToHistory,
    handleUndo,
    handleRedo,
    isUndoingRef,
  };
}
