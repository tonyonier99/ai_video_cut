import { useEffect, useRef } from 'react';
import type { ActiveTool } from '../types';

interface KeyboardHandlers {
  togglePlay: () => void;
  handleSplit: () => void;
  handleDelete: () => void;
  handleUndo: () => void;
  handleRedo: () => void;
  setActiveTool: React.Dispatch<React.SetStateAction<ActiveTool>>;
  handleZoom: (zoomIn: boolean) => void;
}

export function useKeyboardShortcuts(handlers: KeyboardHandlers) {
  const handlersRef = useRef<KeyboardHandlers>(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        handlersRef.current.togglePlay();
      } else if (e.code === 'KeyK' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handlersRef.current.handleSplit();
      } else if (e.code === 'KeyC') {
        if (!e.metaKey && !e.ctrlKey) {
          handlersRef.current.setActiveTool('blade');
        }
      } else if (e.code === 'KeyV') {
        if (!e.metaKey && !e.ctrlKey) {
          handlersRef.current.setActiveTool('select');
        }
      } else if (e.code === 'Backspace' || e.code === 'Delete') {
        handlersRef.current.handleDelete();
      } else if (e.code === 'KeyZ' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        if (e.shiftKey) {
          handlersRef.current.handleRedo();
        } else {
          handlersRef.current.handleUndo();
        }
      } else if ((e.metaKey || e.ctrlKey) && (e.code === 'Equal' || e.code === 'NumpadAdd')) {
        e.preventDefault();
        handlersRef.current.handleZoom(true);
      } else if ((e.metaKey || e.ctrlKey) && (e.code === 'Minus' || e.code === 'NumpadSubtract')) {
        e.preventDefault();
        handlersRef.current.handleZoom(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
}
