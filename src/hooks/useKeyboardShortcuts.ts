import { useEffect, useRef } from 'react';
import type { ActiveTool } from '../types';

interface KeyboardHandlers {
  togglePlay: () => void;
  handleSplit: () => void;
  handleDelete: () => void;
  handleRippleDelete: () => void;
  handleUndo: () => void;
  handleRedo: () => void;
  handleSelectAll: () => void;
  handleDeselectAll: () => void;
  setActiveTool: React.Dispatch<React.SetStateAction<ActiveTool>>;
  handleZoom: (zoomIn: boolean) => void;
  pause?: () => void;
  cycleForwardRate?: () => void;
  cycleReverseRate?: () => void;
  setPlaybackRate?: (rate: number) => void;
  markIn?: () => void;
  markOut?: () => void;
}

export function useKeyboardShortcuts(handlers: KeyboardHandlers) {
  const handlersRef = useRef<KeyboardHandlers>(handlers);
  useEffect(() => {
    handlersRef.current = handlers;
  });

  const kHeldRef = useRef(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        handlersRef.current.togglePlay();
      } else if (e.code === 'KeyK' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        kHeldRef.current = true;
        handlersRef.current.pause?.();
      } else if (e.code === 'KeyK' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handlersRef.current.handleSplit();
      } else if (e.code === 'KeyL' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        if (kHeldRef.current) {
          handlersRef.current.setPlaybackRate?.(0.25);
        } else {
          handlersRef.current.cycleForwardRate?.();
        }
      } else if (e.code === 'KeyJ' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        if (kHeldRef.current) {
          handlersRef.current.setPlaybackRate?.(-0.25);
        } else {
          handlersRef.current.cycleReverseRate?.();
        }
      } else if (e.code === 'KeyI' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        handlersRef.current.markIn?.();
      } else if (e.code === 'KeyO' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        handlersRef.current.markOut?.();
      } else if (e.code === 'KeyA' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handlersRef.current.handleSelectAll();
      } else if (e.code === 'Escape') {
        handlersRef.current.handleDeselectAll();
      } else if (e.code === 'KeyC') {
        if (!e.metaKey && !e.ctrlKey) {
          handlersRef.current.setActiveTool('blade');
        }
      } else if (e.code === 'KeyV') {
        if (!e.metaKey && !e.ctrlKey) {
          handlersRef.current.setActiveTool('select');
        }
      } else if (e.code === 'Backspace' || e.code === 'Delete') {
        if (e.shiftKey) {
          e.preventDefault();
          handlersRef.current.handleRippleDelete();
        } else {
          handlersRef.current.handleDelete();
        }
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

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'KeyK') {
        kHeldRef.current = false;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);
}
