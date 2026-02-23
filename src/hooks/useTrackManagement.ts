import { useState, useMemo, useCallback } from 'react';
import type { TrackConfig, Cut } from '../types';

interface UseTrackManagementOptions {
  setCuts: React.Dispatch<React.SetStateAction<Cut[]>>;
}

export function useTrackManagement({ setCuts }: UseTrackManagementOptions) {
  const [videoTracks, setVideoTracks] = useState<TrackConfig[]>([
    { id: 0, type: 'video', name: 'V1', visible: true, locked: false },
    { id: 1, type: 'text', name: 'T1', visible: true, locked: false },
    { id: 99, type: 'audio', name: 'A1', visible: true, locked: false }
  ]);

  const sortedTracks = useMemo(() => {
    const subtitles = videoTracks.filter(t => t.type === 'text').sort((a, b) => a.id - b.id);
    const videos = videoTracks.filter(t => t.type === 'video').sort((a, b) => b.id - a.id);
    const audios = videoTracks.filter(t => t.type === 'audio').sort((a, b) => a.id - b.id);
    return [...subtitles, ...videos, ...audios];
  }, [videoTracks]);

  const toggleTrackVisibility = useCallback((id: number) => {
    setVideoTracks(prev => prev.map(t => t.id === id ? { ...t, visible: !t.visible } : t));
  }, []);

  const toggleTrackLock = useCallback((id: number) => {
    setVideoTracks(prev => prev.map(t => t.id === id ? { ...t, locked: !t.locked } : t));
  }, []);

  const handleAddTrack = useCallback((type: 'video' | 'audio' | 'text') => {
    setVideoTracks(prev => {
      const sameType = prev.filter(t => t.type === type);
      const prefix = type === 'video' ? 'V' : type === 'audio' ? 'A' : 'T';
      const nextNum = sameType.length + 1;

      let nextId = Math.max(0, ...prev.map(t => t.id < 99 ? t.id : 0)) + 1;
      if (type === 'audio') {
        const audioIds = prev.filter(t => t.type === 'audio').map(t => t.id);
        nextId = audioIds.length > 0 ? Math.max(...audioIds) + 1 : 99;
      }

      return [...prev, {
        id: nextId,
        type,
        name: `${prefix}${nextNum}`,
        visible: true,
        locked: false
      }];
    });
  }, []);

  const handleDeleteTrack = useCallback((id: number) => {
    if (videoTracks.length <= 1) return;
    if (!confirm(`確定要刪除此軌道及其所有內容嗎？`)) return;

    setVideoTracks(prev => prev.filter(t => t.id !== id));
    setCuts(prev => prev.filter(c => c.trackId !== id));
  }, [videoTracks.length, setCuts]);

  return {
    videoTracks,
    setVideoTracks,
    sortedTracks,
    toggleTrackVisibility,
    toggleTrackLock,
    handleAddTrack,
    handleDeleteTrack,
  };
}
