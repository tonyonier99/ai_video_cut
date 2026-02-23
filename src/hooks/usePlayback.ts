import { useState, useEffect, useCallback } from 'react';
import type { Cut } from '../types';

interface UsePlaybackOptions {
  totalDuration: number;
  cuts: Cut[];
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export function usePlayback({ totalDuration, cuts, videoRef }: UsePlaybackOptions) {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);

  // Enhanced Playback Timer
  useEffect(() => {
    let lastTime = performance.now();
    let frameId: number;

    const loop = () => {
      if (isPlaying) {
        const now = performance.now();
        const delta = (now - lastTime) / 1000;
        lastTime = now;

        setCurrentTime(prev => {
          const next = prev + delta;
          if (next >= totalDuration) {
            setIsPlaying(false);
            return totalDuration;
          }
          return next;
        });
        frameId = requestAnimationFrame(loop);
      }
    };

    if (isPlaying) {
      lastTime = performance.now();
      frameId = requestAnimationFrame(loop);
    }
    return () => cancelAnimationFrame(frameId);
  }, [isPlaying, totalDuration]);

  // Stop playback if timeline becomes empty
  useEffect(() => {
    if (cuts.length === 0 && isPlaying) {
      if (videoRef.current) videoRef.current.pause();
      setIsPlaying(false);
    }
  }, [cuts, isPlaying, videoRef]);

  const togglePlay = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  const handleMetadata = useCallback(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  }, [videoRef]);

  const seek = useCallback((time: number) => {
    setCurrentTime(time);
    if (videoRef.current) {
      const activeCut = cuts.find(c => time >= c.start && time < c.end);
      if (activeCut) {
        videoRef.current.currentTime = (activeCut.sourceStart ?? activeCut.start) + (time - activeCut.start);
      } else {
        videoRef.current.currentTime = time;
      }
    }
  }, [cuts, videoRef]);

  return {
    currentTime,
    setCurrentTime,
    isPlaying,
    setIsPlaying,
    duration,
    setDuration,
    togglePlay,
    handleMetadata,
    seek,
  };
}
