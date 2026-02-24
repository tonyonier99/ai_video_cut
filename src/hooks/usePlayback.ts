import { useState, useEffect, useCallback, useRef } from 'react';
import type { Cut } from '../types';

interface UsePlaybackOptions {
  totalDuration: number;
  cuts: Cut[];
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

const FORWARD_RATES = [1, 2, 4, 8];
const REVERSE_RATES = [-1, -2, -4, -8];

export function usePlayback({ totalDuration, cuts, videoRef }: UsePlaybackOptions) {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const forwardIndexRef = useRef(0);
  const reverseIndexRef = useRef(0);

  useEffect(() => {
    let lastTime = performance.now();
    let frameId: number;

    const loop = () => {
      if (isPlaying) {
        const now = performance.now();
        const delta = (now - lastTime) / 1000;
        lastTime = now;

        setCurrentTime(prev => {
          const next = prev + delta * playbackRate;
          if (next >= totalDuration) {
            setIsPlaying(false);
            return totalDuration;
          }
          if (next <= 0) {
            setIsPlaying(false);
            return 0;
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
  }, [isPlaying, totalDuration, playbackRate]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = Math.abs(playbackRate) || 1;
    }
  }, [playbackRate, videoRef]);

  useEffect(() => {
    if (cuts.length === 0 && isPlaying) {
      if (videoRef.current) videoRef.current.pause();
      queueMicrotask(() => setIsPlaying(false));
    }
  }, [cuts, isPlaying, videoRef]);

  const togglePlay = useCallback(() => {
    setIsPlaying(prev => !prev);
    setPlaybackRate(1);
    forwardIndexRef.current = 0;
    reverseIndexRef.current = 0;
  }, []);

  const pause = useCallback(() => {
    setIsPlaying(false);
    setPlaybackRate(1);
    forwardIndexRef.current = 0;
    reverseIndexRef.current = 0;
  }, []);

  const cycleForwardRate = useCallback(() => {
    reverseIndexRef.current = 0;
    const nextIdx = Math.min(forwardIndexRef.current + 1, FORWARD_RATES.length - 1);
    if (!isPlaying || playbackRate < 0) {
      forwardIndexRef.current = 0;
      setPlaybackRate(FORWARD_RATES[0]);
    } else {
      forwardIndexRef.current = nextIdx;
      setPlaybackRate(FORWARD_RATES[nextIdx]);
    }
    setIsPlaying(true);
  }, [isPlaying, playbackRate]);

  const cycleReverseRate = useCallback(() => {
    forwardIndexRef.current = 0;
    const nextIdx = Math.min(reverseIndexRef.current + 1, REVERSE_RATES.length - 1);
    if (!isPlaying || playbackRate > 0) {
      reverseIndexRef.current = 0;
      setPlaybackRate(REVERSE_RATES[0]);
    } else {
      reverseIndexRef.current = nextIdx;
      setPlaybackRate(REVERSE_RATES[nextIdx]);
    }
    setIsPlaying(true);
  }, [isPlaying, playbackRate]);

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
    playbackRate,
    setPlaybackRate,
    togglePlay,
    pause,
    cycleForwardRate,
    cycleReverseRate,
    handleMetadata,
    seek,
  };
}
