import { useRef, useEffect } from 'react';
import type { WaveformData } from '../utils/waveformUtils';
import { downsamplePeaks } from '../utils/waveformUtils';

interface WaveformCanvasProps {
  waveformData: WaveformData;
  sourceStart: number;
  sourceEnd: number;
  width: number;
  height: number;
  color?: string;
}

export function WaveformCanvas({
  waveformData,
  sourceStart,
  sourceEnd,
  width,
  height,
  color = 'rgba(255,255,255,0.5)',
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const barWidth = 3;
    const gap = 1;
    const targetBars = Math.max(1, Math.floor(width / (barWidth + gap)));

    const bars = downsamplePeaks(waveformData, sourceStart, sourceEnd, targetBars);
    const centerY = height / 2;
    const maxHeight = height * 0.8;

    ctx.fillStyle = color;

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const x = i * (barWidth + gap);
      const barH = Math.max(1, bar.max * maxHeight);

      ctx.fillRect(x, centerY - barH / 2, barWidth, barH);
    }
  }, [waveformData, sourceStart, sourceEnd, width, height, color]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width,
        height,
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        opacity: 0.7,
      }}
    />
  );
}
