export interface WaveformData {
  peaks: Float32Array;
  duration: number;
  sampleRate: number;
}

export interface WaveformBar {
  min: number;
  max: number;
}

let audioContextInstance: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!audioContextInstance) {
    audioContextInstance = new AudioContext();
  }
  return audioContextInstance;
}

export async function extractWaveformData(url: string): Promise<WaveformData> {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const audioContext = getAudioContext();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Merge all channels into mono peaks
  const channelCount = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const peaks = new Float32Array(length);

  for (let ch = 0; ch < channelCount; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      peaks[i] += Math.abs(channelData[i]);
    }
  }

  // Average across channels
  if (channelCount > 1) {
    for (let i = 0; i < length; i++) {
      peaks[i] /= channelCount;
    }
  }

  return {
    peaks,
    duration: audioBuffer.duration,
    sampleRate: audioBuffer.sampleRate,
  };
}

export function downsamplePeaks(
  waveform: WaveformData,
  sourceStart: number,
  sourceEnd: number,
  targetBars: number
): WaveformBar[] {
  const { peaks, duration, sampleRate } = waveform;

  const clampedStart = Math.max(0, sourceStart);
  const clampedEnd = Math.min(duration, sourceEnd);

  const startSample = Math.floor(clampedStart * sampleRate);
  const endSample = Math.min(Math.floor(clampedEnd * sampleRate), peaks.length);
  const totalSamples = endSample - startSample;

  if (totalSamples <= 0 || targetBars <= 0) {
    return Array.from({ length: Math.max(1, targetBars) }, () => ({ min: 0, max: 0 }));
  }

  const samplesPerBar = totalSamples / targetBars;
  const bars: WaveformBar[] = [];

  for (let i = 0; i < targetBars; i++) {
    const barStart = startSample + Math.floor(i * samplesPerBar);
    const barEnd = Math.min(startSample + Math.floor((i + 1) * samplesPerBar), endSample);

    let min = 0;
    let max = 0;

    for (let s = barStart; s < barEnd; s++) {
      const val = peaks[s];
      if (val > max) max = val;
      if (val < min) min = val;
    }

    bars.push({ min, max });
  }

  return bars;
}
