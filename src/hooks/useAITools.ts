import { useState, useEffect, useRef, useCallback } from 'react';
import { loadApiKey, saveApiKey } from '../utils/secureApiKeyStorage';
import { API_BASE_URL } from '../config';
import type { Cut, TrackConfig, SubtitleConfig, JobStatus } from '../types';

interface UseAIToolsOptions {
  videoFile: File | null;
  videoTracks: TrackConfig[];
  setCuts: React.Dispatch<React.SetStateAction<Cut[]>>;
}

export function useAITools({ videoFile, videoTracks, setCuts }: UseAIToolsOptions) {
  // API Key
  const [apiKey, setApiKey] = useState('');
  const apiKeyLoaded = useRef(false);

  useEffect(() => {
    let cancelled = false;
    loadApiKey().then(key => {
      if (!cancelled) {
        setApiKey(key);
        apiKeyLoaded.current = true;
      }
    });
    localStorage.removeItem('antigravity_api_key');
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!apiKeyLoaded.current) return;
    saveApiKey(apiKey);
  }, [apiKey]);

  // Highlight params
  const [highlightCount, setHighlightCount] = useState(5);
  const [targetDuration, setTargetDuration] = useState(60);
  const [instruction, setInstruction] = useState('');
  const [geminiModel, setGeminiModel] = useState('gemini-3-flash-preview');

  // Silence Removal Params
  const [silenceThreshold, setSilenceThreshold] = useState(-30);
  const [silenceMinDuration, setSilenceMinDuration] = useState(0.5);

  // Whisper Params
  const [whisperModel, setWhisperModel] = useState('turbo');
  const [whisperLanguage, setWhisperLanguage] = useState('zh');
  const [whisperBeamSize, setWhisperBeamSize] = useState(5);
  const [whisperTemperature, setWhisperTemperature] = useState(0);
  const [whisperCharsPerLine, setWhisperCharsPerLine] = useState(14);
  const [whisperRemovePunc, setWhisperRemovePunc] = useState(true);

  // Font / Style State
  const [availableFonts, setAvailableFonts] = useState<Array<{ name: string; filename: string }>>([]);
  const [subtitleConfig, setSubtitleConfig] = useState<SubtitleConfig>({
    fontFamily: 'Inter',
    fontSize: 48,
    primaryColor: '#FFFFFF',
    useGradient: false,
    gradientStops: [
      { color: '#FFFFFF', offset: 0 },
      { color: '#3ea6ff', offset: 100 }
    ],
    gradientAngle: 180,
    safeZoneMargin: 10,
    useOutline: true,
    outlineColor: '#000000',
    outlineWidth: 4,
    useShadow: true,
    shadowColor: 'rgba(0,0,0,0.5)',
    shadowBlur: 8,
    shadowOffsetX: 2,
    shadowOffsetY: 2,
    useBackground: false,
    backgroundColor: '#000000',
    backgroundOpacity: 0.5,
    borderRadius: 8,
    paddingX: 20,
    paddingY: 10,
    verticalOffset: 80
  });

  const fetchFonts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/list-fonts`);
      if (res.ok) {
        const data = await res.json();
        setAvailableFonts(data.fonts);

        const styleId = 'dynamic-fonts-style';
        let styleEl = document.getElementById(styleId);
        if (styleEl) styleEl.remove();

        styleEl = document.createElement('style');
        styleEl.id = styleId;
        document.head.appendChild(styleEl);

        let css = '';
        data.fonts.forEach((font: { name: string; filename: string }) => {
          css += `
            @font-face {
              font-family: '${font.name}';
              src: url('${API_BASE_URL}/fonts/${encodeURIComponent(font.filename)}');
              font-weight: normal;
              font-style: normal;
            }
          `;
        });
        styleEl.textContent = css;
      }
    } catch (_e) {
      // Font fetch failed silently
    }
  }, []);

  useEffect(() => {
    fetchFonts();
  }, [fetchFonts]);

  // Processing status
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentJobStatus, setCurrentJobStatus] = useState<JobStatus | null>(null);

  // Polling for Progress
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isProcessing) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/job-status`);
          if (res.ok) {
            const status = await res.json();
            setCurrentJobStatus(status);
          }
        } catch (_e) {
          // Polling error silently ignored
        }
      }, 800);
    } else {
      setCurrentJobStatus(null);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  const handleGeminiHighlights = useCallback(async () => {
    if (!videoFile || !apiKey) return;
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('instruction', instruction || "Find the best highlights");
    formData.append('target_count', highlightCount.toString());
    formData.append('target_duration', targetDuration.toString());
    formData.append('api_key', apiKey);
    formData.append('model_name', geminiModel);

    try {
      const res = await fetch(`${API_BASE_URL}/analyze-video`, { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        const aiCuts: Cut[] = data.map((c: { start: number; end: number; label?: string }, i: number) => ({
          id: `ai-${i}`,
          start: c.start,
          end: c.end,
          sourceStart: c.start,
          sourceEnd: c.end,
          label: c.label || `Highlight ${i + 1}`,
          trackId: 0
        }));

        if (confirm(`AI Found ${aiCuts.length} clips. Replace timeline?`)) {
          setCuts(aiCuts);
        }
      } else {
        alert("AI Analysis Failed");
      }
    } catch (_e) {
      alert("Error connecting to backend");
    }
    setIsProcessing(false);
  }, [videoFile, apiKey, instruction, highlightCount, targetDuration, geminiModel, setCuts]);

  const handleSilenceRemoval = useCallback(async () => {
    if (!videoFile) return;
    if (!confirm("這將會移除影片中的靜音部分並取代目前的時間軸。確定嗎？")) return;

    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('threshold_db', silenceThreshold.toString());
    formData.append('min_duration', silenceMinDuration.toString());

    try {
      const res = await fetch(`${API_BASE_URL}/detect-silence`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const segments = await res.json();
        const newCuts: Cut[] = segments.map((seg: { start: number; end: number }, i: number) => ({
          id: `silence-${i}`,
          start: seg.start,
          end: seg.end,
          sourceStart: seg.start,
          sourceEnd: seg.end,
          label: 'Speech',
          trackId: 0
        }));

        if (newCuts.length === 0) {
          alert("未偵測到任何語音片段 (No speech found)");
        } else {
          setCuts(newCuts);
        }
      } else {
        alert("偵測失敗");
      }
    } catch (_e) {
      alert("連線後端失敗");
    } finally {
      setIsProcessing(false);
    }
  }, [videoFile, silenceThreshold, silenceMinDuration, setCuts]);

  const handleAISubtitles = useCallback(async () => {
    if (!videoFile) {
      alert("請先上傳影片以便辨識");
      return;
    }

    setIsProcessing(true);
    setCurrentJobStatus({ progress: 0, message: "準備辨識中...", step: "init" });

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('whisper_language', whisperLanguage);
    formData.append('whisper_model_size', whisperModel);
    formData.append('whisper_beam_size', whisperBeamSize.toString());
    formData.append('whisper_temperature', whisperTemperature.toString());
    formData.append('whisper_chars_per_line', whisperCharsPerLine.toString());
    formData.append('whisper_remove_punctuation', whisperRemovePunc ? 'true' : 'false');

    try {
      const res = await fetch(`${API_BASE_URL}/transcribe`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const segments = await res.json();
        const textTrack = videoTracks.find(t => t.type === 'text');
        const tid = textTrack ? textTrack.id : 1;

        const newCuts: Cut[] = segments.map((seg: { start: number; end: number; text: string }) => ({
          id: `subtitle-${Math.random().toString(36).substr(2, 5)}`,
          start: seg.start,
          end: seg.end,
          sourceStart: 0,
          sourceEnd: 0,
          label: seg.text,
          trackId: tid
        }));

        if (newCuts.length > 0) {
          setCuts(prev => [...prev.filter(c => {
            const tr = videoTracks.find(t => t.id === c.trackId);
            return tr && tr.type !== 'text';
          }), ...newCuts]);
        } else {
          alert("未辨識到任何語音內容");
        }
      } else {
        const err = await res.text();
        alert("辨識失敗: " + err);
      }
    } catch (_e) {
      alert("連線後端失敗");
    } finally {
      setIsProcessing(false);
      setCurrentJobStatus(null);
    }
  }, [videoFile, whisperLanguage, whisperModel, whisperBeamSize, whisperTemperature, whisperCharsPerLine, whisperRemovePunc, videoTracks, setCuts]);

  return {
    apiKey,
    setApiKey,
    apiKeyLoaded,
    highlightCount,
    setHighlightCount,
    targetDuration,
    setTargetDuration,
    instruction,
    setInstruction,
    geminiModel,
    setGeminiModel,
    silenceThreshold,
    setSilenceThreshold,
    silenceMinDuration,
    setSilenceMinDuration,
    whisperModel,
    setWhisperModel,
    whisperLanguage,
    setWhisperLanguage,
    whisperBeamSize,
    setWhisperBeamSize,
    whisperTemperature,
    setWhisperTemperature,
    whisperCharsPerLine,
    setWhisperCharsPerLine,
    whisperRemovePunc,
    setWhisperRemovePunc,
    availableFonts,
    subtitleConfig,
    setSubtitleConfig,
    fetchFonts,
    isProcessing,
    setIsProcessing,
    currentJobStatus,
    setCurrentJobStatus,
    handleGeminiHighlights,
    handleSilenceRemoval,
    handleAISubtitles,
  };
}
