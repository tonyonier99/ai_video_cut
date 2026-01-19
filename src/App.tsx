import React, { useState, useRef, useEffect } from 'react';
import { Download, Video, Scissors, Play, Pause, Loader2, Film, Key, Upload, Wand2, Zap, Languages, Trash, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import WaveSurfer from 'wavesurfer.js';
import { Player } from '@remotion/player';
import { MyComposition } from './remotion/MyComposition';
import './App.css';

interface Cut {
  id: string;
  start: number;
  end: number;
  label: string;
}


function App() {
  const formatTimestamp = (timeStr: string | number) => {
    const time = typeof timeStr === 'string' ? parseFloat(timeStr) : timeStr;
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const milliseconds = Math.floor((time % 1) * 10); // Show 1 decimal place
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds}`;
  };

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);


  const [targetCount, setTargetCount] = useState(1);
  const [targetDuration, setTargetDuration] = useState(15);
  const [instruction, setInstruction] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showApiSettings, setShowApiSettings] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gemini-3.0-flash');

  // AI 核心功能
  const [isFaceTracking, setIsFaceTracking] = useState(true);
  const [isStudioSound, setIsStudioSound] = useState(false); // Default disabled
  const [isAutoCaption, setIsAutoCaption] = useState(true);
  const [isTranslate, setIsTranslate] = useState(true); // Default enabled for Traditional Chinese Optimization
  const [isBurnCaptions, setIsBurnCaptions] = useState(true);
  const [whisperModelSize, setWhisperModelSize] = useState('turbo');
  const [whisperBeamSize, setWhisperBeamSize] = useState(5);
  const [whisperRemovePunctuation, setWhisperRemovePunctuation] = useState(true);
  const [whisperTemperature, setWhisperTemperature] = useState(0);
  const [whisperNoSpeechThreshold, setWhisperNoSpeechThreshold] = useState(0.6);
  const [whisperConditionOnPreviousText, setWhisperConditionOnPreviousText] = useState(true);
  const [whisperBestOf, setWhisperBestOf] = useState(5);
  const [whisperPatience, setWhisperPatience] = useState(1.0);
  const [whisperCompressionRatioThreshold, setWhisperCompressionRatioThreshold] = useState(2.4);
  const [whisperLogprobThreshold, setWhisperLogprobThreshold] = useState(-1.0);
  const [whisperFp16, setWhisperFp16] = useState(true);
  const [showExpertWhisper, setShowExpertWhisper] = useState(false);

  // 進階參數
  const [trackZoom, setTrackZoom] = useState(1.5);
  const [mpMinDetectionCon, setMpMinDetectionCon] = useState(0.5);
  const [dfn3Strength, setDfn3Strength] = useState(100);

  // Silence Removal & Jump Cut Parameters
  const [isSilenceRemoval, setIsSilenceRemoval] = useState(false);
  const [silenceThreshold, setSilenceThreshold] = useState(0.5); // Seconds
  const [isJumpCutZoom, setIsJumpCutZoom] = useState(true);

  // 字幕樣式
  // 字幕樣式 (Advanced)
  const [subtitleFontSize, setSubtitleFontSize] = useState(90);
  const [subtitleFontName, setSubtitleFontName] = useState('Arial');
  const [subtitleFontWeight, setSubtitleFontWeight] = useState<string | number>('normal');
  const [subtitleFontStyle, setSubtitleFontStyle] = useState('normal');

  const [subtitleTextColor, setSubtitleTextColor] = useState('#FFFFFF');
  const [isTextGradient, setIsTextGradient] = useState(false);
  const [textGradientColors, setTextGradientColors] = useState(['#FF0080', '#7928CA']); // Default nice gradient
  const [textGradientDirection, setTextGradientDirection] = useState('to right');

  const [subtitleOutlineWidth, setSubtitleOutlineWidth] = useState(4);
  const [subtitleOutlineColor, setSubtitleOutlineColor] = useState('#000000');
  const [isSubtitleOutline, setIsSubtitleOutline] = useState(true);

  const [subtitleShadowColor, setSubtitleShadowColor] = useState('#000000');
  const [subtitleShadowOpacity, setSubtitleShadowOpacity] = useState(80);
  const [subtitleShadowBlur, setSubtitleShadowBlur] = useState(0);
  const [subtitleShadowOffsetX, setSubtitleShadowOffsetX] = useState(3);
  const [subtitleShadowOffsetY, setSubtitleShadowOffsetY] = useState(3);
  const [isSubtitleShadow, setIsSubtitleShadow] = useState(true);

  const [subtitleLetterSpacing, setSubtitleLetterSpacing] = useState(0);
  const [subtitleLineHeight, setSubtitleLineHeight] = useState(1.2);
  const [subtitleTextTransform, setSubtitleTextTransform] = useState('none');
  const [subtitleTextAlign, setSubtitleTextAlign] = useState('center');

  const [subtitleMarginV, setSubtitleMarginV] = useState(600);
  const [subtitleCharsPerLine, setSubtitleCharsPerLine] = useState(9); // Visual Wrap
  const [whisperCharsPerLine, setWhisperCharsPerLine] = useState(14); // Transcription Limit

  const [subtitleBgEnabled, setSubtitleBgEnabled] = useState(false);
  const [subtitleBgColor, setSubtitleBgColor] = useState('#000000');
  const [subtitleBgOpacity, setSubtitleBgOpacity] = useState(50);
  const [subtitleBgPaddingX, setSubtitleBgPaddingX] = useState(10);
  const [subtitleBgPaddingY, setSubtitleBgPaddingY] = useState(4);
  const [subtitleBgRadius, setSubtitleBgRadius] = useState(4);

  const [subtitleAnimation, setSubtitleAnimation] = useState<'none' | 'pop' | 'fade' | 'slide-up'>('none');
  const [subtitleAnimationDuration, setSubtitleAnimationDuration] = useState(15);
  const [subtitleAnimationSpring, setSubtitleAnimationSpring] = useState(0.5); // Mass / Intensity

  const [showSafeArea, setShowSafeArea] = useState(true);
  const [previewText, setPreviewText] = useState('預覽文字內容預覽文字內容');
  const [isDraggingSubtitle, setIsDraggingSubtitle] = useState(false);


  const [previewCut, setPreviewCut] = useState<Cut | null>(null); // NEW: Remotion Preview State
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewProgress, setPreviewProgress] = useState(0);
  const [previewFaceCenter, setPreviewFaceCenter] = useState(0.5); // NEW for Face Tracking
  const [previewMessage, setPreviewMessage] = useState(''); // NEW: Detailed status
  const [previewSubtitles, setPreviewSubtitles] = useState<any[]>([]); // Temp subs for preview
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [previewVisualSegments, setPreviewVisualSegments] = useState<any[]>([]);
  const [srtSubtitles, setSrtSubtitles] = useState<any[]>([]); // User uploaded SRT content


  const [draggingCut, setDraggingCut] = useState<{ id: string, edge: 'start' | 'end' | 'move', initialX: number, initialStart: number, initialEnd: number } | null>(null);
  const [systemStatus, setSystemStatus] = useState({ progress: 100, status: 'ready', message: '就緒' });
  const [fontList, setFontList] = useState(['Arial', 'Sans-serif']); // Remove client-local fonts that backend doesn't have
  const [outputQuality, setOutputQuality] = useState('high');
  const [outputResolution, setOutputResolution] = useState('1080x1920');
  const [whisperLanguage, setWhisperLanguage] = useState('zh');

  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  // NEW: Precise Subtitle Scaling System
  const [scaleRatio, setScaleRatio] = useState(0.4);
  const previewCanvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!previewCanvasRef.current) return;
    const updateScale = () => {
      if (previewCanvasRef.current) {
        // Standardize scaling: 
        // User sets font size relative to 1080x1920 canvas.
        // We scale the DOM preview linearly based on the canvas height vs 1920 (Height).
        setScaleRatio(previewCanvasRef.current.offsetHeight / 1920);
      }
    };
    const observer = new ResizeObserver(updateScale);
    observer.observe(previewCanvasRef.current);
    updateScale(); // Init relative to current size
    return () => observer.disconnect();
  }, []);

  // Poll Job Status when Exporting to show detailed steps (like "Applying Subtitles")
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isExporting) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('http://localhost:8000/job-status');
          if (res.ok) {
            const status = await res.json();
            if (status.step !== 'idle' && status.step !== 'done') {
              setExportProgress(status.progress);
              setSystemStatus(prev => ({ ...prev, message: status.message }));
            }
          }
        } catch (e) { }
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isExporting]);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;

    const pollStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/model-status');
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
          // Stop polling once model is ready
          if (data.status === 'ready' && timer) {
            clearInterval(timer);
            timer = null;
          }
        }
      } catch (e) { }
    };

    timer = setInterval(pollStatus, 2000);
    pollStatus();

    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  // Load available fonts from server
  useEffect(() => {
    const loadFonts = async () => {
      try {
        const res = await fetch('http://localhost:8000/list-fonts');
        if (res.ok) {
          const data = await res.json();
          if (data.fonts && data.fonts.length > 0) {
            setFontList(() => {
              // Only show fonts that actually exist on backend + basic fallbacks
              const combined = [...new Set([...data.fonts, 'Arial', 'Sans-serif'])];

              // Dynamically load fonts for accurate preview
              data.fonts.forEach(async (fontName: string) => {
                const baseUrl = `http://localhost:8000/fonts/${encodeURIComponent(fontName)}`;
                // Try multiple extensions
                const extensions = ['.ttf', '.otf', '.TTF', '.OTF'];
                let loaded = false;

                for (const ext of extensions) {
                  if (loaded) break;
                  try {
                    const font = new FontFace(fontName, `url(${baseUrl}${ext})`);
                    await font.load();
                    document.fonts.add(font);
                    loaded = true;
                    console.log(`✅ Loaded font: ${fontName} (${ext})`);
                  } catch (e) {
                    // Silently try next extension
                  }
                }
                if (!loaded) {
                  console.warn(`❌ Failed to load font: ${fontName} after trying various extensions.`);
                }
              });

              return combined;
            });
          }
        }
      } catch (e) { /* Server not running */ }
    };
    loadFonts();
  }, []);

  // Initialize WaveSurfer
  useEffect(() => {
    if (videoUrl && waveformRef.current && !wavesurferRef.current) {
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#6366f1',
        progressColor: '#818cf8',
        cursorColor: '#fff',
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        height: 50,
        normalize: true,
        backend: 'MediaElement',
        media: videoRef.current || undefined,
      });
      wavesurferRef.current.load(videoUrl);
    }
    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
  }, [videoUrl]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setCuts([]);
      setSrtSubtitles([]); // Clear SRT when new video
    }
  };

  const handleMakeVideo = async () => {
    if (!videoFile) return;
    setIsProcessing(true);

    // If no API key, generate random cuts based on settings
    if (!apiKey.trim()) {
      const videoDur = duration || 60;
      const count = targetCount || 1;
      const clipLen = targetDuration || 15;
      const newCuts: Cut[] = [];

      for (let i = 0; i < count; i++) {
        const maxStart = Math.max(0, videoDur - clipLen);
        const start = Math.random() * maxStart;
        const end = Math.min(start + clipLen, videoDur);
        newCuts.push({
          id: i.toString(),
          start: parseFloat(start.toFixed(2)),
          end: parseFloat(end.toFixed(2)),
          label: `片段 ${i + 1}`
        });
      }
      setCuts(newCuts);
      setIsProcessing(false);
      return;
    }

    // With API key, use AI analysis
    const formData = new FormData();
    formData.append('file', videoFile as any);
    formData.append('instruction', instruction);
    if (targetCount) formData.append('target_count', targetCount.toString());
    if (targetDuration) formData.append('target_duration', targetDuration.toString());
    formData.append('api_key', apiKey);
    formData.append('model_name', selectedModel);

    try {
      console.log("🚀 Starting AI Analysis...", { model: selectedModel, targetCount });
      const res = await fetch('http://localhost:8000/analyze-video', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        console.log("✅ Analysis Result:", data);
        setCuts(data.map((c: any, i: number) => ({ ...c, id: i.toString() })));
      } else {
        const errData = await res.json();
        alert(`分析失敗: ${errData.detail || res.statusText}`);
      }
    } catch (e) {
      console.error("❌ AI Analysis Error:", e);
      alert('分析失敗，請檢查網路連線或後端服務是否運行於 localhost:8000');
    }
    setIsProcessing(false);
  };


  // NEW: Instant Remotion Preview
  const handlePreview = async (cut: Cut) => {
    if (!videoUrl) return;

    setPreviewCut({ ...cut });
    setShowPreviewModal(false); // Hide Player Modal initially
    setIsPreviewLoading(true); // Show Loading Overlay
    setPreviewProgress(0);
    setPreviewMessage("正在初始化...");

    setPreviewSubtitles([]);
    setPreviewAudioUrl(null);
    setPreviewVisualSegments([]);
    setPreviewFaceCenter(0.5);

    // Strict Pipeline Execution
    if (!videoFile) {
      alert("⚠️ 請重新上傳原始影片以執行完整 AI 預覽流程。");
      setTimeout(() => setShowPreviewModal(false), 500);
      return;
    }

    setPreviewMessage("AI 全流程處理中... (降噪 -> 去氣口 -> 字幕 -> 人臉)");
    setPreviewProgress(10);
    setPreviewSubtitles([]);
    setPreviewFaceCenter(0.5);
    setPreviewAudioUrl(null);
    setPreviewVisualSegments([]);

    try {
      const fd = new FormData();
      fd.append('file', videoFile);
      fd.append('start', cut.start.toString());
      fd.append('end', cut.end.toString());

      // Pass all configs for Strict Execution
      fd.append('is_denoise', isStudioSound.toString());
      fd.append('is_silence_removal', isSilenceRemoval.toString());
      fd.append('silence_threshold', silenceThreshold.toString());

      // Auto Caption
      fd.append('is_auto_caption', (isAutoCaption || isBurnCaptions).toString());
      fd.append('whisper_language', whisperLanguage);
      fd.append('whisper_model_size', whisperModelSize);
      fd.append('whisper_beam_size', whisperBeamSize.toString());
      fd.append('whisper_temperature', whisperTemperature.toString());
      fd.append('whisper_no_speech_threshold', whisperNoSpeechThreshold.toString());
      fd.append('whisper_condition_on_previous_text', whisperConditionOnPreviousText.toString());
      fd.append('whisper_best_of', whisperBestOf.toString());
      fd.append('whisper_patience', whisperPatience.toString());
      fd.append('whisper_compression_ratio_threshold', whisperCompressionRatioThreshold.toString());
      fd.append('whisper_logprob_threshold', whisperLogprobThreshold.toString());
      fd.append('whisper_fp16', whisperFp16.toString());
      fd.append('whisper_chars_per_line', whisperCharsPerLine.toString());
      fd.append('whisper_remove_punctuation', whisperRemovePunctuation.toString());
      fd.append('translate_to_chinese', isTranslate.toString());
      fd.append('api_key', apiKey);
      if (srtSubtitles.length > 0) {
        fd.append('srt_json', JSON.stringify(srtSubtitles));
      }

      // Pass Subtitle Config JSON to ensure Chars Per Line is respected in Transcription
      const subConfigObj = { charsPerLine: whisperCharsPerLine };
      fd.append('subtitle_config', JSON.stringify(subConfigObj));

      // Face Tracking
      fd.append('is_face_tracking', isFaceTracking.toString());

      // Endpoint Call with Streaming Support
      const res = await fetch('http://localhost:8000/process-preview-pipeline', { method: 'POST', body: fd });

      const reader = res.body?.getReader();
      if (!reader) throw new Error("Failed to get reader");

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter(l => l.trim() !== "");

        for (const line of lines) {
          try {
            const msg = JSON.parse(line);
            console.log("[Preview Pipeline Stream]", msg);

            if (msg.status === 'progress') {
              setPreviewMessage(msg.message);
              setPreviewProgress(msg.percent);
            }
            else if (msg.status === 'success') {
              const data = msg.data;
              if (data.subtitles) setPreviewSubtitles(data.subtitles);
              if (data.faceCenterX !== undefined) setPreviewFaceCenter(data.faceCenterX);
              if (data.audioUrl) setPreviewAudioUrl(data.audioUrl);
              if (data.visualSegments) setPreviewVisualSegments(data.visualSegments);

              setPreviewProgress(100);
              setTimeout(() => {
                setIsPreviewLoading(false);
                setShowPreviewModal(true);
              }, 500);
            }
            else if (msg.status === 'error') {
              setPreviewMessage("發生錯誤: " + msg.message);
              console.error("Pipeline Back-end Error", msg.message);
              return;
            }
          } catch (e) {
            console.warn("Stream parse error", e, line);
          }
        }
      }

    } catch (e) {
      console.error("Pipeline Error", e);
      setPreviewMessage("連線錯誤");
    }

  };

  /* LEGACY BACKEND PREVIEW (Disabled for Remotion)
  const handlePreview = async (cut: Cut) => {
     // ... old logic ...
  };
  */

  const handleExport = async (singleCut?: Cut) => {
    const activeCuts = singleCut ? [singleCut] : cuts;
    if (activeCuts.length === 0) {
      alert("請先分析影片或手動新增片段");
      return;
    }

    setIsExporting(true);
    setExportProgress(0);

    // Start Export
    setIsExporting(true);
    setExportProgress(0);

    const formData = new FormData();
    formData.append('file', videoFile!);
    formData.append('cuts_json', JSON.stringify(activeCuts));
    formData.append('face_tracking', isFaceTracking.toString());
    formData.append('studio_sound', isStudioSound.toString());
    formData.append('dfn3_strength', dfn3Strength.toString());
    formData.append('is_silence_removal', isSilenceRemoval.toString());
    formData.append('silence_threshold', silenceThreshold.toString());
    formData.append('is_jump_cut_zoom', isJumpCutZoom.toString());

    formData.append('auto_caption', isAutoCaption.toString());
    formData.append('translate_to_chinese', isTranslate.toString());
    formData.append('burn_captions', isBurnCaptions.toString());
    formData.append('srt_json', JSON.stringify(srtSubtitles)); // Send user's SRT
    formData.append('subtitle_font_name', subtitleFontName);
    formData.append('subtitle_font_size', subtitleFontSize.toString());
    formData.append('subtitle_font_weight', subtitleFontWeight.toString());
    formData.append('subtitle_font_style', subtitleFontStyle);

    formData.append('subtitle_text_color', subtitleTextColor);
    formData.append('is_text_gradient', isTextGradient.toString());
    formData.append('text_gradient_colors', JSON.stringify(textGradientColors));
    formData.append('text_gradient_direction', textGradientDirection);

    formData.append('subtitle_outline_width', (isSubtitleOutline ? subtitleOutlineWidth : 0).toString());
    formData.append('subtitle_outline_color', subtitleOutlineColor);

    formData.append('subtitle_shadow_color', subtitleShadowColor);
    formData.append('subtitle_shadow_opacity', (isSubtitleShadow ? subtitleShadowOpacity : 0).toString()); // Int 0-100
    formData.append('subtitle_shadow_blur', subtitleShadowBlur.toString());
    formData.append('subtitle_shadow_offset_x', subtitleShadowOffsetX.toString());
    formData.append('subtitle_shadow_offset_y', subtitleShadowOffsetY.toString());

    formData.append('subtitle_letter_spacing', subtitleLetterSpacing.toString());
    formData.append('subtitle_line_height', subtitleLineHeight.toString());
    formData.append('subtitle_text_transform', subtitleTextTransform);
    formData.append('subtitle_text_align', subtitleTextAlign);

    formData.append('subtitle_margin_v', subtitleMarginV.toString());
    formData.append('subtitle_chars_per_line', subtitleCharsPerLine.toString());
    formData.append('subtitle_animation', subtitleAnimation);
    formData.append('subtitle_animation_duration', subtitleAnimationDuration.toString());
    formData.append('subtitle_animation_spring', subtitleAnimationSpring.toString());

    formData.append('subtitle_box_enabled', subtitleBgEnabled.toString());
    formData.append('subtitle_box_color', subtitleBgColor);
    formData.append('subtitle_box_opacity', subtitleBgOpacity.toString()); // Int 0-100
    formData.append('subtitle_box_padding_x', subtitleBgPaddingX.toString());
    formData.append('subtitle_box_padding_y', subtitleBgPaddingY.toString());
    formData.append('subtitle_box_radius', subtitleBgRadius.toString());

    formData.append('output_quality', outputQuality);
    formData.append('output_resolution', outputResolution);
    formData.append('whisper_language', whisperLanguage);
    formData.append('whisper_model_size', whisperModelSize);
    formData.append('whisper_beam_size', whisperBeamSize.toString());
    formData.append('whisper_temperature', whisperTemperature.toString());
    formData.append('whisper_no_speech_threshold', whisperNoSpeechThreshold.toString());
    formData.append('whisper_condition_on_previous_text', whisperConditionOnPreviousText.toString());
    formData.append('whisper_best_of', whisperBestOf.toString());
    formData.append('whisper_patience', whisperPatience.toString());
    formData.append('whisper_compression_ratio_threshold', whisperCompressionRatioThreshold.toString());
    formData.append('whisper_logprob_threshold', whisperLogprobThreshold.toString());
    formData.append('whisper_fp16', whisperFp16.toString());
    formData.append('whisper_remove_punctuation', whisperRemovePunctuation.toString());
    formData.append('whisper_chars_per_line', whisperCharsPerLine.toString());

    try {
      const res = await fetch('http://localhost:8000/process-video', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        // Trigger download via direct URL
        const downloadUrl = `http://localhost:8000${data.download_url}`;
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

      } else {
        const errorData = await res.json().catch(() => ({ detail: '匯出失敗' }));
        alert(`匯出失敗: ${errorData.detail || res.statusText}`);
      }
    } catch (e) {
      alert('無法連接後端服務，或處理時間過長導致超時');
    }

    setExportProgress(100);
    setTimeout(() => setExportProgress(0), 500);
    setIsExporting(false);
  };

  const updateCutLabel = (id: string, label: string) => setCuts(cuts.map(c => c.id === id ? { ...c, label } : c));
  const updateCutTime = (id: string, field: 'start' | 'end', value: number) => {
    setCuts(cuts.map(c => c.id === id ? { ...c, [field]: Math.max(0, Math.min(duration, value)) } : c));
  };
  const seekTo = (time: number) => { if (videoRef.current) videoRef.current.currentTime = time; };

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (draggingCut) return; // Don't seek when dragging
    if (timelineRef.current) {
      const rect = timelineRef.current.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      const targetTime = pos * duration;
      seekTo(targetTime);
    }
  };

  const handleCutDragStart = (e: React.MouseEvent, cutId: string, edge: 'start' | 'end' | 'move') => {
    e.stopPropagation();
    const cut = cuts.find(c => c.id === cutId);
    if (cut) {
      setDraggingCut({ id: cutId, edge, initialX: e.clientX, initialStart: cut.start, initialEnd: cut.end });
    }
  };

  const handleCutDrag = (e: React.MouseEvent) => {
    if (!draggingCut || !timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const deltaX = e.clientX - draggingCut.initialX;
    const deltaTime = (deltaX / rect.width) * duration;

    if (draggingCut.edge === 'start') {
      const newStart = Math.max(0, Math.min(draggingCut.initialEnd - 0.5, draggingCut.initialStart + deltaTime));
      updateCutTime(draggingCut.id, 'start', newStart);
    } else if (draggingCut.edge === 'end') {
      const newEnd = Math.max(draggingCut.initialStart + 0.5, Math.min(duration, draggingCut.initialEnd + deltaTime));
      updateCutTime(draggingCut.id, 'end', newEnd);
    } else { // move
      const clipDuration = draggingCut.initialEnd - draggingCut.initialStart;
      let newStart = draggingCut.initialStart + deltaTime;
      newStart = Math.max(0, Math.min(duration - clipDuration, newStart));
      setCuts(cuts.map(c => c.id === draggingCut.id ? { ...c, start: newStart, end: newStart + clipDuration } : c));
    }
  };

  const handleCutDragEnd = () => {
    setDraggingCut(null);
  };

  // Wrap text based on chars per line (each character = 1)
  const wrapText = (text: string, charsPerLine: number): string => {
    if (!text || charsPerLine <= 0) return text;

    // Support existing newlines first
    const paragraphs = text.split('\n');
    const wrappedParagraphs = paragraphs.map(p => {
      if (p.length <= charsPerLine) return p;

      const lines = [];
      let current = p;

      while (current.length > 0) {
        if (current.length <= charsPerLine) {
          lines.push(current);
          break;
        }

        // Find best break point for English (space) within the limit
        let breakAt = charsPerLine;
        const sub = current.substring(0, charsPerLine + 1);
        const lastSpace = sub.lastIndexOf(' ');

        // Only use space break if it's not too far back (e.g. at least 60% of limit)
        if (lastSpace > charsPerLine * 0.6) {
          breakAt = lastSpace;
        }

        lines.push(current.substring(0, breakAt).trim());
        current = current.substring(breakAt).trim();
      }
      return lines.join('\n');
    });

    return wrappedParagraphs.join('\n');
  };

  // Calculate shadow offset from angle


  const handleGradientColorChange = (index: number, color: string) => {
    const newColors = [...textGradientColors];
    newColors[index] = color;
    setTextGradientColors(newColors);
  };

  return (
    <div className="app-container">
      {/* System Status Banner */}
      {systemStatus.status !== 'ready' && (
        <div className="system-status-banner glass">
          <div className="status-info">
            <div className="pulse-dot" />
            <span className="status-message">{systemStatus.message}</span>
            <span className="status-percent">{systemStatus.progress}%</span>
          </div>
          <div className="status-progress-bar">
            <div className="status-progress-fill" style={{ width: `${systemStatus.progress}%` }} />
          </div>
        </div>
      )}

      {/* Preview Modal (Updated for Remotion) */}

      {/* Preview Modal (Updated for Remotion) */}
      <AnimatePresence>
        {showPreviewModal && previewCut && videoUrl && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => { setPreviewCut(null); setShowPreviewModal(false); }}
            style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', zIndex: 9999, display: 'flex', justifyContent: 'center', alignItems: 'center' }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="preview-modal-content"
              style={{
                width: 'auto',
                height: '85vh',
                aspectRatio: '9/16',
                background: '#000',
                borderRadius: '24px',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: '0 8px 32px rgba(0,0,0,0.5)'
              }}
            >
              <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                <Player
                  key={`res-preview-${previewCut.id}-${previewCut.start}-${previewCut.end}-${previewFaceCenter}-${previewAudioUrl}-${subtitleFontSize}-${subtitleFontName}-${subtitleFontWeight}-${subtitleFontStyle}-${subtitleTextColor}-${subtitleBgEnabled}-${subtitleOutlineWidth}-${subtitleShadowOpacity}-${subtitleLetterSpacing}-${subtitleLineHeight}-${subtitleTextAlign}-${subtitleTextTransform}-${isTextGradient}-${subtitleBgColor}-${subtitleBgOpacity}-${subtitleBgPaddingX}-${subtitleBgPaddingY}-${subtitleBgRadius}-${subtitleCharsPerLine}-${isBurnCaptions}`}
                  component={MyComposition}
                  durationInFrames={Math.max(1, Math.floor((previewCut.end - previewCut.start) * 30))}
                  compositionWidth={1080}
                  compositionHeight={1920}
                  fps={30}
                  style={{ width: '100%', height: '100%' }}
                  controls
                  autoPlay
                  loop
                  inputProps={{
                    videoUrl: videoUrl,
                    audioUrl: previewAudioUrl || undefined,
                    startFrom: previewCut.start,
                    visualSegments: previewVisualSegments.length > 0 ? previewVisualSegments : [{
                      startInVideo: previewCut.start,
                      duration: previewCut.end - previewCut.start,
                      zoom: 1.0
                    }],
                    subtitles: (() => {
                      // 1. If we have preview subtitles (from server), use them.
                      if (previewSubtitles && previewSubtitles.length > 0) return previewSubtitles;

                      // 2. If we have globally loaded SRT subtitles, filter and use them.
                      const globalInCut = srtSubtitles.filter(s => s.start <= previewCut.end && s.end >= previewCut.start);
                      if (globalInCut.length > 0) return globalInCut;

                      // 3. Last fallback: ONLY if no real subtitles exist, show the manual preview text.
                      return [{
                        id: 'preview_placeholder',
                        start: previewCut.start,
                        end: previewCut.end,
                        text: wrapText(previewText || '預覽文字內容', subtitleCharsPerLine)
                      }];
                    })(),
                    isFaceTracking: isFaceTracking,
                    faceCenterX: previewFaceCenter, // Force usage of detected face center
                    subtitleConfig: {
                      fontSize: subtitleFontSize,
                      fontFamily: subtitleFontName,
                      fontWeight: subtitleFontWeight,
                      fontStyle: subtitleFontStyle,

                      textColor: subtitleTextColor,
                      isTextGradient: isTextGradient,
                      textGradientColors: textGradientColors,
                      textGradientDirection: textGradientDirection,

                      outlineWidth: isSubtitleOutline ? subtitleOutlineWidth : 0,
                      outlineColor: subtitleOutlineColor,

                      shadowColor: subtitleShadowColor,
                      shadowBlur: subtitleShadowBlur,
                      shadowOffsetX: subtitleShadowOffsetX,
                      shadowOffsetY: subtitleShadowOffsetY,
                      shadowOpacity: isSubtitleShadow ? subtitleShadowOpacity / 100 : 0,

                      letterSpacing: subtitleLetterSpacing,
                      lineHeight: subtitleLineHeight,
                      textTransform: subtitleTextTransform as any,
                      textAlign: subtitleTextAlign as any,

                      marginBottom: subtitleMarginV,
                      charsPerLine: subtitleCharsPerLine,
                      animation: subtitleAnimation,

                      isUnknownBackground: subtitleBgEnabled,
                      backgroundColor: subtitleBgColor,
                      backgroundOpacity: subtitleBgOpacity / 100,
                      backgroundPaddingX: subtitleBgPaddingX,
                      backgroundPaddingY: subtitleBgPaddingY,
                      backgroundBorderRadius: subtitleBgRadius,
                    } as any // Force cast to avoid strict union type errors temporarily
                  }}
                />
              </div>
              {/* Overlay instruction removed to avoid overlap with subtitles */}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <header className="header glass">
        <div className="logo">
          <Scissors className="icon-primary" size={24} />
          <span>Antigravity Cut <small>v3.0</small></span>
        </div>
        <div className="api-status">
          {/* Empty space - API Key moved to sidebar */}
        </div>
      </header>

      <main className="main-content-new">
        <div className="top-row">
          {/* Section 1: Main Video Player (Left) */}
          <div className="panel-column">
            <div className="panel-header-title">
              <Video size={14} style={{ marginRight: '8px' }} /> 16:9 素材影片
            </div>
            <div className="video-panel">
              <div className="video-container-16-9">
                {videoUrl ? (
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    className="video-player"
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                  />
                ) : (
                  <div className="upload-placeholder">
                    <input type="file" accept="video/*" onChange={handleFileUpload} style={{ display: 'none' }} id="video-upload-main" />
                    <label htmlFor="video-upload-main" className="upload-label">
                      <Upload size={48} color="#6366f1" />
                      <div>點擊此處上傳原始影片</div>
                    </label>
                  </div>
                )}
              </div>
              <div className="timeline-inline">
                <div className="timeline-info">
                  <span className="time-display">{new Date(currentTime * 1000).toISOString().substr(14, 5)} / {new Date(duration * 1000).toISOString().substr(14, 5)}</span>
                  <div className="timeline-controls">
                    <button className="btn-icon-sm" onClick={() => { if (videoRef.current) { if (isPlaying) videoRef.current.pause(); else videoRef.current.play(); setIsPlaying(!isPlaying); } }}>
                      {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    </button>
                  </div>
                </div>
                {videoUrl && <div ref={waveformRef} className="waveform-track" />}
                <div
                  ref={timelineRef}
                  className="markers-track"
                  onClick={handleTimelineClick}
                  onMouseMove={handleCutDrag}
                  onMouseUp={handleCutDragEnd}
                  onMouseLeave={handleCutDragEnd}
                >
                  <div className="playhead" style={{ left: `${(currentTime / (duration || 1)) * 100}%` }} />
                  {cuts.map(cut => (
                    <div
                      key={cut.id}
                      className={`cut-marker ${draggingCut?.id === cut.id ? 'dragging' : ''}`}
                      style={{ left: `${(cut.start / (duration || 1)) * 100}%`, width: `${((cut.end - cut.start) / (duration || 1)) * 100}%` }}
                    >
                      <div
                        className="cut-edge cut-edge-start"
                        onMouseDown={(e) => handleCutDragStart(e, cut.id, 'start')}
                      />
                      <div
                        className="cut-center"
                        onMouseDown={(e) => handleCutDragStart(e, cut.id, 'move')}
                      />
                      <div
                        className="cut-edge cut-edge-end"
                        onMouseDown={(e) => handleCutDragStart(e, cut.id, 'end')}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Subtitle Preview (Middle) */}
          <div className="panel-column">
            <div className="preview-916-header">9:16 字幕位置預覽</div>
            <div className="preview-916-panel">
              <div
                className="preview-916-canvas"
                ref={previewCanvasRef} // Attach Observer
                style={{ background: '#00ff00' }}
                onMouseMove={(e) => {
                  if (isDraggingSubtitle && e.buttons === 1) {
                    const rect = e.currentTarget.getBoundingClientRect();
                    // Calculate distance from bottom in visual pixels
                    const visualBottom = rect.height - (e.clientY - rect.top);
                    // Convert to 1080p reference pixels
                    const truePixels = visualBottom / scaleRatio;
                    // Clamp to reasonable range (0 to full height)
                    setSubtitleMarginV(Math.max(0, Math.round(truePixels)));
                  }
                }}
                onMouseUp={() => setIsDraggingSubtitle(false)}
                onMouseLeave={() => setIsDraggingSubtitle(false)}
              >
                {showSafeArea && <div className="safe-area-guide" />}

                {/* Subtitle Display Wrapper matches Remotion Layout */}
                <div
                  className="subtitle-draggable-wrapper"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'flex-end', // Aligns items to bottom
                    alignItems: (subtitleTextAlign === 'left' ? 'flex-start' : (subtitleTextAlign === 'right' ? 'flex-end' : 'center')),
                    paddingBottom: `${subtitleMarginV * scaleRatio}px`, // Scaled pixel margin
                    paddingLeft: '5%',
                    paddingRight: '5%',
                    boxSizing: 'border-box',
                    cursor: 'ns-resize',
                    pointerEvents: 'auto' // Allow dragging on empty space? Maybe better on text only
                  }}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setIsDraggingSubtitle(true);
                  }}
                >
                  <div style={{
                    // Inner content box
                    backgroundColor: subtitleBgEnabled ? `${subtitleBgColor}${Math.round((subtitleBgOpacity / 100) * 255).toString(16).padStart(2, '0')}` : 'transparent',
                    padding: `${subtitleBgPaddingY * scaleRatio}px ${subtitleBgPaddingX * scaleRatio}px`,
                    borderRadius: `${subtitleBgRadius * scaleRatio}px`,
                    maxWidth: '100%',
                    boxSizing: 'border-box',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: (subtitleTextAlign === 'left' ? 'flex-start' : (subtitleTextAlign === 'right' ? 'flex-end' : 'center')),
                  }}>
                    <span style={{
                      fontFamily: subtitleFontName ? `"${subtitleFontName}", Arial, sans-serif` : 'Arial',
                      fontSize: `${subtitleFontSize * scaleRatio}px`,
                      fontWeight: subtitleFontWeight,
                      fontStyle: subtitleFontStyle,
                      color: isTextGradient ? 'transparent' : subtitleTextColor,
                      textAlign: (subtitleTextAlign as any) || 'center',
                      letterSpacing: `${subtitleLetterSpacing * scaleRatio}px`,
                      lineHeight: subtitleLineHeight,
                      textTransform: (subtitleTextTransform as any) || 'none',
                      backgroundImage: isTextGradient ? `linear-gradient(${textGradientDirection}, ${textGradientColors.join(', ')})` : 'none',
                      WebkitBackgroundClip: isTextGradient ? 'text' : undefined,
                      WebkitTextFillColor: isTextGradient ? 'transparent' : undefined,
                      // CSS Stroke simulation
                      WebkitTextStroke: (isSubtitleOutline && subtitleOutlineWidth > 0) ? `${(subtitleOutlineWidth * scaleRatio) * 2}px ${subtitleOutlineColor}` : '0px',
                      paintOrder: 'stroke fill',
                      strokeLinejoin: 'round',
                      strokeLinecap: 'round',
                      // CSS Shadow simulation
                      textShadow: isSubtitleShadow ? `${subtitleShadowOffsetX * scaleRatio}px ${subtitleShadowOffsetY * scaleRatio}px ${subtitleShadowBlur * scaleRatio}px ${subtitleShadowColor}${Math.round((subtitleShadowOpacity / 100) * 255).toString(16).padStart(2, '0')}` : 'none',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {(() => {
                        const currentSub = srtSubtitles.find(s => currentTime >= s.start && currentTime <= s.end);
                        const displayText = currentSub ? currentSub.text : (previewText || '預覽文字內容 (點擊此處可輸入內容測試)');
                        return wrapText(displayText, subtitleCharsPerLine);
                      })()}
                    </span>
                  </div>
                </div>
              </div>
              <div className="preview-controls">
                <label className="checkbox-inline">
                  <input type="checkbox" checked={showSafeArea} onChange={e => setShowSafeArea(e.target.checked)} />
                  <span>顯示安全邊界</span>
                </label>
              </div>
            </div>
          </div>

          {/* Section 3: Controls (Right) */}
          <div className="panel-column">
            <div className="panel-header-title" style={{ opacity: 0 }}>佔位符</div>
            <aside className="controls-panel">
              <div className="controls-scroll">
                {/* 1. Project Section */}
                <div className="controls-section">
                  <div className="section-header">
                    <Film size={16} className="icon-primary" />
                    <span>專案與模型配置</span>
                    <button onClick={() => setShowApiSettings(!showApiSettings)} className={`toggle-btn-sm ${apiKey ? 'active' : ''}`}>
                      <Key size={14} />
                    </button>
                  </div>

                  <AnimatePresence>
                    {showApiSettings && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="settings-drawer">
                        <div className="api-key-row">
                          <Key size={14} className="icon-dim" />
                          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="輸入 Gemini API Key..." className="api-input-inline" />
                        </div>
                        <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="select-sm">
                          <option value="gemini-3.0-flash">Gemini 3.0 Flash</option>
                          <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                        </select>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <div className="settings-row">
                    <div className="setting-mini"><label>片段數量</label><input type="number" value={targetCount} onChange={(e) => setTargetCount(Number(e.target.value))} /></div>
                    <div className="setting-mini"><label>長度(秒)</label><input type="number" value={targetDuration} onChange={(e) => setTargetDuration(Number(e.target.value))} /></div>
                  </div>
                  <textarea className="instruction-input-sm" placeholder="AI 剪輯指令 (例如：找出所有精彩笑點...)" value={instruction} onChange={(e) => setInstruction(e.target.value)} />
                  <button
                    onClick={handleMakeVideo}
                    disabled={!videoFile || isProcessing}
                    className={!videoFile ? "btn-disabled" : "btn-premium-ai"}
                    style={{ width: '100%', marginTop: '16px' }}
                  >
                    {isProcessing ? <><Loader2 className="spin" size={20} /> AI 分析中 (約 30s)...</> :
                      !videoFile ? <><Video size={20} /> 請先上傳素材影片</> :
                        <><Zap size={20} /> 開始 AI 自動分析片段</>}
                  </button>
                </div>

                {/* 2. AI Enhancement Section */}
                <div className="controls-section">
                  <div className="section-header">
                    <Languages size={16} className="icon-primary" />
                    <span>AI 影像與音訊增強</span>
                  </div>
                  <div className="ai-switches">
                    <label className="checkbox-sm">
                      <input type="checkbox" checked={isFaceTracking} onChange={e => setIsFaceTracking(e.target.checked)} />
                      <span>人臉自動追蹤裁切 (9:16)</span>
                    </label>
                    {isFaceTracking && (
                      <div className="mini-settings-panel">
                        <div className="style-row-inline">
                          <span className="style-label">縮放</span>
                          <input type="number" step="0.1" value={trackZoom} onChange={e => setTrackZoom(parseFloat(e.target.value))} className="input-mini-sm" />
                          <span className="style-label">信心</span>
                          <input type="range" min="0.1" max="0.9" step="0.1" value={mpMinDetectionCon} onChange={e => setMpMinDetectionCon(parseFloat(e.target.value))} className="slider-sm" />
                        </div>
                      </div>
                    )}

                    <label className="checkbox-sm">
                      <input type="checkbox" checked={isStudioSound} onChange={e => setIsStudioSound(e.target.checked)} />
                      <span>DFN3 深度智慧降噪 (Beta)</span>
                    </label>
                    {isStudioSound && (
                      <div className="mini-settings-panel">
                        <div className="style-row-inline">
                          <span className="style-label">強度</span>
                          <input type="range" min="0" max="100" value={dfn3Strength} onChange={e => setDfn3Strength(parseInt(e.target.value))} className="slider-sm" />
                          <span className="val-display">{dfn3Strength}%</span>
                        </div>
                      </div>
                    )}

                    <label className="checkbox-sm">
                      <input type="checkbox" checked={isSilenceRemoval} onChange={e => setIsSilenceRemoval(e.target.checked)} />
                      <span>智慧去氣口 & Jump Cuts (Beta)</span>
                    </label>
                    {isSilenceRemoval && (
                      <div className="mini-settings-panel">
                        <div className="style-row-inline" style={{ marginBottom: '8px' }}>
                          <span className="style-label">靜音閾值(秒)</span>
                          <input type="range" min="0.1" max="2.0" step="0.1" value={silenceThreshold} onChange={e => setSilenceThreshold(parseFloat(e.target.value))} className="slider-sm" />
                          <span className="val-display">{silenceThreshold}s</span>
                        </div>
                        <label className="checkbox-inline">
                          <input type="checkbox" checked={isJumpCutZoom} onChange={e => setIsJumpCutZoom(e.target.checked)} />
                          <span>自動變焦</span>
                        </label>
                      </div>
                    )}

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <label className="checkbox-sm">
                        <input type="checkbox" checked={isAutoCaption} onChange={e => setIsAutoCaption(e.target.checked)} />
                        <span>{srtSubtitles.length > 0 ? `已載入 SRT 字幕 (${srtSubtitles.length})` : 'AI 語音轉文字'}</span>
                      </label>
                    </div>

                    {isAutoCaption && (
                      <div className="caption-sub-settings">
                        {srtSubtitles.length === 0 && (
                          <>
                            <div className="style-row-inline" style={{ marginBottom: '8px' }}>
                              <span className="style-label" style={{ minWidth: '60px' }}>轉錄語言</span>
                              <select value={whisperLanguage} onChange={e => setWhisperLanguage(e.target.value)} className="select-xs" style={{ flex: 1 }}>
                                <option value="zh">中文</option>
                                <option value="en">英文</option>
                                <option value="ja">日文</option>
                                <option value="auto">自動偵測</option>
                              </select>
                            </div>

                            {/* Advanced Whisper Settings */}
                            <div className="whisper-adv-panel">
                              <div className="section-subtitle" style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', marginBottom: '12px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px' }}>AI 轉錄引擎調優</div>

                              <div className="whisper-row">
                                <span className="whisper-label">模型大小</span>
                                <select value={whisperModelSize} onChange={e => setWhisperModelSize(e.target.value)} className="select-xs" style={{ flex: 1 }}>
                                  <option value="turbo">Turbo</option>
                                  <option value="large-v3">Large-v3</option>
                                  <option value="medium">Medium</option>
                                  <option value="small">Small</option>
                                  <option value="base">Base</option>
                                </select>
                              </div>

                              <div className="whisper-row">
                                <span className="whisper-label">搜尋強度</span>
                                <input type="range" min="1" max="10" value={whisperBeamSize} onChange={e => setWhisperBeamSize(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                <span className="val-display">{whisperBeamSize}</span>
                              </div>

                              <div className="whisper-row">
                                <span className="whisper-label">轉錄切分字數</span>
                                <input type="number" min="4" max="40" value={whisperCharsPerLine} onChange={e => setWhisperCharsPerLine(parseInt(e.target.value))} className="input-mini-xs" />
                                <input type="range" min="4" max="40" value={whisperCharsPerLine} onChange={e => setWhisperCharsPerLine(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                              </div>

                              <div className="whisper-row">
                                <span className="whisper-label">隨機性</span>
                                <input type="range" min="0" max="1" step="0.1" value={whisperTemperature} onChange={e => setWhisperTemperature(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                <span className="val-display">{whisperTemperature.toFixed(1)}</span>
                              </div>

                              <div className="whisper-row">
                                <span className="whisper-label">靜音過濾</span>
                                <input type="range" min="0" max="1" step="0.1" value={whisperNoSpeechThreshold} onChange={e => setWhisperNoSpeechThreshold(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                <span className="val-display">{whisperNoSpeechThreshold.toFixed(1)}</span>
                              </div>

                              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '4px' }}>
                                <label className="checkbox-inline">
                                  <input type="checkbox" checked={whisperRemovePunctuation} onChange={e => setWhisperRemovePunctuation(e.target.checked)} />
                                  <span className="label-xs">移除標點</span>
                                </label>
                                <label className="checkbox-inline">
                                  <input type="checkbox" checked={whisperConditionOnPreviousText} onChange={e => setWhisperConditionOnPreviousText(e.target.checked)} />
                                  <span className="label-xs">語意關聯</span>
                                </label>
                              </div>

                              {/* Expert Toggle */}
                              <button
                                className={`expert-toggle-btn ${showExpertWhisper ? 'active' : ''}`}
                                onClick={() => setShowExpertWhisper(!showExpertWhisper)}
                              >
                                <span>{showExpertWhisper ? '收起開發者設定' : '開啟專家級調優'}</span>
                                <span className="chevron">▼</span>
                              </button>

                              {showExpertWhisper && (
                                <div className="whisper-expert-container">
                                  <div className="whisper-row">
                                    <span className="whisper-label" style={{ minWidth: '70px' }}>Best of</span>
                                    <input type="number" value={whisperBestOf} onChange={e => setWhisperBestOf(parseInt(e.target.value))} className="input-mini-xs" />
                                    <input type="range" min="1" max="10" value={whisperBestOf} onChange={e => setWhisperBestOf(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                  </div>
                                  <div className="whisper-row">
                                    <span className="whisper-label" style={{ minWidth: '70px' }}>Patience</span>
                                    <input type="number" step="0.1" value={whisperPatience} onChange={e => setWhisperPatience(parseFloat(e.target.value))} className="input-mini-xs" />
                                    <input type="range" min="0" max="3" step="0.1" value={whisperPatience} onChange={e => setWhisperPatience(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                  </div>
                                  <div className="whisper-row">
                                    <span className="whisper-label" style={{ minWidth: '70px' }}>壓縮閾值</span>
                                    <input type="number" step="0.1" value={whisperCompressionRatioThreshold} onChange={e => setWhisperCompressionRatioThreshold(parseFloat(e.target.value))} className="input-mini-xs" />
                                    <input type="range" min="1" max="4" step="0.1" value={whisperCompressionRatioThreshold} onChange={e => setWhisperCompressionRatioThreshold(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                  </div>
                                  <div className="whisper-row">
                                    <span className="whisper-label" style={{ minWidth: '70px' }}>Logprob</span>
                                    <input type="number" step="0.1" value={whisperLogprobThreshold} onChange={e => setWhisperLogprobThreshold(parseFloat(e.target.value))} className="input-mini-xs" />
                                    <input type="range" min="-3" max="0" step="0.1" value={whisperLogprobThreshold} onChange={e => setWhisperLogprobThreshold(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                                  </div>
                                  <label className="checkbox-inline">
                                    <input type="checkbox" checked={whisperFp16} onChange={e => setWhisperFp16(e.target.checked)} />
                                    <span className="label-xs">FP16 硬體加速</span>
                                  </label>
                                </div>
                              )}
                            </div>
                          </>
                        )}
                        {srtSubtitles.length !== 0 && (
                          <div style={{ marginBottom: '8px' }}>
                            <button onClick={() => setSrtSubtitles([])} className="btn-ghost-xs" style={{ color: '#ef4444' }}>移除已載入的 SRT</button>
                          </div>
                        )}

                        <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
                          <label className="checkbox-inline">
                            <input type="checkbox" checked={isTranslate} onChange={e => setIsTranslate(e.target.checked)} />
                            <span>簡轉繁優化</span>
                          </label>
                          <label className="checkbox-inline">
                            <input type="checkbox" checked={isBurnCaptions} onChange={e => setIsBurnCaptions(e.target.checked)} />
                            <span>燒錄至影片</span>
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 3. Subtitle Styles (Advanced) - Redesigned for Pro Layout */}
                {(isBurnCaptions || isAutoCaption) && (
                  <div className="controls-section">
                    <div className="section-header"><Wand2 size={16} className="icon-primary" /><span>字幕視覺設計</span></div>

                    <div className="style-group">
                      <label className="group-label">文字樣式</label>

                      <div className="style-item" style={{ marginBottom: '10px' }}>
                        <select
                          value={subtitleFontName}
                          onChange={e => setSubtitleFontName(e.target.value)}
                          className="select-sm"
                          style={{ width: '100%' }}
                        >
                          {fontList.map(f => <option key={f} value={f}>{f}</option>)}
                        </select>
                      </div>

                      <div className="style-item">
                        <div className="style-row-inline">
                          <span className="style-label" style={{ minWidth: '30px' }}>大小</span>
                          <input type="range" min="12" max="150" value={subtitleFontSize} onChange={e => setSubtitleFontSize(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                          <span className="val-display" style={{ minWidth: '40px' }}>{subtitleFontSize}px</span>
                        </div>
                      </div>

                      <div className="style-grid-2" style={{ marginTop: '10px' }}>
                        <div className="mini-control">
                          <label className="label-xs">粗細</label>
                          <select value={subtitleFontWeight} onChange={e => setSubtitleFontWeight(e.target.value)} className="select-xs">
                            <option value="normal">Normal</option>
                            <option value="bold">Bold</option>
                            <option value="500">Medium</option>
                            <option value="900">Black</option>
                          </select>
                        </div>
                        <div className="mini-control">
                          <label className="label-xs">斜體</label>
                          <select value={subtitleFontStyle} onChange={e => setSubtitleFontStyle(e.target.value)} className="select-xs">
                            <option value="normal">正體</option>
                            <option value="italic">斜體</option>
                          </select>
                        </div>
                        <div className="mini-control">
                          <label className="label-xs">轉換</label>
                          <select value={subtitleTextTransform} onChange={e => setSubtitleTextTransform(e.target.value)} className="select-xs">
                            <option value="none">正常</option>
                            <option value="uppercase">大寫</option>
                            <option value="lowercase">小寫</option>
                            <option value="capitalize">首字大寫</option>
                          </select>
                        </div>
                        <div className="mini-control">
                          <label className="label-xs">對齊</label>
                          <select value={subtitleTextAlign} onChange={e => setSubtitleTextAlign(e.target.value)} className="select-xs">
                            <option value="center">置中</option>
                            <option value="left">靠左</option>
                            <option value="right">靠右</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    <div className="style-group">
                      <label className="group-label">間隔與佈局</label>

                      <div className="style-grid-3">
                        <div className="mini-control">
                          <label className="label-xs">行高</label>
                          <input type="range" min="0.8" max="2.5" step="0.1" value={subtitleLineHeight} onChange={e => setSubtitleLineHeight(parseFloat(e.target.value))} className="slider-xs" />
                          <input type="number" step="0.1" value={subtitleLineHeight} onChange={e => setSubtitleLineHeight(parseFloat(e.target.value))} className="input-xs" />
                        </div>
                        <div className="mini-control">
                          <label className="label-xs">字距</label>
                          <input type="range" min="-5" max="20" step="0.5" value={subtitleLetterSpacing} onChange={e => setSubtitleLetterSpacing(parseFloat(e.target.value))} className="slider-xs" />
                          <input type="number" step="0.5" value={subtitleLetterSpacing} onChange={e => setSubtitleLetterSpacing(parseFloat(e.target.value))} className="input-xs" />
                        </div>
                        <div className="mini-control">
                          <label className="label-xs">視覺分行字數</label>
                          <input type="range" min="4" max="40" step="1" value={subtitleCharsPerLine} onChange={e => setSubtitleCharsPerLine(parseInt(e.target.value))} className="slider-xs" />
                          <input type="number" value={subtitleCharsPerLine} onChange={e => setSubtitleCharsPerLine(parseInt(e.target.value))} className="input-xs" />
                        </div>
                      </div>

                      <div className="style-row-inline" style={{ marginTop: '12px' }}>
                        <span className="label-xs" style={{ minWidth: '50px' }}>垂直邊距</span>
                        <input type="range" min="0" max="600" value={subtitleMarginV} onChange={e => setSubtitleMarginV(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                        <span className="val-display" style={{ minWidth: '30px' }}>{subtitleMarginV}</span>
                      </div>

                      <div className="style-row-inline" style={{ marginTop: '12px' }}>
                        <span className="label-xs" style={{ minWidth: '50px' }}>出現動畫</span>
                        <select
                          value={subtitleAnimation}
                          onChange={e => setSubtitleAnimation(e.target.value as any)}
                          className="select-xs"
                          style={{ flex: 1 }}
                        >
                          <option value="none">無 (Static)</option>
                          <option value="pop">放大彈出 (Pop)</option>
                          <option value="fade">淡入 (Fade)</option>
                          <option value="slide-up">向上滑入 (Slide Up)</option>
                        </select>
                      </div>

                      {subtitleAnimation !== 'none' && (
                        <div className="style-col-flex" style={{ marginTop: '8px', gap: '8px', paddingLeft: '8px', borderLeft: '2px solid rgba(255,255,255,0.1)' }}>
                          <div className="style-row-inline" style={{ justifyContent: 'space-between' }}>
                            <span className="label-xs">動畫時長</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, marginLeft: '12px' }}>
                              <input type="range" min="5" max="60" value={subtitleAnimationDuration} onChange={e => setSubtitleAnimationDuration(parseInt(e.target.value))} className="slider-xs" style={{ flex: 1 }} />
                              <span className="val-display" style={{ minWidth: '24px' }}>{subtitleAnimationDuration}</span>
                            </div>
                          </div>
                          <div className="style-row-inline" style={{ justifyContent: 'space-between' }}>
                            <span className="label-xs">彈力強度</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, marginLeft: '12px' }}>
                              <input type="range" min="0.1" max="2.0" step="0.1" value={subtitleAnimationSpring} onChange={e => setSubtitleAnimationSpring(parseFloat(e.target.value))} className="slider-xs" style={{ flex: 1 }} />
                              <span className="val-display" style={{ minWidth: '24px' }}>{subtitleAnimationSpring.toFixed(1)}</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="style-item" style={{ marginTop: '10px' }}>
                      <div className="input-group-xs">
                        <span className="label-xs">預覽文字</span>
                        <input type="text" value={previewText} onChange={e => setPreviewText(e.target.value)} className="input-xs" style={{ flex: 1, textAlign: 'left' }} placeholder="輸入預覽字樣..." />
                      </div>
                    </div>

                    <div className="style-group">
                      <label className="group-label">外觀與特效</label>

                      <div className="tab-switch" style={{ marginBottom: '10px' }}>
                        <button className={!isTextGradient ? 'active' : ''} onClick={() => setIsTextGradient(false)}>單色</button>
                        <button className={isTextGradient ? 'active' : ''} onClick={() => setIsTextGradient(true)}>漸層</button>
                      </div>

                      {!isTextGradient ? (
                        <div className="style-row-inline">
                          <input type="color" value={subtitleTextColor} onChange={e => setSubtitleTextColor(e.target.value)} className="color-full" style={{ height: '32px' }} />
                        </div>
                      ) : (
                        <div className="style-col-flex">
                          <div className="style-grid-2">
                            <input type="color" value={textGradientColors[0]} onChange={e => handleGradientColorChange(0, e.target.value)} className="color-full" />
                            <input type="color" value={textGradientColors[1]} onChange={e => handleGradientColorChange(1, e.target.value)} className="color-full" />
                          </div>
                          <select value={textGradientDirection} onChange={e => setTextGradientDirection(e.target.value)} className="select-xs" style={{ width: '100%', marginTop: '4px' }}>
                            <option value="to bottom">垂直漸層 (↓)</option>
                            <option value="to right">水平漸層 (→)</option>
                            <option value="to bottom right">對角漸層 (↘)</option>
                            <option value="to top">反向垂直 (↑)</option>
                          </select>
                        </div>
                      )}

                      <div className="style-grid-3" style={{ marginTop: '10px' }}>
                        <div className="mini-control" style={{ gridColumn: 'span 2' }}>
                          <label className="checkbox-inline">
                            <input type="checkbox" checked={isSubtitleOutline} onChange={e => setIsSubtitleOutline(e.target.checked)} />
                            <span className="label-xs">描邊效果</span>
                          </label>
                          <input type="range" min="0" max="30" step="0.5" value={subtitleOutlineWidth} onChange={e => setSubtitleOutlineWidth(parseFloat(e.target.value))} className="slider-sm" disabled={!isSubtitleOutline} />
                        </div>
                        <div className="mini-control">
                          <label className="label-xs">顏色</label>
                          <input type="color" value={subtitleOutlineColor} onChange={e => setSubtitleOutlineColor(e.target.value)} className="color-sm" style={{ width: '100%' }} disabled={!isSubtitleOutline} />
                        </div>
                      </div>
                    </div>

                    <div className="style-group">
                      <label className="checkbox-inline" style={{ marginBottom: '14px', display: 'flex' }}>
                        <input type="checkbox" checked={isSubtitleShadow} onChange={e => setIsSubtitleShadow(e.target.checked)} />
                        <span className="label-xs" style={{ fontWeight: 600, color: '#fafafa' }}>陰影 Shadow</span>
                      </label>
                      {isSubtitleShadow && (
                        <div style={{ paddingLeft: '4px' }}>
                          <div className="style-row-inline" style={{ marginBottom: '16px' }}>
                            <input type="color" value={subtitleShadowColor} onChange={e => setSubtitleShadowColor(e.target.value)} className="color-sm" style={{ width: '36px', height: '36px' }} />
                            <div className="style-col-flex" style={{ flex: 1, marginLeft: '12px' }}>
                              <div className="style-row-inline" style={{ justifyContent: 'space-between', marginBottom: '4px' }}>
                                <span className="label-xs">模糊</span>
                                <input type="range" min="0" max="30" value={subtitleShadowBlur} onChange={e => setSubtitleShadowBlur(parseFloat(e.target.value))} className="slider-sm" style={{ width: '80px' }} />
                              </div>
                              <div className="style-row-inline" style={{ justifyContent: 'space-between' }}>
                                <span className="label-xs">不透</span>
                                <input type="range" min="0" max="100" value={subtitleShadowOpacity} onChange={e => setSubtitleShadowOpacity(parseInt(e.target.value))} className="slider-sm" style={{ width: '80px' }} />
                              </div>
                            </div>
                          </div>
                          <div className="style-col-flex" style={{ marginTop: '14px', gap: '8px' }}>
                            <div className="style-row-inline" style={{ justifyContent: 'space-between' }}>
                              <span className="label-xs">Shadow X</span>
                              <input type="range" min="-50" max="50" value={subtitleShadowOffsetX} onChange={e => setSubtitleShadowOffsetX(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1, margin: '0 8px' }} />
                              <input type="number" value={subtitleShadowOffsetX} onChange={e => setSubtitleShadowOffsetX(parseFloat(e.target.value))} className="input-xs" style={{ width: '36px' }} />
                            </div>
                            <div className="style-row-inline" style={{ justifyContent: 'space-between' }}>
                              <span className="label-xs">Shadow Y</span>
                              <input type="range" min="-50" max="50" value={subtitleShadowOffsetY} onChange={e => setSubtitleShadowOffsetY(parseFloat(e.target.value))} className="slider-sm" style={{ flex: 1, margin: '0 8px' }} />
                              <input type="number" value={subtitleShadowOffsetY} onChange={e => setSubtitleShadowOffsetY(parseFloat(e.target.value))} className="input-xs" style={{ width: '36px' }} />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="style-group">
                      <label className="checkbox-sm">
                        <input type="checkbox" checked={subtitleBgEnabled} onChange={e => setSubtitleBgEnabled(e.target.checked)} />
                        <span>字幕背景框</span>
                      </label>
                      {subtitleBgEnabled && (
                        <div className="nested-panel" style={{ marginTop: '10px' }}>
                          <div className="style-row-inline" style={{ marginBottom: '8px', gap: '8px' }}>
                            <input type="color" value={subtitleBgColor} onChange={e => setSubtitleBgColor(e.target.value)} className="color-sm" style={{ width: '32px' }} />
                            <span className="label-xs">不透</span>
                            <input type="range" min="0" max="100" value={subtitleBgOpacity} onChange={e => setSubtitleBgOpacity(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                          </div>
                          <div className="style-col-flex" style={{ gap: '10px' }}>
                            <div className="slider-row">
                              <label className="label-xs" style={{ minWidth: '40px' }}>Pad X</label>
                              <input type="range" min="0" max="100" value={subtitleBgPaddingX} onChange={e => setSubtitleBgPaddingX(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                              <input type="number" value={subtitleBgPaddingX} onChange={e => setSubtitleBgPaddingX(parseInt(e.target.value))} className="input-xs" style={{ width: '40px' }} />
                            </div>
                            <div className="slider-row">
                              <label className="label-xs" style={{ minWidth: '40px' }}>Pad Y</label>
                              <input type="range" min="0" max="100" value={subtitleBgPaddingY} onChange={e => setSubtitleBgPaddingY(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                              <input type="number" value={subtitleBgPaddingY} onChange={e => setSubtitleBgPaddingY(parseInt(e.target.value))} className="input-xs" style={{ width: '40px' }} />
                            </div>
                            <div className="slider-row">
                              <label className="label-xs" style={{ minWidth: '40px' }}>圓角</label>
                              <input type="range" min="0" max="50" value={subtitleBgRadius} onChange={e => setSubtitleBgRadius(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                              <input type="number" value={subtitleBgRadius} onChange={e => setSubtitleBgRadius(parseInt(e.target.value))} className="input-xs" style={{ width: '40px' }} />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 4. Cuts List */}
                <div className="controls-section no-border">
                  <div className="section-header"><Scissors size={16} className="icon-primary" /><span>自動裁切片段清單</span></div>
                  <div className="cuts-list">
                    {cuts.length === 0 ? <div className="empty-state">尚未分析片段</div> : cuts.map(cut => (
                      <div key={cut.id} className="cut-card">
                        <div className="cut-header">
                          <input className="cut-label-input" value={cut.label} onChange={(e) => updateCutLabel(cut.id, e.target.value)} />
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="cut-duration-badge">{(cut.end - cut.start).toFixed(1)}s</span>
                            <button onClick={() => setCuts(cuts.filter(c => c.id !== cut.id))} className="btn-delete"><Trash size={14} /></button>
                          </div>
                        </div>
                        <div className="cut-time-inputs">
                          <div className="time-input-group">
                            <label>開始</label>
                            <input
                              type="text"
                              value={formatTimestamp(cut.start)}
                              readOnly
                              className="time-input-readonly"
                              style={{ width: '70px', textAlign: 'center', background: 'transparent', border: 'none', color: '#4ade80', fontFamily: 'monospace' }}
                            />
                          </div>
                          <div className="time-input-group">
                            <label>結束</label>
                            <input
                              type="text"
                              value={formatTimestamp(cut.end)}
                              readOnly
                              className="time-input-readonly"
                              style={{ width: '70px', textAlign: 'center', background: 'transparent', border: 'none', color: '#4ade80', fontFamily: 'monospace' }}
                            />
                          </div>
                        </div>
                        {/* Time Range Sliders */}
                        < div className="cut-slider-row" >
                          <label>開始</label>
                          <input
                            type="range"
                            min="0"
                            max={duration}
                            step="0.1"
                            value={cut.start}
                            onChange={(e) => updateCutTime(cut.id, 'start', parseFloat(e.target.value))}
                            className="cut-slider"
                          />
                        </div>
                        <div className="cut-slider-row">
                          <label>結束</label>
                          <input
                            type="range"
                            min="0"
                            max={duration}
                            step="0.1"
                            value={cut.end}
                            onChange={(e) => updateCutTime(cut.id, 'end', parseFloat(e.target.value))}
                            className="cut-slider"
                          />
                        </div>
                        <div className="cut-actions">
                          <button className="btn-seek" onClick={() => seekTo(cut.start)}>
                            <Play size={12} /> 跳轉開始
                          </button>
                          <button className="btn-preview-xs" onClick={() => handlePreview(cut)}>
                            <Play size={12} /> 預覽
                          </button>
                          <button className="btn-preview-xs" style={{ background: 'rgba(16, 185, 129, 0.1)', borderColor: 'rgba(16, 185, 129, 0.3)' }} onClick={() => handleExport(cut)}>
                            <Download size={12} /> 匯出此段
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 5. Output Settings & Export */}
              <div className="controls-footer">
                <div className="output-settings">
                  <div className="output-row">
                    <label>畫質</label>
                    <select value={outputQuality} onChange={e => setOutputQuality(e.target.value)} className="select-xs">
                      <option value="high">高畫質 (20Mbps)</option>
                      <option value="medium">中畫質 (8Mbps)</option>
                      <option value="low">低畫質 (4Mbps)</option>
                    </select>
                  </div>
                  <div className="output-row">
                    <label>解析度 (9:16)</label>
                    <select value={outputResolution} onChange={e => setOutputResolution(e.target.value)} className="select-xs">
                      <option value="1080x1920">1080×1920</option>
                      <option value="720x1280">720×1280</option>
                      <option value="original">原始比例</option>
                    </select>
                  </div>
                </div>
                <button onClick={() => handleExport()} disabled={cuts.length === 0 || isExporting} className="btn-export-main">
                  {isExporting ? <Loader2 className="spin" size={18} /> : <><Download size={18} /> 匯出全部片段</>}
                </button>
              </div>
            </aside>
          </div>
        </div >
      </main >

      {/* Bottom Progress Bar */}
      {
        (isExporting || isPreviewLoading) && (
          <div className="progress-bar-bottom">
            <div className="progress-bar-fill" style={{ width: `${isPreviewLoading ? previewProgress : exportProgress}%` }} />
            <div className="progress-bar-content">
              <Loader2 className="spin" size={16} />
              <span>{isPreviewLoading ? previewMessage : (systemStatus.message || '匯出處理中')}</span>
              <span className="progress-percent">{Math.round(isPreviewLoading ? previewProgress : exportProgress)}%</span>
            </div>
            {/* Cancel Button (Only for export for now) */}
            {!isPreviewLoading && (
              <button className="btn-icon-xs" style={{ marginLeft: '10px', background: 'rgba(255,255,255,0.2)' }} onClick={() => window.location.reload()}>
                <X size={14} />
              </button>
            )}
          </div>
        )
      }
    </div >
  );
}


export default App;
