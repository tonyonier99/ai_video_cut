import React, { useState, useRef, useEffect } from 'react';
import { Download, Video, Scissors, Play, Pause, Loader2, Film, Key, Upload, Wand2, Languages, X, Trash } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import WaveSurfer from 'wavesurfer.js';
import './App.css';

interface Cut {
  id: string;
  start: number;
  end: number;
  label: string;
}

function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [previewProgress, setPreviewProgress] = useState(0);

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
  const [isTranslate, setIsTranslate] = useState(false); // Default false, Whisper directly outputs Chinese
  const [isBurnCaptions, setIsBurnCaptions] = useState(true);

  // 進階參數
  const [trackZoom, setTrackZoom] = useState(1.5);
  const [mpMinDetectionCon, setMpMinDetectionCon] = useState(0.5);
  const [dfn3Strength, setDfn3Strength] = useState(100);

  // 字幕樣式
  const [subtitleFontSize, setSubtitleFontSize] = useState(55);
  const [subtitleColor, setSubtitleColor] = useState('#FFFFFF');
  const [subtitleFontName, setSubtitleFontName] = useState('Heiti TC'); // Default to a Chinese font
  const [subtitleOutlineWidth, setSubtitleOutlineWidth] = useState(4);
  const [subtitleOutlineColor, setSubtitleOutlineColor] = useState('#000000');
  const [subtitleMarginV, setSubtitleMarginV] = useState(50);
  const [subtitleBold, setSubtitleBold] = useState(false); // Default disabled
  const [subtitleItalic, setSubtitleItalic] = useState(false);
  const [subtitleShadowSize, setSubtitleShadowSize] = useState(3);
  // Removed unsupported Angle (Fixed to 135/45 deg in backend)
  const [subtitleShadowColor, setSubtitleShadowColor] = useState('#000000');
  const [subtitleShadowOpacity, setSubtitleShadowOpacity] = useState(80);
  const [subtitleCharsPerLine, setSubtitleCharsPerLine] = useState(8);
  const [subtitleBgEnabled, setSubtitleBgEnabled] = useState(false);
  const [subtitleBgColor, setSubtitleBgColor] = useState('#000000');
  const [subtitleBgOpacity, setSubtitleBgOpacity] = useState(50);
  // Removed unsupported Radius
  const [showSafeArea, setShowSafeArea] = useState(true);
  const [previewText, setPreviewText] = useState('預覽文字內容Test123456');
  const [isDraggingSubtitle, setIsDraggingSubtitle] = useState(false);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewingId, setPreviewingId] = useState<string | null>(null);
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
        // User sets font size relative to 1080p height.
        // We scale the DOM preview linearly based on the canvas height vs 1080.
        setScaleRatio(previewCanvasRef.current.offsetHeight / 1080);
      }
    };
    const observer = new ResizeObserver(updateScale);
    observer.observe(previewCanvasRef.current);
    updateScale(); // Init relative to current size
    return () => observer.disconnect();
  }, []);

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
            setFontList(prev => {
              // Only show fonts that actually exist on backend + basic fallbacks
              const combined = [...new Set([...data.fonts, 'Arial', 'Sans-serif'])];

              // Dynamically load fonts for accurate preview
              data.fonts.forEach(async (fontName: string) => {
                // Try TTF first
                const fontTTF = new FontFace(fontName, `url(http://localhost:8000/fonts/${fontName}.ttf)`);
                try {
                  await fontTTF.load();
                  document.fonts.add(fontTTF);
                  return;
                } catch (e) { }

                // Try OTF fallback
                const fontOTF = new FontFace(fontName, `url(http://localhost:8000/fonts/${fontName}.otf)`);
                try {
                  await fontOTF.load();
                  document.fonts.add(fontOTF);
                } catch (e) { }
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
    formData.append('video', videoFile);
    formData.append('instruction', instruction);
    formData.append('target_count', targetCount.toString());
    formData.append('api_key', apiKey);
    formData.append('model', selectedModel);

    try {
      const res = await fetch('http://localhost:8000/analyze-video', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        setCuts(data.cuts.map((c: any, i: number) => ({ ...c, id: i.toString() })));
      }
    } catch (e) {
      alert('分析失敗，請檢查 API Key 或後端狀態');
    }
    setIsProcessing(false);
  };

  const handlePreview = async (cut: Cut) => {
    if (!videoFile) return;
    setIsPreviewLoading(true);
    setPreviewingId(cut.id);
    setPreviewProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setPreviewProgress(prev => Math.min(prev + Math.random() * 10, 95));
    }, 500);
    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('start', cut.start.toString());
    formData.append('end', cut.end.toString());
    formData.append('face_tracking', isFaceTracking.toString());
    formData.append('studio_sound', isStudioSound.toString());
    formData.append('dfn3_strength', dfn3Strength.toString());
    formData.append('auto_caption', isAutoCaption.toString());
    formData.append('translate', isTranslate.toString());
    formData.append('burn_captions', isBurnCaptions.toString());
    formData.append('subtitle_font_name', subtitleFontName);
    formData.append('subtitle_font_size', subtitleFontSize.toString());
    formData.append('subtitle_color', subtitleColor);
    formData.append('subtitle_outline_width', subtitleOutlineWidth.toString());
    formData.append('subtitle_outline_color', subtitleOutlineColor);
    formData.append('subtitle_chars_per_line', subtitleCharsPerLine.toString());
    formData.append('whisper_language', whisperLanguage);

    // Missing Params Added
    formData.append('subtitle_margin_v', subtitleMarginV.toString());
    formData.append('subtitle_bold', subtitleBold.toString());
    formData.append('subtitle_italic', subtitleItalic.toString());
    formData.append('subtitle_shadow_size', subtitleShadowSize.toString());
    formData.append('subtitle_box_enabled', subtitleBgEnabled.toString());
    formData.append('subtitle_box_color', subtitleBgColor);
    formData.append('subtitle_box_alpha', (subtitleBgOpacity / 100).toString());


    try {
      const res = await fetch('http://localhost:8000/preview-clip', { method: 'POST', body: formData });
      if (res.ok) {
        const blob = await res.blob();
        setPreviewUrl(URL.createObjectURL(blob));
      } else {
        const errorData = await res.json().catch(() => ({ detail: '預覽生成失敗' }));
        alert(`預覽失敗: ${errorData.detail || res.statusText}`);
      }
    } catch (e) {
      alert('無法連接後端服務，請確認 server.py 正在運行');
    }
    clearInterval(progressInterval);
    setPreviewProgress(100);
    setTimeout(() => setPreviewProgress(0), 500);
    setIsPreviewLoading(false);
    setPreviewingId(null);
  };

  const handleExport = async () => {
    if (cuts.length === 0) return;
    setIsExporting(true);
    setExportProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setExportProgress(prev => Math.min(prev + Math.random() * 5, 95));
    }, 800);

    const formData = new FormData();
    formData.append('file', videoFile!);
    formData.append('cuts_json', JSON.stringify(cuts));
    formData.append('face_tracking', isFaceTracking.toString());
    formData.append('studio_sound', isStudioSound.toString());
    formData.append('dfn3_strength', dfn3Strength.toString());
    formData.append('auto_caption', isAutoCaption.toString());
    formData.append('translate', isTranslate.toString());
    formData.append('burn_captions', isBurnCaptions.toString());
    formData.append('subtitle_font_name', subtitleFontName);
    formData.append('subtitle_font_size', subtitleFontSize.toString());
    formData.append('subtitle_color', subtitleColor);
    formData.append('subtitle_outline_width', subtitleOutlineWidth.toString());
    formData.append('subtitle_outline_color', subtitleOutlineColor);
    formData.append('subtitle_chars_per_line', subtitleCharsPerLine.toString());

    // Missing Params Added
    formData.append('subtitle_margin_v', subtitleMarginV.toString());
    formData.append('subtitle_bold', subtitleBold.toString());
    formData.append('subtitle_italic', subtitleItalic.toString());
    formData.append('subtitle_shadow_size', subtitleShadowSize.toString());
    formData.append('subtitle_box_enabled', subtitleBgEnabled.toString());
    formData.append('subtitle_box_color', subtitleBgColor);
    formData.append('subtitle_box_alpha', (subtitleBgOpacity / 100).toString());

    formData.append('output_quality', outputQuality);
    formData.append('output_resolution', outputResolution);
    formData.append('whisper_language', whisperLanguage);

    try {
      const res = await fetch('http://localhost:8000/process-video', { method: 'POST', body: formData });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Antigravity_Cuts_${new Date().getTime()}.zip`;
        a.click();
      } else {
        const errorData = await res.json().catch(() => ({ detail: '匯出失敗' }));
        alert(`匯出失敗: ${errorData.detail || res.statusText}`);
      }
    } catch (e) {
      alert('無法連接後端服務，請確認 server.py 正在運行');
    }
    clearInterval(progressInterval);
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
    const lines: string[] = [];
    for (let i = 0; i < text.length; i += charsPerLine) {
      lines.push(text.slice(i, i + charsPerLine));
    }
    return lines.join('\n');
  };

  // Calculate shadow offset from angle
  const getShadowOffset = (angle: number, size: number) => {
    const rad = (angle * Math.PI) / 180;
    return {
      x: Math.round(Math.cos(rad) * size),
      y: Math.round(Math.sin(rad) * size)
    };
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

      {/* Preview Modal */}
      <AnimatePresence>
        {previewUrl && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => setPreviewUrl(null)}
            style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', zIndex: 9999, display: 'flex', justifyContent: 'center', alignItems: 'center' }}
          >
            <motion.div
              className="modal-content glass"
              onClick={e => e.stopPropagation()}
              style={{ width: '90%', maxWidth: '360px', padding: 24, borderRadius: 20 }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <h3 style={{ margin: 0 }}>預覽片段</h3>
                <button onClick={() => setPreviewUrl(null)} className="btn-ghost"><X size={20} /></button>
              </div>
              <video src={previewUrl} controls autoPlay style={{ width: '100%', borderRadius: 12, boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }} />
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
                    const y = e.clientY - rect.top;
                    const percentage = 100 - (y / rect.height) * 100;
                    setSubtitleMarginV(Math.max(0, Math.min(200, Math.round(percentage * 2))));
                  }
                }}
                onMouseUp={() => setIsDraggingSubtitle(false)}
                onMouseLeave={() => setIsDraggingSubtitle(false)}
              >
                {showSafeArea && <div className="safe-area-guide" />}
                <div
                  className="subtitle-draggable"
                  style={{
                    bottom: `${subtitleMarginV / 2}%`,
                    color: subtitleColor,
                    fontSize: `${subtitleFontSize * scaleRatio}px`,
                    fontFamily: subtitleFontName,
                    fontWeight: subtitleBold ? 'bold' : 'normal',
                    fontStyle: subtitleItalic ? 'italic' : 'normal',
                    // Dynamic scaling for Shadow
                    textShadow: `${Math.round(subtitleShadowSize * scaleRatio * 0.7)}px ${Math.round(subtitleShadowSize * scaleRatio * 0.7)}px 0px ${subtitleShadowColor}${Math.round(subtitleShadowOpacity * 2.55).toString(16).padStart(2, '0')}`,
                    // Dynamic scaling for Stroke
                    // FIX: Multiply by 2. Web CSS stroke is centered (half hidden by fill), ASS outline is outer.
                    // So we must 2x the CSS stroke to match the visual thickness of ASS outline.
                    WebkitTextStroke: subtitleBgEnabled ? '0px' : `${subtitleOutlineWidth * 2 * scaleRatio}px ${subtitleOutlineColor}`,
                    paintOrder: 'stroke fill',
                    cursor: 'ns-resize',
                    backgroundColor: subtitleBgEnabled ? `${subtitleBgColor}${Math.round(subtitleBgOpacity * 2.55).toString(16).padStart(2, '0')}` : 'transparent',
                    // Dynamic scaling for Padding
                    padding: subtitleBgEnabled ? `${4 * scaleRatio}px ${subtitleOutlineWidth * scaleRatio}px` : '0',
                    borderRadius: '0px',
                  }}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setIsDraggingSubtitle(true);
                  }}
                >
                  {wrapText(previewText, subtitleCharsPerLine)}
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
                  <button onClick={handleMakeVideo} disabled={!videoFile || isProcessing} className="btn-action-sm">
                    {isProcessing ? <Loader2 className="spin" size={14} /> : <><Wand2 size={14} /> AI 分析片段</>}
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
                      <span>DFN3 深度智慧降噪</span>
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
                      <input type="checkbox" checked={isAutoCaption} onChange={e => setIsAutoCaption(e.target.checked)} />
                      <span>AI 語音轉文字 (SRT)</span>
                    </label>
                    {isAutoCaption && (
                      <div className="caption-sub-settings">
                        <div className="style-row-inline" style={{ marginBottom: '8px' }}>
                          <span className="style-label" style={{ minWidth: '60px' }}>轉錄語言</span>
                          <select value={whisperLanguage} onChange={e => setWhisperLanguage(e.target.value)} className="select-xs" style={{ flex: 1 }}>
                            <option value="zh">繁體中文 (Chinese)</option>
                            <option value="en">英文 (English)</option>
                            <option value="ja">日文 (Japanese)</option>
                            <option value="auto">自動偵測 (Auto)</option>
                          </select>
                        </div>
                        <label className="checkbox-inline"><input type="checkbox" checked={isTranslate} onChange={e => setIsTranslate(e.target.checked)} /><span>繁體中文翻譯 (Gemini)</span></label>
                        <label className="checkbox-inline"><input type="checkbox" checked={isBurnCaptions} onChange={e => setIsBurnCaptions(e.target.checked)} /><span>燒錄至影片 (硬字幕)</span></label>
                      </div>
                    )}
                  </div>
                </div>

                {/* 3. Subtitle Styles */}
                {isBurnCaptions && (
                  <div className="controls-section">
                    <div className="section-header"><Wand2 size={16} className="icon-primary" /><span>字幕視覺設計</span></div>

                    {/* Font Selection */}
                    <div className="style-item" style={{ marginBottom: '16px' }}>
                      <label>字體系統</label>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <select value={subtitleFontName} onChange={e => setSubtitleFontName(e.target.value)} className="select-sm" style={{ flex: 1 }}>
                          {fontList.map(f => <option key={f} value={f}>{f}</option>)}
                        </select>
                        <input
                          type="file"
                          accept=".ttf,.otf,.woff,.woff2"
                          style={{ display: 'none' }}
                          id="font-upload"
                          onChange={async (e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;
                            const formData = new FormData();
                            formData.append('file', file);
                            try {
                              const res = await fetch('http://localhost:8000/upload-font', { method: 'POST', body: formData });
                              if (res.ok) {
                                const data = await res.json();
                                const fontName = data.font_name || file.name.replace(/\.[^/.]+$/, '');
                                setFontList(prev => [...new Set([...prev, fontName])]);
                                setSubtitleFontName(fontName);
                                alert(`字體 "${fontName}" 上傳成功！`);
                              }
                            } catch {
                              alert('字體上傳失敗，請確認後端服務運行中');
                            }
                          }}
                        />
                        <label htmlFor="font-upload" className="btn-icon-sm" style={{ cursor: 'pointer', padding: '6px 10px' }}>
                          <Upload size={14} />
                        </label>
                      </div>
                    </div>

                    {/* Font Size Slider */}
                    <div className="slider-row">
                      <label>字體大小</label>
                      <input type="range" min="24" max="150" value={subtitleFontSize} onChange={e => setSubtitleFontSize(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="24" max="150" value={subtitleFontSize} onChange={e => setSubtitleFontSize(parseInt(e.target.value) || 24)} className="val-input" />
                    </div>

                    {/* Outline Width / Box Padding Slider (Dual Purpose) */}
                    <div className="slider-row" style={{ marginTop: '14px' }}>
                      <label>{subtitleBgEnabled ? '背景寬度 (Padding)' : '邊框寬度'}</label>
                      <input type="range" min="0" max={subtitleBgEnabled ? 50 : 20} value={subtitleOutlineWidth} onChange={e => setSubtitleOutlineWidth(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="0" max={subtitleBgEnabled ? 50 : 20} value={subtitleOutlineWidth} onChange={e => setSubtitleOutlineWidth(parseInt(e.target.value) || 0)} className="val-input" />
                    </div>

                    {/* Outline Color - Only show if BG is disabled */}
                    {!subtitleBgEnabled && (
                      <div className="style-grid" style={{ marginTop: '14px' }}>
                        <div className="style-item">
                          <label>文字顏色</label>
                          <input type="color" value={subtitleColor} onChange={e => setSubtitleColor(e.target.value)} className="color-sm" />
                        </div>
                        <div className="style-item">
                          <label>邊框顏色</label>
                          <input type="color" value={subtitleOutlineColor} onChange={e => setSubtitleOutlineColor(e.target.value)} className="color-sm" />
                        </div>
                      </div>
                    )}

                    {subtitleBgEnabled && (
                      <div className="style-item" style={{ marginTop: '14px' }}>
                        <label>文字顏色</label>
                        <input type="color" value={subtitleColor} onChange={e => setSubtitleColor(e.target.value)} className="color-sm" />
                      </div>
                    )}

                    {/* Shadow Slider */}
                    <div className="slider-row">
                      <label>陰影大小</label>
                      <input type="range" min="0" max="20" value={subtitleShadowSize} onChange={e => setSubtitleShadowSize(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="0" max="20" value={subtitleShadowSize} onChange={e => setSubtitleShadowSize(parseInt(e.target.value) || 0)} className="val-input" />
                    </div>

                    {/* Shadow Color & Opacity (Color removed, opacity only) */}
                    <div className="style-grid" style={{ marginTop: '14px' }}>
                      <div className="style-item" style={{ width: '100%' }}>
                        <label>陰影透明度</label>
                        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                          <input type="range" min="0" max="100" value={subtitleShadowOpacity} onChange={e => setSubtitleShadowOpacity(parseInt(e.target.value))} className="slider-sm" style={{ flex: 1 }} />
                          <input type="number" min="0" max="100" value={subtitleShadowOpacity} onChange={e => setSubtitleShadowOpacity(parseInt(e.target.value) || 0)} className="val-input" style={{ width: '40px' }} />
                        </div>
                      </div>
                    </div>

                    {/* Vertical Position Slider */}
                    <div className="slider-row">
                      <label>垂直位移</label>
                      <input type="range" min="0" max="200" value={subtitleMarginV} onChange={e => setSubtitleMarginV(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="0" max="200" value={subtitleMarginV} onChange={e => setSubtitleMarginV(parseInt(e.target.value) || 0)} className="val-input" />
                    </div>

                    {/* Chars Per Line Slider */}
                    <div className="slider-row">
                      <label>每行字數</label>
                      <input type="range" min="4" max="30" value={subtitleCharsPerLine} onChange={e => setSubtitleCharsPerLine(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="4" max="30" value={subtitleCharsPerLine} onChange={e => setSubtitleCharsPerLine(parseInt(e.target.value) || 4)} className="val-input" />
                    </div>

                    {/* Font Style */}
                    <div className="style-row-checkboxes" style={{ marginTop: '14px', display: 'flex', gap: '16px' }}>
                      <label className="checkbox-inline">
                        <input type="checkbox" checked={subtitleBold} onChange={e => setSubtitleBold(e.target.checked)} />
                        <span>粗體</span>
                      </label>
                      <label className="checkbox-inline">
                        <input type="checkbox" checked={subtitleItalic} onChange={e => setSubtitleItalic(e.target.checked)} />
                        <span>斜體</span>
                      </label>
                    </div>

                    {/* Background Box */}
                    <div className="style-block" style={{ marginTop: '16px' }}>
                      <label className="checkbox-sm">
                        <input type="checkbox" checked={subtitleBgEnabled} onChange={e => setSubtitleBgEnabled(e.target.checked)} />
                        <span>字幕背景框</span>
                      </label>
                      {subtitleBgEnabled && (
                        <div style={{ marginLeft: '28px', marginTop: '10px' }}>
                          <div className="style-item" style={{ marginBottom: '10px' }}>
                            <label>背景顏色</label>
                            <input type="color" value={subtitleBgColor} onChange={e => setSubtitleBgColor(e.target.value)} className="color-sm" />
                          </div>
                          <div className="slider-row">
                            <label>透明度</label>
                            <input type="range" min="0" max="100" value={subtitleBgOpacity} onChange={e => setSubtitleBgOpacity(parseInt(e.target.value))} className="slider-sm" />
                            <input type="number" min="0" max="100" value={subtitleBgOpacity} onChange={e => setSubtitleBgOpacity(parseInt(e.target.value) || 0)} className="val-input" />
                          </div>
                        </div>

                      )}
                    </div>

                    {/* Preview Text */}
                    <div className="preview-text-box">
                      <label>預覽文字</label>
                      <input type="text" value={previewText} onChange={e => setPreviewText(e.target.value)} className="input-full-sm" />
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
                          <button onClick={() => setCuts(cuts.filter(c => c.id !== cut.id))} className="btn-delete"><Trash size={14} /></button>
                        </div>
                        <div className="cut-time-inputs">
                          <div className="time-input-group">
                            <label>開始</label>
                            <input
                              type="number"
                              step="0.1"
                              min="0"
                              max={cut.end - 0.5}
                              value={cut.start.toFixed(1)}
                              onChange={(e) => updateCutTime(cut.id, 'start', parseFloat(e.target.value) || 0)}
                              className="time-input"
                            />
                            <span>s</span>
                          </div>
                          <div className="time-input-group">
                            <label>結束</label>
                            <input
                              type="number"
                              step="0.1"
                              min={cut.start + 0.5}
                              max={duration}
                              value={cut.end.toFixed(1)}
                              onChange={(e) => updateCutTime(cut.id, 'end', parseFloat(e.target.value) || 0)}
                              className="time-input"
                            />
                            <span>s</span>
                          </div>
                          <span className="cut-duration">({(cut.end - cut.start).toFixed(1)}s)</span>
                        </div>
                        {/* Time Range Sliders */}
                        <div className="cut-slider-row">
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
                          <button className={`btn-preview-xs ${previewingId === cut.id ? 'active' : ''}`} onClick={() => handlePreview(cut)} disabled={isPreviewLoading}>
                            {previewingId === cut.id ? <Loader2 className="spin" size={12} /> : <Play size={12} />} 預覽此片段
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
        (isPreviewLoading || isExporting) && (
          <div className="progress-bar-bottom">
            <div className="progress-bar-fill" style={{ width: `${isExporting ? exportProgress : previewProgress}%` }} />
            <div className="progress-bar-content">
              <Loader2 className="spin" size={16} />
              <span>{isExporting ? '匯出處理中' : '預覽生成中'}</span>
              <span className="progress-percent">{Math.round(isExporting ? exportProgress : previewProgress)}%</span>
            </div>
          </div>
        )
      }
    </div >
  );
}

export default App;
