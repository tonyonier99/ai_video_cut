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
  const [isStudioSound, setIsStudioSound] = useState(true);
  const [isAutoCaption, setIsAutoCaption] = useState(true);
  const [isTranslate, setIsTranslate] = useState(true);
  const [isBurnCaptions, setIsBurnCaptions] = useState(true);

  // 進階參數
  const [trackZoom, setTrackZoom] = useState(1.5);
  const [mpMinDetectionCon, setMpMinDetectionCon] = useState(0.5);
  const [dfn3Strength, setDfn3Strength] = useState(100);

  // 字幕樣式
  const [subtitleFontSize, setSubtitleFontSize] = useState(72);
  const [subtitleColor, setSubtitleColor] = useState('#FFFFFF');
  const [subtitleFontName, setSubtitleFontName] = useState('Arial Black');
  const [subtitleOutlineWidth, setSubtitleOutlineWidth] = useState(4);
  const [subtitleOutlineColor, setSubtitleOutlineColor] = useState('#000000');
  const [subtitleMarginV, setSubtitleMarginV] = useState(100);
  const [subtitleBold, setSubtitleBold] = useState(true);
  const [subtitleItalic, setSubtitleItalic] = useState(false);
  const [subtitleShadowSize, setSubtitleShadowSize] = useState(3);
  const [subtitleShadowAngle, setSubtitleShadowAngle] = useState(135);
  const [subtitleShadowColor, setSubtitleShadowColor] = useState('#000000');
  const [subtitleShadowOpacity, setSubtitleShadowOpacity] = useState(80);
  const [subtitleCharsPerLine, setSubtitleCharsPerLine] = useState(12);
  const [subtitleBgEnabled, setSubtitleBgEnabled] = useState(false);
  const [subtitleBgColor, setSubtitleBgColor] = useState('#000000');
  const [subtitleBgOpacity, setSubtitleBgOpacity] = useState(50);
  const [subtitleBgRadius, setSubtitleBgRadius] = useState(8);
  const [showSafeArea, setShowSafeArea] = useState(true);
  const [previewText, setPreviewText] = useState('預覽文字內容');
  const [isDraggingSubtitle, setIsDraggingSubtitle] = useState(false);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [previewingId, setPreviewingId] = useState<string | null>(null);
  const [systemStatus, setSystemStatus] = useState({ progress: 100, status: 'ready', message: '就緒' });
  const [fontList, setFontList] = useState(['Arial Black', 'Helvetica', 'Times New Roman', 'Inter', 'Outfit']);

  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/status');
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
        }
      } catch (e) { }
    };
    const timer = setInterval(pollStatus, 2000);
    pollStatus();
    return () => clearInterval(timer);
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
    const formData = new FormData();
    formData.append('video', videoFile);
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
    formData.append('subtitle_chars_per_line', subtitleCharsPerLine.toString());

    try {
      const res = await fetch('http://localhost:8000/preview-clip', { method: 'POST', body: formData });
      if (res.ok) {
        const blob = await res.blob();
        setPreviewUrl(URL.createObjectURL(blob));
      }
    } catch (e) { }
    setIsPreviewLoading(false);
    setPreviewingId(null);
  };

  const handleExport = async () => {
    if (cuts.length === 0) return;
    setIsExporting(true);

    const formData = new FormData();
    formData.append('video', videoFile!);
    formData.append('cuts', JSON.stringify(cuts));
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
    formData.append('subtitle_chars_per_line', subtitleCharsPerLine.toString());

    try {
      const res = await fetch('http://localhost:8000/process-video', { method: 'POST', body: formData });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Antigravity_Cuts_${new Date().getTime()}.zip`;
        a.click();
      }
    } catch (e) { }
    setIsExporting(false);
  };

  const updateCutLabel = (id: string, label: string) => setCuts(cuts.map(c => c.id === id ? { ...c, label } : c));
  const updateCutTime = (id: string, field: 'start' | 'end', value: number) => {
    setCuts(cuts.map(c => c.id === id ? { ...c, [field]: Math.max(0, Math.min(duration, value)) } : c));
  };
  const seekTo = (time: number) => { if (videoRef.current) videoRef.current.currentTime = time; };

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (timelineRef.current) {
      const rect = timelineRef.current.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      const targetTime = pos * duration;
      seekTo(targetTime);
    }
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
          {apiKey ? <span className="status-badge success">AI Ready</span> : <span className="status-badge warning" onClick={() => setShowApiSettings(true)}>Set API Key</span>}
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
                >
                  <div className="playhead" style={{ left: `${(currentTime / (duration || 1)) * 100}%` }} />
                  {cuts.map(cut => (
                    <div key={cut.id} className="cut-marker" style={{ left: `${(cut.start / (duration || 1)) * 100}%`, width: `${((cut.end - cut.start) / (duration || 1)) * 100}%` }} />
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
                    fontSize: `${subtitleFontSize / 2.5}px`,
                    fontFamily: subtitleFontName,
                    fontWeight: subtitleBold ? 'bold' : 'normal',
                    fontStyle: subtitleItalic ? 'italic' : 'normal',
                    textShadow: `${getShadowOffset(subtitleShadowAngle, subtitleShadowSize).x}px ${getShadowOffset(subtitleShadowAngle, subtitleShadowSize).y}px ${subtitleShadowSize * 2}px ${subtitleShadowColor}${Math.round(subtitleShadowOpacity * 2.55).toString(16).padStart(2, '0')}`,
                    WebkitTextStroke: `${subtitleOutlineWidth / 2}px ${subtitleOutlineColor}`,
                    paintOrder: 'stroke fill',
                    cursor: 'ns-resize',
                    backgroundColor: subtitleBgEnabled ? `${subtitleBgColor}${Math.round(subtitleBgOpacity * 2.55).toString(16).padStart(2, '0')}` : 'transparent',
                    padding: subtitleBgEnabled ? '6px 14px' : '0',
                    borderRadius: subtitleBgEnabled ? `${subtitleBgRadius}px` : '0',
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
                    <button onClick={() => setShowApiSettings(!showApiSettings)} className={`toggle-btn-sm ${apiKey ? 'active' : ''}`}><Key size={14} /></button>
                  </div>

                  <AnimatePresence>
                    {showApiSettings && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="settings-drawer">
                        <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Gemini API Key..." className="api-input-sm" />
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
                        <label className="checkbox-inline"><input type="checkbox" checked={isTranslate} onChange={e => setIsTranslate(e.target.checked)} /><span>繁體中文翻譯</span></label>
                        <label className="checkbox-inline"><input type="checkbox" checked={isBurnCaptions} onChange={e => setIsBurnCaptions(e.target.checked)} /><span>燒錄至影片</span></label>
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
                      <select value={subtitleFontName} onChange={e => setSubtitleFontName(e.target.value)} className="select-sm">
                        {fontList.map(f => <option key={f} value={f}>{f}</option>)}
                      </select>
                    </div>

                    {/* Font Size Slider */}
                    <div className="slider-row">
                      <label>字體大小</label>
                      <input type="range" min="24" max="150" value={subtitleFontSize} onChange={e => setSubtitleFontSize(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="24" max="150" value={subtitleFontSize} onChange={e => setSubtitleFontSize(parseInt(e.target.value) || 24)} className="val-input" />
                    </div>

                    {/* Colors */}
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

                    {/* Outline Width Slider */}
                    <div className="slider-row">
                      <label>邊框寬度</label>
                      <input type="range" min="0" max="20" value={subtitleOutlineWidth} onChange={e => setSubtitleOutlineWidth(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="0" max="20" value={subtitleOutlineWidth} onChange={e => setSubtitleOutlineWidth(parseInt(e.target.value) || 0)} className="val-input" />
                    </div>

                    {/* Shadow Slider */}
                    <div className="slider-row">
                      <label>陰影大小</label>
                      <input type="range" min="0" max="20" value={subtitleShadowSize} onChange={e => setSubtitleShadowSize(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="0" max="20" value={subtitleShadowSize} onChange={e => setSubtitleShadowSize(parseInt(e.target.value) || 0)} className="val-input" />
                    </div>

                    {/* Shadow Angle */}
                    <div className="slider-row">
                      <label>陰影角度</label>
                      <input type="range" min="0" max="360" value={subtitleShadowAngle} onChange={e => setSubtitleShadowAngle(parseInt(e.target.value))} className="slider-sm" />
                      <input type="number" min="0" max="360" value={subtitleShadowAngle} onChange={e => setSubtitleShadowAngle(parseInt(e.target.value) || 0)} className="val-input" />
                    </div>

                    {/* Shadow Color & Opacity */}
                    <div className="style-grid" style={{ marginTop: '14px' }}>
                      <div className="style-item">
                        <label>陰影顏色</label>
                        <input type="color" value={subtitleShadowColor} onChange={e => setSubtitleShadowColor(e.target.value)} className="color-sm" />
                      </div>
                      <div className="style-item">
                        <label>陰影透明度</label>
                        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                          <input type="range" min="0" max="100" value={subtitleShadowOpacity} onChange={e => setSubtitleShadowOpacity(parseInt(e.target.value))} className="slider-sm" style={{ width: '60px' }} />
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
                          <div className="slider-row">
                            <label>圓角</label>
                            <input type="range" min="0" max="30" value={subtitleBgRadius} onChange={e => setSubtitleBgRadius(parseInt(e.target.value))} className="slider-sm" />
                            <input type="number" min="0" max="30" value={subtitleBgRadius} onChange={e => setSubtitleBgRadius(parseInt(e.target.value) || 0)} className="val-input" />
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
                          <button onClick={() => setCuts(cuts.filter(c => c.id !== cut.id))} className="btn-delete"><Trash size={12} /></button>
                        </div>
                        <div className="cut-time-inputs">
                          <div className="time-input-group">
                            <label>開始</label>
                            <input
                              type="number"
                              step="0.1"
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
                              value={cut.end.toFixed(1)}
                              onChange={(e) => updateCutTime(cut.id, 'end', parseFloat(e.target.value) || 0)}
                              className="time-input"
                            />
                            <span>s</span>
                          </div>
                          <span className="cut-duration">({(cut.end - cut.start).toFixed(1)}s)</span>
                        </div>
                        <button className={`btn-preview-xs ${previewingId === cut.id ? 'active' : ''}`} onClick={() => handlePreview(cut)} disabled={isPreviewLoading}>
                          {previewingId === cut.id ? <Loader2 className="spin" size={10} /> : <Play size={10} />} 預覽此片段
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 5. Footer Export */}
              <div className="controls-footer">
                <button onClick={() => handleExport()} disabled={cuts.length === 0 || isExporting} className="btn-export-main">
                  {isExporting ? <Loader2 className="spin" size={18} /> : <><Download size={18} /> 匯出全部片段</>}
                </button>
              </div>
            </aside>
          </div>
        </div>



      </main>
    </div>
  );
}

export default App;
