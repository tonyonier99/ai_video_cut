import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { DragEvent } from 'react';
import { Play, Pause, Scissors, MousePointer2, ZoomIn, ZoomOut, Upload, Plus, Trash, Save, Film, Loader2, Zap, X, Settings, Download, ChevronRight, ChevronDown, RotateCcw, Monitor, Smartphone, Hand, Magnet, SplitSquareHorizontal, Type } from 'lucide-react';
import './App.css';

interface Cut {
  id: string;
  start: number;
  end: number;
  label: string;
  trackId: number;
  assetId?: string;
}

interface Asset {
  id: string;
  type: 'video' | 'image' | 'audio';
  name: string;
  url: string;
  duration?: number;
  file?: File;
}

interface TrackConfig {
  id: number;
  type: 'video' | 'audio' | 'text';
  name: string;
  visible: boolean;
  locked: boolean;
}

const MAX_HISTORY = 20;

function App() {
  // --- Global State ---
  const [projectAssets, setProjectAssets] = useState<Asset[]>([]);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [originalVideoPath, setOriginalVideoPath] = useState<string | null>(null);

  // --- Layout State ---
  const [isVerticalMode, setIsVerticalMode] = useState(false);
  const [leftPanelTab, setLeftPanelTab] = useState<'project' | 'controls' | 'effects' | 'ai'>('project');
  const [activeTool, setActiveTool] = useState<'select' | 'blade' | 'text'>('select');
  const [appView, setAppView] = useState<'welcome' | 'editor'>('welcome');

  // --- Timeline State ---
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [selectedCutId, setSelectedCutId] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(10);

  const [videoTracks, setVideoTracks] = useState<TrackConfig[]>([
    { id: 0, type: 'video', name: 'V1', visible: true, locked: false },
    { id: 1, type: 'text', name: 'T1', visible: true, locked: false }
  ]);

  const toggleTrackVisibility = (id: number) => {
    setVideoTracks(prev => prev.map(t => t.id === id ? { ...t, visible: !t.visible } : t));
  };

  const toggleTrackLock = (id: number) => {
    setVideoTracks(prev => prev.map(t => t.id === id ? { ...t, locked: !t.locked } : t));
  };

  // History
  const [history, setHistory] = useState<Cut[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Export Modal
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportFormat, setExportFormat] = useState('xml');
  const [exportResolution, setExportResolution] = useState('1080p');

  // Dragging State
  const [dragState, setDragState] = useState<{
    isDragging: boolean;
    type: 'scrub' | 'move' | 'trim-start' | 'trim-end' | null;
    targetId: string | null;
    startX: number;
    initialValue: number;
  }>({ isDragging: false, type: null, targetId: null, startX: 0, initialValue: 0 });

  // --- Export States ---
  const [exportAutoCaption, setExportAutoCaption] = useState(false);
  const [exportFaceTracking, setExportFaceTracking] = useState(false);
  const [exportStudioSound, setExportStudioSound] = useState(false);
  const [exportMergeClips, setExportMergeClips] = useState(true);
  // --- AI / Tools State ---
  const [apiKey, setApiKey] = useState(localStorage.getItem('antigravity_api_key') || '');
  const [highlightCount, setHighlightCount] = useState(5);
  const [targetDuration, setTargetDuration] = useState(60);
  const [instruction, setInstruction] = useState('');

  // Silence Removal Params
  const [silenceThreshold, setSilenceThreshold] = useState(-30);
  const [silenceMinDuration, setSilenceMinDuration] = useState(0.5);

  // --- Inspector State ---

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  // --- Persistence ---
  useEffect(() => {
    localStorage.setItem('antigravity_api_key', apiKey);
  }, [apiKey]);

  // --- Shortcuts ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input/textarea
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'KeyK' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSplit();
      } else if (e.code === 'Backspace' || e.code === 'Delete') {
        handleDelete();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, currentTime, cuts, selectedCutId]);

  // --- Persistence & Auto-Save ---
  useEffect(() => {
    const projectData = {
      cuts,
      videoUrl,
      originalVideoPath,
      isVerticalMode,
      projectAssets: projectAssets.map(a => ({ ...a, file: undefined })) // Can't serialize File objects
    };
    if (cuts.length > 0 || videoUrl) {
      localStorage.setItem('antigravity_current_project', JSON.stringify(projectData));
    }
  }, [cuts, videoUrl, originalVideoPath, isVerticalMode, projectAssets]);

  const loadProject = (data: any) => {
    try {
      if (data.cuts) setCuts(data.cuts);
      if (data.videoUrl) setVideoUrl(data.videoUrl);
      if (data.originalVideoPath) setOriginalVideoPath(data.originalVideoPath);
      if (data.isVerticalMode !== undefined) setIsVerticalMode(data.isVerticalMode);
      if (data.projectAssets) setProjectAssets(data.projectAssets);
      setAppView('editor');
    } catch (e) {
      console.error("Failed to load project", e);
      alert("專案檔案格式錯誤");
    }
  };

  const handleExportProject = () => {
    const projectData = {
      version: '1.0',
      timestamp: Date.now(),
      cuts,
      videoUrl,
      originalVideoPath,
      isVerticalMode,
      projectAssets: projectAssets.map(a => ({ ...a, file: undefined }))
    };
    const blob = new Blob([JSON.stringify(projectData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `project_${new Date().toISOString().split('T')[0]}.agpro`;
    a.click();
  };

  const handleImportProject = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string);
        loadProject(data);
      } catch (e) {
        alert("匯入失敗：無效的專案檔案");
      }
    };
    reader.readAsText(file);
  };

  // --- Player Logic ---
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) videoRef.current.pause();
      else videoRef.current.play();
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    // Only update state from video if NOT scrubbing
    if (videoRef.current && dragState.type !== 'scrub') {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      if (cuts.length === 0) {
        // Initial clip is full video
        setCuts([{ id: 'full', start: 0, end: videoRef.current.duration, label: 'Full Video', trackId: 0 }]);
      }
    }
  };

  // --- File Upload ---
  const processFile = async (file: File) => {
    if (!file) return;
    setVideoFile(file);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/upload-proxy', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        const encodedUrl = data.url; // Server already encodes if needed, but we'll ensure it's a valid URL object
        setVideoUrl(encodedUrl);
        setOriginalVideoPath(data.original_path);
        setCuts([]);
      } else {
        alert("Upload failed");
        // Fallback to client side
        setVideoUrl(URL.createObjectURL(file));
      }
    } catch (err) {
      console.error("Upload error", err);
      // Fallback
      setVideoUrl(URL.createObjectURL(file));
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      processFile(file);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };


  // --- Timeline Operations ---
  const handleSplit = () => {
    // Find which cut includes currentTime
    const targetCut = cuts.find(c => currentTime > c.start + 0.1 && currentTime < c.end - 0.1);
    if (!targetCut) return;

    const newCutId = Math.random().toString(36).substr(2, 9);
    const firstHalf: Cut = { ...targetCut, end: currentTime };
    const secondHalf: Cut = {
      id: newCutId,
      start: currentTime,
      end: targetCut.end,
      label: targetCut.label,
      trackId: targetCut.trackId || 0,
      assetId: targetCut.assetId
    };

    const newCuts = cuts.map(c => c.id === targetCut.id ? firstHalf : c);
    const index = newCuts.findIndex(c => c.id === targetCut.id);
    newCuts.splice(index + 1, 0, secondHalf);

    setCuts([...newCuts]);
    setSelectedCutId(newCutId);
  };

  const handleDelete = () => {
    if (!selectedCutId) return;
    setCuts(cuts.filter(c => c.id !== selectedCutId));
    setSelectedCutId(null);
  };

  // --- DRAG HANDLERS ---
  const handleTimelineMouseDown = (e: React.MouseEvent) => {
    // If clicking on ruler or empty space, start scrubbing
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left + timelineContainerRef.current!.scrollLeft;
    const time = Math.max(0, clickX / zoomLevel);

    // Update immediatley
    setCurrentTime(time);
    if (videoRef.current) videoRef.current.currentTime = time;

    setDragState({
      isDragging: true,
      type: 'scrub',
      startX: e.clientX,
      targetId: null,
      initialValue: 0
    });
  };

  const handleClipMouseDown = (e: React.MouseEvent, cut: Cut, type: 'move' | 'trim-start' | 'trim-end') => {
    e.stopPropagation();
    setSelectedCutId(cut.id);

    setDragState({
      isDragging: true,
      type,
      targetId: cut.id,
      startX: e.clientX,
      initialValue: type === 'move' ? cut.start : (type === 'trim-start' ? cut.start : cut.end)
    });
  };

  // Global Mouse Move / Up Listeners
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragState.isDragging) return;

      const deltaX = e.clientX - dragState.startX;
      const deltaTime = deltaX / zoomLevel;

      if (dragState.type === 'scrub') {
        if (videoRef.current) {
          const newTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + deltaTime));
          // For scrubbing, we simply read the timeline position relative to container is better?
          // Let's use simplified approach: re-calculate from mouse position relative to container
          if (timelineRef.current && timelineContainerRef.current) {
            const rect = timelineRef.current.getBoundingClientRect();
            // We need to account for scroll
            const offsetX = e.clientX - rect.left + timelineContainerRef.current.scrollLeft;
            const absTime = Math.max(0, offsetX / zoomLevel);
            setCurrentTime(absTime);
            videoRef.current.currentTime = absTime;
          }
        }
      }
      else if (dragState.type === 'move' && dragState.targetId) {
        const newStart = Math.max(0, dragState.initialValue + deltaTime);
        const duration = cuts.find(c => c.id === dragState.targetId)!.end - cuts.find(c => c.id === dragState.targetId)!.start;
        setCuts(prev => prev.map(c => c.id === dragState.targetId ? { ...c, start: newStart, end: newStart + duration } : c));
      }
      else if (dragState.type === 'trim-start' && dragState.targetId) {
        const newStart = Math.min(dragState.initialValue + deltaTime, cuts.find(c => c.id === dragState.targetId)!.end - 0.1);
        setCuts(prev => prev.map(c => c.id === dragState.targetId ? { ...c, start: Math.max(0, newStart) } : c));
      }
      else if (dragState.type === 'trim-end' && dragState.targetId) {
        const newEnd = Math.max(dragState.initialValue + deltaTime, cuts.find(c => c.id === dragState.targetId)!.start + 0.1);
        setCuts(prev => prev.map(c => c.id === dragState.targetId ? { ...c, end: newEnd } : c));
      }
    };

    const handleMouseUp = () => {
      if (dragState.isDragging) {
        setDragState({ ...dragState, isDragging: false, type: null });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragState, cuts, zoomLevel, duration]);


  // --- AI Functions ---
  const handleGeminiHighlights = async () => {
    if (!videoFile || !apiKey) return;
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('instruction', instruction || "Find the best highlights");
    formData.append('target_count', highlightCount.toString());
    formData.append('target_duration', targetDuration.toString());
    formData.append('api_key', apiKey);
    formData.append('model_name', 'gemini-2.0-flash-exp');

    try {
      const res = await fetch('http://localhost:8000/analyze-video', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        const aiCuts = data.map((c: any, i: number) => ({
          id: `ai-${i}`,
          start: c.start,
          end: c.end,
          label: c.label || `Highlight ${i + 1}`,
          trackId: 0
        }));

        if (confirm(`AI Found ${aiCuts.length} clips. Replace timeline?`)) {
          setCuts(aiCuts);
        }
      } else {
        alert("AI Analysis Failed");
      }
    } catch (e) {
      console.error(e);
      alert("Error connecting to backend");
    }
    setIsProcessing(false);
  };

  const handleSilenceRemoval = async () => {
    if (!videoFile) return;
    if (!confirm("這將會移除影片中的靜音部分並取代目前的時間軸。確定嗎？")) return;

    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('threshold_db', silenceThreshold.toString());
    formData.append('min_duration', silenceMinDuration.toString());

    try {
      const res = await fetch('http://localhost:8000/detect-silence', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const segments = await res.json();
        const newCuts: Cut[] = segments.map((seg: any, i: number) => ({
          id: `silence-${i}`,
          start: seg.start,
          end: seg.end,
          label: 'Speech',
          trackId: 0
        }));

        if (newCuts.length === 0) {
          alert("未偵測到任何語音片段 (No speech found)");
        } else {
          setCuts(newCuts);
        }
      } else {
        console.error(await res.text());
        alert("偵測失敗");
      }
    } catch (e) {
      console.error(e);
      alert("連線後端失敗");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExportVideo = async () => {
    if (cuts.length === 0 || !videoFile) {
      if (!videoFile) alert("請先上傳影片檔案以進行匯出");
      return;
    }

    setIsProcessing(true);
    setShowExportModal(false);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('cuts_json', JSON.stringify(cuts));
    formData.append('output_resolution', exportResolution);
    formData.append('output_mode', 'video'); // Force video render
    formData.append('whisper_language', 'zh');
    formData.append('vertical_mode', isVerticalMode ? 'true' : 'false');

    // Linked AI options
    formData.append('burn_captions', exportAutoCaption ? 'true' : 'false');
    formData.append('auto_caption', exportAutoCaption ? 'true' : 'false');
    formData.append('face_tracking', exportFaceTracking ? 'true' : 'false');
    formData.append('studio_sound', exportStudioSound ? 'true' : 'false');
    formData.append('merge_clips', exportMergeClips ? 'true' : 'false');

    try {
      const response = await fetch('http://localhost:8000/process-video', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success' && data.download_url) {
          const downloadUrl = `http://localhost:8000${data.download_url}`;
          const a = document.createElement('a');
          a.href = downloadUrl;
          a.download = data.filename || `export_${Date.now()}.zip`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        } else {
          alert("匯出成功但未獲取下載連結: " + JSON.stringify(data));
        }
      } else {
        const errorText = await response.text();
        console.error("Export failed:", errorText);
        alert("匯出失敗，請檢查後端日誌");
      }
    } catch (e) {
      console.error("Export error:", e);
      alert("連線後端失敗");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExportXML = () => {
    if (cuts.length === 0) return;
    const fps = 30;
    const fileName = videoFile?.name || "video.mp4";

    let xml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
    <resources>
        <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
        <asset id="a1" name="${fileName}" src="file://${originalVideoPath || `localhost/path/to/${fileName}`}" duration="${Math.round(duration * fps)}/30s" hasVideo="1" hasAudio="1"/>
    </resources>
    <library>
        <event name="Antigravity Cut Event">
            <project name="Antigravity Project">
                <sequence format="r1" duration="${Math.round(cuts.reduce((a, b) => a + (b.end - b.start), 0) * fps)}/30s" tcStart="0s" tcFormat="NDF">
                    <spine>`;

    let offset = 0;
    cuts.forEach((cut, i) => {
      const dur = cut.end - cut.start;
      const durFrames = Math.round(dur * fps);
      const startFrames = Math.round(cut.start * fps);
      const offsetFrames = Math.round(offset * fps);

      xml += `
                        <video name="${cut.label || 'Clip ' + (i + 1)}" offset="${offsetFrames}/30s" ref="a1" duration="${durFrames}/30s" start="${startFrames}/30s"/>`;
      offset += dur;
    });

    xml += `
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>`;

    const blob = new Blob([xml], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `project_export.xml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="app-container" onDrop={handleDrop} onDragOver={handleDragOver}>
      {/* 1. Compact Header */}
      <header className="header">
        <div className="logo" style={{ fontSize: '14px', paddingLeft: 80 }}>
          <Scissors size={14} className="text-secondary" />
          <span style={{ color: '#ddd' }}>Antigravity Cut</span>
        </div>
        <div className="header-drag-region" />
        <div style={{ display: 'flex', gap: '8px', WebkitAppRegion: 'no-drag', alignItems: 'center' } as any}>
          <button className="btn-ghost-sm" onClick={() => setAppView('welcome')} title="主選單">
            <RotateCcw size={14} />
          </button>
          <div style={{ width: 1, height: 16, background: '#333', margin: '0 4px' }} />
          <button className="btn-ghost-sm" onClick={handleExportProject} title="儲存專案 (Save Project)">
            <Save size={14} /> 儲存專案
          </button>
          <button className="btn-ghost-sm" onClick={handleExportXML} style={{ height: 28, fontSize: 11 }}>
            XML
          </button>
          <button className="btn-primary-sm" onClick={() => setShowExportModal(true)} style={{ height: 28, fontSize: 11, padding: '0 12px', background: 'linear-gradient(135deg, #3ea6ff 0%, #007aff 100%)', border: 'none', fontWeight: 600 }}>
            <Download size={14} /> 匯出影片
          </button>
        </div>
      </header>

      {/* Welcome Screen */}
      {appView === 'welcome' ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#111', color: 'white' }}>
          <div style={{ marginBottom: 40, textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16, marginBottom: 24 }}>
              <Scissors size={64} className="text-secondary" />
              <h1 style={{ fontSize: 48, margin: 0, fontWeight: 800, background: 'linear-gradient(135deg, #fff 0%, #a1a1aa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Antigravity Cut</h1>
            </div>
            <p style={{ color: '#666', fontSize: 16 }}>專業級 AI 智能影音剪輯工具</p>
          </div>

          <div style={{ display: 'flex', gap: 24 }}>
            <div
              onClick={() => {
                const hasSaved = localStorage.getItem('antigravity_current_project');
                if (!hasSaved || confirm('這將會清空目前所有進度，確定嗎？')) {
                  localStorage.removeItem('antigravity_current_project');
                  setCuts([]);
                  setVideoUrl(null);
                  setOriginalVideoPath(null);
                  setProjectAssets([]);
                  setAppView('editor');
                }
              }}
              style={{ width: 200, height: 160, background: '#222', borderRadius: 12, border: '1px solid #333', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'all 0.2s' }}
              className="welcome-card"
            >
              <Plus size={40} color="#3ea6ff" style={{ marginBottom: 16 }} />
              <span style={{ fontWeight: 600, fontSize: 16 }}>建立新專案</span>
              <span style={{ fontSize: 12, color: '#666', marginTop: 8 }}>Start New Project</span>
            </div>

            <div
              onClick={() => {
                const saved = localStorage.getItem('antigravity_current_project');
                if (saved) {
                  loadProject(JSON.parse(saved));
                }
              }}
              style={{ width: 200, height: 160, background: '#222', borderRadius: 12, border: '1px solid #333', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'all 0.2s', opacity: localStorage.getItem('antigravity_current_project') ? 1 : 0.5, pointerEvents: localStorage.getItem('antigravity_current_project') ? 'auto' : 'none' }}
              className="welcome-card"
            >
              <RotateCcw size={40} color="#10b981" style={{ marginBottom: 16 }} />
              <span style={{ fontWeight: 600, fontSize: 16 }}>恢復上次專案</span>
              <span style={{ fontSize: 12, color: '#666', marginTop: 8 }}>Resume Project</span>
            </div>

            <label
              style={{ width: 200, height: 160, background: '#222', borderRadius: 12, border: '1px solid #333', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'all 0.2s' }}
              className="welcome-card"
            >
              <Save size={40} color="#eb64ff" style={{ marginBottom: 16 }} />
              <span style={{ fontWeight: 600, fontSize: 16 }}>從檔案匯入 project</span>
              <span style={{ fontSize: 12, color: '#666', marginTop: 8 }}>Import .agpro File</span>
              <input type="file" hidden accept=".agpro,.json" onChange={handleImportProject} />
            </label>
          </div>

          <div style={{ marginTop: 64, color: '#444', fontSize: 12 }}>
            v1.0.0 Alpha
          </div>
        </div>
      ) : (
        <div className="premiere-layout">
          {/* 2. Main Premiere Layout */}

          {/* Top Section: Panels */}
          <div className="top-panels">

            {/* Left: Project / Inspector Panel */}
            <div className="panel-container" style={{ width: 340, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
              {/* Panel Tabs Header */}
              <div className="panel-header" style={{ gap: 2, padding: '0 4px' }}>
                <div className={`panel-tab ${leftPanelTab === 'project' ? 'active' : ''}`} onClick={() => setLeftPanelTab('project')} style={{ flex: 1, justifyContent: 'center' }}>專案</div>
                <div className={`panel-tab ${leftPanelTab === 'controls' ? 'active' : ''}`} onClick={() => setLeftPanelTab('controls')} style={{ flex: 1, justifyContent: 'center' }}>控制</div>
                <div className={`panel-tab ${leftPanelTab === 'ai' ? 'active' : ''}`} onClick={() => setLeftPanelTab('ai')} style={{ flex: 1, justifyContent: 'center' }}>AI</div>
                <div className={`panel-tab ${leftPanelTab === 'effects' ? 'active' : ''}`} onClick={() => setLeftPanelTab('effects')} style={{ flex: 1, justifyContent: 'center' }}>特效</div>
              </div>

              <div className="panel-content" style={{ padding: 0, overflowY: 'auto' }}>

                {/* 1. PROJECT TAB */}
                {leftPanelTab === 'project' && (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    {/* Toolbar */}
                    <div style={{ padding: 8, borderBottom: '1px solid #333', display: 'flex', gap: 8 }}>
                      <label className="btn-primary-sm" style={{ flex: 1, cursor: 'pointer' }}>
                        <Plus size={12} /> 匯入素材
                        <input type="file" multiple hidden accept="video/*,image/*,audio/*" onChange={(e) => {
                          if (e.target.files) {
                            const newAssets = Array.from(e.target.files).map(f => ({
                              id: Math.random().toString(36).substr(2, 9),
                              type: f.type.startsWith('video') ? 'video' : f.type.startsWith('audio') ? 'audio' : 'image',
                              name: f.name,
                              url: URL.createObjectURL(f),
                              file: f
                            } as Asset));
                            setProjectAssets(prev => [...prev, ...newAssets]);
                            // Auto-load first video if timeline is empty
                            if (cuts.length === 0 && newAssets[0].type === 'video') {
                              setVideoUrl(newAssets[0].url);
                              setVideoFile(newAssets[0].file || null);
                              setOriginalVideoPath(null); // Local file
                              // Need to wait for metadata... handled by onLoadedMetadata
                            }
                          }
                        }} />
                      </label>
                      <button className="btn-icon-sm" onClick={() => { if (confirm('清空素材庫？')) setProjectAssets([]) }} title="清空素材庫"><Trash size={14} /></button>
                    </div>

                    {/* Assets Grid */}
                    <div style={{ flex: 1, overflowY: 'auto', padding: 8, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, alignContent: 'start' }}>
                      {projectAssets.length > 0 ? projectAssets.map(asset => (
                        <div key={asset.id}
                          onClick={() => {
                            if (asset.type === 'video') {
                              if (confirm(`要將 "${asset.name}" 載入到預覽視窗嗎?`)) {
                                setVideoUrl(asset.url);
                                setVideoFile(asset.file || null);
                                // Clean restart
                                setCuts([]);
                              }
                            }
                          }}
                          className="asset-item"
                          title={asset.name}
                          style={{ aspectRatio: '1/1', background: '#222', borderRadius: 4, overflow: 'hidden', position: 'relative', border: '1px solid #444', cursor: 'pointer' }}
                        >
                          {asset.type === 'video' || asset.type === 'image' ? (
                            <video src={asset.url} style={{ width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none' }} />
                          ) : (
                            <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              <Loader2 size={24} color="#666" />
                            </div>
                          )}
                          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: 'rgba(0,0,0,0.8)', fontSize: 10, padding: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {asset.name}
                          </div>
                        </div>
                      )) : (
                        <div style={{ gridColumn: '1/-1', textAlign: 'center', marginTop: 32, color: '#666', fontSize: 11 }}>
                          無素材<br />點擊上方按鈕匯入
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 2. CONTROLS TAB */}
                {leftPanelTab === 'controls' && (
                  <div style={{ padding: 12 }}>
                    {selectedCutId ? (
                      <div style={{ color: '#ddd', fontSize: 11 }}>
                        <div style={{ marginBottom: 16, fontWeight: 'bold', color: 'var(--primary-color)', borderBottom: '1px solid #333', paddingBottom: 8 }}>
                          {cuts.find(c => c.id === selectedCutId)?.label}
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
                          <div>
                            <label style={{ display: 'block', marginBottom: 4, color: '#888' }}>開始時間 (Start)</label>
                            <input className="input-dark" readOnly value={cuts.find(c => c.id === selectedCutId)?.start.toFixed(2)} style={{ width: '100%', fontFamily: 'monospace' }} />
                          </div>
                          <div>
                            <label style={{ display: 'block', marginBottom: 4, color: '#888' }}>結束時間 (End)</label>
                            <input className="input-dark" readOnly value={cuts.find(c => c.id === selectedCutId)?.end.toFixed(2)} style={{ width: '100%', fontFamily: 'monospace' }} />
                          </div>
                        </div>

                        {cuts.find(c => c.id === selectedCutId)?.trackId === 1 && (
                          <div style={{ marginBottom: 16, borderTop: '1px solid #333', paddingTop: 16 }}>
                            <div style={{ fontWeight: 'bold', marginBottom: 8 }}>文字設定</div>
                            {/* Placeholder for future text styling */}
                            <input className="input-dark" value={cuts.find(c => c.id === selectedCutId)?.label}
                              onChange={(e) => {
                                setCuts(prev => prev.map(c => c.id === selectedCutId ? { ...c, label: e.target.value } : c));
                              }}
                              style={{ width: '100%' }}
                            />
                          </div>
                        )}

                        <button className="btn-ghost-sm" style={{ width: '100%', color: '#ef4444', border: '1px solid #ef4444' }} onClick={handleDelete}>
                          <Trash size={12} style={{ marginRight: 4 }} /> 刪除片段
                        </button>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', marginTop: 40, color: '#666', fontSize: 11 }}>
                        未選取任何片段<br />(請點擊時間軸上的片段)
                      </div>
                    )}
                  </div>
                )}

                {/* 3. AI TOOLS TAB (Restored & Moved) */}
                {leftPanelTab === 'ai' && (
                  <div style={{ padding: 12 }}>

                    {/* Silence Removal */}
                    <div className="style-group" style={{ marginTop: 0 }}>
                      <div style={{ marginBottom: 8, borderBottom: '1px solid #333', paddingBottom: 8 }}>
                        <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#10b981', display: 'flex', alignItems: 'center' }}>
                          <Zap size={14} style={{ marginRight: 6 }} /> 智能去氣口
                        </div>
                        <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>自動偵測並刪除影片中的靜音片段。</div>

                        <div className="style-grid-2" style={{ marginBottom: 8 }}>
                          <div>
                            <label className="label-sm">噪音閥值 (dB)</label>
                            <input type="number" className="input-xs" value={silenceThreshold} onChange={e => setSilenceThreshold(Number(e.target.value))} placeholder="-30" />
                          </div>
                          <div>
                            <label className="label-sm">最短保留 (秒)</label>
                            <input type="number" className="input-xs" value={silenceMinDuration} onChange={e => setSilenceMinDuration(Number(e.target.value))} placeholder="0.5" />
                          </div>
                        </div>
                        <button className="btn-primary-sm" style={{ width: '100%' }} onClick={handleSilenceRemoval}>
                          執行去氣口
                        </button>
                      </div>

                      {/* AI Highlights */}
                      <div style={{ marginTop: 16 }}>
                        <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#818cf8', display: 'flex', alignItems: 'center' }}>
                          <Film size={14} style={{ marginRight: 6 }} /> AI 精華生成
                        </div>
                        <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>使用 Gemini AI 分析並剪輯精華片段。</div>

                        <div className="style-grid-2" style={{ marginBottom: 8 }}>
                          <div>
                            <label className="label-sm">片段數量</label>
                            <input type="number" className="input-xs" value={highlightCount} onChange={e => setHighlightCount(Number(e.target.value))} />
                          </div>
                          <div>
                            <label className="label-sm">單片時長 (秒)</label>
                            <input type="number" className="input-xs" value={targetDuration} onChange={e => setTargetDuration(Number(e.target.value))} />
                          </div>
                        </div>
                        <div style={{ marginBottom: 8 }}>
                          <label className="label-sm">AI 提示詞</label>
                          <textarea className="input-dark" rows={2} style={{ width: '100%', fontSize: 11 }}
                            value={instruction}
                            onChange={e => setInstruction(e.target.value)}
                            placeholder="例如：找出最有趣的對話..."
                          />
                        </div>
                        <div style={{ marginBottom: 8 }}>
                          <label className="label-sm">API Key</label>
                          <input type="password" className="input-dark" style={{ width: '100%' }}
                            value={apiKey}
                            onChange={e => setApiKey(e.target.value)}
                            placeholder="Gemini API Key"
                          />
                        </div>

                        <button className="btn-primary-sm" style={{ width: '100%' }} onClick={handleGeminiHighlights}>
                          生成精華短片
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* 4. EFFECTS TAB */}
                {leftPanelTab === 'effects' && (
                  <div style={{ padding: 12, color: '#888', fontSize: 11 }}>
                    <div style={{ marginBottom: 16, borderBottom: '1px solid #333', paddingBottom: 8 }}>
                      <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#fff' }}>標題特效 (Titles)</div>
                      <div className="effect-item" style={{ background: '#222', padding: 12, marginBottom: 8, cursor: 'pointer', borderRadius: 8, border: '1px solid #333', display: 'flex', alignItems: 'center', gap: 10 }} onClick={() => {
                        const newCut: Cut = { id: Math.random().toString(36).substr(2, 9), start: currentTime, end: currentTime + 3, label: '新的文字標題', trackId: 1 };
                        setCuts(prev => [...prev, newCut]);
                      }}>
                        <div style={{ width: 24, height: 24, background: 'var(--primary-color)', borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: 14, fontWeight: 'bold' }}>T</div>
                        <div>
                          <div style={{ color: '#eee', fontWeight: 600 }}>基礎文字標題</div>
                          <div style={{ fontSize: 10, color: '#666' }}>點擊在當前時間增加文字圖層</div>
                        </div>
                      </div>
                    </div>

                    <div style={{ opacity: 0.5 }}>
                      <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#fff' }}>更多轉場即將推出...</div>
                      <div style={{ background: '#1a1a1a', padding: 20, borderRadius: 8, textAlign: 'center', border: '1px dashed #333' }}>
                        AI 智能轉場開發中
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Center: Program Monitor */}
            <div className="panel-container" style={{ flex: 1 }}>
              <div className="panel-header" style={{ justifyContent: 'space-between' }}>
                <div className="panel-tab active">節目檢視: 序列 01</div>
                {/* Layout Toggle */}
                <div style={{ display: 'flex', gap: 4, marginRight: 8 }}>
                  <button className={`btn-icon-sm ${!isVerticalMode ? 'active' : ''}`} onClick={() => setIsVerticalMode(false)} style={{ padding: 4, opacity: !isVerticalMode ? 1 : 0.5 }}>
                    <Monitor size={14} />
                  </button>
                  <button className={`btn-icon-sm ${isVerticalMode ? 'active' : ''}`} onClick={() => setIsVerticalMode(true)} style={{ padding: 4, opacity: isVerticalMode ? 1 : 0.5 }}>
                    <Smartphone size={14} />
                  </button>
                </div>
              </div>
              <div
                className={`preview-area ${isVerticalMode ? 'vertical-mode' : 'horizontal-mode'}`}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', overflow: 'hidden', position: 'relative' }}
              >
                {(videoUrl || cuts.length > 0) && !(cuts.length > 0 && !videoUrl && cuts.some(c => c.trackId === 0)) ? (
                  <div className="video-container" style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
                    {videoUrl ? (
                      <video
                        ref={videoRef}
                        src={videoUrl}
                        crossOrigin="anonymous"
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleMetadata}
                        onClick={togglePlay}
                        onError={(e) => {
                          console.error("Video element error:", e);
                          if (videoUrl && videoUrl.startsWith('http') && videoFile) {
                            console.log("♻️ Server video failed, falling back to local Blob URL...");
                            setVideoUrl(URL.createObjectURL(videoFile));
                          } else if (videoUrl) {
                            alert("影片載入失敗！這可能是由於格式不支援 (例如 HEVC) 或檔案毀損。請嘗試將其轉換為標準 H.264 MP4 後再試。");
                          }
                        }}
                      />
                    ) : (
                      // Text Only / Blank Mode
                      <div style={{ width: '100%', height: '100%', background: '#000', display: 'flex', alignItems: 'center', justifyItems: 'center' }}>
                        {/* Placeholder for size if needed, or just relying on container */}
                      </div>
                    )}

                    {/* Text Overlays Layer */}
                    <div className="text-overlay-renderer" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'none', overflow: 'hidden' }}>
                      {cuts.filter(c => {
                        const track = videoTracks.find(t => t.id === c.trackId);
                        return track && track.type === 'text' && currentTime >= c.start && currentTime < c.end;
                      }).map(cut => (
                        <div key={cut.id} className="text-element" style={{
                          position: 'absolute',
                          top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                          fontSize: 48, color: 'white', fontWeight: 'bold',
                          textShadow: '0 2px 4px rgba(0,0,0,0.8)',
                        }}>
                          {cut.label}
                        </div>
                      ))}
                    </div>

                    {/* Overlay Controls */}
                    {!isPlaying && !isProcessing && (
                      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }}>
                        <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'rgba(0,0,0,0.3)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(255,255,255,0.1)' }}>
                          <Play size={28} fill="white" stroke="none" />
                        </div>
                      </div>
                    )}
                    {isProcessing && (
                      <div style={{ position: 'absolute', bottom: 20, right: 20, background: 'rgba(0,0,0,0.7)', padding: '8px 12px', borderRadius: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Loader2 className="spin" size={14} color="#3ea6ff" />
                        <span style={{ fontSize: 11, color: '#fff' }}>處理中...</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#666', padding: 20 }}>
                    {isUploading ? (
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <Loader2 className="spin" size={48} color="var(--primary-color)" />
                        <div style={{ marginTop: 16, fontSize: 13 }}>檔案處理中...</div>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center' }}>
                        {cuts.length > 0 && cuts.some(c => c.trackId === 0) ? (
                          <div style={{ background: 'rgba(255,50,50,0.1)', padding: 24, borderRadius: 12, border: '1px solid rgba(255,0,0,0.2)' }}>
                            <div style={{ color: '#ff4444', fontWeight: 'bold', marginBottom: 12, fontSize: 20 }}>⚠️ 媒體連結中斷</div>
                            <p style={{ color: '#aaa', fontSize: 13, marginBottom: 20 }}>目前專案中有剪輯進度，但預覽影片未載入。<br />請重新選取影片檔案以繼續編輯。</p>
                            <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
                              <label className="btn-primary-sm" style={{ cursor: 'pointer', background: '#d32f2f', border: 'none' }}>
                                🚀 重新連結影片
                                <input type="file" hidden onChange={handleFileUpload} accept="video/*" />
                              </label>
                              <button className="btn-ghost-sm" onClick={() => { if (confirm('要清空目前進度嗎？')) { setCuts([]); localStorage.removeItem('antigravity_cuts'); window.location.reload(); } }}>
                                捨棄進度
                              </button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div style={{ marginBottom: 20, opacity: 0.5 }}>
                              <Upload size={64} strokeWidth={1} />
                            </div>
                            <label className="btn-primary-sm" style={{ padding: '10px 24px', fontSize: 14 }}>
                              點擊匯入或拖曳影片至此
                              <input type="file" hidden onChange={handleFileUpload} accept="video/*" />
                            </label>
                            <p style={{ marginTop: 12, fontSize: 11, color: '#555' }}>支援 MP4, MOV, WEBM 等常見格式</p>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

          </div>

          {/* Bottom Section: Timeline */}
          <div className="timeline-section" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="timeline-toolbar">
              <div className="timeline-tools-container">
                {/* Edit Tools Group */}
                <div className="tool-group">
                  <div
                    className={`tool-btn ${activeTool === 'select' ? 'active' : ''}`}
                    title="選取工具 (V)"
                    onClick={() => setActiveTool('select')}
                  >
                    <MousePointer2 size={18} />
                  </div>
                  <div
                    className={`tool-btn ${activeTool === 'blade' ? 'active' : ''}`}
                    title="切割工具 (K)"
                    onClick={() => setActiveTool('blade')}
                  >
                    <Scissors size={18} />
                  </div>
                  {/* Hand Tool (Visual only for now) */}
                  <div
                    className={`tool-btn ${dragState.type === 'move' ? '' : ''}`} // Just visual toggle for now
                    title="手形工具 (H) - 暫未開放"
                    style={{ opacity: 0.5, cursor: 'not-allowed' }}
                  >
                    <Hand size={18} />
                  </div>
                </div>

                {/* Actions Group */}
                <div className="tool-group">
                  <div
                    className="tool-btn"
                    title="分割目前片段 (Cmd+K)"
                    onClick={handleSplit}
                  >
                    <SplitSquareHorizontal size={18} />
                  </div>
                  <div
                    className="tool-btn danger"
                    title="刪除選取片段 (Delete)"
                    onClick={handleDelete}
                  >
                    <Trash size={18} />
                  </div>
                </div>

                {/* Insert Group */}
                <div className="tool-group">
                  <div
                    className={`tool-btn ${activeTool === 'text' ? 'active' : ''}`}
                    title="新增文字 (T)"
                    onClick={() => {
                      setActiveTool('text');
                      const newCut: Cut = {
                        id: Math.random().toString(36).substr(2, 9),
                        start: currentTime,
                        end: currentTime + 3,
                        label: '文字圖層',
                        trackId: 1
                      };
                      setCuts(prev => [...prev, newCut]);
                      setActiveTool('select');
                    }}
                  >
                    <Type size={18} />
                  </div>
                  <div className="tool-btn" title="磁吸對齊 (S) - 開發中" style={{ opacity: 0.5 }}>
                    <Magnet size={18} />
                  </div>
                </div>
              </div>

              {/* Right Side: Time & Zoom */}
              <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16 }}>
                <div className="time-counter">
                  {new Date(currentTime * 1000).toISOString().substr(11, 8)}
                  <span style={{ fontSize: 10, opacity: 0.5, marginLeft: 4 }}>
                    {(Math.floor((currentTime % 1) * 30)).toString().padStart(2, '0')}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <button className="btn-icon-sm" onClick={() => setZoomLevel(z => Math.max(z / 1.2, 1))}><ZoomOut size={14} /></button>
                  <input type="range" min="1" max="100" value={zoomLevel} onChange={e => setZoomLevel(Number(e.target.value))} style={{ width: 80, accentColor: '#3ea6ff' }} />
                  <button className="btn-icon-sm" onClick={() => setZoomLevel(z => Math.min(z * 1.2, 100))}><ZoomIn size={14} /></button>
                </div>
              </div>
            </div>

            {/* Timeline Main */}
            <div className="timeline-main">
              {/* Timeline Header toolbar */}
              {/* Timeline Header toolbar (Removed Duplicate) */}

              <div className="timeline-headers-container">
                {/* Track Headers V1/A1 */}
                <div className="track-headers">
                  {/* V1 Header */}
                  <div className={`track-header-item ${videoTracks[0].locked ? 'locked' : ''}`} style={{ opacity: videoTracks[0].visible ? 1 : 0.5 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ color: '#999', fontWeight: 'bold' }}>V1</span>
                      <div style={{ display: 'flex', gap: 4 }}>
                        <div
                          onClick={() => toggleTrackVisibility(0)}
                          style={{ border: '1px solid #555', padding: '0 4px', borderRadius: 2, fontSize: 9, color: videoTracks[0].visible ? '#3ea6ff' : '#666', cursor: 'pointer' }}
                        >
                          {videoTracks[0].visible ? '👁️' : '🚫'}
                        </div>
                        <div
                          onClick={() => toggleTrackLock(0)}
                          style={{ border: '1px solid #555', padding: '0 4px', borderRadius: 2, fontSize: 9, color: videoTracks[0].locked ? '#ef4444' : '#666', cursor: 'pointer' }}
                        >
                          {videoTracks[0].locked ? '🔒' : '🔓'}
                        </div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 2 }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#3ea6ff', opacity: 0.5 }}></div>
                    </div>
                  </div>
                  {/* A1 Header (Subtitles/Text Track) */}
                  <div className={`track-header-item ${videoTracks[1].locked ? 'locked' : ''}`} style={{ opacity: videoTracks[1].visible ? 1 : 0.5 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ color: '#999', fontWeight: 'bold' }}>T1</span>
                      <div style={{ display: 'flex', gap: 4 }}>
                        <div
                          onClick={() => toggleTrackVisibility(1)}
                          style={{ border: '1px solid #555', padding: '0 4px', borderRadius: 2, fontSize: 9, color: videoTracks[1].visible ? '#10b981' : '#666', cursor: 'pointer' }}
                        >
                          {videoTracks[1].visible ? '👁️' : '🚫'}
                        </div>
                      </div>
                    </div>
                    {/* Fake Audio Meter visualization */}
                    <div style={{ height: 4, width: '100%', background: '#333', marginTop: 8, borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: isPlaying ? '70%' : '10%', background: 'linear-gradient(90deg, #10b981, #f59e0b, #ef4444)', transition: 'width 0.1s' }}></div>
                    </div>
                  </div>
                </div>

                {/* Tracks Area */}
                <div className="timeline-tracks-area"
                  ref={timelineContainerRef}
                >
                  {/* Ruler */}
                  <div className="timeline-ruler-container"
                    style={{ width: Math.max(100, duration * zoomLevel) + 'px' }}
                    onMouseDown={handleTimelineMouseDown}
                  >
                    {/* Scrubbing Playhead Head (The Triangle) inside ruler */}
                    <div style={{
                      position: 'absolute',
                      left: currentTime * zoomLevel - 6,
                      top: 12,
                      width: 0, height: 0, borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderTop: '10px solid #3ea6ff',
                      pointerEvents: 'none'
                    }}></div>

                    {/* Ruler Ticks */}
                    {Array.from({ length: Math.ceil(duration) }).map((_, i) => (
                      <React.Fragment key={i}>
                        <div className="time-mark" style={{ left: i * zoomLevel }}></div>
                        {i % 5 === 0 && <div className="time-text" style={{ left: i * zoomLevel + 4 }}>{i}s</div>}
                      </React.Fragment>
                    ))}
                  </div>

                  {/* Tracks Content */}
                  <div className="tracks-content"
                    style={{ width: Math.max(100, duration * zoomLevel) + 'px', cursor: dragState.type === 'scrub' ? 'ew-resize' : 'default' }}
                    onMouseDown={handleTimelineMouseDown}
                    ref={timelineRef}
                  >
                    {/* Playhead Line */}
                    <div className="playhead-marker" style={{ left: currentTime * zoomLevel }}></div>

                    {/* V1 Track */}
                    <div className="track-lane">
                      {cuts.map(cut => (
                        <div
                          key={cut.id}
                          className={`clip-block ${selectedCutId === cut.id ? 'selected' : ''}`}
                          style={{
                            left: cut.start * zoomLevel,
                            width: Math.max(10, (cut.end - cut.start) * zoomLevel) + 'px'
                          }}
                          onMouseDown={(e) => handleClipMouseDown(e, cut, 'move')}
                        >
                          {/* Drag Handles */}
                          <div
                            className="clip-handle-area"
                            style={{ position: 'absolute', left: 0, width: 6, height: '100%', cursor: 'ew-resize', zIndex: 20, background: 'rgba(255,255,255,0.1)' }}
                            onMouseDown={(e) => handleClipMouseDown(e, cut, 'trim-start')}
                          />
                          <div
                            className="clip-handle-area"
                            style={{ position: 'absolute', right: 0, width: 6, height: '100%', cursor: 'ew-resize', zIndex: 20, background: 'rgba(255,255,255,0.1)' }}
                            onMouseDown={(e) => handleClipMouseDown(e, cut, 'trim-end')}
                          />

                          <span className="clip-label">{cut.label}</span>
                        </div>
                      ))}
                    </div>

                    {/* A1 Track (Mirror V1 for now) */}
                    <div className="track-lane" style={{ background: '#1a1a1a' }}>
                      {cuts.map(cut => (
                        <div
                          key={'audio-' + cut.id}
                          className="clip-block"
                          style={{
                            left: cut.start * zoomLevel,
                            width: Math.max(10, (cut.end - cut.start) * zoomLevel) + 'px',
                            background: '#10b981', // Audio Green
                            border: '1px solid #059669',
                            opacity: 0.8
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* Export Modal */}
      {showExportModal && (
        <div className="modal-overlay" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div className="export-modal-content" style={{ width: 420, background: '#121212', borderRadius: 20, border: '1px solid #333', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)' }}>
            <div style={{ padding: '20px 24px', borderBottom: '1px solid #222', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#1a1a1a' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ padding: 8, background: 'rgba(62,166,255,0.1)', borderRadius: 10 }}>
                  <Download size={20} color="#3ea6ff" />
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#fff' }}>匯出影片</h3>
                  <div style={{ fontSize: 11, color: '#666' }}>設定您的匯出偏好</div>
                </div>
              </div>
              <button
                onClick={() => setShowExportModal(false)}
                className="btn-icon-sm"
                style={{ background: '#222', borderRadius: '50%', border: '1px solid #333' }}
              >
                <X size={16} />
              </button>
            </div>

            <div style={{ padding: 24 }}>
              {/* Resolution Toggle */}
              <div style={{ marginBottom: 24 }}>
                <label style={{ display: 'block', marginBottom: 12, fontSize: 13, fontWeight: 600, color: '#aaa' }}>輸出解析度</label>
                <div style={{ display: 'flex', gap: 10 }}>
                  {[{ id: '1080p', label: '1080p', desc: 'Full HD' }, { id: '720p', label: '720p', desc: 'HD Ready' }].map(res => (
                    <div
                      key={res.id}
                      onClick={() => setExportResolution(res.id)}
                      style={{
                        flex: 1, padding: '12px 16px', borderRadius: 12, cursor: 'pointer', transition: 'all 0.2s',
                        border: `1px solid ${exportResolution === res.id ? '#3ea6ff' : '#222'}`,
                        background: exportResolution === res.id ? 'rgba(62,166,255,0.08)' : '#1a1a1a',
                        textAlign: 'center'
                      }}
                    >
                      <div style={{ fontSize: 14, fontWeight: 700, color: exportResolution === res.id ? '#3ea6ff' : '#eee' }}>{res.label}</div>
                      <div style={{ fontSize: 10, color: exportResolution === res.id ? 'rgba(62,166,255,0.6)' : '#555', marginTop: 2 }}>{res.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Advanced Options Grid */}
              <div style={{ marginBottom: 24 }}>
                <label style={{ display: 'block', marginBottom: 12, fontSize: 13, fontWeight: 600, color: '#aaa' }}>AI 加強與處理</label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div
                    onClick={() => setExportAutoCaption(!exportAutoCaption)}
                    style={{ padding: '12px 16px', borderRadius: 12, background: '#1a1a1a', border: `1px solid ${exportAutoCaption ? 'rgba(16,185,129,0.3)' : '#222'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <Zap size={16} color={exportAutoCaption ? '#10b981' : '#555'} />
                      <div style={{ fontSize: 13, color: exportAutoCaption ? '#fff' : '#888' }}>自動生成 AI 字幕</div>
                    </div>
                    <div style={{ width: 14, height: 14, borderRadius: 4, background: exportAutoCaption ? '#10b981' : '#333', border: '1px solid #444' }} />
                  </div>

                  <div
                    onClick={() => setExportFaceTracking(!exportFaceTracking)}
                    style={{ padding: '12px 16px', borderRadius: 12, background: '#1a1a1a', border: `1px solid ${exportFaceTracking ? 'rgba(62,166,255,0.3)' : '#222'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <Monitor size={16} color={exportFaceTracking ? '#3ea6ff' : '#555'} />
                      <div style={{ fontSize: 13, color: exportFaceTracking ? '#fff' : '#888' }}>AI 人臉追蹤與自動取景</div>
                    </div>
                    <div style={{ width: 14, height: 14, borderRadius: 4, background: exportFaceTracking ? '#3ea6ff' : '#333', border: '1px solid #444' }} />
                  </div>

                  <div
                    onClick={() => setExportStudioSound(!exportStudioSound)}
                    style={{ padding: '12px 16px', borderRadius: 12, background: '#1a1a1a', border: `1px solid ${exportStudioSound ? 'rgba(235,100,255,0.3)' : '#222'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <Smartphone size={16} color={exportStudioSound ? '#eb64ff' : '#555'} />
                      <div style={{ fontSize: 13, color: exportStudioSound ? '#fff' : '#888' }}>AI 降噪與錄音室音質</div>
                    </div>
                    <div style={{ width: 14, height: 14, borderRadius: 4, background: exportStudioSound ? '#eb64ff' : '#333', border: '1px solid #444' }} />
                  </div>

                  <div
                    onClick={() => setExportMergeClips(!exportMergeClips)}
                    style={{ padding: '12px 16px', borderRadius: 12, background: '#1a1a1a', border: `1px solid ${exportMergeClips ? 'rgba(255,165,0,0.3)' : '#222'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <Film size={16} color={exportMergeClips ? '#ffa500' : '#555'} />
                      <div style={{ fontSize: 13, color: exportMergeClips ? '#fff' : '#888' }}>合併為單一影片</div>
                    </div>
                    <div style={{ width: 14, height: 14, borderRadius: 4, background: exportMergeClips ? '#ffa500' : '#333', border: '1px solid #444' }} />
                  </div>
                </div>
              </div>

              <button
                className="btn-primary"
                onClick={handleExportVideo}
                disabled={isProcessing}
                style={{
                  width: '100%', height: 48, fontSize: 15, fontWeight: 700, borderRadius: 14,
                  background: 'linear-gradient(135deg, #3ea6ff 0%, #007aff 100%)',
                  boxShadow: '0 8px 20px -5px rgba(0,122,255,0.4)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10
                }}
              >
                {isProcessing ? <><Loader2 className="spin" size={18} /> 處理中...</> : <><Download size={18} /> 開始匯出成品</>}
              </button>

              <div style={{ marginTop: 16, textAlign: 'center', fontSize: 11, color: '#444' }}>
                處理時間取決於影片長度與選取的 AI 功能
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
