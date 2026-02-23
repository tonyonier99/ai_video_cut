import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { DragEvent } from 'react';
import { Play, Scissors, MousePointer2, ZoomIn, ZoomOut, Upload, Plus, Trash, Save, Film, Loader2, Zap, X, Download, RotateCcw, Monitor, Smartphone, Hand, Magnet, SplitSquareHorizontal, Type, Video, Music, ChevronDown } from 'lucide-react';
import { API_BASE_URL } from './config';
import './App.css';

import type { Cut, Asset, DragState, ActiveTool, LeftPanelTab, AppView } from './types';
import { getTrackTheme, getTimelineCursor } from './utils/timelineUtils';
import { generateFCPXML, downloadFile } from './utils/exportUtils';
import { usePlayback } from './hooks/usePlayback';
import { useHistory } from './hooks/useHistory';
import { useTimeline } from './hooks/useTimeline';
import { useTrackManagement } from './hooks/useTrackManagement';
import { useAITools } from './hooks/useAITools';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { WelcomeScreen } from './components/WelcomeScreen';
import { ExportModal } from './components/ExportModal';

function App() {
  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const relinkInputRef = useRef<HTMLInputElement>(null);

  // --- Global State ---
  const [projectAssets, setProjectAssets] = useState<Asset[]>([]);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [originalVideoPath, setOriginalVideoPath] = useState<string | null>(null);

  // --- Layout State ---
  const [isVerticalMode, setIsVerticalMode] = useState(false);
  const [leftPanelTab, setLeftPanelTab] = useState<LeftPanelTab>('project');
  const [activeTool, setActiveTool] = useState<ActiveTool>('select');
  const [appView, setAppView] = useState<AppView>('welcome');
  const [marqueeRect, setMarqueeRect] = useState<{ startX: number; startY: number; currX: number; currY: number } | null>(null);

  // --- Timeline Hook ---
  const timeline = useTimeline({ duration: 0, currentTime: 0 });
  const { cuts, setCuts, selectedCutIds, setSelectedCutIds, zoomLevel, setZoomLevel,
    isMagnetEnabled, setIsMagnetEnabled, totalDuration, getSnapTime,
    handleSplit, handleDelete, handleAlignCuts, handleZoom } = timeline;

  // --- Track Management Hook ---
  const trackMgmt = useTrackManagement({ setCuts });
  const { videoTracks, sortedTracks, toggleTrackVisibility, toggleTrackLock,
    handleAddTrack, handleDeleteTrack } = trackMgmt;

  // --- Playback Hook ---
  const playback = usePlayback({ totalDuration, cuts, videoRef });
  const { currentTime, setCurrentTime, isPlaying, setIsPlaying, duration,
    setDuration, togglePlay, handleMetadata, seek } = playback;

  // --- History Hook ---
  const historyHook = useHistory({ cuts, setCuts });
  const { history, historyIndex, addToHistory, handleUndo, handleRedo } = historyHook;

  // --- AI Tools Hook ---
  const aiTools = useAITools({ videoFile, videoTracks, setCuts });
  const { apiKey, setApiKey, apiKeyLoaded, isProcessing, setIsProcessing,
    currentJobStatus, setCurrentJobStatus,
    highlightCount, setHighlightCount, targetDuration, setTargetDuration,
    instruction, setInstruction, geminiModel, setGeminiModel,
    silenceThreshold, setSilenceThreshold, silenceMinDuration, setSilenceMinDuration,
    whisperModel, setWhisperModel, whisperLanguage, setWhisperLanguage,
    whisperBeamSize, setWhisperBeamSize, whisperTemperature, setWhisperTemperature,
    whisperCharsPerLine, setWhisperCharsPerLine, whisperRemovePunc, setWhisperRemovePunc,
    availableFonts, subtitleConfig, setSubtitleConfig, fetchFonts,
    handleGeminiHighlights, handleSilenceRemoval, handleAISubtitles } = aiTools;

  // --- Export Modal ---
  const [showExportModal, setShowExportModal] = useState(false);

  // Improved Dragging State
  const [dragState, setDragState] = useState<DragState>({ isDragging: false, type: null, targetId: null, startX: 0, initialValue: 0 });

  // Synchronized Ref for heavy background tasks (History/Save)
  const cutsRef = useRef(cuts);
  useEffect(() => {
    cutsRef.current = cuts;
  }, [cuts]);

  // History & Auto-save management (Optimized Release-Only)
  useEffect(() => {
    // ONLY run when isDragging changes from TRUE to FALSE (The Release)
    if (dragState.isDragging) return;

    // We use the Ref to get the latest state without triggering this effect 
    // on every intermediate 'cuts' update during the drag.
    const finalCuts = cutsRef.current;
    if (finalCuts.length === 0) return;

    // 1. History Commit
    if (history.length > 0 && historyIndex >= 0) {
      const currentHistory = history[historyIndex];
      // Quick check to see if anything actually changed
      if (currentHistory.length !== finalCuts.length || JSON.stringify(currentHistory) !== JSON.stringify(finalCuts)) {
        addToHistory(finalCuts);
      }
    } else if (history.length === 0) {
      setHistory([finalCuts]);
      setHistoryIndex(0);
    }

    // 2. Auto-save to localStorage
    const projectData = {
      cuts: finalCuts,
      videoUrl,
      originalVideoPath,
      isVerticalMode,
      projectAssets: projectAssets.map(a => ({ ...a, file: undefined }))
    };
    localStorage.setItem('antigravity_current_project', JSON.stringify(projectData));

  }, [dragState.isDragging]); // Crucial: Only depends on the Drag State change

  // Contextual cursor derived from imported utility
  const timelineCursor = getTimelineCursor(dragState.isDragging, dragState.type, activeTool);




  // --- Project Management ---
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

  // Extended handleMetadata - also creates initial cut
  const handleVideoMetadata = useCallback(() => {
    handleMetadata();
    if (videoRef.current && cuts.length === 0) {
      setCuts([{
        id: 'full',
        start: 0,
        end: videoRef.current.duration,
        sourceStart: 0,
        sourceEnd: videoRef.current.duration,
        label: 'Full Video',
        trackId: 0
      }]);
    }
  }, [handleMetadata, videoRef, cuts.length, setCuts]);

  // --- File Upload ---
  const processFile = async (file: File) => {
    if (!file) return;
    setVideoFile(file);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/upload-proxy`, {
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


  // --- DRAG HANDLERS ---
  const handleTimelineMouseDown = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const time = Math.max(0, clickX / zoomLevel);

    const isRuler = (e.target as HTMLElement).closest('.timeline-ruler-container');

    if (isRuler || activeTool === 'blade') {
      // Scrub or Split
      setCurrentTime(time);
      if (videoRef.current) {
        const activeCut = cuts.find(c => time >= c.start && time < c.end);
        if (activeCut) {
          videoRef.current.currentTime = (activeCut.sourceStart ?? activeCut.start) + (time - activeCut.start);
        } else {
          videoRef.current.currentTime = time;
        }
      }

      if (activeTool === 'blade') {
        handleSplit(time);
        return;
      }

      setDragState({
        isDragging: true,
        type: 'scrub',
        startX: e.clientX,
        targetId: null,
        initialValue: 0
      });
    } else if (activeTool === 'hand') {
      // Start Panning
      setDragState({
        isDragging: true,
        type: 'pan',
        startX: e.clientX,
        targetId: null,
        initialValue: 0,
        initialScrollLeft: timelineContainerRef.current?.scrollLeft || 0
      });
    } else if (activeTool === 'select') {
      // Start Marquee
      const containerRect = timelineRef.current.closest('.timeline-tracks-area')?.getBoundingClientRect();
      if (!containerRect) return;

      const startX = e.clientX;
      const startY = e.clientY;

      setMarqueeRect({ startX, startY, currX: startX, currY: startY });
      setDragState({
        isDragging: true,
        type: 'marquee',
        startX,
        targetId: null,
        initialValue: 0
      });

      if (!e.shiftKey && !e.metaKey) {
        setSelectedCutIds([]);
      }
    }
  };

  const handleClipMouseDown = (e: React.MouseEvent, cut: Cut, type: 'move' | 'trim-start' | 'trim-end') => {
    e.stopPropagation();

    if (activeTool === 'blade') {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const timeAtClick = cut.start + (clickX / zoomLevel);
      handleSplit(timeAtClick);
      return;
    }

    const isShift = e.shiftKey || e.metaKey;
    if (isShift) {
      if (selectedCutIds.includes(cut.id)) {
        setSelectedCutIds(prev => prev.filter(id => id !== cut.id));
      } else {
        setSelectedCutIds(prev => [...prev, cut.id]);
      }
    } else {
      if (!selectedCutIds.includes(cut.id)) {
        setSelectedCutIds([cut.id]);
      }
    }

    setDragState({
      isDragging: true,
      type,
      targetId: cut.id,
      startX: e.clientX,
      initialValue: type === 'move' ? cut.start : (type === 'trim-start' ? cut.start : cut.end),
      initialCuts: [...cuts]
    });
  };

  // Global Mouse Move / Up Listeners
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragState.isDragging && !dragState.type) return;

      const deltaX = e.clientX - dragState.startX;
      const deltaTime = deltaX / zoomLevel;

      if (dragState.type === 'scrub') {
        if (timelineRef.current && timelineContainerRef.current) {
          const rect = timelineRef.current.getBoundingClientRect();
          const offsetX = e.clientX - rect.left;
          let absTime = Math.max(0, offsetX / zoomLevel);

          // Snap Playhead
          const snapped = getSnapTime(absTime);
          if (snapped !== null) absTime = snapped;

          setCurrentTime(absTime);
          if (videoRef.current) {
            const activeCut = cuts.find(c => absTime >= c.start && absTime < c.end);
            if (activeCut) {
              videoRef.current.currentTime = (activeCut.sourceStart ?? activeCut.start) + (absTime - activeCut.start);
            } else {
              videoRef.current.currentTime = absTime;
            }
          }
        }
      }
      else if (dragState.type === 'pan') {
        if (timelineContainerRef.current) {
          const deltaX = e.clientX - dragState.startX;
          timelineContainerRef.current.scrollLeft = (dragState.initialScrollLeft || 0) - deltaX;
        }
      }
      else if (dragState.type === 'marquee') {
        setMarqueeRect(prev => prev ? { ...prev, currX: e.clientX, currY: e.clientY } : null);

        if (marqueeRect) {
          const x1 = Math.min(marqueeRect.startX, e.clientX);
          const x2 = Math.max(marqueeRect.startX, e.clientX);
          const y1 = Math.min(marqueeRect.startY, e.clientY);
          const y2 = Math.max(marqueeRect.startY, e.clientY);

          const newlySelected: string[] = [];
          const clipBlocks = document.querySelectorAll('.clip-block');
          clipBlocks.forEach(el => {
            const r = el.getBoundingClientRect();
            const hasOverlap = !(r.right < x1 || r.left > x2 || r.bottom < y1 || r.top > y2);
            if (hasOverlap) {
              const cid = el.getAttribute('data-cut-id');
              if (cid) newlySelected.push(cid);
            }
          });

          // If shift key is held, merge with previous selection (additive)
          // But usually marquee REPLACES unless Shift is held.
          // For now, let's keep it simple: Marquee replaces selection or adds if shift.
          setSelectedCutIds(newlySelected);
        }
      }
      else if (dragState.type === 'move' && dragState.targetId) {
        let deltaT = deltaTime;
        const baseCuts = dragState.initialCuts || cuts;
        const selectionSet = new Set(selectedCutIds);

        // Pre-calculation for Magnet (Collision Avoidance) Mode
        if (isMagnetEnabled) {
          const nonSelectedCuts = baseCuts.filter(c => !selectionSet.has(c.id));
          let maxBackshift = -Infinity;
          let maxForwardshift = Infinity;

          // Check boundaries for all selected clips across all tracks
          baseCuts.filter(c => selectionSet.has(c.id)).forEach(c => {
            const trackNeighbors = nonSelectedCuts.filter(nc => nc.trackId === c.trackId);

            // Left Bumper
            const lefties = trackNeighbors.filter(ln => ln.end <= c.start + 0.001);
            const nearestLeft = lefties.length > 0 ? Math.max(...lefties.map(l => l.end)) : 0;
            maxBackshift = Math.max(maxBackshift, nearestLeft - c.start);

            // Right Bumper
            const righties = trackNeighbors.filter(rn => rn.start >= c.end - 0.001);
            if (righties.length > 0) {
              const nearestRight = Math.min(...righties.map(r => r.start));
              maxForwardshift = Math.min(maxForwardshift, nearestRight - c.end);
            }
          });

          // Final clamped delta
          deltaT = Math.max(maxBackshift, Math.min(maxForwardshift, deltaT));
        }

        if (selectionSet.has(dragState.targetId)) {
          // Multi-clip movement
          const draggingAnchor = baseCuts.find(c => c.id === dragState.targetId);
          if (!draggingAnchor) return;

          let newAnchorStart = Math.max(0, draggingAnchor.start + deltaT);
          const snapped = getSnapTime(newAnchorStart, dragState.targetId);
          if (snapped !== null) newAnchorStart = snapped;

          const actualDelta = newAnchorStart - draggingAnchor.start;

          setCuts(baseCuts.map(c => {
            if (selectionSet.has(c.id)) {
              return { ...c, start: Math.max(0, c.start + actualDelta), end: Math.max(0.1, c.end + actualDelta) };
            }
            return c;
          }));
        } else {
          // Single clip move
          const targetCut = baseCuts.find(c => c.id === dragState.targetId);
          if (!targetCut) return;

          let newStart = Math.max(0, targetCut.start + deltaT);
          const snapped = getSnapTime(newStart, dragState.targetId);
          if (snapped !== null) newStart = snapped;

          const duration = targetCut.end - targetCut.start;
          setCuts(baseCuts.map(c => c.id === dragState.targetId ? { ...c, start: newStart, end: newStart + duration } : c));
        }
      }
      else if (dragState.type === 'trim-start' && dragState.targetId) {
        const baseCuts = dragState.initialCuts || cuts;
        const targetCut = baseCuts.find(c => c.id === dragState.targetId);
        if (!targetCut) return;

        let newStart = Math.min(targetCut.start + deltaTime, targetCut.end - 0.1);
        const snapped = getSnapTime(newStart, dragState.targetId);
        if (snapped !== null && snapped < targetCut.end) newStart = snapped;

        const clampedStart = Math.max(0, newStart);
        const delta = clampedStart - targetCut.start;
        const newSourceStart = (targetCut.sourceStart ?? targetCut.start) + delta;

        setCuts(baseCuts.map(c => c.id === dragState.targetId ? {
          ...c,
          start: clampedStart,
          sourceStart: Math.max(0, newSourceStart)
        } : c));
      }
      else if (dragState.type === 'trim-end' && dragState.targetId) {
        const baseCuts = dragState.initialCuts || cuts;
        const targetCut = baseCuts.find(c => c.id === dragState.targetId);
        if (!targetCut) return;

        let newEnd = Math.max(targetCut.end + deltaTime, targetCut.start + 0.1);
        const snapped = getSnapTime(newEnd, dragState.targetId);
        if (snapped !== null && snapped > targetCut.start) newEnd = snapped;

        const delta = newEnd - targetCut.end;
        const newSourceEnd = (targetCut.sourceEnd ?? targetCut.end) + delta;

        setCuts(baseCuts.map(c => c.id === dragState.targetId ? {
          ...c,
          end: newEnd,
          sourceEnd: newSourceEnd
        } : c));
      }
      else if (dragState.type === 'new-asset') {
        // Update ghost position for new asset
        if (timelineRef.current && timelineContainerRef.current) {
          const rect = timelineRef.current.getBoundingClientRect();
          // Check if mouse is within timeline y-bounds
          if (e.clientY >= rect.top && e.clientY <= rect.bottom + 300) { // +300 for some leeway
            const offsetX = e.clientX - rect.left;
            let ghostTime = Math.max(0, offsetX / zoomLevel);

            const snapped = getSnapTime(ghostTime);
            if (snapped !== null) ghostTime = snapped;

            // Determine track based on element under cursor
            let ghostTrackId = dragState.ghostTrackId;
            const elementUnderCursor = document.elementFromPoint(e.clientX, e.clientY);
            const trackLane = elementUnderCursor?.closest('.track-lane');
            if (trackLane) {
              const tid = Number(trackLane.getAttribute('data-track-id'));
              if (!isNaN(tid)) ghostTrackId = tid;
            }

            setDragState(prev => ({ ...prev, ghostTime, ghostTrackId }));
          }
        }
      }
    };

    const handleMouseUp = () => {
      setMarqueeRect(null);
      if (dragState.isDragging && dragState.type === 'new-asset' && dragState.ghostTime !== undefined) {
        const asset = projectAssets.find(a => a.id === dragState.newAssetId);
        if (!asset) {
          setDragState({ ...dragState, isDragging: false, type: null, ghostTime: undefined });
          return;
        }

        let trackId = dragState.ghostTrackId || 0;
        const targetTrack = videoTracks.find(t => t.id === trackId);

        // Enforce Track Compatibility
        // Video/Image -> Video Track
        // Audio -> Audio Track
        // Text -> Text Track (Assets shouldn't go here)
        let isValidTrack = false;

        if (targetTrack) {
          if ((asset.type === 'video' || asset.type === 'image') && targetTrack.type === 'video') isValidTrack = true;
          if (asset.type === 'audio' && targetTrack.type === 'audio') isValidTrack = true;
        }

        // If invalid drop target, auto-assign to best track
        if (!isValidTrack) {
          if (asset.type === 'video' || asset.type === 'image') {
            // Find first video track
            const vTrack = videoTracks.find(t => t.type === 'video');
            trackId = vTrack ? vTrack.id : 0;
          } else if (asset.type === 'audio') {
            // Find first audio track
            const aTrack = videoTracks.find(t => t.type === 'audio');
            trackId = aTrack ? aTrack.id : 99;
          }
        }

        const start = dragState.ghostTime;
        // Use asset duration if available, else fallback (images defaults to 3s in loader)
        const duration = asset.duration || dragState.newAssetDuration || 3;

        // Auto-init video source if empty and we are dropping a video
        if (asset.type === 'video' && !videoUrl) {
          setVideoUrl(asset.url);
          setVideoFile(asset.file || null);
          setOriginalVideoPath(null);
        }

        // Create the new cut
        const newCut: Cut = {
          id: Math.random().toString(36).substr(2, 9),
          start: start,
          end: start + duration,
          sourceStart: 0, // Starts at beginning of asset
          sourceEnd: duration,
          label: asset.name,
          trackId: trackId,
          assetId: dragState.newAssetId
        };

        setCuts(prev => [...prev, newCut]);
        setDragState({ ...dragState, isDragging: false, type: null, ghostTime: undefined });
      } else {
        // Handle other drag types reset if needed
        if (dragState.isDragging) {
          setDragState({ ...dragState, isDragging: false, type: null, ghostTime: undefined });
        }
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragState, cuts, zoomLevel, duration]);



  const handleExportXML = useCallback(() => {
    if (cuts.length === 0) return;
    const fileName = videoFile?.name || "video.mp4";
    const xml = generateFCPXML(cuts, duration, fileName, originalVideoPath);
    downloadFile(xml, 'project_export.xml', 'application/xml');
  }, [cuts, videoFile, duration, originalVideoPath]);

  const handleExportVideo = useCallback(async (resolution: string, bitrate: number, selectedFormats: string[]) => {
    if (cuts.length === 0 || !videoFile) {
      if (!videoFile) alert("請先上傳影片檔案以進行匯出");
      return;
    }

    setIsProcessing(true);
    setShowExportModal(false);

    if (selectedFormats.includes('xml')) {
      handleExportXML();
      if (selectedFormats.length === 1) {
        setIsProcessing(false);
        return;
      }
    }

    const cutsForBackend = cuts.map(c => ({
      ...c,
      start: c.sourceStart ?? c.start,
      end: c.sourceEnd ?? c.end
    }));

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('cuts_json', JSON.stringify(cutsForBackend));
    formData.append('output_resolution', resolution);
    formData.append('output_bitrate', `${bitrate}M`);
    formData.append('output_mode', 'video');
    formData.append('whisper_language', whisperLanguage);
    formData.append('whisper_model_size', whisperModel);
    formData.append('whisper_beam_size', whisperBeamSize.toString());
    formData.append('whisper_temperature', whisperTemperature.toString());
    formData.append('whisper_chars_per_line', whisperCharsPerLine.toString());
    formData.append('whisper_remove_punctuation', whisperRemovePunc ? 'true' : 'false');
    formData.append('vertical_mode', isVerticalMode ? 'true' : 'false');
    formData.append('burn_captions', 'false');
    formData.append('auto_caption', 'false');
    formData.append('face_tracking', 'false');
    formData.append('studio_sound', 'false');
    formData.append('merge_clips', 'true');
    formData.append('selected_formats', JSON.stringify(selectedFormats));

    try {
      const response = await fetch(`${API_BASE_URL}/process-video`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success' && data.download_url) {
          const downloadUrl = `${API_BASE_URL}${data.download_url}`;
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
        alert("匯出失敗，請檢查後端日誌");
      }
    } catch (_e) {
      alert("連線後端失敗");
    } finally {
      setIsProcessing(false);
    }
  }, [cuts, videoFile, duration, handleExportXML, setIsProcessing, whisperLanguage, whisperModel, whisperBeamSize, whisperTemperature, whisperCharsPerLine, whisperRemovePunc, isVerticalMode]);

  const handleWheelZoom = (e: React.WheelEvent) => {
    if (e.metaKey || e.ctrlKey) {
      e.preventDefault();
      const timelineContainer = timelineContainerRef.current;
      if (!timelineContainer) return;

      const rect = timelineContainer.getBoundingClientRect();
      const mouseX = e.clientX - rect.left; // Mouse X relative to container viewport
      const totalScrollLeft = timelineContainer.scrollLeft;
      const mouseTime = (totalScrollLeft + mouseX) / zoomLevel;

      // Determine direction
      const delta = e.deltaY;
      const factor = delta < 0 ? 1.1 : 0.9;

      let newZoom = zoomLevel * factor;
      newZoom = Math.max(1, Math.min(newZoom, 300));

      // Calculate new scroll to keep mouseTime under mouseX
      // newScrollLeft = (mouseTime * newZoom) - mouseX
      const newScrollLeft = (mouseTime * newZoom) - mouseX;

      setZoomLevel(newZoom);

      // React state update is async, but we can try to set scroll immediately if we assume zoom updates fast enough?
      // Or use a ref to track pending scroll?
      // Let's try requestAnimationFrame which often syncs well with layout updates.
      requestAnimationFrame(() => {
        if (timelineContainerRef.current) {
          timelineContainerRef.current.scrollLeft = newScrollLeft;
        }
      });
    }
  };

  // --- Keyboard Shortcuts (via hook) ---
  useKeyboardShortcuts({
    togglePlay,
    handleSplit,
    handleDelete,
    handleUndo,
    handleRedo,
    setActiveTool,
    handleZoom,
  });

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
          {appView === 'editor' && (
            <>
              <button className="btn-ghost-sm" onClick={() => setAppView('welcome')} title="主選單">
                <RotateCcw size={14} />
              </button>
              <div style={{ width: 1, height: 16, background: '#333', margin: '0 4px' }} />

              {/* Layout Selector Group */}
              <div style={{ display: 'flex', background: '#222', borderRadius: 6, padding: 2, gap: 2 }}>
                <button
                  className={`btn-ghost-sm ${!isVerticalMode ? 'active' : ''}`}
                  onClick={() => setIsVerticalMode(false)}
                  style={{ padding: '4px 8px', fontSize: 11, background: !isVerticalMode ? '#333' : 'transparent', color: !isVerticalMode ? '#3ea6ff' : '#888' }}
                >
                  <Monitor size={14} style={{ marginRight: 4 }} /> 橫式排版
                </button>
                <button
                  className={`btn-ghost-sm ${isVerticalMode ? 'active' : ''}`}
                  onClick={() => setIsVerticalMode(true)}
                  style={{ padding: '4px 8px', fontSize: 11, background: isVerticalMode ? '#333' : 'transparent', color: isVerticalMode ? '#3ea6ff' : '#888' }}
                >
                  <Smartphone size={14} style={{ marginRight: 4 }} /> 直式排版
                </button>
              </div>

              <div style={{ width: 1, height: 16, background: '#333', margin: '0 4px' }} />
              <button className="btn-ghost-sm" onClick={handleExportProject} title="儲存專案 (Save Project)">
                <Save size={14} /> 儲存專案
              </button>

              <button className="btn-primary-sm" onClick={() => setShowExportModal(true)} style={{ height: 28, fontSize: 11, padding: '0 12px', background: 'linear-gradient(135deg, #3ea6ff 0%, #007aff 100%)', border: 'none', fontWeight: 600 }}>
                <Download size={14} /> 匯出影片
              </button>
            </>
          )}
        </div>
      </header>

      {/* Welcome Screen */}
      {appView === 'welcome' ? (
        <WelcomeScreen
          onNewProject={() => {
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
          onResumeProject={() => {
            const saved = localStorage.getItem('antigravity_current_project');
            if (saved) {
              loadProject(JSON.parse(saved));
            }
          }}
          onImportProject={handleImportProject}
          hasExistingProject={!!localStorage.getItem('antigravity_current_project')}
        />
      ) : (
        <div className={`premiere-layout ${isVerticalMode ? 'vertical-layout-active' : ''}`}>
          {/* 2. Main Premiere Layout */}

          {/* Top Section: Panels */}
          <div className="top-panels">

            {/* Left: Project / Inspector Panel */}
            <div className="panel-container" style={{ width: 360, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
              {/* Panel Tabs Header */}
              <div className="panel-header" style={{ gap: 2, padding: '0 4px', overflowX: 'auto', scrollbarWidth: 'none' }}>
                <div className={`panel-tab ${leftPanelTab === 'project' ? 'active' : ''}`} onClick={() => setLeftPanelTab('project')} style={{ fontSize: 10, flex: 1, justifyContent: 'center', whiteSpace: 'nowrap' }}>專案</div>
                <div className={`panel-tab ${leftPanelTab === 'controls' ? 'active' : ''}`} onClick={() => setLeftPanelTab('controls')} style={{ fontSize: 10, flex: 1, justifyContent: 'center', whiteSpace: 'nowrap' }}>字幕校正</div>
                <div className={`panel-tab ${leftPanelTab === 'effects' ? 'active' : ''}`} onClick={() => setLeftPanelTab('effects')} style={{ fontSize: 10, flex: 1, justifyContent: 'center', whiteSpace: 'nowrap' }}>屬性</div>
                <div className={`panel-tab ${leftPanelTab === 'roughcut' ? 'active' : ''}`} onClick={() => setLeftPanelTab('roughcut')} style={{ fontSize: 10, flex: 1, justifyContent: 'center', whiteSpace: 'nowrap' }}>AI 粗剪</div>
                <div className={`panel-tab ${leftPanelTab === 'subtitles' ? 'active' : ''}`} onClick={() => setLeftPanelTab('subtitles')} style={{ fontSize: 10, flex: 1, justifyContent: 'center', whiteSpace: 'nowrap' }}>AI 字幕</div>
                <div className={`panel-tab ${leftPanelTab === 'highlights' ? 'active' : ''}`} onClick={() => setLeftPanelTab('highlights')} style={{ fontSize: 10, flex: 1, justifyContent: 'center', whiteSpace: 'nowrap' }}>AI 精華</div>
              </div>

              <div className="panel-content" style={{ padding: 0, overflowY: 'auto' }}>

                {/* 1. PROJECT TAB */}
                {leftPanelTab === 'project' && (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    {/* Toolbar */}
                    <div style={{ padding: 8, borderBottom: '1px solid #333', display: 'flex', gap: 8 }}>
                      <label className="btn-primary-sm" style={{ flex: 1, cursor: 'pointer' }}>
                        <Plus size={12} /> 匯入素材
                        <input type="file" multiple hidden accept="*/*" onChange={(e) => {
                          if (e.target.files && e.target.files.length > 0) {
                            const newAssets = Array.from(e.target.files).map(f => ({
                              id: Math.random().toString(36).substr(2, 9),
                              type: f.type.startsWith('video') ? 'video' : (f.type.startsWith('audio') || f.name.endsWith('.mp3') || f.name.endsWith('.wav') || f.name.endsWith('.m4a')) ? 'audio' : 'image',
                              name: f.name,
                              url: URL.createObjectURL(f),
                              file: f,
                              duration: 0 // Init with 0, update later
                            } as Asset));

                            setProjectAssets(prev => [...prev, ...newAssets]);

                            // Reset input value to allow re-selecting same file
                            e.target.value = '';

                            // Async get duration
                            newAssets.forEach(asset => {
                              if (asset.type === 'video' || asset.type === 'audio') {
                                const el = document.createElement(asset.type === 'video' ? 'video' : 'audio');
                                el.src = asset.url;
                                el.onloadedmetadata = () => {
                                  setProjectAssets(prev => prev.map(p => p.id === asset.id ? { ...p, duration: el.duration } : p));
                                };
                              } else {
                                // Image default duration
                                setProjectAssets(prev => prev.map(p => p.id === asset.id ? { ...p, duration: 3 } : p));
                              }
                            });
                          }
                        }} />
                      </label>
                      <button className="btn-icon-sm" onClick={() => {
                        if (confirm('確定要清空所有素材與時間軸嗎？這將會重置專案。')) {
                          setProjectAssets([]);
                          setCuts([]);
                          setVideoUrl(null);
                          setVideoFile(null);
                          setOriginalVideoPath(null);
                          setCurrentTime(0);
                        }
                      }} title="清空素材庫與重置專案"><Trash size={14} /></button>
                    </div>

                    {/* Assets Grid */}
                    <div style={{ flex: 1, overflowY: 'auto', padding: 8, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, alignContent: 'start' }}>
                      {projectAssets.length > 0 ? projectAssets.map(asset => (
                        <div key={asset.id}
                          className="asset-item"
                          draggable="false" // We implement custom dragging
                          onMouseDown={(e) => {
                            e.preventDefault();
                            setDragState({
                              isDragging: true,
                              type: 'new-asset',
                              targetId: null,
                              startX: e.clientX,
                              initialValue: 0,
                              newAssetId: asset.id,
                              newAssetDuration: asset.duration || 5 // Default 5s if unknown
                            });
                          }}
                          onClick={() => {
                            // Only click if not dragged
                            if (!dragState.isDragging && asset.type === 'video') {
                              if (confirm(`要將 "${asset.name}" 載入到預覽視窗嗎? 這將會重置時間軸。`)) {
                                setVideoUrl(asset.url);
                                setVideoFile(asset.file || null);
                                // Clean restart
                                setCuts([]);
                                setCurrentTime(0);
                                setOriginalVideoPath(null);
                              }
                            }
                          }}
                          title={asset.name}
                          style={{ aspectRatio: '1/1', background: '#222', borderRadius: 4, overflow: 'hidden', position: 'relative', border: '1px solid #444', cursor: 'pointer' }}
                        >
                          {asset.type === 'video' ? (
                            <video src={asset.url} style={{ width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none' }} />
                          ) : asset.type === 'image' ? (
                            <img src={asset.url} style={{ width: '100%', height: '100%', objectFit: 'cover', pointerEvents: 'none' }} />
                          ) : (
                            <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#333' }}>
                              <Music size={24} color="#666" />
                              <span style={{ fontSize: 9, color: '#888', marginTop: 4 }}>AUDIO</span>
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

                {/* 2. SUBTITLE CORRECTION TAB (formerly controls) */}
                {leftPanelTab === 'controls' && (
                  <div style={{ padding: 12, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ fontSize: 11, color: '#aaa', fontWeight: 700, marginBottom: 12, paddingLeft: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Type size={14} /> 字幕全清單校正
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8, paddingRight: 4 }}>
                      {cuts.filter(c => {
                        const track = videoTracks.find(t => t.id === c.trackId);
                        return track && track.type === 'text';
                      }).length > 0 ? (
                        cuts.filter(c => {
                          const track = videoTracks.find(t => t.id === c.trackId);
                          return track && track.type === 'text';
                        }).sort((a, b) => a.start - b.start).map(cut => (
                          <div key={cut.id}
                            style={{
                              background: '#1a1a1a',
                              border: selectedCutIds.includes(cut.id) ? '1px solid #3ea6ff' : '1px solid #333',
                              borderRadius: 8, padding: 8,
                              cursor: 'pointer'
                            }}
                            onClick={() => {
                              setSelectedCutIds([cut.id]);
                              setCurrentTime(cut.start);
                              if (videoRef.current) videoRef.current.currentTime = cut.start;
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 10, color: '#666', fontFamily: 'monospace' }}>
                              <span>{cut.start.toFixed(2)}s - {cut.end.toFixed(2)}s</span>
                              <div style={{ display: 'flex', gap: 8 }}>
                                <button onClick={(e) => {
                                  e.stopPropagation();
                                  setCuts(prev => prev.filter(p => p.id !== cut.id));
                                }} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: 0 }}>刪除</button>
                              </div>
                            </div>
                            <textarea
                              className="textarea-modern"
                              value={cut.label}
                              onChange={(e) => {
                                setCuts(prev => prev.map(p => p.id === cut.id ? { ...p, label: e.target.value } : p));
                              }}
                              style={{ height: 36, width: '100%', fontSize: 12, padding: 8, background: '#111', resize: 'none' }}
                            />
                          </div>
                        ))
                      ) : (
                        <div style={{ textAlign: 'center', padding: 40, color: '#555', fontSize: 12, border: '1px dashed #333', borderRadius: 12 }}>
                          尚未生成任何字幕片段
                        </div>
                      )}
                    </div>

                    {selectedCutIds.length > 0 && cuts.find(c => c.id === selectedCutIds[0])?.trackId === 1 && (
                      <div style={{ marginTop: 12, padding: 12, background: '#1c1c1e', borderRadius: 8, border: '1px solid #333' }}>
                        <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>快速調整選中片段</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <button className="btn-ghost-sm" onClick={() => {
                            const sid = selectedCutIds[0];
                            setCuts(prev => prev.map(c => c.id === sid ? { ...c, start: Math.max(0, c.start - 0.1) } : c));
                          }}>-0.1s 開始</button>
                          <button className="btn-ghost-sm" onClick={() => {
                            const sid = selectedCutIds[0];
                            setCuts(prev => prev.map(c => c.id === sid ? { ...c, end: c.end + 0.1 } : c));
                          }}>+0.1s 結束</button>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 3. ROUGH CUT TAB (Silence Removal) */}
                {leftPanelTab === 'roughcut' && (
                  <div style={{ padding: 12, overflowY: 'auto', maxHeight: '100%' }}>
                    <div className="panel-card" style={{ background: '#1c1c1e', padding: 20, borderRadius: 16, border: '1px solid #333', boxShadow: '0 4px 24px rgba(0,0,0,0.2)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
                        <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(62,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: 12 }}>
                          <Zap size={18} color="#3ea6ff" />
                        </div>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: 14, color: '#fff' }}>智能去氣口</div>
                          <div style={{ fontSize: 11, color: '#666' }}>自動移除靜音</div>
                        </div>
                      </div>

                      <div style={{ marginBottom: 20 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                          <label className="label-sm" style={{ color: '#aaa', fontWeight: 500 }}>噪音閥值</label>
                          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#3ea6ff', background: 'rgba(62,166,255,0.1)', padding: '2px 6px', borderRadius: 4 }}>
                            {silenceThreshold} dB
                          </span>
                        </div>
                        <input
                          type="range" min="-60" max="0" step="1"
                          className="slider-modern"
                          style={{ '--fill-percent': `${((silenceThreshold + 60) / 60) * 100}%` } as React.CSSProperties}
                          value={silenceThreshold}
                          onChange={e => setSilenceThreshold(Number(e.target.value))}
                        />
                      </div>

                      <div style={{ marginBottom: 20 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                          <label className="label-sm" style={{ color: '#aaa', fontWeight: 500 }}>最短保留時長</label>
                          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#3ea6ff', background: 'rgba(62,166,255,0.1)', padding: '2px 6px', borderRadius: 4 }}>
                            {silenceMinDuration}s
                          </span>
                        </div>
                        <input
                          type="range" min="0" max="2" step="0.1"
                          className="slider-modern"
                          style={{ '--fill-percent': `${(silenceMinDuration / 2) * 100}%` } as React.CSSProperties}
                          value={silenceMinDuration}
                          onChange={e => setSilenceMinDuration(Number(e.target.value))}
                        />
                      </div>

                      <button className="btn-gradient-animate" onClick={handleSilenceRemoval}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                          <Zap size={16} fill="currentColor" />
                          <span>開始去氣口處理</span>
                        </div>
                      </button>

                      <div style={{ height: 1, background: '#333', margin: '20px 0' }}></div>

                      <div style={{ padding: 12, border: '2px dashed #222', borderRadius: 12, background: 'rgba(255,255,255,0.02)' }}>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#eee', marginBottom: 4 }}>高效剪輯必備</div>
                        <div style={{ fontSize: 11, color: '#666', marginBottom: 16 }}>處理完靜音後，可以一鍵消除所有間隔，讓字幕與畫面緊密對齊。</div>

                        <button
                          className="btn-primary"
                          onClick={handleAlignCuts}
                          style={{
                            width: '100%',
                            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                            boxShadow: '0 4px 15px rgba(16, 185, 129, 0.3)',
                            border: 'none',
                            fontWeight: 600,
                            height: 40
                          }}
                        >
                          <Magnet size={18} /> 一鍵對齊 (消除間隔)
                        </button>
                      </div>


                    </div>
                  </div>
                )}

                {/* 4. AI HIGHLIGHTS TAB */}
                {leftPanelTab === 'highlights' && (
                  <div style={{ padding: 12, overflowY: 'auto', maxHeight: '100%' }}>
                    <div className="panel-card" style={{ background: '#1c1c1e', padding: 20, borderRadius: 16, border: '1px solid #333', boxShadow: '0 4px 24px rgba(0,0,0,0.2)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
                        <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(59,130,246,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: 12 }}>
                          <Film size={18} color="#3b82f6" />
                        </div>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: 14, color: '#fff' }}>AI 精華生成</div>
                          <div style={{ fontSize: 11, color: '#666' }}>使用 Gemini AI 智能分析</div>
                        </div>
                      </div>

                      <div className="style-grid-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                        <div>
                          <label className="label-modern">精華片段數量</label>
                          <input
                            type="number" className="input-modern"
                            value={highlightCount} onChange={e => setHighlightCount(Number(e.target.value))}
                            min="1" max="50"
                          />
                        </div>
                        <div>
                          <label className="label-modern">單片時長 (秒)</label>
                          <input
                            type="number" className="input-modern"
                            value={targetDuration} onChange={e => setTargetDuration(Number(e.target.value))}
                            min="5" max="300"
                          />
                        </div>
                      </div>

                      <div style={{ marginBottom: 16 }}>
                        <label className="label-modern">AI 模型選擇</label>
                        <div className="select-wrapper">
                          <select
                            className="select-modern"
                            value={geminiModel}
                            onChange={(e) => setGeminiModel(e.target.value)}
                          >
                            <option value="gemini-3-pro-preview">Gemini 3 Pro</option>
                            <option value="gemini-3-flash-preview">Gemini 3 Flash</option>
                            <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                            <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                          </select>
                          <div className="select-arrow"><ChevronDown size={14} /></div>
                        </div>
                      </div>

                      <div style={{ marginBottom: 16 }}>
                        <label className="label-modern">AI 提示詞 (Prompt)</label>
                        <textarea
                          className="textarea-modern" rows={3}
                          value={instruction} onChange={e => setInstruction(e.target.value)}
                          placeholder="例如：找出最有趣的對話，或是動作最精彩的片段..."
                        />
                      </div>

                      <div style={{ marginBottom: 20 }}>
                        <label className="label-modern">API Key</label>
                        <div style={{ position: 'relative' }}>
                          <input
                            className="input-modern" type="password"
                            value={apiKey} onChange={e => setApiKey(e.target.value)}
                            placeholder="輸入您的 Gemini API Key"
                            style={{ paddingRight: 36 }}
                          />
                          <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', opacity: 0.5 }}>
                            <Zap size={14} />
                          </div>
                        </div>
                      </div>

                      <button
                        className="btn-gradient-animate btn-gradient-blue"
                        onClick={handleGeminiHighlights}
                        disabled={isProcessing}
                        style={{ opacity: isProcessing ? 0.7 : 1, cursor: isProcessing ? 'wait' : 'pointer', width: '100% ' }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                          {isProcessing ? (
                            <Loader2 size={16} className="spin" />
                          ) : (
                            <Film size={16} fill="currentColor" />
                          )}
                          <span>{isProcessing ? 'AI 分析生成中...' : '生成精華短片'}</span>
                        </div>
                      </button>
                    </div>
                  </div>
                )}

                {/* 5. SUBTITLES TAB */}
                {leftPanelTab === 'subtitles' && (
                  <div style={{ padding: 12, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <div className="panel-card" style={{ background: '#1c1c1e', padding: 20, borderRadius: 16, border: '1px solid #333', boxShadow: '0 4px 24px rgba(0,0,0,0.2)', marginBottom: 16 }}>
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                        <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(59,130,246,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: 12 }}>
                          <Type size={18} color="#3b82f6" />
                        </div>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: 14, color: '#fff' }}>SRT 語音辨識</div>
                          <div style={{ fontSize: 11, color: '#666' }}>自動提取影片對話</div>
                        </div>
                      </div>

                      <div style={{ marginBottom: 16 }}>
                        <label className="label-sm" style={{ display: 'block', marginBottom: 8, color: '#aaa' }}>辨識模型</label>
                        <div className="select-wrapper">
                          <select
                            className="select-modern"
                            value={whisperModel}
                            onChange={(e) => setWhisperModel(e.target.value)}
                          >
                            <option value="turbo">Turbo</option>
                            <option value="large">Large</option>
                            <option value="medium">Medium</option>
                            <option value="base">Base</option>
                            <option value="tiny">Tiny</option>
                          </select>
                          <div className="select-arrow"><ChevronDown size={14} /></div>
                        </div>
                      </div>

                      <div style={{ marginBottom: 20 }}>
                        <label className="label-sm" style={{ display: 'block', marginBottom: 8, color: '#aaa' }}>辨識語言</label>
                        <div className="select-wrapper">
                          <select
                            className="select-modern"
                            value={whisperLanguage}
                            onChange={(e) => setWhisperLanguage(e.target.value)}
                          >
                            <option value="zh">繁體中文</option>
                            <option value="en">英文</option>
                            <option value="ja">日文</option>
                            <option value="auto">自動偵測</option>
                          </select>
                          <div className="select-arrow"><ChevronDown size={14} /></div>
                        </div>
                      </div>

                      <div style={{ marginBottom: 16 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                          <div>
                            <label className="label-sm" style={{ display: 'block', marginBottom: 8, color: '#aaa' }}>Beam Size</label>
                            <input
                              type="number" className="input-modern" style={{ height: 32, fontSize: 12 }}
                              value={whisperBeamSize} onChange={e => setWhisperBeamSize(Number(e.target.value))}
                              min="1" max="10"
                            />
                          </div>
                          <div>
                            <label className="label-sm" style={{ display: 'block', marginBottom: 8, color: '#aaa' }}>溫度 (Temp)</label>
                            <input
                              type="number" className="input-modern" style={{ height: 32, fontSize: 12 }}
                              value={whisperTemperature} onChange={e => setWhisperTemperature(Number(e.target.value))}
                              step="0.1" min="0" max="1"
                            />
                          </div>
                        </div>
                      </div>

                      <div style={{ marginBottom: 20 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                          <label className="label-sm" style={{ color: '#aaa' }}>每行字數限制</label>
                          <span style={{ fontSize: 11, color: '#3ea6ff', fontWeight: 700 }}>{whisperCharsPerLine}</span>
                        </div>
                        <input
                          type="range" min="5" max="30" step="1"
                          className="slider-modern"
                          style={{ '--fill-percent': `${((whisperCharsPerLine - 5) / 25) * 100}%` } as React.CSSProperties}
                          value={whisperCharsPerLine}
                          onChange={e => setWhisperCharsPerLine(Number(e.target.value))}
                        />
                      </div>

                      <div style={{ marginBottom: 20 }}>
                        <div
                          onClick={() => setWhisperRemovePunc(!whisperRemovePunc)}
                          style={{
                            padding: '10px 14px', borderRadius: 12, background: whisperRemovePunc ? 'rgba(62,166,255,0.08)' : 'rgba(255,255,255,0.02)',
                            border: `1px solid ${whisperRemovePunc ? 'rgba(62,166,255,0.3)' : 'rgba(255,255,255,0.05)'}`,
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer'
                          }}
                        >
                          <span style={{ fontSize: 12, color: whisperRemovePunc ? '#eee' : '#666' }}>自動移除標點符號</span>
                          <div style={{ width: 12, height: 12, borderRadius: 3, background: whisperRemovePunc ? '#3ea6ff' : '#333' }} />
                        </div>
                      </div>

                      <button
                        className="btn-gradient-animate btn-gradient-blue"
                        onClick={handleAISubtitles}
                        disabled={isProcessing}
                        style={{ width: '100%', opacity: isProcessing ? 0.7 : 1 }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                          <RotateCcw size={16} />
                          <span>開始 AI 字幕辨識</span>
                        </div>
                      </button>
                    </div>

                    {/* Redundant Editor Section Removed: Moved to Subtitle Correction Tab */}
                  </div>
                )}

                {/* 3. EFFECTS / INSPECTOR TAB */}
                {leftPanelTab === 'effects' && (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    {(() => {
                      const activeCut = cuts.find(c => selectedCutIds.includes(c.id));

                      if (!activeCut) {
                        return (
                          <div style={{ padding: 20, textAlign: 'center', color: '#666', fontSize: 13, marginTop: 40 }}>
                            請選擇時間軸上的一個片段<br />以編輯屬性
                          </div>
                        );
                      }

                      if (activeCut.trackId === 1) {
                        // Text Track -> Show Subtitle Settings
                        return (
                          <div style={{ padding: 12, height: '100%', overflowY: 'auto' }}>
                            {/* ... Existing Subtitle UI ... */}
                            <div className="panel-card" style={{ background: '#1c1c1e', padding: 16, borderRadius: 12, border: '1px solid #333', marginBottom: 12 }}>
                              <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16, gap: 8 }}>
                                <Type size={16} color="#3ea6ff" />
                                <div style={{ fontWeight: 700, fontSize: 13, color: '#fff' }}>字體設定</div>
                              </div>
                              {/* Existing Font Controls - Reusing state directly */}
                              <div style={{ marginBottom: 16 }}>
                                <label className="label-sm" style={{ display: 'block', marginBottom: 6, color: '#888' }}>選擇字體</label>
                                <div className="select-wrapper">
                                  <select
                                    className="select-modern"
                                    style={{ height: 32, fontSize: 12 }}
                                    value={subtitleConfig.fontFamily}
                                    onChange={(e) => setSubtitleConfig(prev => ({ ...prev, fontFamily: e.target.value }))}
                                  >
                                    <option value="Inter, system-ui, sans-serif">系統預設 (Inter)</option>
                                    {availableFonts.map((font: any) => (
                                      <option key={font.name} value={font.name}>{font.name}</option>
                                    ))}
                                  </select>
                                  <ChevronDown size={14} className="select-arrow" color="#888" />
                                </div>
                              </div>

                              <div>
                                <label className="label-sm" style={{ display: 'block', marginBottom: 6, color: '#888' }}>上傳字體 (.ttf, .otf, .woff2)</label>
                                <button
                                  className="btn-secondary-sm"
                                  style={{ width: '100%', justifyContent: 'center', cursor: 'pointer', border: 'none' }}
                                  onClick={() => document.getElementById('font-upload-input')?.click()}
                                >
                                  <Upload size={12} /> 上傳字體檔案
                                </button>
                                <input
                                  id="font-upload-input"
                                  type="file"
                                  hidden
                                  accept=".ttf,.otf,.woff2"
                                  onChange={async (e) => {
                                    const file = e.target.files?.[0];
                                    if (!file) return;
                                    const formData = new FormData();
                                    formData.append('file', file);
                                    try {
                                      const res = await fetch(`${API_BASE_URL}/upload-font`, { method: 'POST', body: formData });
                                      if (res.ok) {
                                        const data = await res.json();
                                        alert('字體上傳成功！');
                                        fetchFonts();
                                        setSubtitleConfig(prev => ({ ...prev, fontFamily: data.font_name }));
                                      }
                                    } catch (err) { alert('上傳失敗'); }
                                  }}
                                />
                              </div>

                              {/* Safe Zone Control */}
                              <div style={{ marginTop: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                  <label className="label-sm" style={{ display: 'block', marginBottom: 6, color: '#888' }}>安全框邊距 (Safe Zone)</label>
                                  <span style={{ fontSize: 10, color: '#555' }}>{subtitleConfig.safeZoneMargin}%</span>
                                </div>
                                <input
                                  type="range"
                                  min="0"
                                  max="30"
                                  step="1"
                                  className="slider-modern"
                                  style={{ width: '100%' }}
                                  value={subtitleConfig.safeZoneMargin}
                                  onChange={(e) => setSubtitleConfig(prev => ({ ...prev, safeZoneMargin: Number(e.target.value) }))}
                                />
                              </div>
                            </div>

                            {/* Styling Params Panel (The existing one below will need to be rendered here or keep structure) */}
                            {/* Moving the logic from below to here is tricky if I want to keep the file structure clean. 
                                 For now, I will just render the Subtitle Properites Panel here directly.
                             */}
                            <div className="panel-card" style={{ background: '#1c1c1e', padding: 16, borderRadius: 12, border: '1px solid #333' }}>
                              <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16, gap: 8 }}>
                                <Zap size={16} color="#3ea6ff" />
                                <div style={{ fontWeight: 700, fontSize: 13, color: '#fff' }}>樣式詳細參數 (全域)</div>
                              </div>

                              {/* Basic Size & Pos */}
                              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                                <div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                    <label className="label-sm" style={{ color: '#888' }}>字體大小</label>
                                    <input
                                      type="number"
                                      className="input-modern"
                                      style={{ width: 60, padding: 4, height: 24, fontSize: 12, textAlign: 'right' }}
                                      value={subtitleConfig.fontSize}
                                      onChange={(e) => setSubtitleConfig(prev => ({ ...prev, fontSize: Number(e.target.value) }))}
                                    />
                                  </div>
                                  <input type="range" min="12" max="200" className="slider-modern" value={subtitleConfig.fontSize} onChange={(e) => setSubtitleConfig(prev => ({ ...prev, fontSize: Number(e.target.value) }))} />
                                </div>
                                <div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                    <label className="label-sm" style={{ color: '#888' }}>垂直位置 (%)</label>
                                    <input
                                      type="number"
                                      className="input-modern"
                                      style={{ width: 60, padding: 4, height: 24, fontSize: 12, textAlign: 'right' }}
                                      value={subtitleConfig.verticalOffset}
                                      onChange={(e) => setSubtitleConfig(prev => ({ ...prev, verticalOffset: Number(e.target.value) }))}
                                    />
                                  </div>
                                  <input type="range" min="0" max="100" className="slider-modern" value={subtitleConfig.verticalOffset} onChange={(e) => setSubtitleConfig(prev => ({ ...prev, verticalOffset: Number(e.target.value) }))} />
                                </div>
                              </div>

                              {/* A. Text Color & Gradient */}
                              <div style={{ marginBottom: 16, borderBottom: '1px solid #222', paddingBottom: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: '#aaa' }}>文字顏色</div>
                                  <div
                                    onClick={() => setSubtitleConfig(prev => ({ ...prev, useGradient: !prev.useGradient }))}
                                    style={{
                                      fontSize: 10, padding: '2px 8px', borderRadius: 4,
                                      background: subtitleConfig.useGradient ? '#3ea6ff' : '#333',
                                      color: subtitleConfig.useGradient ? '#fff' : '#666',
                                      cursor: 'pointer', transition: '0.2s'
                                    }}
                                  >
                                    漸層模式 {subtitleConfig.useGradient ? 'ON' : 'OFF'}
                                  </div>
                                </div>

                                {!subtitleConfig.useGradient ? (
                                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                    <input
                                      type="color"
                                      style={{ width: '100%', height: 36, padding: 0, border: 'none', background: 'none' }}
                                      value={subtitleConfig.primaryColor}
                                      onChange={(e) => setSubtitleConfig(prev => ({ ...prev, primaryColor: e.target.value }))}
                                    />
                                    <input className="input-modern" style={{ fontSize: 10, width: 85 }} value={subtitleConfig.primaryColor} readOnly />
                                  </div>
                                ) : (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                    {/* Gradient Preview Bar */}
                                    <div style={{
                                      height: 12, borderRadius: 6, width: '100%',
                                      background: `linear-gradient(90deg, ${subtitleConfig.gradientStops.map(s => `${s.color} ${s.offset}%`).join(', ')})`,
                                      border: '1px solid #444', marginBottom: 4
                                    }} />

                                    {/* Stops List */}
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                      {subtitleConfig.gradientStops.map((stop, idx) => (
                                        <div key={idx} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                          <div style={{ width: 24, height: 24, borderRadius: 4, overflow: 'hidden', border: '1px solid #444', flexShrink: 0 }}>
                                            <input type="color" style={{ width: 40, height: 40, cursor: 'pointer', margin: -8 }}
                                              value={stop.color}
                                              onChange={e => {
                                                const newStops = [...subtitleConfig.gradientStops];
                                                newStops[idx].color = e.target.value;
                                                setSubtitleConfig(prev => ({ ...prev, gradientStops: newStops }));
                                              }}
                                            />
                                          </div>
                                          <div style={{ flex: 1 }}>
                                            <input type="range" min="0" max="100" className="slider-modern"
                                              value={stop.offset}
                                              onChange={e => {
                                                const newStops = [...subtitleConfig.gradientStops];
                                                newStops[idx].offset = Number(e.target.value);
                                                // Auto sort would be annoying while dragging, maybe sort after?
                                                setSubtitleConfig(prev => ({ ...prev, gradientStops: newStops }));
                                              }}
                                            />
                                          </div>
                                          <span style={{ fontSize: 10, color: '#666', width: 28, textAlign: 'right' }}>{stop.offset}%</span>
                                          {subtitleConfig.gradientStops.length > 2 && (
                                            <button onClick={() => {
                                              setSubtitleConfig(prev => ({ ...prev, gradientStops: prev.gradientStops.filter((_, i) => i !== idx) }));
                                            }} style={{ background: 'none', border: 'none', color: '#ef4444', padding: 4, cursor: 'pointer', fontSize: 14 }}>×</button>
                                          )}
                                        </div>
                                      ))}

                                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
                                        <button className="btn-ghost-sm" style={{ fontSize: 10, padding: '4px 8px', borderColor: '#444' }} onClick={() => {
                                          // Add a stop at 50% or appropriate gap
                                          const lastStop = subtitleConfig.gradientStops[subtitleConfig.gradientStops.length - 1];
                                          const newOffset = Math.min(100, lastStop.offset + 10);
                                          setSubtitleConfig(prev => ({ ...prev, gradientStops: [...prev.gradientStops, { color: '#ffffff', offset: newOffset }].sort((a, b) => a.offset - b.offset) }));
                                        }}>+ 新增顏色點</button>

                                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                          <span style={{ fontSize: 10, color: '#666' }}>角度: {subtitleConfig.gradientAngle}°</span>
                                          <input type="range" min="0" max="360" className="slider-modern" style={{ width: 60 }} value={subtitleConfig.gradientAngle} onChange={e => setSubtitleConfig(prev => ({ ...prev, gradientAngle: Number(e.target.value) }))} />
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* B. Outline Section */}
                              <div style={{ marginBottom: 16, borderBottom: '1px solid #222', paddingBottom: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: '#aaa' }}>描邊 (Outline)</div>
                                  <input type="checkbox" checked={subtitleConfig.useOutline} onChange={e => setSubtitleConfig(prev => ({ ...prev, useOutline: e.target.checked }))} />
                                </div>
                                {subtitleConfig.useOutline && (
                                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                    <input
                                      type="color"
                                      style={{ width: 40, height: 32, padding: 0, border: 'none', background: 'none' }}
                                      value={subtitleConfig.outlineColor}
                                      onChange={(e) => setSubtitleConfig(prev => ({ ...prev, outlineColor: e.target.value }))}
                                    />
                                    <div style={{ flex: 1 }}>
                                      <input
                                        type="range" min="0" max="20" step="0.5"
                                        className="slider-modern"
                                        value={subtitleConfig.outlineWidth}
                                        onChange={(e) => setSubtitleConfig(prev => ({ ...prev, outlineWidth: Number(e.target.value) }))}
                                      />
                                    </div>
                                    <span style={{ fontSize: 10, color: '#666', width: 20 }}>{subtitleConfig.outlineWidth}</span>
                                  </div>
                                )}
                              </div>

                              {/* C. Shadow Section */}
                              <div style={{ marginBottom: 16, borderBottom: '1px solid #222', paddingBottom: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: '#aaa' }}>陰影 (Shadow)</div>
                                  <input type="checkbox" checked={subtitleConfig.useShadow} onChange={e => setSubtitleConfig(prev => ({ ...prev, useShadow: e.target.checked }))} />
                                </div>
                                {subtitleConfig.useShadow && (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                      <input type="color" style={{ width: 40, height: 32, border: 'none', background: 'none' }} value={subtitleConfig.shadowColor.startsWith('rgba') ? '#000000' : subtitleConfig.shadowColor} onChange={e => setSubtitleConfig(prev => ({ ...prev, shadowColor: e.target.value }))} />
                                      <div style={{ flex: 1 }}>
                                        <label className="label-sm" style={{ fontSize: 9 }}>模糊度</label>
                                        <input type="range" min="0" max="40" className="slider-modern" value={subtitleConfig.shadowBlur} onChange={e => setSubtitleConfig(prev => ({ ...prev, shadowBlur: Number(e.target.value) }))} />
                                      </div>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                                      <div>
                                        <label className="label-sm" style={{ fontSize: 9 }}>X 偏移: {subtitleConfig.shadowOffsetX}</label>
                                        <input type="range" min="-20" max="20" className="slider-modern" value={subtitleConfig.shadowOffsetX} onChange={e => setSubtitleConfig(prev => ({ ...prev, shadowOffsetX: Number(e.target.value) }))} />
                                      </div>
                                      <div>
                                        <label className="label-sm" style={{ fontSize: 9 }}>Y 偏移: {subtitleConfig.shadowOffsetY}</label>
                                        <input type="range" min="-20" max="20" className="slider-modern" value={subtitleConfig.shadowOffsetY} onChange={e => setSubtitleConfig(prev => ({ ...prev, shadowOffsetY: Number(e.target.value) }))} />
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* D. Background Section */}
                              <div style={{ marginBottom: 8 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: '#aaa' }}>背景底盒 (Background)</div>
                                  <input type="checkbox" checked={subtitleConfig.useBackground} onChange={e => setSubtitleConfig(prev => ({ ...prev, useBackground: e.target.checked }))} />
                                </div>
                                {subtitleConfig.useBackground && (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                      <input type="color" style={{ width: 40, height: 32, border: 'none', background: 'none' }} value={subtitleConfig.backgroundColor} onChange={e => setSubtitleConfig(prev => ({ ...prev, backgroundColor: e.target.value }))} />
                                      <div style={{ flex: 1 }}>
                                        <label className="label-sm" style={{ fontSize: 9 }}>透明度: {subtitleConfig.backgroundOpacity}</label>
                                        <input type="range" min="0" max="1" step="0.1" className="slider-modern" value={subtitleConfig.backgroundOpacity} onChange={e => setSubtitleConfig(prev => ({ ...prev, backgroundOpacity: Number(e.target.value) }))} />
                                      </div>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                                      <div>
                                        <label className="label-sm" style={{ fontSize: 9 }}>圓角弧度: {subtitleConfig.borderRadius}</label>
                                        <input type="range" min="0" max="50" className="slider-modern" value={subtitleConfig.borderRadius} onChange={e => setSubtitleConfig(prev => ({ ...prev, borderRadius: Number(e.target.value) }))} />
                                      </div>
                                      <div>
                                        <label className="label-sm" style={{ fontSize: 9 }}>上下內距 (Y): {subtitleConfig.paddingY}</label>
                                        <input type="range" min="0" max="50" className="slider-modern" value={subtitleConfig.paddingY} onChange={e => setSubtitleConfig(prev => ({ ...prev, paddingY: Number(e.target.value) }))} />
                                      </div>
                                      <div>
                                        <label className="label-sm" style={{ fontSize: 9 }}>左右內距 (X): {subtitleConfig.paddingX}</label>
                                        <input type="range" min="0" max="100" className="slider-modern" value={subtitleConfig.paddingX} onChange={e => setSubtitleConfig(prev => ({ ...prev, paddingX: Number(e.target.value) }))} />
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      }

                      // Video/Image Inspector (The New Request)
                      const style = activeCut.style || { scale: 1, x: 0, y: 0, rotation: 0, opacity: 1, mirror: false };
                      const updateStyle = (key: string, val: any) => {
                        setCuts(prev => prev.map(c => c.id === activeCut.id ? { ...c, style: { ...style, [key]: val } } : c));
                      };

                      return (
                        <div style={{ padding: 12, height: '100%', overflowY: 'auto' }}>
                          <div className="panel-card" style={{ background: '#1c1c1e', padding: 16, borderRadius: 12, border: '1px solid #333' }}>
                            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16, gap: 8 }}>
                              <Monitor size={16} color="#3ea6ff" />
                              <div style={{ fontWeight: 700, fontSize: 13, color: '#fff' }}>變形 (Transform)</div>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
                              <div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                  <label className="label-sm" style={{ color: '#888' }}>縮放 (Scale)</label>
                                  <span className="label-sm" style={{ color: '#ccc' }}>{(style.scale * 100).toFixed(0)}%</span>
                                </div>
                                <input type="range" min="0" max="3" step="0.01" className="slider-modern" value={style.scale} onChange={e => updateStyle('scale', Number(e.target.value))} />
                              </div>

                              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                                <div>
                                  <label className="label-sm" style={{ display: 'block', marginBottom: 6, color: '#888' }}>X 位置</label>
                                  <input type="number" className="input-modern" value={style.x} onChange={e => updateStyle('x', Number(e.target.value))} />
                                </div>
                                <div>
                                  <label className="label-sm" style={{ display: 'block', marginBottom: 6, color: '#888' }}>Y 位置</label>
                                  <input type="number" className="input-modern" value={style.y} onChange={e => updateStyle('y', Number(e.target.value))} />
                                </div>
                              </div>

                              <div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                  <label className="label-sm" style={{ color: '#888' }}>旋轉 (Rotation)</label>
                                  <span className="label-sm" style={{ color: '#ccc' }}>{style.rotation}°</span>
                                </div>
                                <input type="range" min="-180" max="180" className="slider-modern" value={style.rotation} onChange={e => updateStyle('rotation', Number(e.target.value))} />
                              </div>
                            </div>
                          </div>

                          <div className="panel-card" style={{ background: '#1c1c1e', padding: 16, borderRadius: 12, border: '1px solid #333', marginTop: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16, gap: 8 }}>
                              <Zap size={16} color="#3ea6ff" />
                              <div style={{ fontWeight: 700, fontSize: 13, color: '#fff' }}>不透明度 (Opacity)</div>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                <label className="label-sm" style={{ color: '#888' }}>透明度</label>
                                <span className="label-sm" style={{ color: '#ccc' }}>{(style.opacity * 100).toFixed(0)}%</span>
                              </div>
                              <input type="range" min="0" max="1" step="0.01" className="slider-modern" value={style.opacity} onChange={e => updateStyle('opacity', Number(e.target.value))} />
                            </div>
                          </div>

                          <div className="panel-card" style={{ background: '#1c1c1e', padding: 16, borderRadius: 12, border: '1px solid #333', marginTop: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8, gap: 8 }}>
                              <input type="checkbox" checked={style.mirror || false} onChange={e => updateStyle('mirror', e.target.checked)} />
                              <span style={{ fontSize: 13, color: '#ddd' }}>水平鏡像 (Mirror)</span>
                            </div>
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>

            {/* Center: Program Monitor */}
            <div className="panel-container" style={{ flex: 1 }}>
              <div className="panel-header">
                <div className="panel-tab active">節目檢視: 序列 01</div>
              </div>
              <div
                className={`preview-area ${isVerticalMode ? 'vertical-mode' : 'horizontal-mode'}`}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', overflow: 'hidden', position: 'relative' }}
              >
                {(cuts.length > 0 || (videoUrl && duration === 0)) ? (
                  <div className="video-container" style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>

                    {/* Multi-Track Media Renderer */}
                    <div className="media-renderer-layer" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
                      {sortedTracks.filter(t => t.type !== 'text' && t.visible).map(track => {
                        // 1. Try to find an active cut
                        const activeCut = cuts.find(c => c.trackId === track.id && currentTime >= c.start && currentTime < c.end);

                        // 2. Handle Initialization Case:
                        // If we are on the main video track (0), and there are NO cuts at all (first load),
                        // and we have a videoUrl, we MUST render the video to trigger onLoadedMetadata.
                        const isInit = !activeCut && track.id === 0 && cuts.length === 0 && !!videoUrl;

                        if (!activeCut && !isInit) return null;

                        const asset = activeCut?.assetId ? projectAssets.find(a => a.id === activeCut.assetId) : null;

                        // Default to track type if no asset (source video track 0)
                        const mediaType = asset ? asset.type : track.type;
                        const url = asset ? asset.url : (track.id === 0 ? videoUrl : null);
                        if (!url) return null;

                        // Calculate local time for the media element
                        const localTime = activeCut
                          ? (currentTime - activeCut.start) + (activeCut.sourceStart || 0)
                          : 0; // In init mode, start at 0

                        if (mediaType === 'video') {
                          return (
                            <video
                              key={activeCut?.id || 'init-video'}
                              src={url}
                              onLoadedMetadata={track.id === 0 ? handleVideoMetadata : undefined}
                              style={{
                                position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                                objectFit: 'contain', zIndex: track.id,
                                // Apply Transform
                                transform: activeCut?.style ? `translate(${activeCut.style.x}px, ${activeCut.style.y}px) scale(${activeCut.style.scale}) rotate(${activeCut.style.rotation}deg) scaleX(${activeCut.style.mirror ? -1 : 1})` : 'none',
                                opacity: activeCut?.style ? activeCut.style.opacity : 1
                              }}
                              ref={(el) => {
                                if (el) {
                                  if (track.id === 0) (videoRef as any).current = el;

                                  // Only sync time if established, otherwise let it load
                                  if (Math.abs(el.currentTime - localTime) > 0.3) el.currentTime = localTime;

                                  if (isPlaying && el.paused) el.play().catch(() => { });
                                  else if (!isPlaying && !el.paused) el.pause();
                                }
                              }}
                            />
                          );
                        }

                        if (mediaType === 'image') {
                          return (
                            <img
                              key={activeCut?.id || 'init-image'}
                              src={url}
                              style={{
                                position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                                objectFit: 'contain', zIndex: track.id,
                                // Apply Transform for Images too
                                transform: activeCut?.style ? `translate(${activeCut.style.x}px, ${activeCut.style.y}px) scale(${activeCut.style.scale}) rotate(${activeCut.style.rotation}deg) scaleX(${activeCut.style.mirror ? -1 : 1})` : 'none',
                                opacity: activeCut?.style ? activeCut.style.opacity : 1
                              }}
                            />
                          );
                        }

                        if (mediaType === 'audio') {
                          return (
                            <audio
                              key={activeCut?.id || 'init-audio'}
                              src={url}
                              ref={(el) => {
                                if (el) {
                                  if (Math.abs(el.currentTime - localTime) > 0.3) el.currentTime = localTime;
                                  if (isPlaying && el.paused) el.play().catch(() => { });
                                  else if (!isPlaying && !el.paused) el.pause();
                                }
                              }}
                            />
                          );
                        }
                        return null;
                      })}
                    </div>

                    {/* Text Overlays Layer */}
                    <div className="text-overlay-renderer" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'none', overflow: 'hidden' }}>
                      {cuts.filter(c => {
                        const track = videoTracks.find(t => t.id === c.trackId);
                        return track && track.type === 'text' && currentTime >= c.start && currentTime < c.end;
                      }).map(cut => (
                        <React.Fragment key={cut.id}>
                          {/* Safe Zone Visualizer (Optional: Only show when editing subtitles?) */}
                          {selectedCutIds.includes(cut.id) && (
                            <div style={{
                              position: 'absolute',
                              top: 0, left: 0, right: 0, bottom: 0,
                              margin: `0 ${subtitleConfig.safeZoneMargin}%`,
                              borderLeft: '1px dashed rgba(255, 255, 0, 0.3)',
                              borderRight: '1px dashed rgba(255, 255, 0, 0.3)',
                              pointerEvents: 'none',
                              zIndex: 0
                            }} />
                          )}
                          <div key={cut.id} className="text-element" style={{
                            position: 'absolute',
                            top: `${subtitleConfig.verticalOffset}%`,
                            left: '50%', transform: 'translate(-50%, -50%)',
                            fontSize: `${subtitleConfig.fontSize}px`,
                            fontFamily: `'${subtitleConfig.fontFamily}', sans-serif`,
                            fontWeight: 'bold',
                            textAlign: 'center',
                            width: `${100 - (subtitleConfig.safeZoneMargin * 2)}%`, // Apply Safe Zone Width
                            pointerEvents: 'none',
                            display: 'flex',
                            justifyContent: 'center',
                            // Debug / Visual Guide for safe zone (only when selected?)
                            // outline: '1px dashed rgba(255,255,0,0.3)' 
                          }}>
                            <span style={{
                              padding: subtitleConfig.useBackground ? `${subtitleConfig.paddingY}px ${subtitleConfig.paddingX}px` : 0,
                              borderRadius: subtitleConfig.useBackground ? `${subtitleConfig.borderRadius}px` : 0,
                              background: subtitleConfig.useBackground
                                ? `${subtitleConfig.backgroundColor}${Math.round(subtitleConfig.backgroundOpacity * 255).toString(16).padStart(2, '0')}`
                                : 'transparent',
                              // Text Color & Gradient
                              color: subtitleConfig.useGradient ? 'transparent' : subtitleConfig.primaryColor,
                              backgroundImage: subtitleConfig.useGradient
                                ? `linear-gradient(${subtitleConfig.gradientAngle}deg, ${subtitleConfig.gradientStops.map(s => `${s.color} ${s.offset}%`).join(', ')})`
                                : 'none',
                              WebkitBackgroundClip: subtitleConfig.useGradient ? 'text' : 'unset',
                              // Outline - Only if enabled
                              // Outline - proper outward stroke using paint-order
                              WebkitTextStroke: subtitleConfig.useOutline ? `${subtitleConfig.outlineWidth * 2}px ${subtitleConfig.outlineColor}` : 'unset',
                              paintOrder: 'stroke fill',
                              WebkitPaintOrder: 'stroke fill', // Important for reliable "outward" look

                              // Shadow - Only if enabled (independent of outline now)
                              textShadow: subtitleConfig.useShadow ? `${subtitleConfig.shadowOffsetX}px ${subtitleConfig.shadowOffsetY}px ${subtitleConfig.shadowBlur}px ${subtitleConfig.shadowColor}` : 'none',
                              WebkitFontSmoothing: 'antialiased',
                              display: 'inline-block',
                              whiteSpace: 'pre-wrap'
                            } as React.CSSProperties}>
                              {cut.label}
                            </span>
                          </div>
                        </React.Fragment>
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
                              <button
                                className="btn-primary-sm"
                                style={{ cursor: 'pointer', background: '#d32f2f', border: 'none' }}
                                onClick={() => relinkInputRef.current?.click()}
                              >
                                🚀 重新連結影片
                              </button>
                              <input
                                type="file"
                                hidden
                                ref={relinkInputRef}
                                onChange={handleFileUpload}
                                accept="video/*"
                              />
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
                            <button
                              className="btn-primary-sm"
                              style={{ padding: '10px 24px', fontSize: 14, cursor: 'pointer', border: 'none' }}
                              onClick={() => fileInputRef.current?.click()}
                            >
                              點擊匯入或拖曳影片至此
                            </button>
                            <input
                              type="file"
                              hidden
                              ref={fileInputRef}
                              onChange={handleFileUpload}
                              accept="video/*,audio/*,image/*,.mkv,.ts,.flv"
                            />
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
                  {/* Hand Tool */}
                  <div
                    className={`tool-btn ${activeTool === 'hand' ? 'active' : ''}`}
                    title="手形工具 (H) - 拖拽平移時間軸"
                    onClick={() => setActiveTool('hand')}
                  >
                    <Hand size={18} />
                  </div>
                </div>

                {/* Actions Group */}
                <div className="tool-group">
                  <div
                    className="tool-btn action-btn-accent"
                    title="分割 (Cmd+K) - 在播放頭位置切割"
                    onClick={() => handleSplit()}
                    style={{ color: '#3ea6ff' }}
                  >
                    <SplitSquareHorizontal size={18} />
                  </div>
                  <div
                    className="tool-btn danger"
                    title="刪除 (Delete) - 移除選定片段"
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
                      const duration = 3;
                      const start = currentTime;
                      const end = start + duration;

                      // Smart Track Allocation
                      let targetTrackId = -1;
                      // Check existing text tracks (id > 0)
                      const textTracks = videoTracks.filter(t => t.type === 'text');

                      for (const track of textTracks) {
                        const hasOverlap = cuts.some(c => c.trackId === track.id && !(c.end <= start || c.start >= end));
                        if (!hasOverlap) {
                          targetTrackId = track.id;
                          break;
                        }
                      }

                      // If all occupied or none, create new track
                      if (targetTrackId === -1) {
                        const newId = videoTracks.length > 0 ? Math.max(...videoTracks.map(t => t.id)) + 1 : 1;
                        setVideoTracks(prev => [...prev, { id: newId, type: 'text', name: `T${newId}`, visible: true, locked: false }]);
                        targetTrackId = newId;
                      }

                      const newCut: Cut = {
                        id: Math.random().toString(36).substr(2, 9),
                        start,
                        end,
                        sourceStart: 0,
                        sourceEnd: 0,
                        label: '文字圖層',
                        trackId: targetTrackId
                      };
                      setCuts(prev => [...prev, newCut]);
                      setActiveTool('select');
                    }}
                  >
                    <Type size={18} />
                  </div>
                  <div
                    className={`tool-btn ${isMagnetEnabled ? 'active' : ''}`}
                    title="磁吸防重疊模組 (S)"
                    onClick={() => setIsMagnetEnabled(!isMagnetEnabled)}
                    style={{
                      color: isMagnetEnabled ? '#10b981' : undefined,
                      background: isMagnetEnabled ? 'rgba(16,185,129,0.1)' : undefined
                    }}
                  >
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
                  {/* Track Add Toolbar Optimized */}
                  <div style={{
                    display: 'flex',
                    borderBottom: '1px solid #333',
                    background: '#1a1a1a',
                    height: 24,
                    alignItems: 'center',
                    padding: '0 8px',
                    gap: 6
                  }}>
                    <span style={{ fontSize: 9, fontWeight: 800, color: '#555', letterSpacing: 1 }}>ADD</span>
                    <div style={{ display: 'flex', gap: 2, background: '#111', padding: 2, borderRadius: 4 }}>
                      <button
                        onClick={() => handleAddTrack('text')}
                        title="新增字幕軌"
                        className="track-add-btn"
                        style={{ background: 'none', border: 'none', color: '#6366f1', cursor: 'pointer', padding: 3, display: 'flex', borderRadius: 3 }}
                      >
                        <Type size={11} strokeWidth={2.5} />
                      </button>
                      <button
                        onClick={() => handleAddTrack('video')}
                        title="新增影像軌"
                        className="track-add-btn"
                        style={{ background: 'none', border: 'none', color: '#3ea6ff', cursor: 'pointer', padding: 3, display: 'flex', borderRadius: 3 }}
                      >
                        <Video size={11} strokeWidth={2.5} />
                      </button>
                      <button
                        onClick={() => handleAddTrack('audio')}
                        title="新增音訊軌"
                        className="track-add-btn"
                        style={{ background: 'none', border: 'none', color: '#3ea6ff', cursor: 'pointer', padding: 3, display: 'flex', borderRadius: 3 }}
                      >
                        <Music size={11} strokeWidth={2.5} />
                      </button>
                    </div>
                  </div>
                  {sortedTracks.map(track => {
                    const isThin = track.type === 'text';
                    const theme = getTrackTheme(track.type);
                    return (
                      <div
                        key={track.id}
                        className={`track-header-item ${track.locked ? 'locked' : ''}`}
                        style={{
                          height: isThin ? 24 : 60,
                          opacity: track.visible ? 1 : 0.5,
                          marginBottom: 1
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: isThin ? 0 : 4, alignItems: 'center', width: '100%' }}>
                          <div style={{
                            background: theme.secondary,
                            padding: '1px 6px',
                            borderRadius: 4,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4
                          }}>
                            <span style={{
                              color: theme.primary,
                              fontWeight: 800,
                              fontSize: 9,
                              fontFamily: 'monospace'
                            }}>{track.name}</span>
                          </div>

                          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                            <div
                              onClick={() => toggleTrackVisibility(track.id)}
                              className="track-control-btn"
                              style={{ color: track.visible ? '#fff' : '#444' }}
                            >
                              {track.visible ? '👁️' : '🚫'}
                            </div>
                            {!isThin && (
                              <div
                                onClick={() => toggleTrackLock(track.id)}
                                className="track-control-btn"
                                style={{ color: track.locked ? '#ef4444' : '#444' }}
                              >
                                {track.locked ? '🔒' : '🔓'}
                              </div>
                            )}
                            <div
                              onClick={() => handleDeleteTrack(track.id)}
                              className="track-control-btn delete-track"
                              title="刪除軌道"
                            >
                              <X size={10} />
                            </div>
                          </div>
                        </div>
                        {track.type === 'video' && !isThin && (
                          <div style={{ display: 'flex', gap: 2 }}>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#3ea6ff', opacity: 0.5 }}></div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>


                {/* Tracks Area */}
                <div className="timeline-tracks-area"
                  ref={timelineContainerRef}
                  onWheel={handleWheelZoom}
                >
                  {/* Ruler */}
                  <div className="timeline-ruler-container"
                    style={{
                      minWidth: '100%',
                      width: Math.max(100, duration * zoomLevel) + 'px',
                      cursor: timelineCursor
                    }}
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
                    style={{
                      minWidth: '100%',
                      width: Math.max(100, duration * zoomLevel) + 'px',
                      cursor: timelineCursor
                    }}
                    onMouseDown={handleTimelineMouseDown}
                    ref={timelineRef}
                  >
                    {/* Playhead Line */}
                    <div className="playhead-marker" style={{ left: currentTime * zoomLevel }}>
                      {/* Floating time code */}
                      <div style={{ position: 'absolute', top: -14, left: 4, background: '#3ea6ff', color: 'white', padding: '1px 4px', borderRadius: 2, fontSize: 8, fontWeight: 'bold', pointerEvents: 'none', whiteSpace: 'nowrap' }}>
                        {currentTime.toFixed(2)}s
                      </div>
                    </div>

                    {/* Header Alignment Spacer */}
                    <div style={{ height: 24, borderBottom: '1px solid #333', background: 'rgba(255,255,255,0.02)' }}></div>

                    {/* Dynamic Tracks Loop */}
                    {sortedTracks.map(track => {
                      // For A1 (Audio) track, we also show V1 (Video) clips as "Audio"
                      const trackCuts = track.id === 99
                        ? cuts.filter(c => c.trackId === track.id || c.trackId === 0)
                        : cuts.filter(c => c.trackId === track.id);

                      const isThin = track.type === 'text';

                      return (
                        <div
                          key={track.id}
                          data-track-id={track.id}
                          className={`track-lane ${isThin ? 'thin' : ''}`}
                          style={{
                            height: isThin ? 24 : 48,
                            background: track.id % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)',
                            borderBottom: '1px solid #333',
                            position: 'relative',
                            display: 'flex',
                            alignItems: 'center'
                          }}
                        >
                          {/* Ghost Overlay for New Asset Dragging */}
                          {dragState.type === 'new-asset' && dragState.ghostTime !== undefined && dragState.ghostTrackId === track.id && (
                            <div
                              style={{
                                position: 'absolute',
                                left: dragState.ghostTime * zoomLevel,
                                width: (dragState.newAssetDuration || 5) * zoomLevel,
                                height: isThin ? 16 : 36,
                                top: isThin ? 4 : 6,
                                background: 'rgba(255, 255, 255, 0.2)',
                                border: '1px dashed rgba(255, 255, 255, 0.6)',
                                pointerEvents: 'none',
                                zIndex: 100
                              }}
                            />
                          )}
                          {trackCuts.map(cut => (
                            <div
                              key={cut.id + (track.id === 99 ? '-audio' : '')} // Unique key for ghost
                              data-cut-id={cut.id}
                              className={`clip-block ${selectedCutIds.includes(cut.id) ? 'selected' : ''}`}
                              style={{
                                position: 'absolute',
                                left: cut.start * zoomLevel,
                                width: Math.max(10, (cut.end - cut.start) * zoomLevel) + 'px',
                                background: getTrackTheme(track.type).primary,
                                border: `1px solid ${getTrackTheme(track.type).border}`,
                                opacity: track.visible ? 0.9 : 0.4,
                                height: isThin ? '16px' : '36px',
                                top: isThin ? '4px' : '6px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                overflow: 'visible',
                                borderColor: selectedCutIds.includes(cut.id) ? '#fff' : undefined,
                                boxShadow: selectedCutIds.includes(cut.id) ? '0 0 10px rgba(255,255,255,0.4)' : 'none',
                                zIndex: selectedCutIds.includes(cut.id) ? 10 : 1,
                                cursor: activeTool === 'select' ? 'move' : 'inherit'
                              }}
                              onMouseDown={(e) => !track.locked && handleClipMouseDown(e, cut, 'move')}
                            >
                              {!track.locked && (
                                <>
                                  <div
                                    className="clip-handle-area"
                                    style={{
                                      position: 'absolute', left: 0, width: 8, height: '100%',
                                      cursor: 'ew-resize', zIndex: 20,
                                      background: 'rgba(255,255,255,0.2)',
                                      borderRight: '1px solid rgba(0,0,0,0.1)'
                                    }}
                                    onMouseDown={(e) => handleClipMouseDown(e, cut, 'trim-start')}
                                  />
                                  <div
                                    className="clip-handle-area"
                                    style={{
                                      position: 'absolute', right: 0, width: 8, height: '100%',
                                      cursor: 'ew-resize', zIndex: 20,
                                      background: 'rgba(255,255,255,0.2)',
                                      borderLeft: '1px solid rgba(0,0,0,0.1)'
                                    }}
                                    onMouseDown={(e) => handleClipMouseDown(e, cut, 'trim-end')}
                                  />


                                </>
                              )}

                              <span className="clip-label" style={{
                                fontSize: isThin ? '9px' : '11px',
                                padding: '0 10px',
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                maxWidth: '100%',
                                pointerEvents: 'none'
                              }}>{cut.label}</span>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>





          {/* Export Sidebar Panel */}
          {showExportModal && (
            <ExportModal
              cuts={cuts}
              videoUrl={videoUrl}
              isProcessing={isProcessing}
              onClose={() => setShowExportModal(false)}
              onExport={handleExportVideo}
            />
          )}

          {/* 6. GLOBAL PROCESSING OVERLAY */}
          {isProcessing && currentJobStatus && (
            <div style={{
              position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
              background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
              zIndex: 9999, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', color: 'white'
            }}>
              <div style={{ width: 400, textAlign: 'center' }}>
                <div style={{ marginBottom: 24, position: 'relative', display: 'inline-block' }}>
                  <div style={{ width: 80, height: 80, borderRadius: '50%', border: '4px solid #333', borderTopColor: '#3ea6ff' }} className="spin" />
                  <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', fontWeight: 800, fontSize: 18, color: '#3ea6ff' }}>
                    {currentJobStatus.progress}%
                  </div>
                </div>

                <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>AI 處理中...</h2>
                <p style={{ color: '#aaa', fontSize: 14, marginBottom: 32, minHeight: 40 }}>{currentJobStatus.message}</p>

                {/* Progress Bar Container */}
                <div style={{ width: '100%', height: 6, background: '#222', borderRadius: 3, overflow: 'hidden', marginBottom: 12 }}>
                  <div style={{
                    height: '100%', background: 'linear-gradient(90deg, #3ea6ff, #007aff)',
                    width: `${currentJobStatus.progress}%`, transition: 'width 0.4s ease-out'
                  }} />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
                  <span>{
                    currentJobStatus.step === 'idle' ? '待機' :
                      currentJobStatus.step === 'upload' ? '上傳中' :
                        currentJobStatus.step === 'audio_extract' ? '音訊提取' :
                          currentJobStatus.step === 'model_init' ? '模型初始化' :
                            currentJobStatus.step === 'transcribing' ? '語音辨識中' :
                              currentJobStatus.step === 'optimizing' ? '優化斷句' :
                                currentJobStatus.step === 'translating' ? '翻譯轉換' :
                                  currentJobStatus.step === 'done' ? '完成' :
                                    currentJobStatus.step === 'error' ? '出錯' :
                                      currentJobStatus.step === 'init' ? '啟動中' :
                                        currentJobStatus.step
                  }</span>
                  <span>估計時間中...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )
      }
      {/* 7. MARQUEE OVERLAY */}
      {
        marqueeRect && (
          <div style={{
            position: 'fixed',
            left: Math.min(marqueeRect.startX, marqueeRect.currX),
            top: Math.min(marqueeRect.startY, marqueeRect.currY),
            width: Math.abs(marqueeRect.currX - marqueeRect.startX),
            height: Math.abs(marqueeRect.currY - marqueeRect.startY),
            background: 'rgba(62,166,255,0.2)',
            border: '1px solid #3ea6ff',
            pointerEvents: 'none',
            zIndex: 1000
          }}></div>
        )
      }
    </div >
  );
}

export default App;
