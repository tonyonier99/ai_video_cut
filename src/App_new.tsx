
return (
    <div className="app-container">
        {/* System Status Banner */}
        {systemStatus.status !== 'ready' && (
            <div className="system-status-banner glass">
                <div className="status-info">
                    <span className="pulse-dot"></span>
                    <span className="status-message">{systemStatus.message}</span>
                    <span className="status-percent">{systemStatus.progress}%</span>
                </div>
                <div className="status-progress-bar">
                    <div className="status-progress-fill" style={{ width: `${systemStatus.progress}%` }}></div>
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
                    style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.8)', zIndex: 9999, display: 'flex', justifyContent: 'center', alignItems: 'center' }}
                >
                    <motion.div
                        className="modal-content glass"
                        onClick={e => e.stopPropagation()}
                        style={{ width: '90%', maxWidth: '300px', padding: 20, borderRadius: 16 }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                            <h3>預覽片段</h3>
                            <button onClick={() => setPreviewUrl(null)} className="btn-ghost"><X size={20} /></button>
                        </div>
                        <video src={previewUrl} controls autoPlay style={{ width: '100%', borderRadius: 8 }} />
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>

        {/* Header */}
        <header className="header glass">
            <div className="logo">
                <Scissors className="icon-primary" size={24} />
                <span>Antigravity Cut</span>
            </div>
        </header>

        <main className="main-content-new">
            <div className="top-row">
                {/* 1. Left Controls Panel */}
                <aside className="controls-panel glass">
                    <div className="controls-scroll">
                        <div className="controls-section">
                            <div className="section-header">
                                <Film size={16} className="icon-primary" />
                                <span>專案設定</span>
                                <button onClick={() => setShowApiSettings(!showApiSettings)} className={`toggle-btn-sm ${apiKey ? 'active' : ''}`}><Key size={14} /></button>
                            </div>
                            <AnimatePresence>
                                {showApiSettings && (
                                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
                                        <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Gemini API Key..." className="api-input-sm" />
                                        <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="model-select-sm" style={{ marginTop: '8px', width: '100%' }}>
                                            <option value="gemini-3.0-flash">Gemini 3.0 Flash</option>
                                            <option value="gemini-3.0-pro">Gemini 3.0 Pro</option>
                                            <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                                            <option value="gemini-2.0-flash-exp">Gemini 2.0 Flash-Exp</option>
                                        </select>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                            <div className="settings-row">
                                <div className="setting-mini"><label>片段數</label><input type="number" value={targetCount} onChange={(e) => setTargetCount(Number(e.target.value))} /></div>
                                <div className="setting-mini"><label>長度(秒)</label><input type="number" value={targetDuration} onChange={(e) => setTargetDuration(Number(e.target.value))} /></div>
                            </div>

                            <div style={{ marginBottom: '12px' }}>
                                <textarea
                                    className="instruction-input-sm"
                                    placeholder="輸入剪輯指令 (例如：找出所有精彩的擊殺畫面)..."
                                    value={instruction}
                                    onChange={(e) => setInstruction(e.target.value)}
                                    style={{ marginBottom: '12px', width: '100%' }}
                                />
                            </div>

                            <button onClick={handleMakeVideo} disabled={!videoFile || isProcessing} className="btn-action-sm" style={{ marginBottom: '16px' }}>
                                {isProcessing ? <><Loader2 className="loader" size={14} /> 處理中...</> : <>1. AI 分析影片</>}
                            </button>

                            <div style={{ marginBottom: '12px' }}>
                                <div className="section-header" style={{ marginBottom: '8px', fontSize: '11px', color: '#666' }}>
                                    <Languages size={14} className="icon-primary" />
                                    <span>AI 核心功能系統</span>
                                </div>
                                <div className="checkbox-group">
                                    <label className="checkbox-sm">
                                        <input type="checkbox" checked={isFaceTracking} onChange={(e) => setIsFaceTracking(e.target.checked)} />
                                        <span>MediaPipe 人臉追蹤</span>
                                    </label>
                                    {isFaceTracking && (
                                        <div className="indent settings-block-mini">
                                            <div className="mini-row">
                                                <label>縮放</label>
                                                <input type="number" step="0.1" style={{ width: '40px' }} value={trackZoom} onChange={e => setTrackZoom(parseFloat(e.target.value))} />
                                                <label style={{ marginLeft: '8px' }}>信心</label>
                                                <input type="range" min="0.1" max="0.9" step="0.1" value={mpMinDetectionCon} onChange={e => setMpMinDetectionCon(parseFloat(e.target.value))} />
                                            </div>
                                        </div>
                                    )}
                                    <label className="checkbox-sm">
                                        <input type="checkbox" checked={isStudioSound} onChange={(e) => setIsStudioSound(e.target.checked)} />
                                        <span>DFN3 頂級降噪</span>
                                    </label>
                                    {isStudioSound && (
                                        <div className="indent settings-block-mini">
                                            <div className="mini-row">
                                                <label>強度</label>
                                                <input type="range" min="0" max="100" value={dfn3Strength} onChange={e => setDfn3Strength(parseInt(e.target.value))} style={{ flex: 1 }} />
                                                <span className="unit-hint">{dfn3Strength}%</span>
                                            </div>
                                        </div>
                                    )}
                                    <label className="checkbox-sm">
                                        <input type="checkbox" checked={isAutoCaption} onChange={(e) => setIsAutoCaption(e.target.checked)} />
                                        <span>生成 SRT</span>
                                    </label>
                                    {isAutoCaption && (
                                        <>
                                            <label className="checkbox-sm indent"><input type="checkbox" checked={isTranslate} onChange={(e) => setIsTranslate(e.target.checked)} /><span>翻譯繁中</span></label>
                                            <label className="checkbox-sm indent"><input type="checkbox" checked={isBurnCaptions} onChange={(e) => setIsBurnCaptions(e.target.checked)} /><span>燒錄字幕</span></label>
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>

                        {isBurnCaptions && (
                            <div className="controls-section">
                                <div className="section-header"><Wand2 size={16} className="icon-primary" /><span>字幕樣式</span></div>
                                <div className="style-block">
                                    <div className="style-header">🔤 字體與排版</div>
                                    <div className="style-row-inline" style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
                                        <label className="btn-upload-sm" style={{ flex: 1, textAlign: 'center' }}>
                                            📁 上傳
                                            <input type="file" accept=".ttf,.otf" onChange={handleFontUpload} style={{ display: 'none' }} />
                                        </label>
                                        <select value={subtitleFontName} onChange={(e) => setSubtitleFontName(e.target.value)} className="style-select" style={{ flex: 2 }}>
                                            {fontList.map(f => <option key={f} value={f}>{f}</option>)}
                                        </select>
                                    </div>
                                    <div className="style-row-triple" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px' }}>
                                        <div className="style-mini-item"><span>大小</span><input type="number" value={subtitleFontSize} onChange={e => setSubtitleFontSize(parseInt(e.target.value))} /></div>
                                        <div className="style-mini-item"><span>行字</span><input type="number" value={subtitleCharsPerLine} onChange={e => setSubtitleCharsPerLine(parseInt(e.target.value))} /></div>
                                        <div className="style-mini-item"><span>邊粗</span><input type="number" value={subtitleOutlineWidth} onChange={e => setSubtitleOutlineWidth(parseInt(e.target.value))} /></div>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="controls-section">
                            <div className="section-header"><Scissors size={16} className="icon-primary" /><span>裁切片段</span></div>
                            <div className="cuts-list">
                                {cuts.map(cut => (
                                    <div key={cut.id} className="cut-card">
                                        <div className="cut-header">
                                            <input className="cut-label-input" value={cut.label} onChange={(e) => updateCutLabel(cut.id, e.target.value)} />
                                            <button className="btn-icon-xs" onClick={() => setCuts(cuts.filter(c => c.id !== cut.id))}><Trash2 size={12} /></button>
                                        </div>
                                        <div className="cut-actions">
                                            <button className={`btn-preview-xs ${previewingId === cut.id ? 'active' : ''}`} onClick={() => handlePreview(cut)} disabled={isPreviewLoading}>
                                                {previewingId === cut.id ? <Loader2 className="spin" size={10} /> : <Video size={10} />} 預覽
                                            </button>
                                            <div className="time-inputs">
                                                <input type="number" step="0.1" value={cut.start.toFixed(1)} onChange={(e) => updateCutTime(cut.id, 'start', parseFloat(e.target.value))} />
                                                <span>-</span>
                                                <input type="number" step="0.1" value={cut.end.toFixed(1)} onChange={(e) => updateCutTime(cut.id, 'end', parseFloat(e.target.value))} />
                                            </div>
                                        </div>
                                        {previewProgresses[cut.id] !== undefined && (
                                            <div className="cut-card-progress">
                                                <div className="cut-card-progress-bar"><div className="cut-card-progress-fill" style={{ width: `${previewProgresses[cut.id]}%` }} /></div>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="controls-footer">
                        <button onClick={() => handleExport('zip')} disabled={cuts.length === 0 || isExporting} className="btn-export">
                            {isExporting ? <Loader2 className="spin" size={16} /> : <Download size={16} />} 2. 匯出影片
                        </button>
                    </div>
                </aside>

                {/* 2. Middle Video Panel */}
                <div className="video-panel-wrapper">
                    <div className="panel-header-title">16:9 素材影片</div>
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
                                    <input type="file" accept="video/*" onChange={handleFileUpload} style={{ display: 'none' }} id="video-upload" />
                                    <label htmlFor="video-upload" style={{ cursor: 'pointer', textAlign: 'center' }}>
                                        <div className="upload-icon"><Upload size={48} /></div>
                                        <div style={{ marginTop: 16 }}>拖放影片或點擊上傳</div>
                                    </label>
                                </div>
                            )}
                            {videoUrl && (
                                <div style={{ position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)' }}>
                                    <button className="btn-round" onClick={() => { if (videoRef.current) { if (isPlaying) videoRef.current.pause(); else videoRef.current.play(); setIsPlaying(!isPlaying); } }}>
                                        {isPlaying ? <Pause size={20} fill="white" /> : <Play size={20} fill="white" />}
                                    </button>
                                </div>
                            )}
                        </div>
                        <div className="timeline-inline">
                            <div className="timeline-time">
                                {new Date(currentTime * 1000).toISOString().substr(11, 8)} / {new Date(duration * 1000).toISOString().substr(11, 8)}
                            </div>
                            <div ref={waveformRef} className="waveform-track" onClick={handleTimelineClick} />
                            <div ref={timelineRef} className="markers-track" onClick={handleTimelineClick}>
                                {cuts.map(cut => (
                                    <div
                                        key={cut.id}
                                        className="cut-marker"
                                        style={{
                                            left: `${(cut.start / (duration || 1)) * 100}%`,
                                            width: `${((cut.end - cut.start) / (duration || 1)) * 100}%`
                                        }}
                                        onClick={(e) => e.stopPropagation()}
                                    />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* 3. Right Subtitle Preview Panel */}
                <div className="preview-916-panel">
                    <div className="preview-916-header">9:16 字幕預覽</div>
                    <div className="preview-916-canvas" style={{ position: 'relative', background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
                        <div
                            className="subtitle-draggable"
                            style={{
                                position: 'absolute',
                                left: '50%',
                                bottom: `${subtitleMarginV}px`,
                                transform: 'translateX(-50%)',
                                color: subtitleColor,
                                fontSize: `${subtitleFontSize / 2}px`,
                                fontFamily: subtitleFontName,
                                textAlign: 'center',
                                width: '90%',
                                whiteSpace: 'pre-wrap'
                            }}
                        >
                            {previewText || "預覽文字內容"}
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>
);
}

export default App;
