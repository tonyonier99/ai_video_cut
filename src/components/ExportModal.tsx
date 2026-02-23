import { useState } from 'react';
import { Download, X, Loader2, Video, Music, SplitSquareHorizontal, Type } from 'lucide-react';
import type { Cut } from '../types';

interface ExportModalProps {
  cuts: Cut[];
  videoUrl: string | null;
  isProcessing: boolean;
  onClose: () => void;
  onExport: (resolution: string, bitrate: number, formats: string[]) => void;
}

export function ExportModal({ cuts, videoUrl, isProcessing, onClose, onExport }: ExportModalProps) {
  const [exportResolution, setExportResolution] = useState('1080p');
  const [exportBitrate, setExportBitrate] = useState(16);
  const [selectedFormats, setSelectedFormats] = useState<string[]>(['video']);

  return (
    <div className="modal-overlay" style={{ display: 'flex', alignItems: 'stretch', justifyContent: 'flex-end', zIndex: 1001, background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(2px)' }}>
      <div
        className="export-side-panel"
        style={{
          width: 400,
          background: '#121212',
          borderLeft: '1px solid #333',
          overflowY: 'auto',
          boxShadow: '-10px 0 30px rgba(0,0,0,0.5)',
          animation: 'slideInRight 0.3s ease-out'
        }}
      >
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
            onClick={onClose}
            className="btn-icon-sm"
            style={{ background: '#222', borderRadius: '50%', border: '1px solid #333' }}
          >
            <X size={16} />
          </button>
        </div>

        <div style={{ padding: 24 }}>
          {/* Project Status Summary */}
          <div className="panel-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid #333', padding: 20, borderRadius: 12, marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 16, fontSize: 13, fontWeight: 700, color: '#3ea6ff', textTransform: 'uppercase', letterSpacing: 0.5 }}>專案匯出概覽</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>總片段數</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#fff' }}>{cuts.length} <span style={{ fontSize: 11, fontWeight: 400, color: '#555' }}>segments</span></div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>預估匯出時長</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#fff' }}>
                  {Math.max(0, ...cuts.map(c => c.end)).toFixed(1)} <span style={{ fontSize: 11, fontWeight: 400, color: '#555' }}>sec</span>
                </div>
              </div>
            </div>
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #222', display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: videoUrl ? '#3ea6ff' : '#ef4444', boxShadow: videoUrl ? '0 0 8px #3ea6ff' : 'none' }} />
              <span style={{ fontSize: 11, color: '#aaa' }}>{videoUrl ? '媒體來源已連結，準備就緒' : '請注意：尚未連結原始影片素材'}</span>
            </div>
          </div>

          {/* Resolution Toggle */}
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 12, fontSize: 13, fontWeight: 600, color: '#aaa' }}>輸出解析度</label>
            <div style={{ display: 'flex', gap: 10 }}>
              {[
                { id: '4k', label: '4K', desc: 'Ultra HD' },
                { id: '1080p', label: '1080p', desc: 'Full HD' },
                { id: '720p', label: '720p', desc: 'HD Ready' }
              ].map(res => (
                <div
                  key={res.id}
                  onClick={() => {
                    setExportResolution(res.id);
                    if (res.id === '4k' && exportBitrate < 30) setExportBitrate(50);
                    if (res.id === '1080p' && exportBitrate > 30) setExportBitrate(16);
                  }}
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

          {/* Bitrate Presets */}
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 12, fontSize: 13, fontWeight: 600, color: '#aaa' }}>輸出畫質碼率</label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
              {[
                { val: 8, label: '標準', desc: '8M' },
                { val: 16, label: '高清', desc: '16M' },
                { val: 32, label: '極致', desc: '32M' },
                { val: 50, label: '大師', desc: '50M' }
              ].map(opt => (
                <div
                  key={opt.val}
                  onClick={() => setExportBitrate(opt.val)}
                  style={{
                    padding: '10px 4px', borderRadius: 10, cursor: 'pointer', transition: 'all 0.2s',
                    border: `1px solid ${exportBitrate === opt.val ? '#3ea6ff' : '#222'}`,
                    background: exportBitrate === opt.val ? 'rgba(62,166,255,0.08)' : '#1a1a1a',
                    textAlign: 'center'
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700, color: exportBitrate === opt.val ? '#3ea6ff' : '#eee' }}>{opt.label}</div>
                  <div style={{ fontSize: 9, color: exportBitrate === opt.val ? 'rgba(62,166,255,0.6)' : '#555', marginTop: 2 }}>{opt.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Format Selector */}
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 12, fontSize: 13, fontWeight: 700, color: '#aaa', textTransform: 'uppercase' }}>匯出格式選擇 (可多選)</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                { id: 'video', label: '影片 (MP4)', icon: <Video size={16} /> },
                { id: 'audio', label: '音訊 (MP3)', icon: <Music size={16} /> },
                { id: 'xml', label: '工程 (XML)', icon: <SplitSquareHorizontal size={16} /> },
                { id: 'srt', label: '字幕 (SRT)', icon: <Type size={16} /> },
              ].map(format => {
                const isSelected = selectedFormats.includes(format.id);
                return (
                  <div
                    key={format.id}
                    onClick={() => {
                      setSelectedFormats(prev =>
                        prev.includes(format.id)
                          ? prev.filter(f => f !== format.id)
                          : [...prev, format.id]
                      );
                    }}
                    style={{
                      padding: '12px', borderRadius: 12, background: isSelected ? 'rgba(62,166,255,0.08)' : '#1a1a1a',
                      border: `1px solid ${isSelected ? '#3ea6ff' : '#222'}`,
                      cursor: 'pointer', transition: 'all 0.2s',
                      display: 'flex', alignItems: 'center', gap: 10
                    }}
                  >
                    <span style={{ color: isSelected ? '#3ea6ff' : '#555' }}>{format.icon}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: isSelected ? '#eee' : '#666' }}>{format.label}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={() => onExport(exportResolution, exportBitrate, selectedFormats)}
            disabled={isProcessing || selectedFormats.length === 0}
            style={{
              width: '100%', height: 48, fontSize: 15, fontWeight: 700, borderRadius: 14,
              background: 'linear-gradient(135deg, #3ea6ff 0%, #007aff 100%)',
              boxShadow: '0 8px 20px -5px rgba(0,122,255,0.4)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
              opacity: (isProcessing || selectedFormats.length === 0) ? 0.6 : 1
            }}
          >
            {isProcessing ? <><Loader2 className="spin" size={18} /> 處理中...</> : <><Download size={18} /> 開始匯出 ({selectedFormats.length} 個檔案)</>}
          </button>

          <div style={{ marginTop: 16, textAlign: 'center', fontSize: 11, color: '#444' }}>
            處理時間取決於影片長度與選取的 AI 功能
          </div>
        </div>
      </div>
    </div>
  );
}
