import React from 'react';
import { Scissors, Plus, RotateCcw, Save } from 'lucide-react';

interface WelcomeScreenProps {
  onNewProject: () => void;
  onResumeProject: () => void;
  onImportProject: (e: React.ChangeEvent<HTMLInputElement>) => void;
  hasExistingProject: boolean;
}

export function WelcomeScreen({ onNewProject, onResumeProject, onImportProject, hasExistingProject }: WelcomeScreenProps) {
  return (
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
          onClick={onNewProject}
          style={{ width: 200, height: 160, background: '#222', borderRadius: 12, border: '1px solid #333', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'all 0.2s' }}
          className="welcome-card"
        >
          <Plus size={40} color="#3ea6ff" style={{ marginBottom: 16 }} />
          <span style={{ fontWeight: 600, fontSize: 16 }}>建立新專案</span>
          <span style={{ fontSize: 12, color: '#666', marginTop: 8 }}>Start New Project</span>
        </div>

        <div
          onClick={onResumeProject}
          style={{ width: 200, height: 160, background: '#222', borderRadius: 12, border: '1px solid #333', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'all 0.2s', opacity: hasExistingProject ? 1 : 0.5, pointerEvents: hasExistingProject ? 'auto' : 'none' }}
          className="welcome-card"
        >
          <RotateCcw size={40} color="#3ea6ff" style={{ marginBottom: 16 }} />
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
          <input type="file" hidden accept=".agpro,.json" onChange={onImportProject} />
        </label>
      </div>

      <div style={{ marginTop: 64, color: '#444', fontSize: 12 }}>
        v1.0.0 Alpha
      </div>
    </div>
  );
}
