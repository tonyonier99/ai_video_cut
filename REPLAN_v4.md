# Antigravity Cut v4.0 - Re-Architecture Plan

## 1. 核心目標 (Core Objective)
打造一個「Pre-Premiere」的 AI 輔助剪輯工具。使用者在此完成繁瑣的粗剪、去氣口、AI 挑選精華工作，然後選擇：
A. 匯出 XML -> 進入 Premiere Pro 進行精細後製。
B. 匯出 MP4 -> 直接由 Remotion 算繪出片（含 AI 字幕與特效）。

## 2. 功能需求 (Requirements)

### 2.1 萬能匯入 (Universal Import)
*   前端：支援拖曳上傳。
*   後端：Python FastAPI + FFmpeg。
    *   接收任意格式 (MOV, MKV, AVI, MP4...)。
    *   若瀏覽器無法播放，後端自動轉碼為 Proxy (低解析 MP4) 供預覽。
    *   **關鍵**：XML 匯出時，必須指向「原始檔案路徑」，而非 Proxy。

### 2.2 編輯器介面 (Timeline Editor UI)
模仿 CapCut / Premiere 的標準佈局：
*   **Player**：播放器、時間碼、播放控制。
*   **Timeline**：
    *   支援多個 Clips（片段）。
    *   游標 (Playhead) 控制。
    *   **操作**：
        *   `Space`：播放/暫停。
        *   `Commnad+K` / `S`：分割 (Split) 目前片段。
        *   `Delete`：刪除選中片段。
        *   `Drag`：拖曳調整片段順序（以磁性吸附為主）。
        *   `Trim`：拖曳片段邊緣調整 In/Out 點。
*   **Toolbar**：
    *   智慧去氣口 (Silence Removal)。
    *   AI 精華剪輯 (Gemini Highlights)。
    *   生成字幕 (Whisper Transcribe)。

### 2.3 AI 功能整合
*   **智慧去氣口**：
    *   後端分析 VAD (Voice Activity Detection)。
    *   前端將分析結果轉換為 Timeline 上的多個 Clips（自動刪除無聲段）。
*   **Gemini 3 API 精華剪輯**：
    *   使用者輸入：「挑選 5 支，每支 30 秒，關於...」。
    *   後端：Gemini 分析影片內容 (Video Understanding)。
    *   前端：接收 Timecodes，將 Timeline 重組為這 5 個精華片段。
*   **Whisper 字幕**：
    *   生成 SRT 數據。
    *   在 Timeline 上顯示為字幕軌 (可選)。

### 2.4 輸出與備份 (Export & Persistence)
*   **FCPXML 匯出**：
    *   生成相容於 Premiere Pro 的 XML 檔案。
    *   確保 Frame Rate 與 Timecode 準確。
*   **本地備份**：
    *   專案狀態 (Cuts, Subtitles, Settings) 自動存入 `localStorage` 或 `IndexedDB`。

## 3. 畫面佈局 (Wireframe)

```
+---------------------------------------------------------------+
|  [Logo] Antigravity Cut    [Project Name]      [Export XML/MP4]|
+---------------------------------------------------------------+
|                               |                               |
|        VIDEO PREVIEW          |       INSPECTOR / AI TAB      |
|           (Source)            |                               |
|                               |  [ Tab: 屬性 | AI | 字幕 ]   |
|                               |                               |
|        [Play Controls]        |  - Gemini Prompt Input        |
|                               |  - Enhance Switches           |
|                               |  - Subtitle Styles            |
+---------------------------------------------------------------+
|  [Toolbar: Split | Delete | Silence Remove | Undo | Redo ]    |
+---------------------------------------------------------------+
|  TIMELINE TRACKS                                              |
|  -----------------------------------------------------------  |
|  [V1] [Clip 1] [Clip 2] ...      [Clip 3] ...                 |
|  -----------------------------------------------------------  |
|  [A1] (Waveform visualization if possible)                    |
|  -----------------------------------------------------------  |
|  [Sub] [Title 1] [Title 2] ...                                |
+---------------------------------------------------------------+
```

## 4. 執行步驟 (Execution Steps)

1.  **Layout 重構**：建立新的 Grid 佈局，將現有組件拆解並重新安置。
2.  **Timeline 元件強化**：重寫 `Timeline`，從單純的「標記顯示」改為「片段操作」。支援 Split/Delete 操作。
3.  **後端對接**：確保 `server.py` 的 `/analyze-video` (Gemini) 與 `/transcribe` (Whisper) 能正確回傳數據並更新 Timeline。
4.  **XML 產生器優化**：重寫 XML 生成邏輯，支援多片段拼接 (Sequence) 的結構。

