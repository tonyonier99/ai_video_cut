"""Transcription and model status routes."""

import os
import shutil
import time

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip

from backend import models
from backend.config import UPLOAD_DIR
from backend.services.whisper import ensure_whisper_model, get_transcribe_options
from backend.utils.subtitles import optimize_segments, translate_subtitles

router = APIRouter()


@router.get("/model-status")
async def get_model_status():
    """Return current Whisper model loading status."""
    return models.model_status


@router.get("/job-status")
async def get_job_status():
    """Return current processing job status."""
    return models.current_job_status


@router.post("/transcribe")
def transcribe_only(
    file: UploadFile = File(...),
    whisper_language: str = Form("zh"),
    whisper_model_size: str = Form("turbo"),
    whisper_beam_size: int = Form(5),
    whisper_temperature: float = Form(0.0),
    whisper_remove_punctuation: str = Form("true"),
    whisper_chars_per_line: int = Form(14),
):
    """Transcribe audio from uploaded video file."""
    try:
        # 1. Save
        video_path = os.path.join(UPLOAD_DIR, f"transcribe_{int(time.time())}_{file.filename}")
        models.current_job_status = {"progress": 1, "message": "正在上傳並儲存檔案...", "step": "upload"}
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract Audio
        models.current_job_status = {"progress": 5, "message": "正在提取音訊軌道...", "step": "audio_extract"}
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            video_clip.close()
            raise ValueError("影片沒有音訊軌道")

        temp_audio_path = os.path.join(UPLOAD_DIR, f"temp_whisper_full_{int(time.time())}.mp3")
        video_clip.audio.write_audiofile(temp_audio_path, logger=None)
        video_clip.close()

        # 3. Whisper Init
        models.current_job_status = {"progress": 15, "message": f"正在載入 Whisper {whisper_model_size} 模型...", "step": "model_init"}
        model = ensure_whisper_model(whisper_model_size)

        # 4. Transcribe
        models.current_job_status = {"progress": 25, "message": "AI 正在辨識語音內容 (這可能需要幾分鐘)...", "step": "transcribing"}
        transcribe_options = get_transcribe_options(whisper_language, whisper_beam_size, whisper_temperature)
        result = model.transcribe(temp_audio_path, **transcribe_options)

        # 5. Optimize & Translate
        models.current_job_status = {"progress": 85, "message": "正在優化字幕斷句與格式...", "step": "optimizing"}
        remove_punc = str(whisper_remove_punctuation).lower() == "true"
        raw_segments = optimize_segments(result["segments"], max_chars=whisper_chars_per_line, remove_punctuation=remove_punc)

        models.current_job_status = {"progress": 95, "message": "正在轉換為繁體中文...", "step": "translating"}
        final_segments = translate_subtitles(raw_segments, None)

        # 6. Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        models.current_job_status = {"progress": 100, "message": "辨識完成！", "step": "done"}
        return final_segments

    except Exception as e:
        models.current_job_status = {"progress": 0, "message": f"辨識失敗: {str(e)}", "step": "error"}
        return JSONResponse(status_code=500, content={"detail": str(e)})
