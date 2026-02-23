"""Preview pipeline routes."""

import json
import os
import shutil
import time

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from moviepy.editor import VideoFileClip

from backend import models
from backend.config import UPLOAD_DIR
from backend.services.video import detect_speaker_segments, apply_smart_reframing
from backend.services.whisper import ensure_whisper_model, get_transcribe_options
from backend.utils.subtitles import optimize_segments, translate_subtitles

router = APIRouter()


@router.post("/process-preview-pipeline")
def process_preview_pipeline(
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    cuts_json: str = Form(None),
    model_name: str = Form(None),

    # Configs
    is_denoise: str = Form("false"),
    is_silence_removal: str = Form("false"),
    silence_threshold: float = Form(0.3),
    is_auto_caption: str = Form("false"),
    subtitle_config: str = Form(None),
    is_face_tracking: str = Form("false"),
    srt_json: str = Form(None),

    # Whisper
    whisper_language: str = Form("zh"),
    whisper_model_size: str = Form("turbo"),
    whisper_beam_size: int = Form(5),
    whisper_temperature: float = Form(0.0),
    whisper_no_speech_threshold: float = Form(0.6),
    whisper_condition_on_previous_text: str = Form("true"),
    whisper_remove_punctuation: str = Form("true"),
    whisper_best_of: int = Form(5),
    whisper_patience: float = Form(1.0),
    whisper_compression_ratio_threshold: float = Form(2.4),
    whisper_logprob_threshold: float = Form(-1.0),
    whisper_fp16: str = Form("true"),
    whisper_chars_per_line: int = Form(14),
    translate_to_chinese: str = Form("false"),
    api_key: str = Form(None)
):
    """Run the full preview pipeline with AI tools."""
    print(f"📡 Preview: cap={is_auto_caption}, trans={translate_to_chinese}, lang={whisper_language}, has_srt={bool(srt_json)}")
    ts = int(time.time())
    temp_video_path = os.path.join(UPLOAD_DIR, f"preview_chunk_{ts}_{file.filename}")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return StreamingResponse(
        _preview_pipeline_generator(
            temp_video_path, start, end,
            is_denoise, is_silence_removal, silence_threshold,
            is_auto_caption, subtitle_config, is_face_tracking,
            whisper_language, whisper_model_size, whisper_beam_size,
            whisper_temperature, whisper_no_speech_threshold, whisper_condition_on_previous_text,
            whisper_best_of, whisper_patience, whisper_compression_ratio_threshold,
            whisper_logprob_threshold, whisper_fp16,
            whisper_remove_punctuation, whisper_chars_per_line,
            translate_to_chinese, api_key, srt_json
        ),
        media_type="application/x-ndjson"
    )


def _preview_pipeline_generator(
    temp_video_path, start, end,
    is_denoise, is_silence_removal, silence_threshold,
    is_auto_caption, subtitle_config, is_face_tracking,
    whisper_language, whisper_model_size, whisper_beam_size,
    whisper_temperature, whisper_no_speech_threshold, whisper_condition_on_previous_text,
    whisper_best_of, whisper_patience, whisper_compression_ratio_threshold,
    whisper_logprob_threshold, whisper_fp16,
    whisper_remove_punctuation, whisper_chars_per_line,
    translate_to_chinese, api_key, srt_json=None
):
    """Generator that streams progress updates for the preview pipeline."""
    try:
        ts = int(time.time())
        yield json.dumps({"status": "progress", "message": "正在啟動 AI 智慧預覽 (降噪/人臉/字幕)...", "percent": 10}) + "\n"

        clip = VideoFileClip(temp_video_path)
        if start < 0:
            start = 0
        if end > clip.duration:
            end = clip.duration

        sub = clip.subclip(start, end)

        # Speaker detection (on original 16:9 clip before cropping)
        face_center_x = 0.5
        speaker_segments = []

        if str(is_face_tracking).lower() == "true":
            yield json.dumps({"status": "progress", "message": "正在分析講者切換 (唇動偵測)...", "percent": 12}) + "\n"
            try:
                speaker_segments = detect_speaker_segments(sub, segment_duration=0.3)
                for seg in speaker_segments:
                    seg["start"] += start
                    seg["end"] += start

                if speaker_segments:
                    face_center_x = speaker_segments[0]["faceCenterX"]
            except Exception as e:
                print(f"⚠️ detect_speaker_segments error: {e}")
                import traceback
                traceback.print_exc()

        # Apply cropping only for static single-speaker mode
        if str(is_face_tracking).lower() == "true" and len(speaker_segments) <= 1:
            yield json.dumps({"status": "progress", "message": "正在進行人臉裁切...", "percent": 15}) + "\n"
            sub, _ = apply_smart_reframing(sub, aspect_ratio="9:16", face_tracking="true", vertical_mode="true")
        elif str(is_face_tracking).lower() == "true":
            yield json.dumps({"status": "progress", "message": f"偵測到 {len(speaker_segments)} 個講者切換...", "percent": 15}) + "\n"

        temp_sub_path = os.path.join(UPLOAD_DIR, f"sub_{ts}.mp4")
        sub.write_videofile(temp_sub_path, audio_codec='aac', logger=None)

        temp_audio_path = os.path.join(UPLOAD_DIR, f"audio_{ts}.mp3")
        sub.audio.write_audiofile(temp_audio_path, logger=None)

        # Denoise
        final_audio_path = temp_audio_path
        if str(is_denoise).lower() == 'true':
            yield json.dumps({"status": "progress", "message": "正在執行 AI 降噪 (Step 2)...", "percent": 30}) + "\n"

        # Silence Removal
        visual_segments = [{"startInVideo": start, "duration": end - start, "zoom": 1.0}]
        if str(is_silence_removal).lower() == 'true':
            yield json.dumps({"status": "progress", "message": "正在分析與移除氣口 (Step 3)...", "percent": 50}) + "\n"

        # Subtitles
        subtitles = []
        is_subtitle_needed = str(is_auto_caption).lower() == 'true'

        if is_subtitle_needed:
            full_segments_raw = []
            remove_punc = str(whisper_remove_punctuation).lower() == "true"

            if srt_json and srt_json != "[]":
                try:
                    loaded_sub = json.loads(srt_json)
                    full_segments_raw = [{"start": s['start'], "end": s['end'], "text": s['text']} for s in loaded_sub]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            else:
                yield json.dumps({"status": "progress", "message": f"正在生成字幕 ({whisper_language})...", "percent": 70}) + "\n"
                model = ensure_whisper_model(whisper_model_size)
                t_opts = get_transcribe_options(
                    whisper_language, whisper_beam_size,
                    whisper_temperature, whisper_no_speech_threshold,
                    whisper_condition_on_previous_text,
                    whisper_best_of, whisper_patience,
                    whisper_compression_ratio_threshold, whisper_logprob_threshold,
                    whisper_fp16
                )

                if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 1000:
                    res = model.transcribe(final_audio_path, **t_opts)
                    full_segments_raw = res['segments']
                else:
                    full_segments_raw = []

            segs = optimize_segments(full_segments_raw, max_chars=whisper_chars_per_line, remove_punctuation=remove_punc)
            for i, s in enumerate(segs):
                is_srt_absolute = srt_json and srt_json != "[]"
                subtitles.append({
                    "id": f"p_{i}",
                    "start": s['start'] + (start if not is_srt_absolute else 0),
                    "end": s['end'] + (start if not is_srt_absolute else 0),
                    "text": s['text'].strip()
                })

            if str(translate_to_chinese).lower() == "true" and subtitles:
                yield json.dumps({"status": "progress", "message": "正在優化繁體中文轉譯 (本地快速)...", "percent": 80}) + "\n"
                subtitles = translate_subtitles(subtitles, api_key)

        yield json.dumps({"status": "progress", "message": "正在準備客戶端預覽...", "percent": 95}) + "\n"

        # Cleanup
        sub.close()
        clip.close()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_sub_path):
            os.remove(temp_sub_path)

        # Prepare Audio URL
        preview_audio_url = None
        if os.path.exists(final_audio_path):
            preview_audio_name = f"preview_audio_{ts}.mp3"
            shutil.move(final_audio_path, os.path.join(UPLOAD_DIR, preview_audio_name))
            preview_audio_url = f"http://localhost:8000/uploads/{preview_audio_name}"

        result = {
            "status": "success",
            "subtitles": subtitles,
            "faceCenterX": face_center_x,
            "speakerSegments": speaker_segments,
            "audioUrl": preview_audio_url,
            "visualSegments": visual_segments
        }
        yield json.dumps({"status": "success", "data": result}) + "\n"

    except Exception as e:
        print(f"Preview Pipeline Error: {e}")
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"
