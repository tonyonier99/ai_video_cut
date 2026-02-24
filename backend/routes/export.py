"""Export and upload routes: /process-video, /upload-proxy."""

import json
import os
import shutil
import subprocess
import time
import uuid
import zipfile

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip

from backend import models
from backend.config import UPLOAD_DIR, OUTPUT_DIR
from backend.services.whisper import ensure_whisper_model, get_transcribe_options
from backend.utils.subtitles import optimize_segments, translate_subtitles, format_timestamp

router = APIRouter()


@router.post("/upload-proxy")
def upload_proxy_endpoint(file: UploadFile = File(...)):
    """Upload video and optionally transcode to proxy MP4."""
    try:
        safe_filename = f"{int(time.time())}_{file.filename}"
        original_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        filename_lower = safe_filename.lower()
        needs_transcode = not filename_lower.endswith('.mp4') and not filename_lower.endswith('.webm')

        final_filename = safe_filename
        final_path = original_path

        if needs_transcode:
            print(f"🔄 Transcoding {safe_filename} to proxy MP4...")
            proxy_filename = f"proxy_{safe_filename}.mp4"
            proxy_path = os.path.join(UPLOAD_DIR, proxy_filename)

            cmd = [
                "ffmpeg", "-y",
                "-i", original_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                proxy_path
            ]

            subprocess.run(cmd, check=True)

            final_filename = proxy_filename
            final_path = proxy_path
            print(f"✅ Transcode complete: {proxy_path}")

        # Probe media metadata
        fps = 30.0
        width = 1920
        height = 1080
        try:
            cap = cv2.VideoCapture(final_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                cap.release()
        except Exception as probe_err:
            print(f"⚠️ Metadata probe failed: {probe_err}")

        return {
            "url": f"http://localhost:8000/uploads/{final_filename}",
            "filename": file.filename,
            "original_path": os.path.abspath(original_path),
            "proxy_path": os.path.abspath(final_path),
            "is_proxy": needs_transcode,
            "fps": round(fps, 3),
            "width": width,
            "height": height
        }
    except Exception as e:
        print(f"❌ Upload/Transcode failed: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post("/process-video")
def process_video(
    file: UploadFile = File(...),
    cuts_json: str = Form(...),
    model_name: str = Form(None),
    vertical_mode: str = Form("false"),
    aspect_ratio: str = Form("9:16"),
    face_tracking: str = Form("true"),
    auto_caption: str = Form("false"),
    translate_to_chinese: str = Form("false"),
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
    api_key: str = Form(None),
    merge_clips: str = Form("false"),
    burn_captions: str = Form("false"),
    dynamic_zoom: str = Form("false"),
    studio_sound: str = Form("false"),
    viz_tracking: str = Form("false"),
    track_zoom: float = Form(1.5),
    track_weight: float = Form(5.0),
    track_stickiness: float = Form(2.0),
    min_shot_duration: float = Form(2.0),
    output_mode: str = Form("zip"),
    subtitle_style: str = Form("classic"),
    subtitle_font_size: int = Form(90),
    subtitle_font_weight: str = Form("normal"),
    subtitle_font_style: str = Form("normal"),
    subtitle_text_color: str = Form("#FFFFFF"),
    is_text_gradient: str = Form("false"),
    text_gradient_colors: str = Form('["#FFFFFF", "#FFFFFF"]'),
    text_gradient_direction: str = Form("to right"),
    subtitle_shadow_blur: int = Form(0),
    subtitle_shadow_offset_x: int = Form(3),
    subtitle_shadow_offset_y: int = Form(3),
    subtitle_letter_spacing: float = Form(0),
    subtitle_line_height: float = Form(1.2),
    subtitle_text_transform: str = Form("none"),
    subtitle_text_align: str = Form("center"),
    subtitle_margin_v: int = Form(600),
    subtitle_font_name: str = Form("Arial"),
    subtitle_chars_per_line: int = Form(9),
    subtitle_box_opacity: int = Form(50),
    subtitle_box_padding_x: int = Form(10),
    subtitle_box_padding_y: int = Form(4),
    subtitle_box_radius: int = Form(4),
    subtitle_shadow_opacity: int = Form(80),
    dfn3_strength: int = Form(100),
    subtitle_box_color: str = Form("#000000"),
    subtitle_box_enabled: str = Form("false"),
    codec: str = Form("h264"),
    is_silence_removal: str = Form("false"),
    silence_threshold: float = Form(0.5),
    is_jump_cut_zoom: str = Form("true"),
    output_resolution: str = Form("1080p"),
    output_quality: str = Form("high"),
    srt_json: str = Form(None),
    subtitle_animation: str = Form("pop"),
    subtitle_animation_duration: int = Form(15),
    subtitle_animation_spring: float = Form(0.5),
    subtitle_outline_width: float = Form(0),
    subtitle_outline_color: str = Form("#000000"),
    subtitle_shadow_color: str = Form("#000000"),
    output_bitrate: str = Form("16M")
):
    """Process video: render clips with Remotion, apply AI tools, and export zip."""
    import google.generativeai as genai

    print("\n" + "=" * 50)
    print("🚀 [BACKEND] NEW PROCESSING REQUEST RECEIVED")
    print("=" * 50)
    print(f"📁 File: {file.filename}")
    print(f"🎬 Mode: {output_mode} | Resolution: {output_resolution} ({output_bitrate}) | Vertical: {vertical_mode}")

    try:
        if output_mode == "preview_url":
            print("ℹ️ legacy preview mode requested, ignoring...")

        effective_key = api_key if (api_key and api_key != "null") else os.environ.get('GEMINI_API_KEY')
        if effective_key:
            genai.configure(api_key=effective_key)

        # 1. Save uploaded video
        video_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Parse cuts
        cuts = json.loads(cuts_json)

        if not cuts:
            models.current_job_status = {"progress": 100, "message": "No cuts", "step": "done"}
            return {"message": "No cuts provided"}

        models.current_job_status = {"progress": 5, "message": "正在分析字幕與音訊...", "step": "transcribing"}

        # 3. Whisper Transcribe
        full_subtitles = []
        full_segments_raw = []

        model = ensure_whisper_model(whisper_model_size)
        transcribe_options = get_transcribe_options(
            whisper_language, whisper_beam_size,
            whisper_temperature, whisper_no_speech_threshold,
            whisper_condition_on_previous_text
        )
        remove_punc = str(whisper_remove_punctuation).lower() == "true"

        try:
            is_subtitle_needed = str(burn_captions).lower() == "true" or str(auto_caption).lower() == "true"
            if is_subtitle_needed:
                if srt_json and srt_json != "[]":
                    json_data = json.loads(srt_json)
                    full_segments_raw = [{"start": s['start'], "end": s['end'], "text": s['text']} for s in json_data]
                else:
                    video_clip = VideoFileClip(video_path)
                    if video_clip.audio is None:
                        video_clip.close()
                        raise ValueError("影片沒有音訊軌道，無法生成字幕。")
                    full_segments_raw = []

                    total_cuts_count = len(cuts)
                    for idx, cut in enumerate(cuts):
                        c_start = float(cut['start'])
                        c_end = float(cut['end'])

                        models.current_job_status = {
                            "progress": 10 + int((idx / total_cuts_count) * 10),
                            "message": f"正在轉錄片段 {idx + 1}/{total_cuts_count}...",
                            "step": "transcribing"
                        }

                        temp_audio_path = os.path.join(UPLOAD_DIR, f"temp_whisper_{idx}.mp3")
                        video_clip.audio.subclip(c_start, c_end).write_audiofile(temp_audio_path, logger=None)

                        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 1000:
                            result = models.whisper_model.transcribe(temp_audio_path, **transcribe_options)
                        else:
                            result = {"segments": []}

                        for seg in result["segments"]:
                            seg['start'] += c_start
                            seg['end'] += c_start
                            full_segments_raw.append(seg)

                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)

                    video_clip.close()

                raw_segments = optimize_segments(full_segments_raw, max_chars=whisper_chars_per_line, remove_punctuation=remove_punc)
                full_subtitles = [
                    {"id": str(i), "start": seg['start'], "end": seg['end'], "text": seg['text'].strip()}
                    for i, seg in enumerate(raw_segments)
                ]

            if str(translate_to_chinese).lower() == "true" and full_subtitles:
                models.current_job_status = {"progress": 25, "message": "正在優化繁體中文轉譯 (本地快速)...", "step": "translating"}
                full_subtitles = translate_subtitles(full_subtitles, api_key)
            else:
                if not is_subtitle_needed:
                    full_subtitles = []
                    full_segments_raw = []
        except Exception as e:
            print(f"⚠️ Transcription failed: {e}")
            models.current_job_status = {"progress": 15, "message": f"字幕生成失敗: {e}", "step": "error"}
            full_subtitles = []
            full_segments_raw = []

        # Clear previous exports
        for f in os.listdir(OUTPUT_DIR):
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except OSError:
                pass

        processed_files = []

        models.current_job_status = {"progress": 18, "message": "正在套用字幕樣式設定...", "step": "styling"}

        # Pre-calculate Remotion config
        try:
            gradient_colors_list = json.loads(text_gradient_colors)
        except (json.JSONDecodeError, TypeError):
            gradient_colors_list = ["#FFFFFF", "#FFFFFF"]

        remotion_config = {
            "fontSize": subtitle_font_size,
            "fontFamily": subtitle_font_name,
            "fontWeight": subtitle_font_weight,
            "fontStyle": subtitle_font_style,
            "textColor": subtitle_text_color,
            "isTextGradient": str(is_text_gradient).lower() == "true",
            "textGradientColors": gradient_colors_list,
            "textGradientDirection": text_gradient_direction,
            "outlineWidth": subtitle_outline_width,
            "outlineColor": subtitle_outline_color,
            "animation": subtitle_animation,
            "animationDuration": subtitle_animation_duration,
            "animationSpring": subtitle_animation_spring,
            "shadowColor": subtitle_shadow_color,
            "shadowBlur": subtitle_shadow_blur,
            "shadowOffsetX": subtitle_shadow_offset_x,
            "shadowOffsetY": subtitle_shadow_offset_y,
            "shadowOpacity": subtitle_shadow_opacity / 100.0,
            "letterSpacing": subtitle_letter_spacing,
            "lineHeight": subtitle_line_height,
            "textTransform": subtitle_text_transform,
            "textAlign": subtitle_text_align,
            "marginBottom": int(subtitle_margin_v),
            "isUnknownBackground": str(subtitle_box_enabled).lower() == "true",
            "backgroundColor": subtitle_box_color,
            "backgroundOpacity": subtitle_box_opacity / 100.0,
            "backgroundPaddingX": subtitle_box_padding_x,
            "backgroundPaddingY": subtitle_box_padding_y,
            "backgroundBorderRadius": subtitle_box_radius,
            "charsPerLine": int(subtitle_chars_per_line) if subtitle_chars_per_line else 0,
        }

        abs_video_path = os.path.abspath(video_path)
        total_cuts = len(cuts)

        for i, cut in enumerate(cuts):
            progress_pct = 20 + int((i / total_cuts) * 70)
            models.current_job_status = {
                "progress": progress_pct,
                "message": f"正在渲染片段 {i + 1} / {total_cuts}...",
                "step": "rendering"
            }

            start = float(cut.get('start', 0))
            end = float(cut.get('end', 0))
            raw_label = cut.get('label', f'Clip{i + 1}')
            safe_label = "".join([c for c in raw_label if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')

            output_filename = f"Highlight_{i + 1}_{safe_label}.mp4"
            output_app_path = os.path.join(OUTPUT_DIR, output_filename)
            abs_output_path = os.path.abspath(output_app_path)

            clip_subs = []
            if str(burn_captions).lower() == "true":
                clip_subs = [s for s in full_subtitles if s['start'] < end and s['end'] > start]

            # Silence Removal Logic
            visual_segments = []
            final_duration = end - start

            # Face Tracking Logic
            face_center_x = 0.5
            is_tracking_enabled = str(face_tracking).lower() in ["true", "1", "yes", "on"]

            if is_tracking_enabled and models.face_detector:
                models.current_job_status = {
                    "progress": 20 + int((i / total_cuts) * 70),
                    "message": f"正在分析人臉位置 (片段 {i + 1}/{total_cuts})...",
                    "step": "face_tracking"
                }
                try:
                    timestamps = np.arange(start, end, 0.5)
                    if len(timestamps) == 0:
                        timestamps = [start + final_duration / 2]

                    detected_centers = []

                    for ts in timestamps:
                        try:
                            rgb_frame = None
                            cap = cv2.VideoCapture(abs_video_path)
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                                ret, frame = cap.read()
                                if ret:
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                cap.release()

                            if rgb_frame is not None:
                                h, w, _ = rgb_frame.shape
                                face_detector = models.face_detector

                                if hasattr(face_detector, 'detect'):
                                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                                    detection_result = face_detector.detect(mp_image)
                                    if detection_result.detections:
                                        bbox = detection_result.detections[0].bounding_box
                                        center = bbox.origin_x + (bbox.width / 2)
                                        detected_centers.append(center / w)

                                elif hasattr(face_detector, 'process'):
                                    results = face_detector.process(rgb_frame)
                                    if results.detections:
                                        bboxC = results.detections[0].location_data.relative_bounding_box
                                        detected_centers.append(bboxC.xmin + (bboxC.width / 2))

                                elif face_detector == "opencv_fallback" and models.face_cascade:
                                    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                                    faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
                                    if len(faces) > 0:
                                        (x, y, w_f, h_f) = faces[0]
                                        detected_centers.append((x + w_f / 2) / w)
                        except Exception as e:
                            print(f"⚠️ Tracking frame error at {ts}: {e}")

                    if detected_centers:
                        face_center_x = sum(detected_centers) / len(detected_centers)
                except Exception as e:
                    print(f"⚠️ Face detection ERROR for clip {i}: {e}")

            if str(is_silence_removal).lower() == "true":
                active_segments = []
                for seg in full_segments_raw:
                    s_ov = max(start, seg['start'])
                    e_ov = min(end, seg['end'])
                    if e_ov > s_ov:
                        active_segments.append({'start': s_ov, 'end': e_ov})

                if not active_segments:
                    visual_segments = [{'startInVideo': start, 'duration': end - start, 'zoom': 1.0}]
                else:
                    merged_blocks = []
                    current_block = active_segments[0].copy()

                    for j in range(1, len(active_segments)):
                        next_seg = active_segments[j]
                        gap = next_seg['start'] - current_block['end']

                        if gap <= silence_threshold:
                            current_block['end'] = next_seg['end']
                        else:
                            merged_blocks.append(current_block)
                            current_block = next_seg.copy()
                    merged_blocks.append(current_block)

                    zoom_level = 1.0
                    total_dur = 0

                    for block in merged_blocks:
                        dur = block['end'] - block['start']
                        visual_segments.append({
                            'startInVideo': block['start'],
                            'duration': dur,
                            'zoom': zoom_level
                        })
                        total_dur += dur

                        if str(is_jump_cut_zoom).lower() == "true":
                            zoom_level = 1.15 if zoom_level == 1.0 else 1.0

                    final_duration = total_dur
            else:
                visual_segments = [{'startInVideo': start, 'duration': end - start, 'zoom': 1.0}]

            # Construct Remotion input props
            unique_id = str(uuid.uuid4())
            sanitized_video_name = f"render_source_{unique_id}.mp4"

            public_dir = os.path.abspath("public")
            os.makedirs(public_dir, exist_ok=True)
            public_video_path = os.path.join(public_dir, sanitized_video_name)

            if not os.path.exists(public_video_path):
                shutil.copy2(video_path, public_video_path)

            input_props = {
                "videoUrl": sanitized_video_name,
                "startFrom": start,
                "visualSegments": visual_segments,
                "duration": final_duration,
                "subtitles": clip_subs,
                "subtitleConfig": remotion_config,
                "isFaceTracking": is_tracking_enabled,
                "faceCenterX": face_center_x,
                "durationInFrames": int(final_duration * 30),
                "verticalMode": vertical_mode
            }

            props_file = os.path.abspath(f"temp_props_{unique_id}.json")
            with open(props_file, "w") as pf:
                json.dump(input_props, pf)

            dur_in_frames = int(final_duration * 30)

            scale = 1.0
            if output_resolution == "4k":
                scale = 2.0
            elif output_resolution == "720p":
                scale = 0.666666667

            # Map codec to Remotion codec name
            codec_map = {
                "h264": "h264",
                "prores": "prores",
                "dnxhd": "h264",  # Remotion doesn't support DNxHD natively, fallback
            }
            remotion_codec = codec_map.get(codec, "h264")

            cmd = [
                "npx", "remotion", "render",
                "src/remotion/index.ts",
                "MainComposition",
                abs_output_path,
                f"--props={props_file}",
                f"--duration={dur_in_frames}",
                f"--scale={scale}",
                f"--vbitrate={output_bitrate}",
                f"--codec={remotion_codec}",
                "--log=verbose",
                "--concurrency=1",
                "--timeout=120000"
            ]

            try:
                subprocess.run(cmd, check=True, cwd=os.getcwd())
                processed_files.append(output_app_path)
                print(f"✅ Rendered: {output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Remotion render failed for clip {i}: {e}")
            finally:
                if os.path.exists(props_file):
                    os.remove(props_file)
                if os.path.exists(public_video_path):
                    os.remove(public_video_path)

        # Generate global SRT file
        if full_subtitles:
            srt_filename = "transcription.srt"
            srt_path = os.path.join(OUTPUT_DIR, srt_filename)
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, s in enumerate(full_subtitles):
                    f.write(f"{idx + 1}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{s['text']}\n\n")
            processed_files.append(srt_path)

        if not processed_files:
            raise ValueError("Render failed: No files generated.")

        zip_filename = f"Antigravity_Shorts_{len(cuts)}_Clips.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for f in processed_files:
                if os.path.exists(f):
                    zipf.write(f, os.path.basename(f))

        models.current_job_status = {"progress": 100, "message": "處理完成！", "step": "done"}

        return {
            "status": "success",
            "download_url": f"/exports/{zip_filename}",
            "filename": zip_filename
        }

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
