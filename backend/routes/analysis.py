"""Video analysis routes: Gemini highlights, face detection, silence detection."""

import json
import os
import shutil
import subprocess
import time

import cv2
import mediapipe as mp
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as genai

from backend import models
from backend.config import UPLOAD_DIR
from backend.utils.silence import detect_silence_ffmpeg
from backend.utils.subtitles import parse_time

router = APIRouter()


@router.post("/analyze-video")
def analyze_video(
    file: UploadFile = File(...),
    api_key: str = Form(None),
    model_name: str = Form("gemini-1.5-flash-latest"),
    instruction: str = Form("Highlight interesting moments"),
    target_count: int = Form(None),
    target_duration: int = Form(None)
):
    """Analyze video with Gemini AI and extract highlight clips."""
    video_path = None
    try:
        # 1. Config Gemini
        effective_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not effective_key:
            raise HTTPException(
                status_code=400,
                detail="API key required: provide via request or set GEMINI_API_KEY environment variable"
            )
        genai.configure(api_key=effective_key)

        # 2. Save video temporarily
        video_path = f"{UPLOAD_DIR}/temp_analyze_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"🧠 Analysis started: {file.filename} using {model_name}")

        # 3. Upload to Gemini
        video_file = genai.upload_file(path=video_path)
        print(f"📤 Uploaded to Gemini: {video_file.name}")

        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Gemini failed to process video.")

        # Build constraint text
        constraint_text = ""
        if target_count:
            constraint_text += f"\n- MANDATORY: You MUST find exactly {target_count} clips. No more, no less."
        if target_duration:
            constraint_text += f"\n- MANDATORY: Each clip MUST be exactly {target_duration} seconds long (end - start = {target_duration})."
            constraint_text += f"\n- STRICTLY ADHERE to the duration of {target_duration}s. If a scene is longer, cut it at the {target_duration}s mark."

        prompt = f"""
        You are a professional video editor. Analyze the video and extract exactly the best clips based on this instruction:
        "{instruction}"
        {constraint_text}

        Return the result strictly as a JSON list of objects.
        Each object must have:
        - "start": start time in SECONDS (number, e.g., 12.5) . DO NOT use MM:SS format.
        - "end": end time in SECONDS (number, e.g., 25.0). MUST be start + {target_duration if target_duration else "duration"}.
        - "label": A short description of the clip in Traditional Chinese (STRICTLY MAX 10 characters).

        Example:
        [
            {{ "start": 10.5, "end": 20.0, "label": "開場介紹" }},
            {{ "start": 65.0, "end": 90.0, "label": "重點精華" }}
        ]
        """

        # Model fallback logic
        candidate_models = [
            model_name,
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
        ]
        candidate_models = list(dict.fromkeys(candidate_models))

        response = None
        last_error = None
        generation_config = {"response_mime_type": "application/json"}

        for model_try in candidate_models:
            if not model_try:
                continue
            try:
                print(f"🔄 Trying model: {model_try}...")
                gemini = genai.GenerativeModel(model_try)
                response = gemini.generate_content(
                    [prompt, video_file],
                    generation_config=generation_config
                )
                print(f"✅ Success with: {model_try}")
                break
            except Exception as e:
                print(f"⚠️ Failed with {model_try}: {e}")
                last_error = e
                continue

        if not response:
            raise ValueError(f"All model attempts failed. Last error: {last_error}")

        # Parse JSON
        cuts = json.loads(response.text)

        if isinstance(cuts, list):
            for cut in cuts:
                start_t = parse_time(cut.get("start", 0))
                cut["start"] = start_t
                label = cut.get("label", "片段")
                if len(label) > 10:
                    cut["label"] = label[:10]

                if target_duration and target_duration > 0:
                    cut["end"] = round(start_t + target_duration, 2)
                else:
                    cut["end"] = parse_time(cut.get("end", 0))

        # Cleanup
        genai.delete_file(video_file.name)

        return cuts
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


@router.post("/detect-face-clip")
def detect_face_clip(
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...)
):
    """Detect face position in a video clip."""
    try:
        temp_path = f"{UPLOAD_DIR}/temp_preview_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        mid_point = start + (end - start) / 2
        cap = cv2.VideoCapture(temp_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_point * 1000)
        ret, frame = cap.read()
        cap.release()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not ret:
            return {"faceCenterX": 0.5}

        h, w, _ = frame.shape
        face_center_x = 0.5
        face_detector = models.face_detector

        # MediaPipe Tasks API
        if hasattr(face_detector, 'detect'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = face_detector.detect(mp_image)
            if detection_result.detections:
                bbox = detection_result.detections[0].bounding_box
                face_center_x = (bbox.origin_x + bbox.width / 2) / w

        # Legacy API
        elif hasattr(face_detector, 'process'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                face_center_x = bbox.xmin + (bbox.width / 2)

        # OpenCV Fallback
        elif face_detector == "opencv_fallback":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if models.face_cascade:
                faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    (x, y, wf, hf) = faces[0]
                    face_center_x = (x + wf / 2) / w

        print(f"👁️ Preview Face Detect: {face_center_x:.2f}")
        return {"faceCenterX": face_center_x}

    except Exception as e:
        print(f"❌ Preview Detect Error: {e}")
        return {"faceCenterX": 0.5}


@router.post("/detect-silence")
def detect_silence_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(-30.0),
    min_duration: float = Form(0.5),
    padding: float = Form(0.1)
):
    """Detect silence in uploaded audio/video file."""
    try:
        temp_path = os.path.join(UPLOAD_DIR, f"silence_{int(time.time())}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"🤫 Analyzing silence: {file.filename} (Threshold: {threshold}dB, Min: {min_duration}s)")

        silences = detect_silence_ffmpeg(temp_path, noise_db=threshold, duration=min_duration)

        # Get total duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", temp_path],
            stdout=subprocess.PIPE, text=True
        )
        total_duration = float(probe.stdout.strip())

        # Invert to get KEEP segments (Speech)
        keep_segments = []
        current_pos = 0.0

        for s_start, s_end in silences:
            cut_end = max(0, s_start + padding)
            cut_start = min(total_duration, s_end - padding)

            speech_dur = cut_end - current_pos
            if speech_dur > 0.1:
                keep_segments.append({
                    "start": round(current_pos, 2),
                    "end": round(cut_end, 2),
                    "label": "Speech"
                })

            current_pos = cut_start

        if current_pos < total_duration:
            keep_segments.append({
                "start": round(current_pos, 2),
                "end": round(total_duration, 2),
                "label": "Speech"
            })

        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"✅ Found {len(keep_segments)} speech segments")
        return keep_segments

    except Exception as e:
        print(f"❌ Detect Silence Failed: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})
