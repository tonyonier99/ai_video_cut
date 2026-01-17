import os
import imageio_ffmpeg
# CRITICAL: Set FFmpeg path BEFORE importing moviepy to fix dependency issues
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# Also Add to PATH for Whisper (which calls 'ffmpeg' subprocess)
try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Create local bin dir
    local_bin = os.path.abspath("local_bin")
    os.makedirs(local_bin, exist_ok=True)
    
    # Symlink as 'ffmpeg'
    target_link = os.path.join(local_bin, "ffmpeg")
    if os.path.exists(target_link):
        os.remove(target_link) # Refresh
    os.symlink(ffmpeg_exe, target_link)
    
    # Add to PATH
    os.environ["PATH"] += os.pathsep + local_bin
    
    print(f"✅ Created Symlink and Added to PATH: {target_link}")
except Exception as e:
    print(f"⚠️ Failed to add FFmpeg to PATH: {e}")

import shutil
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Definitions
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "exports"
FONTS_DIR = "fonts"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
from pydantic import BaseModel
import json
import zipfile
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
import datetime
import google.generativeai as genai
import time
import subprocess
import mediapipe as mp
import numpy as np

# ... (Previous imports remained, but ensuring order)

def hex_to_ass(hex_color, alpha=1.0):
    """Convert #RRGGBB to &HAABBGGRR for ASS subtitles"""
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a_val = int((1.0 - alpha) * 255)
        return f"&H{a_val:02X}{b:02X}{g:02X}{r:02X}"
    except:
        return "&H00FFFFFF" # Fallback White

def wrap_text(text, max_chars=12):
    """Simple wrapping for subtitles"""
    if not text: return ""
    if max_chars <= 0: max_chars = 12 # Safety fallback
    import re
    # If CJK characters found, do character-based wrap
    if re.search(r'[\u4e00-\u9fff]', text):
        return "\n".join([text[i:i+max_chars] for i in range(0, len(text), max_chars)])
    else:
        # Standard word wrap for English
        import textwrap
        return "\n".join(textwrap.wrap(text, width=max_chars))

# Helper for robustness
def fallback_cut_video(input_path, output_path, start, end):
    """Fallback to direct FFmpeg command if MoviePy fails"""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        duration = end - start
        
        # Simple cut re-encoding
        cmd = [
            ffmpeg_exe, "-y",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac",
            "-strict", "experimental",
            output_path
        ]
        print(f"⚠️ MoviePy failed, trying fallback FFmpeg cut: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"❌ Fallback cut also failed: {e}")
        return False

# ... (Existing code)



# Global variable to track model status
model_status = {"progress": 100, "status": "initializing", "message": "正準備下載模型..."}
whisper_model = None

# MediaPipe Initialization
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Studio Sound (DFN3)
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    df_model, df_state, _ = init_df()
    print("✅ Studio Sound (DFN3) Ready")
except Exception as e:
    print(f"⚠️ DFN3 Init Failed: {e}")
    df_model = None

def download_whisper_model(model_name="turbo"):
    global model, model_status
    import whisper.utils as whisper_utils
    import urllib.request
    
    # Standard location for whisper models
    import os
    download_root = os.path.expanduser("~/.cache/whisper")
    os.makedirs(download_root, exist_ok=True)
    
    # Map model name to URL
    urls = {
        "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff2414bc3a43665576983508339ca91069903bc036573e87002bc0519098906/large-v3-turbo.pt"
    }
    
    url = urls.get(model_name)
    model_path = os.path.join(download_root, f"{model_name}.pt")
    
    if os.path.exists(model_path):
        model_status = {"progress": 100, "status": "ready", "message": "極速模型 Turbo 已就緒"}
        model = whisper.load_model(model_name)
        return

    def progress_callback(count, block_size, total_size):
        progress = int(count * block_size * 100 / total_size)
        model_status["progress"] = min(100, progress)
        model_status["status"] = "downloading"
        model_status["message"] = f"正在下載頂規模型 {model_name} ({progress}%)..."

    try:
        print(f"📥 Starting custom download for {model_name}...")
        urllib.request.urlretrieve(url, model_path, reporthook=progress_callback)
        model_status = {"progress": 100, "status": "loading", "message": "下載完成，正在載入記憶體..."}
        whitelist_model = whisper.load_model(model_path)
        whisper_model = whitelist_model
        model_status["status"] = "ready"
        model_status["message"] = "Whisper Turbo 載入成功！"
    except Exception as e:
        model_status = {"progress": 0, "status": "error", "message": f"下載失敗: {str(e)}"}
        print(f"❌ Download error: {e}")

# Start loading in background thread (starting with turbo)
import threading
threading.Thread(target=download_whisper_model, args=("turbo",), daemon=True).start()

app = FastAPI()

@app.get("/model-status")
async def get_model_status():
    return model_status

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable Static Access to Exports for Preview (Placed here after app init)
app.mount("/exports", StaticFiles(directory=OUTPUT_DIR), name="exports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload-font")
async def upload_font(file: UploadFile = File(...)):
    try:
        font_path = os.path.join(FONTS_DIR, file.filename)
        with open(font_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Font uploaded: {file.filename}")
        return {"message": f"Font {file.filename} uploaded successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})



def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = ','):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3600000
    milliseconds -= hours * 3600000
    minutes = milliseconds // 60000
    milliseconds -= minutes * 60000
    seconds = milliseconds // 1000
    milliseconds -= seconds * 1000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else "00:"
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def parse_time(value):
    """
    Smartly converts various time formats to total seconds (float).
    Supports:
    - 88.5 (float) -> 88.5
    - "88.5" (string) -> 88.5
    - "1:28" (MM:SS) -> 88.0
    - "1:28.5" (MM:SS.ms) -> 88.5
    - "01:02:03" (HH:MM:SS) -> 3723.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    value = str(value).strip()
    
    try:
        # Try direct float conversion first
        return float(value)
    except ValueError:
        # Handle time format with colons
        if ":" in value:
            parts = value.split(":")
            parts = [float(p) for p in parts]
            if len(parts) == 2: # MM:SS
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3: # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
    
    # Fallback/Error
    print(f"⚠️ Could not parse time: {value}, defaulting to 0")
    return 0.0

# --- Helper for Smart Cropping ---
def get_smart_crop_center(clip, width, height, target_w):
    """
    Robust face detection to find the best horizontal center for cropping.
    Uses median of multiple frames to ignore outliers.
    """
    import cv2
    import numpy as np

    # Try improved cascade first, fall back to default
    cascades = [
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_default.xml'
    ]
    face_cascade = None
    for c_name in cascades:
        try:
            path = cv2.data.haarcascades + c_name
            if os.path.exists(path):
                face_cascade = cv2.CascadeClassifier(path)
                break
        except: pass
    
    # Fallback if cv2 data not found or error
    if face_cascade is None or face_cascade.empty():
        print("⚠️ Smart Crop: No valid cascade found, defaulting to center.")
        return width / 2

    # Sample 7 frames for robust preview/export
    start_t = 0 # Preview clip is already cut, so relative 0 is start
    end_t = clip.duration
    duration = end_t - start_t
    
    count = 7
    detected_centers = []
    
    print(f"🤖 Analyzing {count} frames for smart crop...")
    
    for i in range(count):
        t = start_t + (duration * i) / (count - 1)
        # Avoid end of video (black frame)
        if t >= clip.duration: t = clip.duration - 0.1
        
        try:
            frame = clip.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Min size: 10% of height (filtering noise like lamps)
            min_size = int(height * 0.1)
            
            # Detect
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6, # Higher = less false positives
                minSize=(min_size, min_size)
            )
            
            if len(faces) > 0:
                # Pick largest face
                fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
                center_x = fx + (fw / 2)
                detected_centers.append(center_x)
        except Exception as e:
            pass

    if not detected_centers:
        print("🤖 Smart Crop: No faces found. Defaulting to center.")
        return width / 2
    
    # Use Median to reject outliers (e.g. lamp detected once)
    detected_centers.sort()
    median_center = detected_centers[len(detected_centers) // 2]
    
    # Additional smoothing/clamping
    safe_min = target_w / 2
    safe_max = width - (target_w / 2)
    final_center = max(safe_min, min(median_center, safe_max))
    
    print(f"🤖 Smart Crop: Locked on x={final_center:.1f} (Median of {detected_centers})")
    return final_center

# --- Helper for Smart Cropping (v6: Anti-Ghosting & High Confidence) ---
# --- Helper for Smart Cropping (v6: Anti-Ghosting & High Confidence) ---
def get_smart_crop_plan(clip, width, height, target_w, speaker_weight=5.0, stickiness_weight=2.0, min_shot_duration=2.0):
    """
    Analyzes the clip using OpenCV (Industrial Standard) for robust face tracking.
    Fallback from MediaPipe due to environment incompatibility.
    v7.2: Pure Visual Tracking (Size Priority).
    """
    import cv2
    import numpy as np

    try:
        # Load Haar Cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            print("⚠️ Failed to load Haar Cascade. Tracking disabled.")
            return [(0, width/2), (clip.duration, width/2)]
        
        step_sec = 0.2
        times = np.arange(0, clip.duration, step_sec)
        
        robust_faces = [] 
        
        print(f"🤖 AI Tracking v7.2 (OpenCV): High-Confidence Analysis...")
        
        for t in times:
            try:
                frame = clip.get_frame(t)
                if frame is None: continue
                
                # Convert to Gray
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # MediaPipe Face Detection (Instead of Haar)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        # Convert relative to absolute
                        fw = bbox.width * width
                        fh = bbox.height * height
                        fx_center = (bbox.xmin + bbox.width/2) * width
                        fy_center = (bbox.ymin + bbox.height/2) * height
                        
                        robust_faces.append({
                            't': t, 'x': fx_center, 'w': fw, 'area': fw*fh
                        })
                        # Pick only the first detected face for tracking speed
                        break
            except Exception as e:
                pass # Specific frame error
                
        if not robust_faces:
            print("⚠️ No faces found in clip (or tracking failed). Defaulting to center.")
            return [(0, width/2), (clip.duration, width/2)]

        # --- 3. Build Safe Timeline ---
        
        def get_faces_at(t_query):
            return [f for f in robust_faces if abs(f['t'] - t_query) < step_sec * 0.8]
            
        trajectory = []
        
        # Init center
        start_candidates = [f for f in robust_faces if f['t'] < 1.0]
        if not start_candidates: start_candidates = robust_faces[:5]
        
        current_target_x = width/2
        if start_candidates:
            best_start = max(start_candidates, key=lambda f: f['area'])
            current_target_x = best_start['x']
            print(f"🤖 Trace Start: Locking on Main Subject (Area {best_start['area']:.0f}) at X={current_target_x:.1f}")
        
        # --- Shot Logic (Cut Switching) ---
        print("✂️ Generating Cut-Based Camera Plan...")
        
        # Deadband Logic (Dynamic Stability)
        normalized_stickiness = max(0, min(stickiness_weight, 10)) / 10.0
        deadband = width * (0.05 + 0.45 * normalized_stickiness)
        min_dur = float(min_shot_duration)
        last_cut_t = -999 
        
        for t in times:
            faces_now = get_faces_at(t)
            
            if not faces_now:
                trajectory.append((t, current_target_x))
                continue
                
            best_face = None
            highest_score = -1
            
            for face in faces_now:
                # Score: Size Priority
                area_score = (face['area'] / (width * height)) * float(speaker_weight)
                
                # Stickiness
                stickiness = 0
                if abs(face['x'] - current_target_x) < width * 0.1:
                    stickiness = float(stickiness_weight)
                    
                total_score = area_score + stickiness
                
                if total_score > highest_score:
                    highest_score = total_score
                    best_face = face
            
            if best_face:
                dist = abs(best_face['x'] - current_target_x)
                time_since_cut = t - last_cut_t
                
                if dist > deadband and time_since_cut >= min_dur:
                    current_target_x = best_face['x'] # CUT!
                    last_cut_t = t
            
            trajectory.append((t, current_target_x))
            
        return trajectory

    except Exception as e:
        print(f"❌ Smart Crop Logic Failed (Global): {e}")
        return [(0, width/2), (clip.duration, width/2)]
    


# --- Shared Logic for Reframing (v7) ---
def apply_smart_reframing(clip, aspect_ratio, face_tracking, vertical_mode_legacy="false", viz_tracking="false", track_zoom=1.5, track_weight=5.0, track_stickiness=2.0, min_shot_duration=2.0):
    w, h = clip.size
    should_track = (face_tracking == "true")
    print(f"🎥 Smart Reframing Config: Ratio={aspect_ratio}, Track={should_track}, Viz={viz_tracking}, Zoom={track_zoom}, Weight={track_weight}")
    # Interpret 9:16 request
    is_vertical_9_16 = (aspect_ratio == "9:16") or (vertical_mode_legacy == "true")
    
    target_w = w
    target_h = h
    do_resize_back = False
    
    if is_vertical_9_16:
         target_w = h * (9/16)
    
    if should_track:
         # If 16:9 and Tracking, apply Zoom to allow panning
         if not is_vertical_9_16:
             zoom_factor = float(track_zoom)
             print(f"🎥 16:9 AI Tracking: Zoom {zoom_factor}x Active")
             target_w = w / zoom_factor
             target_h = h / zoom_factor
             do_resize_back = True
         
         # AI Trajectory
         trajectory = get_smart_crop_plan(clip, w, h, target_w, speaker_weight=float(track_weight), stickiness_weight=float(track_stickiness), min_shot_duration=float(min_shot_duration))
         
         import numpy as np
         times = [p[0] for p in trajectory]
         centers = [p[1] for p in trajectory]
         
         # --- VISUALIZE MODE (Debug) ---
         if viz_tracking == "true":
             print("👁️ Visualizing Tracking Path...")
             def viz_render(get_frame, t):
                 frame = get_frame(t).copy()
                 hf, wf = frame.shape[:2]
                 if frame is None: return frame
                 
                 current_center_x = np.interp(t, times, centers)
                 tw, th = int(target_w), int(target_h)
                 x1 = int(current_center_x - tw / 2)
                 y1 = int((hf - th) / 2)
                 x1 = max(0, min(x1, wf - tw))
                 y1 = max(0, min(y1, hf - th))
                 
                 import cv2
                 # Draw Green Box
                 cv2.rectangle(frame, (x1, y1), (x1+tw, y1+th), (0, 255, 0), 5)
                 cv2.putText(frame, "AI TRACKING", (x1+10, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                 return frame
             return clip.fl(viz_render)
         
         def pan_crop(get_frame, t):
             frame = get_frame(t)
             if frame is None: return frame
             
             # Step Interpolation (Jump Cuts) for "Cut Reframing"
             idx = np.searchsorted(times, t, side='right') - 1
             idx = max(0, min(idx, len(centers)-1))
             current_center_x = centers[idx]
             
             hf, wf = frame.shape[:2]
             tw = int(target_w)
             th = int(target_h)
             
             x1 = int(current_center_x - tw / 2)
             y1 = int((hf - th) / 2) # Center Vertically default
             
             # Clamping
             x1 = max(0, min(x1, wf - tw))
             y1 = max(0, min(y1, hf - th))
             
             cropped = frame[y1:y1+th, x1:x1+tw]
             
             # Resize back if needed (Virtual Zoom)
             if do_resize_back:
                 import cv2
                 return cv2.resize(cropped, (wf, hf), interpolation=cv2.INTER_LINEAR)
             return cropped

         new_clip = clip.fl(pan_crop, apply_to=['mask'])
         if not do_resize_back:
             # Fix: Do not set .w/.h directly as they are read-only properties
             new_clip.size = (int(target_w), int(target_h))
         return new_clip
         
    elif is_vertical_9_16:
         # Static Vertical Crop (Center)
         x1 = w/2 - target_w/2
         return clip.crop(x1=x1, y1=0, width=target_w, height=h)
         
    return clip

@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model_name: str = Form("gemini-1.5-flash"),
    instruction: str = Form("找出影片中的精華片段"),
    target_count: int = Form(None),
    target_duration: int = Form(None)
):
    try:
        # 1. Config Gemini
        genai.configure(api_key=api_key)
        
        # 2. Save video temporarily
        video_path = f"{UPLOAD_DIR}/temp_analyze_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"🧠 Analysis started: {file.filename} using {model_name}")

        # 3. Upload to Gemini
        video_file = genai.upload_file(path=video_path)
        print(f"📤 Uploaded to Gemini: {video_file.name}")
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            print("⏳ Gemini Processing video...")

        if video_file.state.name == "FAILED":
            raise ValueError("Gemini failed to process video.")

        # --- Model Fallback Logic ---
        real_model_name = model_name
        # Fallback for futuristic model names to stable ones if needed
        if "3.0" in model_name or "2.5" in model_name: 
             pass 

        # Explicit Prompt aimed at JSON structure
        constraint_text = ""
        if target_count:
            constraint_text += f"\n- Please find exactly {target_count} clips."
        if target_duration:
            constraint_text += f"\n- Each clip should be approximately {target_duration} seconds long."

        prompt = f"""
        You are a professional video editor. Analyze the video and extract clips based on this instruction:
        "{instruction}"
        {constraint_text}
        
        Return the result strictly as a JSON list of objects.
        
        Return the result strictly as a JSON list of objects.
        Each object must have:
        - "start": start time in SECONDS (number, e.g., 12.5) . DO NOT use MM:SS format.
        - "end": end time in SECONDS (number, e.g., 25.0). DO NOT use MM:SS format.
        - "label": A short description of the clip in Traditional Chinese.
        
        Example:
        [
            {{ "start": 10.5, "end": 20.0, "label": "開場介紹" }},
            {{ "start": 65.0, "end": 90.0, "label": "重點精華" }}
        ]
        """

        # Restore smart fallback logic to ensure success even if user selects non-existent model (like 3.0)
        candidate_models = [
            real_model_name,              # 1. User selection (e.g. gemini-3.0-flash)
            "gemini-2.0-flash-exp",       # 2. Smart Fallback: Newest Experimental
            "gemini-1.5-pro",             # 3. Stable Pro
            "gemini-1.5-flash",           # 4. Stable Flash
        ]
        
        candidate_models = list(dict.fromkeys(candidate_models)) # Remove dupes
        
        response = None
        last_error = None
        
        # Generation Config to enforce JSON
        generation_config = {
            "response_mime_type": "application/json"
        }

        print(f"🤖 Attempting models in order: {candidate_models}")

        for model_try in candidate_models:
            if not model_try: continue
            try:
                print(f"🔄 Trying model: {model_try}...")
                gemini = genai.GenerativeModel(model_try)
                # Pass generation_config for structured output
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

        print("🤖 Gemini Response:", response.text)
        
        # 5. Parse JSON
        try:
            cuts = json.loads(response.text)
            
            # Post-processing: Normalize times using smart parser
            if isinstance(cuts, list):
                for cut in cuts:
                    cut["start"] = parse_time(cut.get("start", 0))
                    cut["end"] = parse_time(cut.get("end", 0))
            
        except json.JSONDecodeError as je:
             raise ValueError(f"AI returned invalid JSON: {response.text}")
        
        # Cleanup
        genai.delete_file(video_file.name)
        
        return cuts

    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        # Return the actual error message to frontend
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/preview-clip")
async def preview_clip(
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    vertical_mode: str = Form("false"),
    aspect_ratio: str = Form("9:16"),
    face_tracking: str = Form("true"),
    viz_tracking: str = Form("false"),
    track_zoom: float = Form(1.5),
    track_weight: float = Form(5.0),
    track_stickiness: float = Form(2.0),
    min_shot_duration: float = Form(2.0),
    dynamic_zoom: str = Form("false"),
    burn_captions: str = Form("false"),
    subtitle_style: str = Form("classic"),
    subtitle_margin_v: int = Form(220),
    subtitle_font_name: str = Form("Arial"),
    subtitle_font_size: int = Form(24),
    subtitle_color: str = Form("&HFFFFFF"),
    subtitle_outline_color: str = Form("&H000000"),
    subtitle_bold: str = Form("false"),
    subtitle_shadow_size: int = Form(1),
    subtitle_outline_width: int = Form(2),
    subtitle_chars_per_line: int = Form(12),
    subtitle_box_enabled: str = Form("false"),
    subtitle_box_color: str = Form("#000000"),
    subtitle_box_alpha: float = Form(0.8),
    subtitle_box_padding: int = Form(10),
    subtitle_box_radius: int = Form(0),
    subtitle_margin_h: int = Form(40),
    output_resolution: str = Form("1080p"),
    output_quality: str = Form("high")
):
    try:
        print(f"👁️ Preview Request: Vertical={vertical_mode}, Ratio={aspect_ratio}")
        # Save temp file
        temp_input = f"{UPLOAD_DIR}/temp_preview_{file.filename}"
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        video = VideoFileClip(temp_input)
        
        # Parse time (safety)
        start = max(0, float(start))
        end = min(video.duration, float(end))
        
        # Ensure we don't accidentally export whole video if end is 0 or wrong
        if end <= start:
             end = start + 10 # Default fallback if invalid duration
        
        print(f"👁️ Preview Requested: {start}s - {end}s")

        # Cut
        # Note: subclip(t1, t2) should interpret t2 as end time.
        new_clip = video.subclip(start, end)
        
        # 1. Vertical Crop (Smart Face)
        # --- Smart Reframing System (v7) ---
        new_clip = apply_smart_reframing(new_clip, aspect_ratio, face_tracking, vertical_mode, viz_tracking, track_zoom, track_weight, track_stickiness, min_shot_duration)



        # 2. Dynamic Zoom
        if dynamic_zoom == "true":
             def scroll_zoom(get_frame, t):
                 img = get_frame(t)
                 h, w = img.shape[:2]
                 scale = 1.0 + (0.1 * (t / new_clip.duration))
                 cw, ch = w / scale, h / scale
                 x1, y1 = (w - cw) / 2, (h - ch) / 2
                 import cv2
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 cw, ch = min(w-x1, int(cw)), min(h-y1, int(ch))
                 if cw < 2 or ch < 2: return img
                 cropped = img[y1:y1+ch, x1:x1+cw]
                 return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
             new_clip = new_clip.fl(scroll_zoom)

        # Write output
        output_filename = f"preview_{int(time.time())}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        
        # 3. Burn Captions (Simplified for Preview)
        # We don't have SRT here easily unless we generate it. 
        # For Speed, we might skip actual text burning or generate a dummy "Preview Caption".
        # Let's generate real one if requested
        if burn_captions == "true" or burn_captions is True:
            try:
                temp_audio = f"{OUTPUT_DIR}/preview_audio.mp3"
                new_clip.audio.write_audiofile(temp_audio, logger=None)
                # Use the upgraded Whisper Turbo model
                result = whisper_model.transcribe(temp_audio)
                srt_path = output_path.replace(".mp4", ".srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    # Smart Wrap based on margins
                    # On 9:16 (1080 height), logical width is ~608 units
                    total_w_units = 608 if aspect_ratio == "9:16" else 1920
                    available_w = total_w_units - (int(subtitle_margin_h) * 2)
                    # Heuristic: Chinese char is roughly 0.9 * fontsize in ASS units
                    max_safe_chars = max(1, int(available_w / (float(subtitle_font_size) * 0.9)))
                    
                    # Use the smaller of user limit vs safety limit
                    effective_chars = min(int(subtitle_chars_per_line), max_safe_chars) if int(subtitle_chars_per_line) > 0 else max_safe_chars
                    
                    for idx, s in enumerate(result["segments"]):
                        wrapped = wrap_text(s['text'], max_chars=effective_chars)
                        f.write(f"{idx+1}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{wrapped}\n\n")
                
                # 3. Burn Captions (Optimized Quality & Style)
                burned_path = output_path.replace(".mp4", "_burned.mp4")
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                
                abs_srt = os.path.abspath(srt_path)
                safe_srt_path = abs_srt.replace("\\", "/").replace(":", "\\:")
                abs_in = os.path.abspath(output_path)
                abs_out = os.path.abspath(burned_path)
                abs_fonts = os.path.abspath(FONTS_DIR)

                # Consistent Style Calculation
                bold_val = 1 if str(subtitle_bold).lower() == "true" else 0
                final_color_ass = hex_to_ass(subtitle_color)
                outline_color_ass = hex_to_ass(subtitle_outline_color)
                box_color_ass = hex_to_ass(subtitle_box_color, float(subtitle_box_alpha))
                shadow_color_ass = "&H00000000" # Solid Black Shadow
                
                is_box = (subtitle_box_enabled == "true") or (subtitle_style == "box") or (subtitle_box_enabled == "on")
                border_style = 3 if is_box else 1
                
                try: box_pad_int = int(subtitle_box_padding)
                except: box_pad_int = 10
                try: outline_width_int = int(subtitle_outline_width)
                except: outline_width_int = 2
                
                outline_val = box_pad_int if is_box else outline_width_int
                shadow_val = 0 if is_box else int(subtitle_shadow_size)

                # Standard ASS Style String
                style_parts = [
                    "PlayResY=1080", f"Fontname={subtitle_font_name}", f"Fontsize={subtitle_font_size}",
                    f"PrimaryColour={final_color_ass}", f"SecondaryColour={final_color_ass}",
                    f"OutlineColour={outline_color_ass if not is_box else box_color_ass}",
                    f"BackColour={shadow_color_ass if not is_box else box_color_ass}",
                    f"BorderStyle={border_style}", f"Outline={outline_val}", f"Shadow={shadow_val}",
                    f"MarginV={subtitle_margin_v}", f"MarginL={subtitle_margin_h}", f"MarginR={subtitle_margin_h}",
                    "Alignment=2", f"Bold={bold_val}"
                ]
                escaped_style = ",".join(style_parts).replace(",", "\\,")

                cmd = [
                    ffmpeg_exe, "-y", "-i", abs_in, 
                    "-vf", f"subtitles='{safe_srt_path}':fontsdir='{abs_fonts}':force_style='{escaped_style}'",
                    "-c:a", "copy", "-c:v", "libx264", "-crf", "17", "-preset", "slow", abs_out
                ]
                
                subprocess.run(cmd, check=True, env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if os.path.exists(burned_path):
                    os.remove(output_path)
                    os.rename(burned_path, output_path)
            except Exception as e:
                print(f"🔥 Preview burn failed: {e}")

        return FileResponse(output_path, media_type="video/mp4")
        
    except Exception as e:
        print(f"Preview Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    cuts_json: str = Form(...),
    model_name: str = Form(None),
    vertical_mode: str = Form("false"),
    aspect_ratio: str = Form("9:16"),
    face_tracking: str = Form("true"),
    auto_caption: str = Form("false"),
    translate_to_chinese: str = Form("false"), # NEW
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
    subtitle_font_size: int = Form(24),
    subtitle_color: str = Form("&HFFFFFF"),
    subtitle_outline_color: str = Form("&H000000"),
    subtitle_margin_v: int = Form(220),
    subtitle_font_name: str = Form("Arial"),
    subtitle_bold: str = Form("false"),
    subtitle_shadow_size: int = Form(1),
    subtitle_outline_width: int = Form(2),
    subtitle_chars_per_line: int = Form(12),
    subtitle_box_enabled: str = Form("false"),
    subtitle_box_color: str = Form("#000000"),
    subtitle_box_alpha: float = Form(0.8),
    subtitle_box_padding: int = Form(10),
    subtitle_box_radius: int = Form(0),
    subtitle_margin_h: int = Form(40),
    output_resolution: str = Form("1080p"),
    output_quality: str = Form("high")
):
    try:
        print(f"🎬 Processing Request: Vertical={vertical_mode}, Mode={output_mode}, Style={subtitle_style}")
        print(f"   Quality: {output_quality}, Res: {output_resolution}, BoxColor: {subtitle_box_color}, BoxAlpha: {subtitle_box_alpha}")
        
        if output_mode == "preview_url":
            merge_clips = "true"
            # Force Burn-in for Preview so users can see the transcript
            if auto_caption == "true" or auto_caption is True:
                burn_captions = "true"
                print("🔥 Force enabling Burn-In for Preview Mode")
        # Configure GenAI if key provided for Smart Renaming
        if api_key and api_key != "null":
            genai.configure(api_key=api_key)

        # 1. Save uploaded video
        video_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Parse cuts
        cuts = json.loads(cuts_json)
        
        if not cuts:
            return {"message": "No cuts provided"}

        print(f"🎬 Processing video: {file.filename}")
        
        # 3. Process video
        processed_files = []
        clips_for_merge = [] # Store MoviePy clips for concatenation
        
        try:
            video = VideoFileClip(video_path)
            video_duration = video.duration
        except Exception as e:
            print(f"❌ Failed to load video file: {e}")
            raise ValueError(f"Could not load video: {e}")

        # Clear previous exports in a safer way
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp4") or f.endswith(".zip") or f.endswith(".txt") or f.endswith(".srt"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except: pass

        for i, cut in enumerate(cuts):
            try:
                # Use robust parse_time for safety
                start = parse_time(cut.get('start', 0))
                end = parse_time(cut.get('end', 0))
                
                # --- 1. Define Paths ---
                raw_label = cut.get('label', 'Clip')
                safe_label = "".join([c for c in raw_label if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
                output_filename = f"Highlight_{i+1}_{safe_label}.mp4"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                srt_path = output_path.replace(".mp4", ".srt")
                
                print(f"🎬 Processing Cut {i+1}: {start}-{end}")

                # --- 2. Create Subclip & Effects ---
                new_clip = video.subclip(start, end)

                if new_clip.audio is not None:
                     from moviepy.audio.fx.all import audio_fadein, audio_fadeout
                     # Only fade AUDIO to avoid clicks. Video should be instant cut.
                     new_clip.audio = new_clip.audio.fx(audio_fadein, 0.15).fx(audio_fadeout, 0.15)

                # --- Smart Reframing System (v7) ---
                new_clip = apply_smart_reframing(new_clip, aspect_ratio, face_tracking, vertical_mode, viz_tracking, track_zoom, track_weight, track_stickiness, min_shot_duration)
                if False: # Legacy Block Disabled (vertical_mode == "true"...)
                    w, h = new_clip.size
                    target_ratio = 9/16
                    if w/h > target_ratio:
                         new_w = h * target_ratio
                         
                         # Use Global Helper for Robust Detection (Auto-Pan Plan)
                         trajectory = get_smart_crop_plan(new_clip, w, h, new_w)
                         
                         traj_x = [p[1] for p in trajectory]
                         x_min, x_max = min(traj_x), max(traj_x)
                         
                         if (x_max - x_min) < 1.0:
                             # Static 
                             x1 = max(0, trajectory[0][1] - new_w/2)
                             new_clip = new_clip.crop(x1=x1, y1=0, width=new_w, height=h)
                         else:
                             # Dynamic Pan
                             import numpy as np
                             times = [p[0] for p in trajectory]
                             centers = [p[1] for p in trajectory]
                             
                             def pan_crop(get_frame, t):
                                  frame = get_frame(t)
                                  current_center = np.interp(t, times, centers)
                                  
                                  x1 = int(current_center - new_w / 2)
                                  height_f, width_f = frame.shape[:2]
                                  x1 = max(0, min(x1, width_f - int(new_w)))
                                  return frame[0:height_f, x1:x1+int(new_w)]
                             new_clip = new_clip.fl(pan_crop, apply_to=['mask'])

                if (dynamic_zoom == "true" or dynamic_zoom is True) and (face_tracking == "false" or face_tracking is False):
                     # Subtler zoom (1.05x instead of 1.1x)
                     def scroll_zoom(get_frame, t):
                         img = get_frame(t)
                         h, w = img.shape[:2]
                         # Reduced zoom factor for better quality
                         scale = 1.0 + (0.05 * (t / new_clip.duration))
                         cw, ch = w / scale, h / scale
                         x1, y1 = (w - cw) / 2, (h - ch) / 2
                         import cv2
                         x1, y1 = max(0, int(x1)), max(0, int(y1))
                         cw, ch = min(w-x1, int(cw)), min(h-y1, int(ch))
                         if cw < 2 or ch < 2: return img
                         cropped = img[y1:y1+ch, x1:x1+cw]
                         return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                     new_clip = new_clip.fl(scroll_zoom)

                # Determine Bitrate based on Quality Setting
                bitrate_map = {
                    "high": "20000k",   # Upgraded to 20M as requested
                    "medium": "8000k",
                    "low": "4000k"
                }
                target_bitrate = bitrate_map.get(output_quality, "8000k")

                # --- 3. Write RAW Video First (Required for FFmpeg Burning) ---
                try:
                    new_clip.write_videofile(
                        output_path, 
                        codec="libx264", 
                        audio_codec="aac",
                        bitrate=target_bitrate_mw,
                        audio_bitrate="192k",
                        preset="fast",
                        logger=None,
                        threads=4
                    )
                except Exception as write_err:
                    print(f"⚠️ MoviePy Write Error: {write_err}. Switching to Fallback...")
                    success = fallback_cut_video(video_path, output_path, start, end)
                    if not success:
                        raise ValueError(f"Both MoviePy and FFmpeg fallback failed for this clip.")
                    try: 
                        new_clip.close()
                        new_clip = VideoFileClip(output_path)
                    except: pass
                
                # --- 4. Transcription & SRT Generation ---
                has_audio = new_clip.audio is not None
                if (auto_caption == "true" or auto_caption is True or burn_captions == "true" or burn_captions is True) and has_audio:
                    try:
                        print(f"🎙️ Transcribing clip {i+1}... (Burn={burn_captions})")
                        
                        # 4a. Run Whisper (Only Once)
                        # Use temp audio for speed and to avoid codec issues in direct transcription
                        temp_audio_path = os.path.join(OUTPUT_DIR, f"temp_audio_{i}.mp3")
                        new_clip.audio.write_audiofile(temp_audio_path, logger=None)
                        # Use the upgraded Whisper Turbo model
                        result = whisper_model.transcribe(temp_audio_path)
                        
                        # 4b. AI Translation (Gemini)
                        if (translate_to_chinese == "true" or translate_to_chinese is True) and api_key and api_key != "null":
                             try:
                                 print(f"🇨🇳 Translating SRT for clip {i+1} using Gemini...")
                                 segments_text = ""
                                 for idx, s in enumerate(result["segments"]):
                                     segments_text += f"{idx+1}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{s['text']}\n\n"
                                 
                                 target_model = model_name if model_name else "gemini-2.0-flash-exp"
                                 ver_model = genai.GenerativeModel(target_model)
                                 prompt = f"Translate the following SRT subtitles into Traditional Chinese (Taiwan). Keep the format exactly: \n{segments_text}"
                                 resp = ver_model.generate_content(prompt)
                                 translated_text = resp.text.strip().replace("```srt", "").replace("```", "")
                                 
                                 if "-->" in translated_text:
                                      with open(srt_path, "w", encoding="utf-8") as f:
                                          f.write(translated_text)
                                      print("✅ Translation applied.")
                                 else:
                                      # Fallback: Save original
                                      with open(srt_path, "w", encoding="utf-8") as f:
                                          f.write(segments_text)
                             except Exception as te:
                                 print(f"⚠️ Translation failed: {te}")
                                 with open(srt_path, "w", encoding="utf-8") as f:
                                     for idx, s in enumerate(result["segments"]):
                                         wrapped = wrap_text(s['text'], max_chars=subtitle_chars_per_line)
                                         f.write(f"{idx+1}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{wrapped}\n\n")
                        else:
                             # Just save original SRT
                             with open(srt_path, "w", encoding="utf-8") as f:
                                 # Smart Wrap (Same as preview)
                                 total_w_units = 608 if aspect_ratio == "9:16" else 1920
                                 available_w = total_w_units - (int(subtitle_margin_h) * 2)
                                 max_safe_chars = max(1, int(available_w / (float(subtitle_font_size) * 0.9)))
                                 effective_chars = min(int(subtitle_chars_per_line), max_safe_chars) if int(subtitle_chars_per_line) > 0 else max_safe_chars

                                 for idx, s in enumerate(result["segments"]):
                                     wrapped = wrap_text(s['text'], max_chars=effective_chars)
                                     f.write(f"{idx+1}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{wrapped}\n\n")

                        try: os.remove(temp_audio_path)
                        except: pass
                        
                        # --- 5. Burn Captions (Optimized Quality & Style) ---
                        if (burn_captions == "true" or burn_captions is True) and os.path.exists(srt_path):
                            try:
                                print(f"🔥 Burning captions for clip {i+1} using {subtitle_style} (Size: {subtitle_font_size})...")
                                
                                # Use Consistent Global hex_to_ass calculations
                                bold_val = 1 if str(subtitle_bold).lower() == "true" else 0
                                final_color_ass = hex_to_ass(subtitle_color)
                                box_color_ass = hex_to_ass(subtitle_box_color, subtitle_box_alpha)
                                outline_color_ass = hex_to_ass(subtitle_outline_color)
                                shadow_color_ass = "&H00000000" # Black shadow

                                is_box = (subtitle_box_enabled == "true") or (subtitle_style == "box") or (subtitle_box_enabled == "on")
                                border_style = 3 if is_box else 1
                                
                                try: box_pad_int = int(subtitle_box_padding)
                                except: box_pad_int = 10
                                try: outline_width_int = int(subtitle_outline_width)
                                except: outline_width_int = 2
                
                                outline_val = box_pad_int if is_box else outline_width_int
                                shadow_val = 0 if is_box else int(subtitle_shadow_size)
                                
                                # Standard ASS Style String
                                style_parts = [
                                    "PlayResY=1080", f"Fontname={subtitle_font_name}", f"Fontsize={subtitle_font_size}",
                                    f"PrimaryColour={final_color_ass}", f"SecondaryColour={final_color_ass}",
                                    f"OutlineColour={outline_color_ass if not is_box else box_color_ass}",
                                    f"BackColour={shadow_color_ass if not is_box else box_color_ass}",
                                    f"BorderStyle={border_style}", f"Outline={outline_val}", f"Shadow={shadow_val}",
                                    f"MarginV={subtitle_margin_v}", f"MarginL={subtitle_margin_h}", f"MarginR={subtitle_margin_h}",
                                    "Alignment=2", f"Bold={bold_val}"
                                ]
                                escaped_style = ",".join(style_parts).replace(",", "\\,")
                                
                                abs_srt_path = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
                                abs_input_path = os.path.abspath(output_path)
                                burned_output_path = output_path.replace(".mp4", "_burned.mp4")
                                abs_output_path = os.path.abspath(burned_output_path)
                                abs_fonts = os.path.abspath(FONTS_DIR)
                                
                                cmd = [
                                    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", abs_input_path,
                                    "-vf", f"subtitles='{abs_srt_path}':fontsdir='{abs_fonts}':force_style='{escaped_style}'",
                                    "-c:a", "copy", "-c:v", "libx264", "-crf", "17", "-preset", "slow", abs_output_path
                                ]
                                
                                subprocess.run(cmd, check=True, env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                                if os.path.exists(burned_output_path):
                                    os.remove(output_path)
                                    os.rename(burned_output_path, output_path)
                            except Exception as e:
                                print(f"⚠️ Burn failed for clip {i+1}: {e}")
                            else:
                                print(f"✅ Captions burned for clip {i+1}")
                                
                    except Exception as we:
                        print(f"⚠️ Transcription process failed: {we}")

                # --- 6. Studio Sound (DeepFilterNet 3) ---
                if studio_sound == "true" or studio_sound is True:
                     try:
                        print(f"🎙️ DFN3 AI Audio Enhancement for clip {i+1}...")
                        audio_path = "temp_audio_dfn.wav"
                        video.audio.write_audiofile(audio_path, logger=None)
                        
                        audio, spec_orig = load_audio(audio_path, sr=df_state.sr())
                        enhanced = enhance(df_model, df_state, audio)
                        save_audio("enhanced_dfn.wav", enhanced, sr=df_state.sr())
                        
                        optimized_path = output_path.replace(".mp4", "_studio.mp4")
                        cmd = [
                            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", output_path, "-i", "enhanced_dfn.wav",
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", optimized_path
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if os.path.exists(optimized_path):
                            os.remove(output_path)
                            os.rename(optimized_path, output_path)
                        
                        # Cleanup
                        for f in [audio_path, "enhanced_dfn.wav"]:
                            if os.path.exists(f): os.remove(f)
                        print(f"✅ Studio Sound (DFN3) Ready")
                     except Exception as dfne:
                        print(f"⚠️ Studio Sound failed: {dfne}")

                # Finalize
                processed_files.append(output_path)
                if (auto_caption == "true" or auto_caption is True) and os.path.exists(srt_path):
                     processed_files.append(srt_path)

                if merge_clips == "true" or merge_clips is True:
                     clips_for_merge.append(VideoFileClip(output_path))
                
                # Cleanup leftover SRT if not needed in ZIP
                if not (auto_caption == "true" or auto_caption is True) and os.path.exists(srt_path):
                     try: os.remove(srt_path)
                     except: pass

            except Exception as e:
                print(f"❌ Error processing clip {i}: {e}")
                continue

        video.close()
        
        # FAIL-SAFE: Check if any output exists
        valid_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4') and not f.startswith('temp_')]
        if not valid_files:
             # Look for the last error in logs or just raise generic
             raise ValueError("所有片段處理失敗 (All clips failed). 請檢查後端 Log 或參數設定。")
        
        # --- 3. PROFESSIONAL FEATURE: MERGE ALL ---
        if (merge_clips == "true" or merge_clips is True) and clips_for_merge:
            print(f"🔗 Merging {len(clips_for_merge)} clips...")
            try:
                final_concatenated = concatenate_videoclips(clips_for_merge)
                merge_filename = "Full_Highlight_Reel.mp4"
                merge_output_path = os.path.join(OUTPUT_DIR, merge_filename)
                
                # HIGH QUALITY MERGE (Medium Preset for speed/stability)
                final_concatenated.write_videofile(merge_output_path, codec="libx264", audio_codec="aac", bitrate="8000k", audio_bitrate="192k", preset="medium", logger=None)
                processed_files.append(merge_output_path)
                
                # Clean up temps
                for c in clips_for_merge: c.close()
                for i in range(len(cuts)):
                    try: os.remove(os.path.join(OUTPUT_DIR, f"temp_merge_{i}.mp4"))
                    except: pass
            except Exception as me:
                print(f"⚠️ Merge failed: {me}")
        
        # Generate Summary
        summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
        # ... (summary writing logic)
        processed_files.append(summary_path) # Dummy append to ensure list logic works

        if output_mode == "preview_url":
             merge_path = os.path.join(OUTPUT_DIR, "Full_Highlight_Reel.mp4")
             if os.path.exists(merge_path):
                 return JSONResponse({
                     "status": "success",
                     "preview_url": "http://localhost:8000/exports/Full_Highlight_Reel.mp4",
                     "message": "渲染完成 (Preview Ready)"
                 })

        # 4. Zip results
        if len(processed_files) == 0:
             print("⚠️ No video files generated!")
             # Create a dummy file to avoid empty zip error
             dummy_path = os.path.join(OUTPUT_DIR, "error_log.txt")
             with open(dummy_path, "w") as f: f.write("No clips were successfully generated. Check console logs.")
             processed_files.append(dummy_path)
        
        zip_filename = f"Antigravity_Shorts_{len(cuts)}_Clips.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in processed_files:
                if os.path.exists(file) and os.path.getsize(file) > 0:
                    zipf.write(file, os.path.basename(file))
                
        return FileResponse(zip_path, filename=zip_filename, media_type='application/zip')

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Video Processing Server Running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
