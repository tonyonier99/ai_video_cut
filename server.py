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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
from fontTools.ttLib import TTFont

# -- NEW: Font Mapping Logic --
FONT_FILE_MAP = {} # "wt009" -> "WangHanZong..."

def scan_fonts():
    global FONT_FILE_MAP
    FONT_FILE_MAP = {}
    if not os.path.exists(FONTS_DIR): return
    
    print("🔍 Scanning fonts for real Family Names...")
    for f in os.listdir(FONTS_DIR):
        if f.lower().endswith(('.ttf', '.otf', '.ttc')):
            base_name = os.path.splitext(f)[0]
            try:
                path = os.path.join(FONTS_DIR, f)
                font = TTFont(path)
                family = ""
                # Priority: ID 16 (Typographic Family) -> ID 1 (Family)
                # Iterate all namerecords to find best match
                for record in font['name'].names:
                    try:
                         # We specifically look for English names (platformID=1, langID=0) or (platformID=3, langID=0x409)
                         # to ensure FFmpeg compatibility
                         if record.nameID in [1, 16]:
                             decoded = record.toUnicode()
                             # Simple heuristic: ASCII names are safer for ASS
                             if all(ord(c) < 128 for c in decoded):
                                 if record.nameID == 16: # Typographic Preferred
                                     family = decoded
                                     break
                                 if record.nameID == 1 and not family:
                                     family = decoded
                    except: pass
                
                # If no English name found, try any name
                if not family:
                     for record in font['name'].names:
                         if record.nameID == 1:
                             family = record.toUnicode()
                             break

                real_name = family if family else base_name
                FONT_FILE_MAP[base_name] = real_name
                print(f"   Fetched Font: {base_name} -> {real_name}")
            except Exception as e:
                print(f"   Error parsing font {f}: {e}")
                FONT_FILE_MAP[base_name] = base_name

# Initial scan
scan_fonts()

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
    """Improved wrapping for subtitles (supports CJK and English)"""
    if not text or max_chars <= 0: return text
    
    # Check if there are existing newlines
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        if len(line) <= max_chars:
            new_lines.append(line)
            continue
            
        import re
        # If CJK characters found, do character-based wrap
        if re.search(r'[\u4e00-\u9fff]', line):
            # For CJK, we just cut at max_chars
            for i in range(0, len(line), max_chars):
                new_lines.append(line[i:i+max_chars])
        else:
            # Standard word wrap for English
            import textwrap
            new_lines.extend(textwrap.wrap(line, width=max_chars))
            
    return "\n".join(new_lines)


def optimize_segments(segments, max_chars=7):
    """
    Splits long whisper segments into shorter ones based on character count and duration.
    Essential for vertical video to prevent subtitle walls.
    """
    new_segments = []
    import re
    for s in segments:
        text = s['text'].strip()
        if not text: continue
        
        # CLEANUP: Remove punctuation for cleaner short-video subtitles
        text = re.sub(r'[，。、？！：；,.?!:;"\']', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip() # Collapse spaces
        
        if not text: continue
        
        # Threshold: if text is longer than max_chars (e.g. 20 for vertical)
        if len(text) > max_chars:
            # Simple split strategy: Split into N parts
            num_parts = (len(text) // max_chars) + 1
            duration = s['end'] - s['start']
            time_per_part = duration / num_parts
            chars_per_part = len(text) // num_parts
            
            for i in range(num_parts):
                start_t = s['start'] + (i * time_per_part)
                end_t = s['start'] + ((i + 1) * time_per_part)
                
                # Slicing text safely
                start_char = i * chars_per_part
                # Make sure last part gets the rest
                if i == num_parts - 1:
                    sub_text = text[start_char:]
                else:
                    sub_text = text[start_char : start_char + chars_per_part]
                
                new_segments.append({
                    "start": start_t,
                    "end": end_t,
                    "text": sub_text.strip()
                })
        else:
            new_segments.append(s)
    return new_segments
    return new_segments

def translate_subtitles(full_subtitles, api_key):
    """Translate list of subtitle dicts using Gemini"""
    if not api_key or api_key == "null" or not full_subtitles:
        return full_subtitles
        
    print("🤖 Translating transcription with Gemini...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash") # Use fast model for translation
        
        # Translate in batches to avoid rate limits/prompt length issues
        batch_size = 20
        for i in range(0, len(full_subtitles), batch_size):
            batch = full_subtitles[i : i+batch_size]
            text_to_translate = "\n".join([f"{idx}: {s['text']}" for idx, s in enumerate(batch)])
            
            prompt = f"請將以下逐字稿內容翻譯成繁體中文（台灣習慣用語），保持簡短與原始意思。只回傳翻譯內容，格式為 '索引: 翻譯'：\n{text_to_translate}"
            try:
                response = model.generate_content(prompt)
                # Parse response (expected each line: "0: translated text")
                for line in response.text.strip().split('\n'):
                    if ':' in line:
                        idx_str, trans = line.split(':', 1)
                        try:
                            # Use regex to find index and text
                            import re
                            m = re.match(r'(\d+)\s*:\s*(.*)', line)
                            if m:
                                b_idx = int(m.group(1).strip())
                                trans_val = m.group(2).strip()
                                if b_idx < len(batch):
                                    batch[b_idx]['text'] = trans_val
                        except: pass
            except Exception as te:
                print(f"⚠️ Translation batch {i} failed: {te}")
    except Exception as e:
        print(f"⚠️ Global translation failed: {e}")
    
    return full_subtitles
# Fallback logic removed as we use Remotion

# ... (Existing code)



# Global variable to track model status
model_status = {"progress": 100, "status": "initializing", "message": "正準備下載模型..."}
whisper_model = None

# MediaPipe Initialization (New Tasks API for v0.10+)
face_detector = None
mp_face_detection = None

try:
    # Try new MediaPipe Tasks API first (v0.10+)
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    from mediapipe.tasks.python import BaseOptions
    import urllib.request
    
    # Download the face detection model if not present
    model_path = "blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        print("📥 Downloading MediaPipe Face Detection model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)
    
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.5
    )
    face_detector = FaceDetector.create_from_options(options)
    print("✅ MediaPipe Face Detection Ready (Tasks API)")
except Exception as e:
    print(f"⚠️ MediaPipe Tasks API Init Failed: {e}")
    # Try legacy solutions API
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
            mp_face_detection = mp.solutions.face_detection
            face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            print("✅ MediaPipe Face Detection Ready (Legacy API)")
        else:
            raise ImportError("No solutions API available")
    except Exception as e2:
        print(f"⚠️ MediaPipe Legacy Init Also Failed: {e2}")
        # Initialize fallback OpenCV detector as last resort
        try:
            import cv2
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_detector = "opencv_fallback"
            print("⚠️ Used OpenCV Haar Classifier as fallback")
        except:
            face_detector = None

# Studio Sound (DFN3)
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    df_model, df_state, _ = init_df()
    print("✅ Studio Sound (DFN3) Ready")
except Exception as e:
    print(f"⚠️ DFN3 Init Failed: {e}")
    df_model = None

def download_whisper_model(model_name="turbo"):
    global whisper_model, model_status
    
    try:
        # Check if model is already cached
        import os
        download_root = os.path.expanduser("~/.cache/whisper")
        
        # Try to load directly using whisper's built-in download
        model_status = {"progress": 50, "status": "downloading", "message": f"正在載入 Whisper {model_name} 模型..."}
        print(f"📥 Loading Whisper model: {model_name}...")
        
        # Use official whisper API
        # Priority: turbo -> large-v3 -> medium
        model_options = ["turbo", "large-v3", "medium", "base"]
        
        loaded_model = None
        for try_model in model_options:
            try:
                print(f"🔄 Trying model: {try_model}...")
                loaded_model = whisper.load_model(try_model)
                print(f"✅ Successfully loaded: {try_model}")
                break
            except Exception as model_err:
                print(f"⚠️ Failed to load {try_model}: {model_err}")
                continue
        
        if loaded_model:
            whisper_model = loaded_model
            model_status = {"progress": 100, "status": "ready", "message": f"Whisper 模型載入成功！"}
        else:
            model_status = {"progress": 0, "status": "error", "message": "所有模型載入失敗"}
            print("❌ All Whisper model options failed to load")
            
    except Exception as e:
        model_status = {"progress": 0, "status": "error", "message": f"下載失敗: {str(e)}"}
        print(f"❌ Download error: {e}")

# Start loading in background thread (starting with turbo)
import threading
threading.Thread(target=download_whisper_model, args=("turbo",), daemon=True).start()

app = FastAPI()

# Global Job Status
current_job_status = {"progress": 0, "message": "Idle", "step": "idle"}

@app.get("/model-status")
async def get_model_status():
    return model_status

@app.get("/job-status")
async def get_job_status():
    return current_job_status

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable Static Access to Exports & Uploads for Preview
app.mount("/exports", StaticFiles(directory=OUTPUT_DIR), name="exports")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload-font")
async def upload_font(file: UploadFile = File(...)):
    try:
        font_path = os.path.join(FONTS_DIR, file.filename)
        with open(font_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        font_name = file.filename.rsplit('.', 1)[0]  # Remove extension
        print(f"✅ Font uploaded: {file.filename} (name: {font_name})")
        return {"message": f"Font {file.filename} uploaded successfully", "font_name": font_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/list-fonts")
async def list_fonts():
    try:
        # Get custom uploaded fonts
        custom_fonts = []
        if os.path.exists(FONTS_DIR):
            for f in os.listdir(FONTS_DIR):
                if f.endswith(('.ttf', '.otf', '.woff', '.woff2', '.TTF', '.OTF')):
                    font_name = f.rsplit('.', 1)[0]
                    custom_fonts.append(font_name)
        return {"fonts": custom_fonts}
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

# --- All legacy MoviePy/OpenCV reframing logic removed ---
# Video processing is now handled via Remotion in 'process_video' endpoint.

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
            
            # Post-processing: Normalize times and ENFORCE target_duration if specified
            if isinstance(cuts, list):
                for cut in cuts:
                    start_t = parse_time(cut.get("start", 0))
                    cut["start"] = start_t
                    if target_duration and target_duration > 0:
                        cut["end"] = round(start_t + target_duration, 2)
                    else:
                        cut["end"] = parse_time(cut.get("end", 0))
            
        except json.JSONDecodeError as je:
             raise ValueError(f"AI returned invalid JSON: {response.text}")
        
        # Cleanup
        genai.delete_file(video_file.name)
        
        return cuts
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)


@app.post("/transcribe")
def transcribe_only(
    file: UploadFile = File(...),
    whisper_language: str = Form("zh"),
    subtitle_chars_per_line: int = Form(7),
    translate_to_chinese: str = Form("false"),
    api_key: str = Form(None),
    cuts_json: str = Form(None)
):
    try:
        temp_path = os.path.join(UPLOAD_DIR, f"transcribe_{int(time.time())}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract Audio
        video_clip = VideoFileClip(temp_path)
        if video_clip.audio is None:
            video_clip.close()
            if os.path.exists(temp_path): os.remove(temp_path)
            return JSONResponse(status_code=400, content={"detail": "影片沒有音軌，無法進行語音轉文字。"})

        # Determine segments to transcribe
        cuts = []
        if cuts_json:
            try:
                cuts = json.loads(cuts_json)
            except: pass

        # Whisper Options
        transcribe_options = {}
        if whisper_language and whisper_language != "auto":
            lang_arg = "zh" if whisper_language == "zh-tw" else whisper_language
            transcribe_options["language"] = lang_arg
            if whisper_language == "zh" or whisper_language == "zh-tw":
                 transcribe_options["initial_prompt"] = "以下是繁體中文的字幕，請使用台灣繁體中文。"

        full_segments_raw = []

        if cuts and len(cuts) > 0:
            print(f"🗣️ Transcribing {len(cuts)} selected segments...")
            for idx, cut in enumerate(cuts):
                c_start = float(cut.get('start', 0))
                c_end = float(cut.get('end', 0))
                if c_end <= c_start: continue
                
                temp_segment_audio = os.path.join(UPLOAD_DIR, f"seg_trans_{idx}_{int(time.time())}.mp3")
                try:
                    audio_sub = video_clip.audio.subclip(c_start, c_end)
                    audio_sub.write_audiofile(temp_segment_audio, logger=None)
                    
                    if os.path.exists(temp_segment_audio) and os.path.getsize(temp_segment_audio) > 1000:
                        result = whisper_model.transcribe(temp_segment_audio, **transcribe_options)
                        for seg in result["segments"]:
                            seg['start'] += c_start
                            seg['end'] += c_start
                            full_segments_raw.append(seg)
                except Exception as e:
                    print(f"⚠️ Segment {idx} transcription skipped: {e}")
                finally:
                    if os.path.exists(temp_segment_audio): os.remove(temp_segment_audio)
        else:
            print(f"🗣️ Transcribing full video...")
            temp_audio = os.path.join(UPLOAD_DIR, f"full_{int(time.time())}.mp3")
            video_clip.audio.write_audiofile(temp_audio, logger=None)
            result = whisper_model.transcribe(temp_audio, **transcribe_options)
            full_segments_raw = result["segments"]
            if os.path.exists(temp_audio): os.remove(temp_audio)

        video_clip.close()

        raw_segments = optimize_segments(full_segments_raw, max_chars=subtitle_chars_per_line)
        full_subtitles = []
        for i, seg in enumerate(raw_segments):
            full_subtitles.append({
                "id": str(i),
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text'].strip()
            })

        # Translation Logic
        if str(translate_to_chinese).lower() == "true":
            full_subtitles = translate_subtitles(full_subtitles, api_key)

        # Cleanup
        if os.path.exists(temp_path): os.remove(temp_path)

        return full_subtitles
    except Exception as e:
        print(f"❌ Transcribe Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/process-preview-pipeline")
async def process_preview_pipeline(
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    
    # Configs
    is_denoise: str = Form("false"),
    is_silence_removal: str = Form("false"),
    silence_threshold: float = Form(0.3),
    is_auto_caption: str = Form("false"),
    subtitle_config: str = Form(None), # JSON string
    is_face_tracking: str = Form("false"),
    
    # Whisper
    whisper_language: str = Form("zh"),
    translate_to_chinese: str = Form("false"),
    api_key: str = Form(None)
):
    # 1. Save Temp Chunk synchronously first
    ts = int(time.time())
    temp_video_path = os.path.join(UPLOAD_DIR, f"preview_chunk_{ts}_{file.filename}")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return StreamingResponse(
        preview_pipeline_generator(
            temp_video_path, start, end,
            is_denoise, is_silence_removal, silence_threshold,
            is_auto_caption, subtitle_config, is_face_tracking,
            whisper_language, translate_to_chinese, api_key
        ),
        media_type="application/x-ndjson"
    )

def preview_pipeline_generator(
    temp_video_path, start, end,
    is_denoise, is_silence_removal, silence_threshold,
    is_auto_caption, subtitle_config, is_face_tracking,
    whisper_language, translate_to_chinese, api_key
):
    try:
        ts = int(time.time())
        yield json.dumps({"status": "progress", "message": "正在前處理與依需求切割...", "percent": 10}) + "\n"

        # Extract Subclip
        clip = VideoFileClip(temp_video_path)
        if start < 0: start = 0
        if end > clip.duration: end = clip.duration
        
        sub = clip.subclip(start, end)
        temp_sub_path = os.path.join(UPLOAD_DIR, f"sub_{ts}.mp4")
        sub.write_videofile(temp_sub_path, audio_codec='aac', logger=None)
        
        # Audio Extraction
        temp_audio_path = os.path.join(UPLOAD_DIR, f"audio_{ts}.mp3")
        sub.audio.write_audiofile(temp_audio_path, logger=None)
        
        # 2. Denoise (Step 2)
        final_audio_path = temp_audio_path
        if str(is_denoise).lower() == 'true':
            yield json.dumps({"status": "progress", "message": "正在執行 AI 降噪 (Step 2)...", "percent": 30}) + "\n"
            # TODO: Integrate DFN3
            pass

        # 3. Silence Removal (Step 3) - Analysis Only
        visual_segments = [{ "startInVideo": start, "duration": end - start, "zoom": 1.0 }]
        if str(is_silence_removal).lower() == 'true':
            yield json.dumps({"status": "progress", "message": "正在分析與移除氣口 (Step 3)...", "percent": 50}) + "\n"
            # TODO: Integrate Silence Detect
            pass
            
        # 4. Transcribe (Step 3/4)
        subtitles = []
        if str(is_auto_caption).lower() == 'true':
            yield json.dumps({"status": "progress", "message": f"正在生成字幕 ({whisper_language})...", "percent": 70}) + "\n"
            t_opts = {}
            if whisper_language and whisper_language != "auto":
                # Map zh-tw to zh for Whisper, but keep prompt
                lang_arg = "zh" if whisper_language == "zh-tw" else whisper_language
                t_opts["language"] = lang_arg
                
                if whisper_language == "zh" or whisper_language == "zh-tw":
                    t_opts["initial_prompt"] = "以下是繁體中文的字幕，請使用台灣繁體中文。"
            
            if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 1000:
                res = whisper_model.transcribe(final_audio_path, **t_opts)
            else:
                res = {"segments": []}
            
            chars_limit = 14
            if subtitle_config:
                try:
                    c = json.loads(subtitle_config)
                    if 'charsPerLine' in c: chars_limit = c['charsPerLine']
                    yield json.dumps({"status": "progress", "message": "正在套用字幕設定與每行字數...", "percent": 75}) + "\n"
                except: pass
            
            segs = optimize_segments(res['segments'], max_chars=chars_limit)
            
            for i, s in enumerate(segs):
                subtitles.append({
                    "id": str(i),
                    "start": s['start'] + start,
                    "end": s['end'] + start,
                    "text": s['text'].strip()
                })
        
        # 5. Face Tracking (Step 6)
        face_center = 0.5
        if str(is_face_tracking).lower() == 'true':
             yield json.dumps({"status": "progress", "message": "正在偵測人臉位置...", "percent": 90}) + "\n"
             if face_detector:
                 mid_t = (end - start) / 2
                 sc = VideoFileClip(temp_sub_path) # Reopen strict subclip
                 frame = sc.get_frame(sc.duration / 2)
                 
                 import mediapipe as mp # Ensure imported
                 image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                 detect_res = face_detector.detect(image)
                 if detect_res.detections:
                     box = detect_res.detections[0].bounding_box
                     face_center = (box.origin_x + box.width / 2) / frame.shape[1]
                 sc.close()

        # Cleanup
        sub.close()
        clip.close()
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        if os.path.exists(temp_sub_path): os.remove(temp_sub_path)
        
        # Prepare Audio URL
        preview_audio_url = None
        if os.path.exists(final_audio_path):
             preview_audio_name = f"preview_audio_{ts}.mp3"
             shutil.move(final_audio_path, os.path.join(UPLOAD_DIR, preview_audio_name))
             preview_audio_url = f"http://localhost:8000/uploads/{preview_audio_name}"

        result = {
            "status": "success",
            "subtitles": subtitles,
            "faceCenterX": face_center,
            "audioUrl": preview_audio_url,
            "visualSegments": visual_segments
        }
        yield json.dumps({"status": "success", "data": result}) + "\n"

    except Exception as e:
        print(f"Preview Pipeline Error: {e}")
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"

@app.post("/preview-clip")
async def preview_clip_legacy(
    whisper_language: str = Form("zh"),
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
    subtitle_italic: str = Form("false"),
    subtitle_shadow_color: str = Form("#000000"),
    subtitle_shadow_opacity: float = Form(0.8),
    dfn3_strength: int = Form(100),
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
                # --- Handle Output Resolution (Vertical 9:16) for Preview ---
                # To ensure preview looks correct (not shrunk)
                target_w, target_h = (1080, 1920) # Default Preview Resolution
                current_w, current_h = new_clip.size
                if (current_w, current_h) != (target_w, target_h):
                    # Only resize if significantly different to save preview time?
                    # No, better be accurate.
                    new_clip = new_clip.resize(newsize=(target_w, target_h))

                temp_audio = f"{OUTPUT_DIR}/preview_audio.mp3"
                if new_clip.audio:
                    new_clip.audio.write_audiofile(temp_audio, logger=None)
                
                # Use the upgraded Whisper Turbo model with language hint
                transcribe_options = {}
                if whisper_language and whisper_language != "auto":
                     # Map zh-tw to zh for Whisper, but keep prompt
                     lang_arg = "zh" if whisper_language == "zh-tw" else whisper_language
                     transcribe_options["language"] = lang_arg
                     
                     if whisper_language == "zh" or whisper_language == "zh-tw":
                         transcribe_options["initial_prompt"] = "以下是繁體中文的字幕，請使用台灣繁體中文。"

                print(f"🗣️ Preview Transcribing with language: {whisper_language}")
                
                if new_clip.audio:
                     result = whisper_model.transcribe(temp_audio, **transcribe_options)
                else:
                     result = {"segments": []} # No audio
                
                # --- OPTIMIZE SEGMENTS FOR VERTICAL VIDEO ---
                # Force split long segments to avoid "subtitle wall"
                opt_max_chars = 18 if aspect_ratio == "9:16" else 30
                optimized_segments = optimize_segments(result["segments"], max_chars=opt_max_chars)
                
                # --- AI Translation for Preview ---
                # ... (Can be added later if needed)
                
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
                    
                    for idx, s in enumerate(optimized_segments):
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
                abs_fonts_f = abs_fonts.replace("\\", "/").replace(":", "\\:") # Use correct variable for filter

                # Consistent Style Calculation
                bold_val = 1 if str(subtitle_bold).lower() == "true" else 0
                italic_val = 1 if str(subtitle_italic).lower() == "true" else 0
                final_color_ass = hex_to_ass(subtitle_color)
                
                # Boost outline for visibility if white-on-white risk
                try: outline_width_int = int(subtitle_outline_width)
                except: outline_width_int = 2
                if final_color_ass == "&H00FFFFFF" and outline_width_int < 2:
                    outline_width_int = 3

                outline_color_ass = hex_to_ass(subtitle_outline_color)
                box_color_ass = hex_to_ass(subtitle_box_color, float(subtitle_box_alpha))
                
                # Custom Shadow
                shadow_color_ass = hex_to_ass(subtitle_shadow_color, float(subtitle_shadow_opacity))
                
                is_box = (subtitle_box_enabled == "true") or (subtitle_style == "box") or (subtitle_box_enabled == "on")
                border_style = 3 if is_box else 1
                
                try: box_pad_int = int(subtitle_box_padding)
                except: box_pad_int = 10
                
                outline_val = box_pad_int if is_box else outline_width_int
                shadow_val = 0 if is_box else int(subtitle_shadow_size)
                
                # FIX: Scale Params for Consistency (Same as Export)
                target_res_y = 1080
                if aspect_ratio == "9:16":
                     target_res_y = 1920
                
                scale_factor = target_res_y / 1080.0
                
                # CRITICAL FIX: MarginV is 0-200 relative value from Frontend (100 = 50% height)
                # We must convert this to pixels relative to target_res_y
                raw_scaled_margin = int(target_res_y * (float(subtitle_margin_v) / 200.0))
                # CLAMP: Ensure it doesn't fly off screen (max 90% height)
                scaled_margin_v = min(raw_scaled_margin, int(target_res_y * 0.9))
                
                # Font size is 1080p-based pixels, so we scale it normally for 1920p
                scaled_fontsize = int(float(subtitle_font_size) * scale_factor)
                scaled_outline = int(outline_val * scale_factor)
                scaled_shadow = int(shadow_val * scale_factor)

                # FIX: Use Real Family Name for ASS
                real_family = FONT_FILE_MAP.get(subtitle_font_name, subtitle_font_name)
                escaped_font = real_family.replace("'", "").replace(":", "") 
                
                # Standard ASS Style String
                style_parts = [
                    f"PlayResY={target_res_y}", f"Fontname={escaped_font}", f"Fontsize={scaled_fontsize}",
                    f"PrimaryColour={final_color_ass}", f"SecondaryColour={final_color_ass}",
                    f"OutlineColour={outline_color_ass if not is_box else box_color_ass}",
                    f"BackColour={shadow_color_ass if not is_box else box_color_ass}",
                    f"BorderStyle={border_style}", f"Outline={scaled_outline}", f"Shadow={scaled_shadow}",
                    f"MarginV={scaled_margin_v}", f"MarginL={subtitle_margin_h}", f"MarginR={subtitle_margin_h}",
                    "Alignment=2", f"Bold={bold_val}", f"Italic={italic_val}"
                ]
                force_style_arg = ",".join(style_parts)

                # SMART FONT HANDLING (Same as Export)
                font_in_dir = False
                for ext in [".ttf", ".otf", ".ttc"]:
                    if os.path.exists(os.path.join(FONTS_DIR, subtitle_font_name + ext)):
                        font_in_dir = True
                        break
                
                vf_string = f"subtitles='{safe_srt_path}':force_style='{force_style_arg}'"
                if font_in_dir:
                     # Only append fontsdir if we have the file
                     vf_string = f"subtitles='{safe_srt_path}':fontsdir='{abs_fonts_f}':force_style='{force_style_arg}'"
                else:
                     print(f"ℹ️ [Preview] Font '{subtitle_font_name}' not found locally, trying system fonts...")

                cmd = [
                    ffmpeg_exe, "-y", "-i", abs_in, 
                    "-vf", vf_string,
                    "-c:a", "copy", "-c:v", "libx264", "-crf", "17", "-preset", "slow", abs_out
                ]
                
                try:
                    subprocess.run(cmd, check=True, env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if os.path.exists(burned_path):
                        os.remove(output_path)
                        os.rename(burned_path, output_path)
                except subprocess.CalledProcessError as e:
                    err_msg = e.stderr.decode() if e.stderr else "Unknown"
                    print(f"⚠️ Primary preview burn failed: {err_msg}")
                    print("🔄 Retrying with system fonts fallback...")
                    
                    # Fallback: Remove fontsdir and force_style complexity if needed, or just fontsdir
                    vf_simple = f"subtitles='{safe_srt_path}':force_style='{force_style_arg}'"
                    cmd_fallback = [
                        ffmpeg_exe, "-y", "-i", abs_in, 
                        "-vf", vf_simple,
                        "-c:a", "copy", "-c:v", "libx264", "-crf", "17", "-preset", "slow", abs_out
                    ]
                    try:
                        subprocess.run(cmd_fallback, check=True, env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        if os.path.exists(burned_path):
                            os.remove(output_path)
                            os.rename(burned_path, output_path)
                            print("✅ Preview burn succeeded with fallback.")
                    except subprocess.CalledProcessError as e2:
                        print(f"❌ Preview fallback burn also failed: {e2.stderr.decode() if e2.stderr else 'Unknown'}")

            except Exception as e:
                print(f"🔥 Preview burn exception: {e}")

        return FileResponse(output_path, media_type="video/mp4")
        
    except Exception as e:
        print(f"Preview Error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/detect-face-clip")
def detect_face_clip(
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...)
):
    try:
        # Save temp video
        temp_path = f"{UPLOAD_DIR}/temp_preview_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        mid_point = start + (end - start) / 2
        cap = cv2.VideoCapture(temp_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_point * 1000)
        ret, frame = cap.read()
        cap.release()
        
        # Cleanup
        if os.path.exists(temp_path): os.remove(temp_path)
        
        if not ret:
            return {"faceCenterX": 0.5}
            
        h, w, _ = frame.shape
        face_center_x = 0.5
        
        # MediaPipe Tasks API
        if hasattr(face_detector, 'detect'):
            import mediapipe as mp
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
             faces = face_cascade.detectMultiScale(gray, 1.1, 4)
             if len(faces) > 0:
                 (x, y, wf, hf) = faces[0]
                 face_center_x = (x + wf / 2) / w
                 
        print(f"👁️ Preview Face Detect: {face_center_x:.2f}")
        return {"faceCenterX": face_center_x}

    except Exception as e:
        print(f"❌ Preview Detect Error: {e}")
        return {"faceCenterX": 0.5}

@app.post("/process-video")
def process_video(
    file: UploadFile = File(...),
    cuts_json: str = Form(...),
    model_name: str = Form(None),
    vertical_mode: str = Form("false"),
    aspect_ratio: str = Form("9:16"),
    face_tracking: str = Form("true"),
    auto_caption: str = Form("false"),
    translate_to_chinese: str = Form("false"),
    whisper_language: str = Form("zh"), # NEW: Default to Traditional Chinese (zh) which usually handles both, but we can hint it
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
    subtitle_chars_per_line: int = Form(7),
    subtitle_box_opacity: int = Form(50), 
    subtitle_box_padding_x: int = Form(10),
    subtitle_box_padding_y: int = Form(4),
    subtitle_box_radius: int = Form(4),
    subtitle_shadow_opacity: int = Form(80),
    dfn3_strength: int = Form(100),
    subtitle_box_color: str = Form("#000000"),
    subtitle_box_enabled: str = Form("false"),
    is_silence_removal: str = Form("false"),
    silence_threshold: float = Form(0.5),
    is_jump_cut_zoom: str = Form("true"),
    output_resolution: str = Form("1080p"),
    output_quality: str = Form("high"),
    srt_json: str = Form(None),
    subtitle_animation: str = Form("pop"),
    subtitle_outline_width: float = Form(0),
    subtitle_outline_color: str = Form("#000000"),
    subtitle_shadow_color: str = Form("#000000")
):
    global current_job_status
    try:
        print(f"🎬 Processing Request: Vertical={vertical_mode}, Mode={output_mode}, Style={subtitle_style}")
        print(f"📊 DEBUG FORM DATA: Font={subtitle_font_size}, MarginV={subtitle_margin_v}, Chars={subtitle_chars_per_line}, LineHeight={subtitle_line_height}")
        print(f"📊 DEBUG FORM DATA: Shadow={subtitle_shadow_opacity}, Outline={subtitle_outline_width}, Spacing={subtitle_letter_spacing}")
        # print(f"   Quality: {output_quality}, Res: {output_resolution}")
        
        if output_mode == "preview_url":
            print("ℹ️ legacy preview mode requested, ignoring...")

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
            current_job_status = {"progress": 100, "message": "No cuts", "step": "done"}
            return {"message": "No cuts provided"}

        print(f"🎬 Processing video: {file.filename} with Remotion")
        print(f"🔍 Cuts payload: {json.dumps(cuts, indent=2)}") 
        print(f"🔍 Debug Props: FaceTracking={face_tracking}, DetectorReady={face_detector is not None}")
        print(f"🎨 Debug Config: Outline={subtitle_outline_width}, Shadow={subtitle_shadow_opacity}")
        
        current_job_status = {"progress": 5, "message": "正在分析字幕與音訊...", "step": "transcribing"}

        # 3. Analyze Audio & Transcribe (Keep AI Logic in Python)
        full_subtitles = []
        full_segments_raw = [] # Keep raw segments for silence detection logic

        # Extract Audio for Whisper
        transcribe_options = {}
        if whisper_language and whisper_language != "auto":
            # Map zh-tw to zh for Whisper, but keep prompt
            lang_arg = "zh" if whisper_language == "zh-tw" else whisper_language
            transcribe_options["language"] = lang_arg
            
            if whisper_language == "zh" or whisper_language == "zh-tw":
                 transcribe_options["initial_prompt"] = "以下是繁體中文的字幕，請使用台灣繁體中文。"
        
        # Determine if we need to force word timestamps? 
        # For better silence removal, word-level is ideal. 
        # But standard segments usually break on pause. Let's use standard for speed MVP.

        try:
             # ONLY transcribe if burn_captions is true or auto_caption is explicitly on
             is_subtitle_needed = str(burn_captions).lower() == "true" or str(auto_caption).lower() == "true"
             
             if is_subtitle_needed:
                 if srt_json and srt_json != "[]":
                     print("📄 Using provided SRT JSON...")
                     full_subtitles = json.loads(srt_json)
                     full_segments_raw = [{"start": s['start'], "end": s['end'], "text": s['text']} for s in full_subtitles]
                 else:
                     print("🎙️ Extracting Audio & Transcribing by Cuts...")
                     video_clip = VideoFileClip(video_path)
                     if video_clip.audio is None:
                         video_clip.close()
                         raise ValueError("影片沒有音訊軌道，無法生成字幕。")
                     full_segments_raw = []
                     
                     total_cuts_count = len(cuts)
                     
                     for idx, cut in enumerate(cuts):
                         c_start = float(cut['start'])
                         c_end = float(cut['end'])
                         print(f"   Processing Cut {idx+1}/{total_cuts_count}: {c_start}-{c_end}s")
                         
                         current_job_status = {
                             "progress": 10 + int((idx / total_cuts_count) * 10), 
                             "message": f"正在轉錄片段 {idx+1}/{total_cuts_count}...", 
                             "step": "transcribing"
                         }
                         
                         # Extract Audio for specific cut
                         temp_audio_path = os.path.join(UPLOAD_DIR, f"temp_whisper_{idx}.mp3")
                         # Use subclip.audio to avoid loading full video into RAM if possible, but VideoFileClip is efficient
                         video_clip.audio.subclip(c_start, c_end).write_audiofile(temp_audio_path, logger=None)
                         
                         # Transcribe
                         if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 1000:
                             result = whisper_model.transcribe(temp_audio_path, **transcribe_options)
                         else:
                             result = {"segments": []}
                         
                         # Shift timestamps and Add
                         for seg in result["segments"]:
                             seg['start'] += c_start
                             seg['end'] += c_start
                             full_segments_raw.append(seg)
                             
                         # Cleanup temp file
                         if os.path.exists(temp_audio_path):
                             os.remove(temp_audio_path)
                             
                     video_clip.close()

                     # Convert to Remotion Subtitle format
                     raw_segments = optimize_segments(full_segments_raw, max_chars=subtitle_chars_per_line)
                     for i, seg in enumerate(raw_segments):
                         full_subtitles.append({
                             "id": str(i),
                             "start": seg['start'],
                             "end": seg['end'],
                             "text": seg['text'].strip()
                         })
                      
                     # Apply Translation if requested
                     if str(translate_to_chinese).lower() == "true":
                         current_job_status = {"progress": 15, "message": "正在翻譯字幕...", "step": "translating"}
                         full_subtitles = translate_subtitles(full_subtitles, api_key)

                     if os.path.exists(temp_audio_path):
                         os.remove(temp_audio_path)
             else:
                 print("⏭️ Subtitles disabled, skipping transcription")
                 full_subtitles = []
                 full_segments_raw = []
        except Exception as e:
             print(f"⚠️ Transcription failed: {e}")
             current_job_status = {"progress": 15, "message": f"字幕生成失敗: {e}", "step": "error"}
             full_subtitles = []
             full_segments_raw = []

        # Clear previous exports
        for f in os.listdir(OUTPUT_DIR):
             try: os.remove(os.path.join(OUTPUT_DIR, f))
             except: pass
        
        processed_files = []

        # 4. Render Each Cut using Remotion
        # We will iterate cuts and call remotion CLI for each
        
        current_job_status = {"progress": 18, "message": "正在套用字幕樣式設定...", "step": "styling"}
        print(f"DEBUG: burn_captions={burn_captions}, auto_caption={auto_caption}")

        # Pre-calculate Props for Remotion
        # Color converting & Config packing
        try:
             gradient_colors_list = json.loads(text_gradient_colors)
        except:
             gradient_colors_list = ["#FFFFFF", "#FFFFFF"] # Fallback

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

        # Resolve Absolute Path for Video (Remotion needs absolute or http)
        abs_video_path = os.path.abspath(video_path)
        
        total_cuts = len(cuts)
        for i, cut in enumerate(cuts):
            # Update Progress (20% to 90%)
            progress_pct = 20 + int((i / total_cuts) * 70)
            current_job_status = {
                "progress": progress_pct, 
                "message": f"正在渲染片段 {i+1} / {total_cuts}...", 
                "step": "rendering"
            }
            
            start = float(cut.get('start', 0))
            end = float(cut.get('end', 0))
            print(f"✂️ Processing Cut {i+1}: Start={start}s, End={end}s, Duration={end-start}s")
            raw_label = cut.get('label', f'Clip{i+1}')
            safe_label = "".join([c for c in raw_label if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
            
            output_filename = f"Highlight_{i+1}_{safe_label}.mp4"
            output_app_path = os.path.join(OUTPUT_DIR, output_filename)
            abs_output_path = os.path.abspath(output_app_path)
            
            # Filter subtitles for this specific clip range
            # ONLY pass to Remotion if burn_captions is True
            clip_subs = []
            if str(burn_captions).lower() == "true":
                clip_subs = [s for s in full_subtitles if s['start'] < end and s['end'] > start]
            
            # --- Silence Removal Logic ---
            visual_segments = []
            final_duration = end - start
            
            # --- Face Tracking Logic ---
            face_center_x = 0.5 # Default center
            is_tracking_enabled = str(face_tracking).lower() in ["true", "1", "yes", "on"]
            
            if is_tracking_enabled and face_detector:
                current_job_status = {
                    "progress": 20 + int((i / total_cuts) * 70), 
                    "message": f"正在分析人臉位置 (片段 {i+1}/{total_cuts})...", 
                    "step": "face_tracking"
                }
                try:
                    import cv2
                    import mediapipe as mp
                    
                    # Analyze the middle frame of the clip for face position
                    mid_point = start + (final_duration / 2)
                    
                    # DIRECT OPENCV CAPTURE (Faster & Safer)
                    rgb_frame = None
                    try:
                        cap = cv2.VideoCapture(abs_video_path)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_POS_MSEC, mid_point * 1000)
                            ret, frame = cap.read()
                            if ret:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            cap.release()
                    except Exception as ve:
                         print(f"⚠️ OpenCV Frame Extract Failed: {ve}")

                    # Fallback to FFmpeg ONLY if OpenCV failed
                    if rgb_frame is None:
                        print("⚠️ OpenCV failed to grab frame, falling back to FFmpeg CLI...")
                        temp_frame_path = os.path.join(UPLOAD_DIR, f"temp_face_{i}_{int(time.time())}.jpg")
                        try:
                            cmd = [
                                "ffmpeg", "-y", 
                                "-ss", str(mid_point),
                                "-i", abs_video_path,
                                "-frames:v", "1",
                                "-q:v", "2",
                                "-update", "1", # Ensure single image update
                                temp_frame_path
                            ]
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                            
                            if os.path.exists(temp_frame_path):
                                bgr_frame = cv2.imread(temp_frame_path)
                                if bgr_frame is not None:
                                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                                os.remove(temp_frame_path)
                        except Exception as fe:
                             print(f"⚠️ FFmpeg Fallback also failed: {fe}")

                    if rgb_frame is not None:
                        h, w, _ = rgb_frame.shape
                        
                        # Use Tasks API or Legacy or OpenCV
                        if hasattr(face_detector, 'detect'): # Tasks API
                            # MoviePy returns RGB, perfect for MediaPipe
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            detection_result = face_detector.detect(mp_image)
                            if detection_result.detections:
                                bbox = detection_result.detections[0].bounding_box
                                center = bbox.origin_x + (bbox.width / 2)
                                face_center_x = center / w
                                print(f"👤 Face detected (Tasks API) at X={face_center_x:.2f}")
                            else:
                                print(f"👤 No face detected (Tasks API) in frame")
                        
                        elif hasattr(face_detector, 'process'): # Legacy API
                            results = face_detector.process(rgb_frame)
                            if results.detections:
                                bboxC = results.detections[0].location_data.relative_bounding_box
                                face_center_x = bboxC.xmin + (bboxC.width / 2)
                                print(f"👤 Face detected at X={face_center_x:.2f}")
                                
                        elif face_detector == "opencv_fallback": # OpenCV
                            # Convert RGB back to BGR/GRAY for OpenCV
                            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            if len(faces) > 0:
                                (x, y, w_f, h_f) = faces[0]
                                face_center_x = (x + w_f / 2) / w
                                print(f"👤 Face detected (OpenCV) at X={face_center_x:.2f}")
                            else:
                                print(f"👤 No face detected (OpenCV) in frame")
                    else:
                        print(f"⚠️ Could not read frame at {mid_point}s")

                except Exception as e:
                    print(f"⚠️ Face detection ERROR for clip {i}: {e}")
            else:
                 print(f"ℹ️ Face tracking skipped. Enabled: {face_tracking}, Detector: {face_detector is not None}")

            if str(is_silence_removal).lower() == "true":
                # Find segments within this cut [start, end]
                # We use raw segments from Whisper because they represent "Speech Activity"
                active_segments = []
                for seg in full_segments_raw:
                    # Check overlap
                    s_ov = max(start, seg['start'])
                    e_ov = min(end, seg['end'])
                    if e_ov > s_ov:
                        active_segments.append({'start': s_ov, 'end': e_ov})
                
                # If no speech found, backup to full clip
                if not active_segments:
                    visual_segments = [{'startInVideo': start, 'duration': end - start, 'zoom': 1.0}]
                else:
                    # Construct keep intervals based on speech
                    # Logic: Fill the timeline with speech blocks. 
                    # If gap between blocks > silence_threshold, it's a gap (jump cut). 
                    # If gap < silence_threshold, we bridge it (keep as one block).
                    
                    merged_blocks = []
                    if active_segments:
                        current_block = active_segments[0].copy()
                        
                        for i in range(1, len(active_segments)):
                            next_seg = active_segments[i]
                            gap = next_seg['start'] - current_block['end']
                            
                            if gap <= silence_threshold:
                                # Bridge small gap
                                current_block['end'] = next_seg['end']
                            else:
                                # Finalize current block
                                merged_blocks.append(current_block)
                                current_block = next_seg.copy()
                        merged_blocks.append(current_block)
                    
                    # Convert blocks to Remotion visual segments
                    # Alternate Zoom for Jump Cuts
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
                        
                        # Toggle zoom if enabled
                        if str(is_jump_cut_zoom).lower() == "true":
                            zoom_level = 1.15 if zoom_level == 1.0 else 1.0
                            
                    final_duration = total_dur
            else:
                 # Default: One single continuous segment
                 visual_segments = [{'startInVideo': start, 'duration': end - start, 'zoom': 1.0}]

            print(f"🚀 Rendering Clip {i+1}: {safe_label} (Dur: {final_duration:.2f}s, Segments: {len(visual_segments)})")

            # Construct Input Props
            # Use UUID based filename for public access (Handled below)
            video_filename = os.path.basename(video_path)
            
            # === Copy video to public folder for Remotion access ===
            public_dir = os.path.abspath("public")
            os.makedirs(public_dir, exist_ok=True)
            
            # Use a simple sanitized filename to avoid URL encoding issues
            import uuid
            unique_id = str(uuid.uuid4())
            sanitized_video_name = f"render_source_{unique_id}.mp4"
            public_video_path = os.path.join(public_dir, sanitized_video_name)
            
            # Copy video to public folder if not already there
            if not os.path.exists(public_video_path):
                shutil.copy2(video_path, public_video_path)
            
            # Use staticFile path for Remotion
            input_props = {
                "videoUrl": sanitized_video_name,  # Remotion will use staticFile() for this
                "startFrom": start,
                "visualSegments": visual_segments,
                "duration": final_duration, 
                "subtitles": clip_subs,
                "subtitleConfig": remotion_config,
                "isFaceTracking": is_tracking_enabled,
                "faceCenterX": face_center_x,
                "durationInFrames": int(final_duration * 30) # Explicitly pass frames for metadata
            }

            print(f"📦 Debug Input Props: FaceTracking={input_props['isFaceTracking']}, Center={input_props['faceCenterX']}")
            print(f"📦 Debug Config: {json.dumps(remotion_config)}")
            
            # Write props to temp file
            props_file = os.path.abspath(f"temp_props_{unique_id}.json")
            with open(props_file, "w") as pf:
                json.dump(input_props, pf)

            # Calculate Duration in Frames (30fps)
            dur_in_frames = int(final_duration * 30)
            print(f"⏱️ Calculated Duration: {final_duration}s -> {dur_in_frames} frames")
            
            # Build Remotion Command
            cmd = [
                "npx", "remotion", "render",
                "src/remotion/index.ts",
                "MainComposition",
                abs_output_path,
                f"--props={props_file}",
                f"--duration={dur_in_frames}",
                "--log=verbose",
                "--concurrency=1",
                "--timeout=120000"
            ]
            
            # Execute Remotion
            try:
                subprocess.run(cmd, check=True, cwd=os.getcwd())
                processed_files.append(output_app_path)
                print(f"✅ Rendered: {output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Remotion render failed for clip {i}: {e}")
            finally:
                if os.path.exists(props_file): os.remove(props_file)
                # Cleanup public video copy
                if os.path.exists(public_video_path): os.remove(public_video_path)

        # 5. Generate global SRT file if auto_caption or burn_captions is on
        if full_subtitles:
            srt_filename = "transcription.srt"
            srt_path = os.path.join(OUTPUT_DIR, srt_filename)
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, s in enumerate(full_subtitles):
                    # Use the same format_timestamp helper
                    f.write(f"{idx+1}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{s['text']}\n\n")
            processed_files.append(srt_path)

        # 6. Zip results
        if not processed_files:
             raise ValueError("Render failed: No files generated.")

        zip_filename = f"Antigravity_Shorts_{len(cuts)}_Clips.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)
        
        print(f"📦 Packaging {len(processed_files)} clips into: {zip_path}")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in processed_files:
                if os.path.exists(file):
                    zipf.write(file, os.path.basename(file))
        
        current_job_status = {"progress": 100, "message": "處理完成！", "step": "done"}
                
        # Return JSON with download URL instead of FileResponse for better flexibility
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Video Processing Server Running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
