import os
import imageio_ffmpeg

# CRITICAL: Set FFmpeg path BEFORE importing moviepy to fix dependency issues
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# Also Add to PATH for Whisper (which calls 'ffmpeg' subprocess)
try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    local_bin = os.path.abspath("local_bin")
    os.makedirs(local_bin, exist_ok=True)
    target_link = os.path.join(local_bin, "ffmpeg")
    if os.path.exists(target_link):
        os.remove(target_link)
    os.symlink(ffmpeg_exe, target_link)
    os.environ["PATH"] += os.pathsep + local_bin
    print(f"✅ Created Symlink and Added to PATH: {target_link}")
except Exception as e:
    print(f"⚠️ Failed to add FFmpeg to PATH: {e}")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "exports"
FONTS_DIR = "fonts"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
