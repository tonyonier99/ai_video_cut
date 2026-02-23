"""Antigravity Cut — Video Processing Server entry point."""

import threading

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import UPLOAD_DIR, OUTPUT_DIR, FONTS_DIR
from backend.middleware import setup_middleware
from backend.models import init_all_models
from backend.routes import register_routes
from backend.services.fonts import scan_fonts
from backend.services.whisper import download_whisper_model

# Initial font scan
scan_fonts()

# Start Whisper model loading in background
threading.Thread(target=download_whisper_model, args=("turbo",), daemon=True).start()

# Initialize ML models (face detection, face mesh, studio sound)
init_all_models()

# Create FastAPI app
app = FastAPI()

# Register all routes
register_routes(app)

# Setup CORS and custom middleware
setup_middleware(app)

# Mount static file directories
app.mount("/exports", StaticFiles(directory=OUTPUT_DIR), name="exports")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Video Processing Server Running on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
