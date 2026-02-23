"""Route registration for all API endpoints."""

from fastapi import FastAPI

from backend.routes.fonts import router as fonts_router
from backend.routes.transcription import router as transcription_router
from backend.routes.analysis import router as analysis_router
from backend.routes.preview import router as preview_router
from backend.routes.export import router as export_router


def register_routes(app: FastAPI) -> None:
    """Include all route modules into the FastAPI app."""
    app.include_router(fonts_router)
    app.include_router(transcription_router)
    app.include_router(analysis_router)
    app.include_router(preview_router)
    app.include_router(export_router)
