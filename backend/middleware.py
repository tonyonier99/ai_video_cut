"""Custom middleware for the backend."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware


class CustomCORSMiddleware(BaseHTTPMiddleware):
    """Handle CORS for static file paths that StaticFiles doesn't support."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(("/fonts", "/exports", "/uploads")):
            if request.method == "OPTIONS":
                response = Response(status_code=200)
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "*"
                response.headers["Access-Control-Allow-Headers"] = "*"
                return response

            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response

        return await call_next(request)


def setup_middleware(app) -> None:
    """Configure all middleware for the FastAPI app."""
    app.add_middleware(CustomCORSMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
