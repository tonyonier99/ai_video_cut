"""Font management routes."""

import os
import shutil
from fastapi import APIRouter, UploadFile, File
from fontTools.ttLib import TTFont

from backend.config import FONTS_DIR
from backend.services.fonts import scan_fonts, FONT_FILE_MAP

router = APIRouter()


@router.post("/upload-font")
async def upload_font(file: UploadFile = File(...)):
    """Upload a font file and register it."""
    filename = file.filename
    save_path = os.path.join(FONTS_DIR, filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    base_name = os.path.splitext(filename)[0]
    real_name = base_name
    try:
        font = TTFont(save_path)
        for record in font['name'].names:
            if record.nameID == 1:
                real_name = record.toUnicode()
                break
    except Exception:
        pass

    from backend.services import fonts
    fonts.FONT_FILE_MAP[base_name] = real_name
    scan_fonts()

    return {"status": "ok", "name": real_name, "filename": filename}


@router.get("/list-fonts")
async def list_fonts():
    """Return list of available custom fonts."""
    fonts_list = []
    for base_name, real_name in FONT_FILE_MAP.items():
        for ext in ['.ttf', '.otf', '.ttc']:
            fpath = os.path.join(FONTS_DIR, base_name + ext)
            if os.path.exists(fpath):
                fonts_list.append({
                    "name": real_name,
                    "filename": base_name + ext
                })
                break
    return {"fonts": fonts_list}
