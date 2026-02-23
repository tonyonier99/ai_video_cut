"""Font scanning and management service."""

import os
from fontTools.ttLib import TTFont
from backend.config import FONTS_DIR

FONT_FILE_MAP: dict[str, str] = {}


def scan_fonts() -> None:
    """Scan fonts directory and extract real font family names using fontTools."""
    global FONT_FILE_MAP
    FONT_FILE_MAP = {}
    if not os.path.exists(FONTS_DIR):
        return

    print("🔍 Scanning fonts for real Family Names...")
    for f in os.listdir(FONTS_DIR):
        if f.lower().endswith(('.ttf', '.otf', '.ttc')):
            base_name = os.path.splitext(f)[0]
            try:
                path = os.path.join(FONTS_DIR, f)
                font = TTFont(path)
                family = ""
                for record in font['name'].names:
                    try:
                        if record.nameID in [1, 16]:
                            decoded = record.toUnicode()
                            if all(ord(c) < 128 for c in decoded):
                                if record.nameID == 16:
                                    family = decoded
                                    break
                                if record.nameID == 1 and not family:
                                    family = decoded
                    except (UnicodeDecodeError, Exception):
                        pass

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
