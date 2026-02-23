"""Subtitle utility functions: formatting, wrapping, optimization, translation."""

import re
import math


def hex_to_ass(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB to &HAABBGGRR for ASS subtitles."""
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a_val = int((1.0 - alpha) * 255)
        return f"&H{a_val:02X}{b:02X}{g:02X}{r:02X}"
    except (ValueError, IndexError):
        return "&H00FFFFFF"


def wrap_text(text: str, max_chars: int = 12) -> str:
    """Improved wrapping for subtitles (supports CJK and English)."""
    if not text or max_chars <= 0:
        return text

    lines = text.split('\n')
    new_lines = []

    for line in lines:
        if len(line) <= max_chars:
            new_lines.append(line)
            continue

        if re.search(r'[\u4e00-\u9fff]', line):
            for i in range(0, len(line), max_chars):
                new_lines.append(line[i:i + max_chars])
        else:
            import textwrap
            new_lines.extend(textwrap.wrap(line, width=max_chars))

    return "\n".join(new_lines)


def optimize_segments(
    segments: list,
    max_chars: int = 14,
    remove_punctuation: bool = True
) -> list:
    """
    Splits long whisper segments into shorter ones based on character count and duration.
    MERGES tiny segments unless there is a strong sentence pause.
    Uses punctuation cues for better splitting before optionally removing them.
    """
    if not segments:
        return []

    print(f"🛠️ optimize_segments: input={len(segments)} segs, max_chars={max_chars}, remove_punc={remove_punctuation}")

    CJK_STOPS = r'[。？！]'
    ENG_STOPS = r'[.?!]'
    ALL_STOPS = CJK_STOPS + r'|' + ENG_STOPS

    working_segs = []
    for s in segments:
        text = s['text'].strip()
        if text:
            working_segs.append({"start": s['start'], "end": s['end'], "text": text})

    if not working_segs:
        return []

    # SMART MERGE
    merged = []
    current = working_segs[0].copy()

    for i in range(1, len(working_segs)):
        next_seg = working_segs[i]
        curr_text = current['text']

        has_stop = re.search(ALL_STOPS + r'$', curr_text.strip())
        combined_len = len(curr_text) + len(next_seg['text'])

        if not has_stop and combined_len <= max_chars:
            t1_end = curr_text[-1]
            t2_start = next_seg['text'][0]
            is_latin_bound = re.match(r'[a-zA-Z0-9]', t1_end) and re.match(r'[a-zA-Z0-9]', t2_start)
            joiner = " " if is_latin_bound else ""
            current['text'] = curr_text + joiner + next_seg['text']
            current['end'] = next_seg['end']
        else:
            merged.append(current)
            current = next_seg.copy()
    merged.append(current)

    # SMART SPLIT
    new_segments = []
    tolerance_factor = 1.2
    soft_limit = max_chars * tolerance_factor
    SPLIT_PATTERN = re.compile(r'([，。、？！：；,.?!:;"\'\s])')

    for s in merged:
        text = s['text']

        if len(text) > soft_limit:
            duration = s['end'] - s['start']
            tokens = SPLIT_PATTERN.split(text)
            current_chunk = ""
            chunks = []

            for token in tokens:
                if not token:
                    continue
                if len(current_chunk) + len(token) > max_chars and len(current_chunk) > 0:
                    is_punc = SPLIT_PATTERN.match(token)
                    if is_punc and len(current_chunk) + len(token) <= soft_limit:
                        current_chunk += token
                    else:
                        chunks.append(current_chunk)
                        current_chunk = token
                else:
                    current_chunk += token

            if current_chunk:
                chunks.append(current_chunk)

            total_chars = sum(len(c) for c in chunks)
            curr_t = s['start']
            for c in chunks:
                if not c.strip():
                    continue
                c_len = len(c)
                c_dur = (c_len / total_chars) * duration if total_chars > 0 else 0
                new_segments.append({"start": curr_t, "end": curr_t + c_dur, "text": c})
                curr_t += c_dur
        else:
            new_segments.append(s)

    # CLEANUP
    final_output = []
    for s in new_segments:
        final_text = s['text']
        if remove_punctuation:
            final_text = re.sub(r'[，。、？！：；]', '', final_text)
            final_text = re.sub(r'\s+', ' ', final_text).strip()
            final_text = re.sub(r'(?<=[\d])\s*\.\s*(?=[\d])', '.', final_text)
            final_text = re.sub(r'(?<=[\d])\s+(?=[\d])', '', final_text)
            final_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', final_text)
            final_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\d])', '', final_text)
            final_text = re.sub(r'(?<=[\d])\s+(?=[\u4e00-\u9fff])', '', final_text)
        else:
            final_text = final_text.strip()

        if final_text:
            s['text'] = final_text
            final_output.append(s)

    print(f"✅ optimize_segments: output={len(final_output)} segs")
    return final_output


def translate_subtitles(full_subtitles: list, api_key: str) -> list:
    """Convert subtitles from Simplified to Traditional Chinese using OpenCC."""
    if not full_subtitles:
        return full_subtitles

    print(f"變換 🔀 Converting {len(full_subtitles)} segments to Traditional Chinese...")
    try:
        from opencc import OpenCC
        try:
            cc = OpenCC('s2tw')
        except Exception:
            cc = OpenCC('s2t')

        for i, sub in enumerate(full_subtitles):
            if 'text' in sub:
                old_text = sub['text']
                sub['text'] = cc.convert(sub['text'])
                if i == 0:
                    print(f"   [DEBUG] Conversion Sample: '{old_text}' -> '{sub['text']}'")
    except Exception as e:
        print(f"⚠️ OpenCC conversion failed: {e}")

    return full_subtitles


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ','
) -> str:
    """Formats seconds to HH:MM:SS,MMM for SRT files."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"
    return f"{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


def parse_time(value) -> float:
    """Converts various time formats (float, string, MM:SS) to seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
        if ':' in value:
            parts = value.split(':')
            try:
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                if len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                pass
    return 0.0
