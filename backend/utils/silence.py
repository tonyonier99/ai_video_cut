"""Silence detection utility using ffmpeg."""

import subprocess


def detect_silence_ffmpeg(
    input_path: str,
    noise_db: float = -30,
    duration: float = 0.5
) -> list[tuple[float, float]]:
    """
    Detects SILENCE using ffmpeg silencedetect filter.
    Returns list of SILENT segments: [(start, end), ...]
    """
    cmd = [
        "ffmpeg", "-i", input_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={duration}",
        "-f", "null", "-"
    ]

    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    output = result.stderr

    silence_starts = []
    silence_ends = []

    for line in output.splitlines():
        if "silence_start" in line:
            try:
                t = float(line.split("silence_start: ")[1])
                silence_starts.append(t)
            except (ValueError, IndexError):
                pass
        elif "silence_end" in line:
            try:
                t = float(line.split("silence_end: ")[1].split("|")[0])
                silence_ends.append(t)
            except (ValueError, IndexError):
                pass

    if len(silence_starts) > len(silence_ends):
        silence_starts.pop()

    segments = []
    for s, e in zip(silence_starts, silence_ends):
        segments.append((s, e))

    return segments
