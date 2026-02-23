"""Whisper model management and transcription options."""

import whisper
from backend import models


def download_whisper_model(model_name: str = "turbo") -> None:
    """Download/load a Whisper model with fallback logic."""
    models.CURRENT_WHISPER_NAME = model_name

    try:
        models.model_status = {
            "progress": 50,
            "status": "downloading",
            "message": f"正在載入 Whisper {model_name} 模型..."
        }
        print(f"📥 Loading Whisper model: {model_name}...")

        model_options = [model_name, "turbo", "large-v3", "medium", "small", "base"]
        unique_options = []
        for opt in model_options:
            if opt not in unique_options:
                unique_options.append(opt)

        loaded_model = None
        current_loaded_name = ""
        for try_model in unique_options:
            try:
                print(f"🔄 Trying model: {try_model}...")
                loaded_model = whisper.load_model(try_model)
                print(f"✅ Successfully loaded: {try_model}")
                current_loaded_name = try_model
                break
            except Exception as model_err:
                print(f"⚠️ Failed to load {try_model}: {model_err}")
                continue

        if loaded_model:
            models.whisper_model = loaded_model
            models.CURRENT_WHISPER_NAME = current_loaded_name
            models.model_status = {
                "progress": 100,
                "status": "ready",
                "message": f"Whisper {current_loaded_name} 模型載入成功！"
            }
        else:
            models.model_status = {
                "progress": 0,
                "status": "error",
                "message": "所有模型載入失敗"
            }
            print("❌ All Whisper model options failed to load")

    except Exception as e:
        models.model_status = {
            "progress": 0,
            "status": "error",
            "message": f"下載失敗: {str(e)}"
        }
        print(f"❌ Download error: {e}")


def ensure_whisper_model(requested_model: str = "turbo"):
    """Check if model matches requested, if not reload."""
    if models.whisper_model is None or (
        requested_model and requested_model != models.CURRENT_WHISPER_NAME
    ):
        print(f"🔄 Model Switch Requested: {models.CURRENT_WHISPER_NAME} -> {requested_model}")
        download_whisper_model(requested_model)
    return models.whisper_model


def get_transcribe_options(
    lang: str,
    beam_size: int,
    temperature: float = 0,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = True,
    best_of: int = 5,
    patience: float = 1.0,
    compression_ratio_threshold: float = 2.4,
    logprob_threshold: float = -1.0,
    fp16: bool = True
) -> dict:
    """Helper to build whisper transcribe options dictionary."""
    opts: dict = {}
    if lang and lang != "auto":
        lang_arg = "zh" if lang == "zh-tw" else lang
        opts["language"] = lang_arg

    try:
        opts["beam_size"] = int(beam_size) if beam_size else 5
    except (TypeError, ValueError):
        opts["beam_size"] = 5

    try:
        opts["temperature"] = float(temperature) if temperature is not None else 0
    except (TypeError, ValueError):
        opts["temperature"] = 0

    try:
        opts["no_speech_threshold"] = float(no_speech_threshold) if no_speech_threshold is not None else 0.6
    except (TypeError, ValueError):
        opts["no_speech_threshold"] = 0.6

    try:
        val = str(condition_on_previous_text).lower() == "true" if isinstance(condition_on_previous_text, str) else bool(condition_on_previous_text)
        opts["condition_on_previous_text"] = val
    except (TypeError, ValueError):
        opts["condition_on_previous_text"] = True

    try:
        opts["best_of"] = int(best_of) if best_of else 5
    except (TypeError, ValueError):
        opts["best_of"] = 5

    try:
        opts["patience"] = float(patience) if patience else 1.0
    except (TypeError, ValueError):
        opts["patience"] = 1.0

    try:
        opts["compression_ratio_threshold"] = float(compression_ratio_threshold) if compression_ratio_threshold else 2.4
    except (TypeError, ValueError):
        opts["compression_ratio_threshold"] = 2.4

    try:
        opts["logprob_threshold"] = float(logprob_threshold) if logprob_threshold else -1.0
    except (TypeError, ValueError):
        opts["logprob_threshold"] = -1.0

    try:
        val_fp16 = str(fp16).lower() == "true" if isinstance(fp16, str) else bool(fp16)
        opts["fp16"] = val_fp16
    except (TypeError, ValueError):
        opts["fp16"] = True

    return opts
