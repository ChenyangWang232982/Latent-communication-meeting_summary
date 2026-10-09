"""Optional audio-to-transcript component. It is deliberately opt-in."""

from __future__ import annotations

from pathlib import Path

from .config import ASRConfig


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def transcribe(audio_path: Path, config: ASRConfig) -> str:
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError("--enable-asr accepts audio input only (.wav, .mp3, .m4a, .flac, .ogg, .opus, .aac).")
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "ASR is enabled but faster-whisper is unavailable. Install it with: pip install faster-whisper"
        ) from error
    model = WhisperModel(config.model, device=config.device, compute_type=config.compute_type)
    segments, _ = model.transcribe(str(audio_path), vad_filter=True)
    transcript = "\n".join(segment.text.strip() for segment in segments if segment.text.strip())
    if not transcript:
        raise RuntimeError("ASR produced an empty transcript.")
    return transcript

