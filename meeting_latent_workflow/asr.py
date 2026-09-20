"""Local Whisper large-v3 transcription for the meeting workflow."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


ASR_MODEL_ID = "openai/whisper-large-v3"


class WhisperLargeV3Transcriber:
    """Transcribe long meeting audio in 30-second overlapping windows."""

    def __init__(self, model_directory: Path, device: str = "auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_directory)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_directory,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=0 if self.device.startswith("cuda") else -1,
        )

    @torch.inference_mode()
    def transcribe(self, audio_path: Path) -> dict:
        return self.pipeline(
            str(audio_path),
            chunk_length_s=30,
            stride_length_s=(5, 5),
            return_timestamps=True,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )


def ensure_whisper_large_v3(model_directory: Path, download: bool) -> Path:
    """Return a local Whisper snapshot, downloading only with explicit consent."""
    if (model_directory / "config.json").is_file():
        return model_directory
    if not download:
        raise FileNotFoundError(
            f"Whisper large-v3 is missing: {model_directory}. "
            "Run again with --download-asr-model to download openai/whisper-large-v3."
        )
    model_directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=ASR_MODEL_ID, local_dir=model_directory)
    return model_directory
