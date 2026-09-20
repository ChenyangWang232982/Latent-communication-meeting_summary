"""Run the LangGraph meeting workflow on a transcript text file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from meeting_latent_workflow.asr import WhisperLargeV3Transcriber, ensure_whisper_large_v3
from meeting_latent_workflow.config import WorkflowConfig
from meeting_latent_workflow.graph import build_workflow


WORKFLOW_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = WORKFLOW_ROOT / "input"
OUTPUT_ROOT = WORKFLOW_ROOT / "output"
MODEL_ROOT = WORKFLOW_ROOT / "model"
ASR_MODEL_ROOT = MODEL_ROOT / "whisper-large-v3"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_name",
        help="Name of a .txt file or folder under meeting_latent_workflow/input",
    )
    parser.add_argument("--mode", choices=["text", "cipher"], default="text")
    parser.add_argument("--cipher-checkpoint", type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="Name of a local Hugging Face model folder under meeting_latent_workflow/model",
    )
    parser.add_argument("--max-refinement-rounds", type=int, default=1)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compressed-latent-len", type=int, default=8)
    parser.add_argument("--chunk-tokens", type=int, default=2800)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=256)
    parser.add_argument("--reduce-group-size", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--download-asr-model",
        action="store_true",
        help="Download openai/whisper-large-v3 into workflow/model/whisper-large-v3",
    )
    return parser.parse_args()


def resolve_input_path(input_name: str) -> Path:
    """Restrict input to the workflow input directory."""
    input_root = INPUT_ROOT.resolve()
    candidate = (input_root / input_name).resolve()
    if candidate != input_root and input_root not in candidate.parents:
        raise ValueError("input_name must stay inside meeting_latent_workflow/input")
    if not candidate.exists():
        raise FileNotFoundError(f"Input file or folder not found: {candidate}")
    return candidate


def resolve_model_path(model_name: str) -> Path:
    """Restrict model selection to the workflow model directory."""
    model_root = MODEL_ROOT.resolve()
    candidate = (model_root / model_name).resolve()
    if candidate == model_root or model_root not in candidate.parents:
        raise ValueError("model must name a folder inside meeting_latent_workflow/model")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Local model folder not found: {candidate}")
    return candidate


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS | {".txt"}:
            raise ValueError("Input must be a .txt transcript or supported audio file")
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS | {".txt"}
    )


def output_path_for(transcript_path: Path) -> Path:
    relative_path = transcript_path.relative_to(INPUT_ROOT.resolve())
    return OUTPUT_ROOT / relative_path.with_suffix(".summary.txt")


def transcript_output_path_for(audio_path: Path) -> Path:
    relative_path = audio_path.relative_to(INPUT_ROOT.resolve())
    return OUTPUT_ROOT / relative_path.with_suffix(".transcript.txt")


def transcript_metadata_path_for(audio_path: Path) -> Path:
    relative_path = audio_path.relative_to(INPUT_ROOT.resolve())
    return OUTPUT_ROOT / relative_path.with_suffix(".transcript.json")


def main():
    args = parse_args()
    input_path = resolve_input_path(args.input_name)
    model_path = resolve_model_path(args.model)
    inputs = collect_inputs(input_path)
    if not inputs:
        raise ValueError(f"No transcript or supported audio found in: {input_path}")
    if args.max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")

    prepared_transcripts: list[tuple[Path, str]] = []
    transcriber = None
    for source_path in inputs:
        if source_path.suffix.lower() == ".txt":
            transcript = source_path.read_text(encoding="utf-8").strip()
        else:
            if transcriber is None:
                asr_model_path = ensure_whisper_large_v3(
                    ASR_MODEL_ROOT,
                    args.download_asr_model,
                )
                transcriber = WhisperLargeV3Transcriber(asr_model_path)
            asr_result = transcriber.transcribe(source_path)
            transcript = asr_result["text"].strip()
            transcript_path = transcript_output_path_for(source_path)
            metadata_path = transcript_metadata_path_for(source_path)
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(transcript + "\n", encoding="utf-8")
            metadata_path.write_text(
                json.dumps(asr_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Transcript saved: {transcript_path}")
        if transcript:
            prepared_transcripts.append((source_path, transcript))
        else:
            print(f"Skipped empty transcript: {source_path.name}")

    # Do not keep Whisper resident while the two LongT5 agents run.
    del transcriber
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = WorkflowConfig(
        model_name=str(model_path),
        communication_mode=args.mode,
        cipher_checkpoint=args.cipher_checkpoint,
        use_compression=args.compression,
        compressed_latent_len=args.compressed_latent_len,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        reduce_group_size=args.reduce_group_size,
        max_refinement_rounds=args.max_refinement_rounds,
    )
    workflow = build_workflow(config)
    for source_path, transcript in prepared_transcripts:
        result = workflow.invoke(
            {"transcript": transcript},
            {"max_concurrency": args.max_concurrency},
        )
        output_path = output_path_for(source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result["final_summary"] + "\n", encoding="utf-8")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
