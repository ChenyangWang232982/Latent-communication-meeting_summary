"""Run the LangGraph meeting workflow on transcript or audio files."""

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
    parser.add_argument("input_name", help="File or folder name below meeting_latent_workflow/input")
    parser.add_argument("--model", required=True, help="Local model folder name below meeting_latent_workflow/model")
    parser.add_argument("--mode", choices=["text", "cipher"], default="cipher")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sender-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-refinement-rounds", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=2800)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=256)
    parser.add_argument("--reduce-group-size", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--download-asr-model", action="store_true")
    return parser.parse_args()


def resolve_child(root: Path, name: str, kind: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{kind} must stay inside {root}")
    return path


def resolve_input_path(input_name: str) -> Path:
    path = resolve_child(INPUT_ROOT, input_name, "input_name")
    if not path.exists():
        raise FileNotFoundError(f"Input file or folder not found: {path}")
    return path


def resolve_model_path(model_name: str) -> Path:
    path = resolve_child(MODEL_ROOT, model_name, "model")
    if not path.is_dir():
        raise FileNotFoundError(f"Local model folder not found: {path}")
    return path


def collect_inputs(input_path: Path) -> list[Path]:
    supported_extensions = AUDIO_EXTENSIONS | {".txt"}
    if input_path.is_file():
        if input_path.suffix.lower() not in supported_extensions:
            raise ValueError("Input must be a .txt transcript or supported audio file")
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in supported_extensions)


def output_path_for(source_path: Path, suffix: str) -> Path:
    return OUTPUT_ROOT / source_path.relative_to(INPUT_ROOT.resolve()).with_suffix(suffix)


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
                transcriber = WhisperLargeV3Transcriber(
                    ensure_whisper_large_v3(ASR_MODEL_ROOT, args.download_asr_model), args.device
                )
            asr_result = transcriber.transcribe(source_path)
            transcript = asr_result["text"].strip()
            transcript_path = output_path_for(source_path, ".transcript.txt")
            metadata_path = output_path_for(source_path, ".transcript.json")
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(transcript + "\n", encoding="utf-8")
            metadata_path.write_text(json.dumps(asr_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Transcript saved: {transcript_path}")
        if transcript:
            prepared_transcripts.append((source_path, transcript))
        else:
            print(f"Skipped empty transcript: {source_path.name}")

    del transcriber
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    workflow = build_workflow(WorkflowConfig(
        model_name=str(model_path), communication_mode=args.mode, device=args.device,
        max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
        sender_max_new_tokens=args.sender_max_new_tokens, temperature=args.temperature,
        chunk_tokens=args.chunk_tokens, chunk_overlap_tokens=args.chunk_overlap_tokens,
        reduce_group_size=args.reduce_group_size, max_refinement_rounds=args.max_refinement_rounds,
    ))
    for source_path, transcript in prepared_transcripts:
        result = workflow.invoke({"transcript": transcript}, {"max_concurrency": args.max_concurrency})
        summary_path = output_path_for(source_path, ".summary.txt")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(result["final_summary"] + "\n", encoding="utf-8")
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
