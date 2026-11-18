"""CLI settings and reproducibility metadata; no model is loaded here."""
import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from receiver_kv_workflow.engine import ReceiverKVConfig

VARIANTS = ("routed_kv", "specialist_text")
ROOT = Path(__file__).resolve().parents[2]


def add_workflow_arguments(parser):
    """Use identical model/routing defaults in formal and diagnostic entry points."""
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--role-budget-ratio", type=float, default=0.25)
    parser.add_argument("--speaker-neighbor-blocks", type=int, default=1)
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--query-batch-tokens", type=int, default=8)
    parser.add_argument("--fp32-lm-head", action="store_true",
                        help="Numerical control: compute only the vocabulary output projection in FP32")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 retains greedy decoding; Qwen3 non-thinking sampling control: 0.7")
    parser.add_argument("--top-p", type=float, default=1.0, help="Shared generation nucleus threshold; sampling control: 0.8")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k filtering; sampling control: 20")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Generated-only token penalty shared across sender/receivers; 1 disables, try 1.1")
    parser.add_argument('--chunk-specialists', action='store_true', help='Paired chronological specialists for both KV and text')
    parser.add_argument('--specialist-parts', type=int, default=8)
    parser.add_argument('--specialist-overlap', type=float, default=0.25)
    parser.add_argument('--specialist-temperature', type=float, default=0.3)
    parser.add_argument('--specialist-top-p', type=float, default=0.9)
    parser.add_argument('--specialist-repetition-penalty', type=float, default=1.1)
    parser.add_argument('--summary-thinking', action='store_true', help='Thinking at final synthesis in both chunk variants')
    parser.add_argument('--summary-seed', type=int, default=42)
    parser.add_argument('--summary-temperature', type=float, default=0.3)
    parser.add_argument('--summary-top-p', type=float, default=0.9)
    from vector.cli import add_arguments
    add_arguments(parser)


def arguments():
    parser = argparse.ArgumentParser(description="Experiment 1: paired draft-summary quality and resource use; review is disabled.")
    parser.add_argument("--input", type=Path, required=True, help="Normalized speaker-labelled JSONL with reference summaries")
    parser.add_argument("--benchmark", choices=("meetingbank", "ami", "icsi", "qmsum_committee"), default="meetingbank")
    parser.add_argument("--output-dir", type=Path, default=Path("test_result/experiment_1"))
    parser.add_argument("--resume", type=Path, help="Resume an existing experiment directory with identical inputs/settings/code")
    parser.add_argument("--num-samples", type=int, default=25, help="First N records in the prepared input; 0 means all")
    add_workflow_arguments(parser)
    args = parser.parse_args()
    if args.num_samples < 0:
        parser.error("--num-samples must be non-negative")
    if not 0 < args.role_budget_ratio <= 1 or not 0 <= args.vector_weight <= 1:
        parser.error("Invalid role budget or vector weight")
    if args.speaker_neighbor_blocks < 0 or min(args.span_tokens, args.query_batch_tokens) < 1:
        parser.error("Neighbor count must be non-negative and token budgets positive")
    if args.summary_thinking and not args.chunk_specialists:
        parser.error('--summary-thinking requires --chunk-specialists')
    return args


def model_config(args):
    return ReceiverKVConfig(model=args.model, device=args.device, seed=args.seed,
        role_budget_ratio=args.role_budget_ratio, speaker_neighbor_blocks=args.speaker_neighbor_blocks,
        span_tokens=args.span_tokens, query_batch_tokens=args.query_batch_tokens,
        vector_routing=args.vector_routing, embedding_model=args.embedding_model, vector_weight=args.vector_weight,
        fp32_lm_head=getattr(args, "fp32_lm_head", False),
        temperature=getattr(args, "temperature", 0.0), top_p=getattr(args, "top_p", 1.0),
        top_k=getattr(args, "top_k", 0),
        repetition_penalty=getattr(args, "repetition_penalty", 1.0),
        chunk_specialists=getattr(args, 'chunk_specialists', False),
        specialist_parts=getattr(args, 'specialist_parts', 8),
        specialist_overlap=getattr(args, 'specialist_overlap', 0.25),
        specialist_temperature=getattr(args, 'specialist_temperature', 0.3),
        specialist_top_p=getattr(args, 'specialist_top_p', 0.9),
        specialist_repetition_penalty=getattr(args, 'specialist_repetition_penalty', 1.1),
        summary_thinking=getattr(args, 'summary_thinking', False),
        review_seed=getattr(args, 'summary_seed', 42),
        review_temperature=getattr(args, 'summary_temperature', 0.3),
        review_top_p=getattr(args, 'summary_top_p', 0.9))


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def reproducibility(args, records, config):
    paths = sorted(path for folder in ("experiment_1", "receiver_kv_workflow", "vector", "meeting_evaluation", "chunk_specialist_workflow")
                   for path in (ROOT / folder).rglob("*.py"))
    code = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path.read_bytes()) for path in paths}
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema_version": 1, "benchmark": args.benchmark, "review": False,
            "variants": list(VARIANTS), "config": asdict(config),
            "input_sha256": sha256(args.input.read_bytes()),
            "sample_ids": [r.id for r in records], "code_sha256": code,
            "environment": {"python": platform.python_version(), "packages": packages},
            "git_commit": commit}
