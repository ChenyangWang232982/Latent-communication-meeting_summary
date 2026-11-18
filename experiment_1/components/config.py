"""CLI settings and reproducibility metadata; no model is loaded here."""
import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from receiver_kv_workflow.cli import add_workflow_arguments, model_config

VARIANTS = ("chunk_kv", "chunk_text")
ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description="Experiment 1: paired global-context chunk specialists: KV vs text, quality and resource use.")
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
    return args


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
    return {"schema_version": 2, "benchmark": args.benchmark, "review": False,
            "variants": list(VARIANTS), "config": asdict(config),
            "input_sha256": sha256(args.input.read_bytes()),
            "sample_ids": [r.id for r in records], "code_sha256": code,
            "environment": {"python": platform.python_version(), "packages": packages},
            "git_commit": commit}
