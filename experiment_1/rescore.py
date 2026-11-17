"""Rebuild reports from saved pairs without loading the generation model."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .components.config import VARIANTS, sha256
from .components.reporting import write_reports
from .components.storage import atomic_json


def rescore(run_dir, output_dir=None, *, bertscore_model=None, bertscore_layers=12, device="cpu"):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["sample_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in manifest")
    completed, failures, missing = [], [], []
    inputs = {}
    for index, sample_id in enumerate(ids):
        path = run_dir / "samples" / f"{index:05d}.json"
        if not path.exists():
            missing.append(sample_id)
            continue
        inputs[str(path.relative_to(run_dir))] = sha256(path.read_bytes())
        sample = json.loads(path.read_text(encoding="utf-8"))
        if sample["id"] != sample_id:
            raise ValueError(f"Sample ID mismatch: {path}")
        if sample["status"] == "completed":
            if not sample["reference"].strip():
                raise ValueError(f"Missing reference: {sample_id}")
            if any(v not in sample["result"]["answers"] or v not in sample["result"]["variants"] for v in VARIANTS):
                raise ValueError(f"Incomplete paired result: {sample_id}")
            sample.setdefault("quality", {})
            for v in VARIANTS:
                sample["quality"].setdefault(v, {})
                # Recompute requested metrics; do not inherit semantic scores from unknown runs.
                for key in list(sample["quality"][v]):
                    if key.startswith("bertscore_"):
                        del sample["quality"][v][key]
            completed.append(sample)
        else:
            failures.append(sample)
    if not completed:
        raise ValueError("No completed sample pairs to evaluate")
    semantic = None
    if bertscore_model:
        from .components.semantic_quality import add_bertscore
        semantic = add_bertscore(completed, bertscore_model, bertscore_layers, device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(output_dir).resolve() if output_dir else run_dir / "evaluation" / stamp
    # Keep source checkpoints, original reports and manual annotations untouched.
    if destination == run_dir or run_dir.is_relative_to(destination):
        raise ValueError("Choose a separate evaluation output directory")
    destination.mkdir(parents=True, exist_ok=False)
    write_reports(SimpleNamespace(directory=destination), completed, failures, len(ids))
    atomic_json(destination / "evaluation_manifest.json", {
        "source_run": str(run_dir), "source_manifest_sha256": sha256((run_dir / "manifest.json").read_bytes()),
        "sample_sha256": inputs, "completed_sample_ids": [s["id"] for s in completed],
        "missing_sample_ids": missing, "semantic_evaluation": semantic,
        "lexical_metrics": "ASCII alphanumeric lowercase tokens; no stemming; whole-summary LCS, not ROUGE-Lsum",
        "quality_aggregation": "macro mean over completed pairs",
        "evaluation_code_sha256": {str(p.relative_to(Path(__file__).parent)): sha256(p.read_bytes())
            for p in Path(__file__).parent.rglob("*.py")},
        "note": "No generation or review was rerun. Original resource and token measurements reused."})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bertscore-model", help="Optional encoder ID/path; downloads if uncached. Use one model for all comparisons.")
    parser.add_argument("--bertscore-layers", type=int, default=12)
    parser.add_argument("--device", default="cpu", help="Optional BERTScore encoder device only")
    args = parser.parse_args()
    if args.bertscore_layers < 1:
        parser.error("--bertscore-layers must be positive")
    print(f"Reports: {rescore(args.run_dir, args.output_dir, bertscore_model=args.bertscore_model, bertscore_layers=args.bertscore_layers, device=args.device)}")


if __name__ == "__main__":
    main()
