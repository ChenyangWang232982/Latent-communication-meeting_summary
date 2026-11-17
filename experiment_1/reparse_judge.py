"""Reparse saved judge responses locally into a NEW report; never calls an API."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .components.config import sha256
from .components.judge_rubric import validate
from .components.judge_runner import summarize
from .components.storage import atomic_json


def reparse(judge_dir, output_dir=None):
    judge_dir = Path(judge_dir).resolve()
    manifest_path = judge_dir / "judge_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = Path(manifest["source_run"])
    if sha256((run / "manifest.json").read_bytes()) != manifest["source_manifest_sha256"]:
        raise ValueError("Source run manifest changed")
    samples = {}
    for index, expected in manifest["source_checkpoint_sha256"].items():
        path = run / "samples" / f"{int(index):05d}.json"
        if sha256(path.read_bytes()) != expected:
            raise ValueError("Source checkpoint changed")
        sample = json.loads(path.read_text(encoding="utf-8"))
        samples[sample["id"]] = sample
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(output_dir).resolve() if output_dir else judge_dir / "reparsed" / stamp
    if destination == judge_dir or judge_dir.is_relative_to(destination):
        raise ValueError("Use a new report directory")
    destination.mkdir(parents=True, exist_ok=False)
    jobs, hashes = [], {}
    for key, identity in sorted(manifest["mapping"].items()):
        if not key.replace("_", "").isalnum():
            raise ValueError("Invalid case name")
        sample, variant = samples[identity["id"]], identity["variant"]
        path = judge_dir / "judgments" / f"{key}.json"
        hashes[key] = sha256(path.read_bytes())
        record = json.loads(path.read_text(encoding="utf-8"))
        record["previous_validation"] = {"status": record["status"], "error": record.get("error")}
        try:
            response = record.get("response", {})
            if response.get("finish_reason") != "stop":
                raise ValueError("No complete saved response to reparse")
            record["judgment"] = validate(response["text"], sample["source"], sample["result"]["answers"][variant])
            record["status"] = "completed"
            record.pop("error", None)
        except Exception as exc:
            record["status"] = "failed"
            record.pop("judgment", None)
            record["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(destination / "judgments" / f"{key}.json", record)
        jobs.append((key, sample, variant))
        print(f"{key}: {record['status']}", flush=True)
    atomic_json(destination / "judge_manifest.json", manifest)
    atomic_json(destination / "reparse_manifest.json", {
        "original_judge_directory": str(judge_dir), "original_manifest_sha256": sha256(manifest_path.read_bytes()),
        "saved_response_sha256": hashes, "validator_sha256": sha256(Path(__file__).with_name("components").joinpath("judge_rubric.py").read_bytes()),
        "api_calls": 0, "new_api_tokens": 0,
        "note": "Usage in reports belongs to the original calls; no new generation. Original files unchanged."})
    summarize(destination, jobs, manifest["settings"])
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Reports: {reparse(args.judge_dir, args.output_dir)}")


if __name__ == "__main__":
    main()
