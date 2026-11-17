"""Offline paired judging with auditable checkpoints and separate costs."""
import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .config import VARIANTS, sha256
from .storage import atomic_json
from .judge_rubric import DIMENSIONS, SYSTEM, messages, validate


def run_judge(run_dir, destination, settings, client):
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples, hashes, excluded = [], {}, []
    for index, sample_id in enumerate(manifest["sample_ids"]):
        path = run_dir / "samples" / f"{index:05d}.json"
        if not path.exists():
            excluded.append(sample_id)
            continue
        sample = json.loads(path.read_text(encoding="utf-8"))
        if sample["id"] != sample_id:
            raise ValueError("Checkpoint ID mismatch")
        hashes[str(index)] = sha256(path.read_bytes())
        if sample["status"] != "completed":
            excluded.append(sample_id)
            continue
        if not sample["source"].strip() or any(not sample["result"]["answers"].get(v, "").strip() for v in VARIANTS):
            raise ValueError(f"Incomplete source/paired outputs: {sample_id}")
        samples.append(sample)
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Need nonempty unique completed sample pairs")
    if destination == run_dir or run_dir.is_relative_to(destination):
        raise ValueError("Use a separate judge output directory")
    if settings.get("num_samples", 0):
        samples = samples[:settings["num_samples"]]
    rng = random.Random(settings["seed"])
    rng.shuffle(samples)
    jobs = []
    for index, sample in enumerate(samples):
        order = list(VARIANTS)
        rng.shuffle(order)
        for label, variant in zip(("A", "B"), order):
            jobs.append((f"{index:05d}_{label}", sample, variant))
    signature = {"source_run": str(run_dir), "source_manifest_sha256": sha256(manifest_path.read_bytes()),
                 "source_checkpoint_sha256": hashes, "settings": settings, "rubric_sha256": sha256(SYSTEM.encode()),
                 "code_sha256": {p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("judge_*.py")},
                 "excluded_sample_ids": excluded,
                 "method": "Task-adapted SIEVE-inspired automatic evaluation; not original SIEVE or human scores",
                 "mapping": {key: {"id": sample["id"], "variant": v} for key, sample, v in jobs}}
    destination.mkdir(parents=True, exist_ok=True)
    saved = destination / "judge_manifest.json"
    if saved.exists():
        if json.loads(saved.read_text(encoding="utf-8")) != signature:
            raise ValueError("Cannot resume: inputs, evaluator settings or code changed; choose new output directory")
    elif any(destination.iterdir()):
        raise ValueError("Output directory is nonempty and has no matching judge manifest")
    else:
        atomic_json(saved, signature)
    for number, (key, sample, variant) in enumerate(jobs, 1):
        path = destination / "judgments" / f"{key}.json"
        if path.exists():
            print(f"{number}/{len(jobs)} {key}: reused", flush=True)
            continue
        source, candidate = sample["source"], sample["result"]["answers"][variant]
        record = {"case": key, "status": "failed", "request_sha256": sha256(json.dumps(messages(source, candidate), ensure_ascii=False).encode())}
        started = time.perf_counter()
        try:
            response = client(messages(source, candidate))
            record["response"] = response  # Preserve usage/raw output even if parsing fails.
            if response["finish_reason"] != "stop":
                raise ValueError(f"Judge response incomplete: {response['finish_reason']}")
            record["judgment"] = validate(response["text"], source, candidate)
            record["status"] = "completed"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["attempt_seconds"] = time.perf_counter() - started
        atomic_json(path, record)
        print(f"{number}/{len(jobs)} {key}: {record['status']}", flush=True)
    return summarize(destination, jobs, settings)


def summarize(destination, jobs, settings):
    rows, costs, flags = [], [], {}
    groups = defaultdict(dict)
    lookup = {}
    for key, sample, variant in jobs:
        record = json.loads((destination / "judgments" / f"{key}.json").read_text(encoding="utf-8"))
        groups[sample["id"]][variant] = record
        lookup[sample["id"]] = sample
        costs.append({**record.get("response", {}), "attempt_seconds": record.get("attempt_seconds", 0)})
        reasons = []
        if record["status"] != "completed":
            reasons.append("judge_failed")
        else:
            for dimension, item in record["judgment"]["dimensions"].items():
                rows.append({"id": sample["id"], "variant": variant, "dimension": dimension,
                             **{k: item[k] for k in ("coverage", "factuality", "uncertain", "reason")}})
                if item["uncertain"]:
                    reasons.append(dimension + ":uncertain")
                if item["factuality"] == 0:
                    reasons.append(dimension + ":major_error")
        if reasons:
            flags.setdefault(sample["id"], []).extend(reasons)
    report = {"method": "SIEVE-inspired automatic judge; scores 0-2, higher is better; no composite score",
              "planned_pairs": len(groups), "completed_pairs": 0, "dimensions": {},
              "cost": {"total_attempt_seconds": sum(c["attempt_seconds"] for c in costs),
                       "seconds_with_response": sum(c.get("seconds", 0) for c in costs),
                       "responses_with_usage": sum(isinstance(c.get("usage"), dict) for c in costs),
                       "requests_without_usage": sum(not isinstance(c.get("usage"), dict) for c in costs)},
              "note": "Judge costs excluded from original generation costs. Human checklist is unfilled. Scores are not verified facts."}
    complete = [g for g in groups.values() if all(g[v]["status"] == "completed" for v in VARIANTS)]
    report["completed_pairs"] = len(complete)
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [c["usage"][field] for c in costs if isinstance(c.get("usage"), dict) and field in c["usage"]]
        report["cost"][field + "_reported_sum"] = sum(values) if values else None
    for dim in DIMENSIONS:
        report["dimensions"][dim] = {}
        for field in ("coverage", "factuality"):
            pairs = [(g[VARIANTS[0]]["judgment"]["dimensions"][dim], g[VARIANTS[1]]["judgment"]["dimensions"][dim]) for g in complete]
            eligible = [(a[field], b[field]) for a, b in pairs if a[field] is not None and b[field] is not None and not a["uncertain"] and not b["uncertain"]]
            report["dimensions"][dim][field] = {"paired_n": len(eligible),
                "excluded_null_or_uncertain_pairs": len(pairs) - len(eligible),
                "null_applicability_disagreements": sum((a[field] is None) != (b[field] is None) for a,b in pairs),
                "routed_kv": mean(a for a,b in eligible) if eligible else None,
                "specialist_text": mean(b for a,b in eligible) if eligible else None,
                "paired_difference_kv_minus_text": mean(a-b for a,b in eligible) if eligible else None}
    random_ids = random.Random(settings["seed"] + 1).sample(sorted(groups), min(settings["audit_pairs"], len(groups)))
    for sample_id in random_ids:
        flags.setdefault(sample_id, []).insert(0, "random_audit")
    for sample_id, group in groups.items():
        if all(group[v]["status"] == "completed" for v in VARIANTS):
            for dim in DIMENSIONS:
                a, b = [group[v]["judgment"]["dimensions"][dim] for v in VARIANTS]
                if any((a[f] is None) != (b[f] is None) for f in ("coverage", "factuality")):
                    flags.setdefault(sample_id, []).append(dim + ":applicability_disagreement")
    report["audit_random_pairs"] = len(random_ids)
    report["audit_total_pairs"] = len(flags)
    atomic_json(destination / "judge_summary.json", report)
    with (destination / "judge_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "variant", "dimension", "coverage", "factuality", "uncertain", "reason"))
        writer.writeheader(); writer.writerows(rows)
    lines = [report["method"], f"Completed pairs: {len(complete)}/{len(groups)}", "dimension | measure | paired n | KV | text | KV-text"]
    for dim, measures in report["dimensions"].items():
        for measure, values in measures.items():
            lines.append(f"{dim} | {measure} | {values['paired_n']} | {values['routed_kv']} | {values['specialist_text']} | {values['paired_difference_kv_minus_text']}")
    lines.extend(["", "Evaluator cost (separate): " + json.dumps(report["cost"]), f"Human audit: {len(random_ids)} random pairs; {len(flags)} total including flags."])
    (destination / "judge_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    audit = destination / "audit"
    audit.mkdir(exist_ok=True)
    # Preserve user annotations on resume. Only fill missing review packets.
    for key, sample, variant in jobs:
        if sample["id"] not in flags:
            continue
        packet = audit / f"{key}.md"
        if not packet.exists():
            source = "\n".join(f"L{i}: {line}" for i,line in enumerate(sample["source"].splitlines(),1))
            judgment_text = (destination / "judgments" / f"{key}.json").read_text(encoding="utf-8")
            packet.write_text(f"# Case {key}\n\n## Source\n{source}\n\n## Candidate\n{sample['result']['answers'][variant]}\n\n## Automated judgment (verify against source)\n```json\n{judgment_text}\n```\n", encoding="utf-8")
    sheet = audit / "checklist.csv"
    if not sheet.exists():
        with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(("case", "selection_reason", "judge_correct", "correction", "source_lines", "reviewer"))
            for key, sample, variant in jobs:
                if sample["id"] in flags:
                    writer.writerow((key, ";".join(sorted(set(flags[sample["id"]]))), "", "", "", ""))
    return destination
