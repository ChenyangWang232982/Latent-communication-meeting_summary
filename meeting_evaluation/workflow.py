"""Adapter around the validated Prefix-KV + StateBridge runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import WorkflowConfig
from .records import MeetingRecord
from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner


def run_statebridge(records: list[MeetingRecord], config: WorkflowConfig, *, meeting_mode: bool) -> list[dict]:
    """Run all communication controls and return machine-readable runner results."""
    selected = records[: config.num_samples] if config.num_samples else records
    if not selected:
        raise ValueError("No records selected for evaluation.")
    with tempfile.TemporaryDirectory(prefix="meeting_eval_") as temp_dir:
        temp = Path(temp_dir)
        data_path = temp / "records.jsonl"
        raw_output = temp / "runner.txt"
        with data_path.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

        command = [
            sys.executable, "-m", "prophetkv_statebridge.run_prefix_cache",
            "--meeting-data" if meeting_mode else "--data", str(data_path), "--output", str(raw_output),
            "--model", config.model,
            # Records have already been limited above. The underlying runner
            # defaults to one record, so explicitly request all of this set.
            "--num-samples", "0",
            "--source-max-tokens", str(config.source_max_tokens),
            "--prefix-tokens-per-agent", str(config.prefix_tokens_per_agent),
            "--evidence-policy", config.evidence_policy,
            "--alignment-scope", config.alignment_scope,
            "--variants", "statebridge", "text", "no_comm",
        ]
        if config.device:
            command.extend(["--device", config.device])
        subprocess.run(command, check=True)
        result_path = raw_output.with_suffix(".jsonl")
        return [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_receiver_kv(records: list[MeetingRecord], config: ReceiverKVConfig, *, num_samples: int) -> list[dict]:
    """Batch MeetingBank records while holding one Qwen3 model in GPU memory."""
    selected = records[:num_samples] if num_samples else records
    if not selected:
        raise ValueError("No records selected for evaluation.")
    total = len(selected)
    runner = ReceiverKVRunner(config, progress=lambda message: print(f"[receiver_kv] {message}", flush=True))
    results = []
    for index, record in enumerate(selected, start=1):
        print(f"[receiver_kv] record {index}/{total}: {record.id}", flush=True)
        result = runner.run(record.id, record.context, variants=("routed_kv", "full_kv", "selected_text", "specialist_text"))
        result["gold_answer"] = record.answer
        results.append(result)
    return results
