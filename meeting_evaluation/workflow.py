"""Adapter around the validated Prefix-KV + StateBridge runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import WorkflowConfig
from .records import MeetingRecord


def run_statebridge(records: list[MeetingRecord], config: WorkflowConfig) -> list[dict]:
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
            "--data", str(data_path), "--output", str(raw_output),
            "--model", config.model,
            # Records have already been limited above. The underlying runner
            # defaults to one record, so explicitly request all of this set.
            "--num-samples", "0",
            "--source-max-tokens", str(config.source_max_tokens),
            "--prefill-chunk-tokens", str(config.prefill_chunk_tokens),
            "--backfill-rounds", str(config.backfill_rounds),
            "--agent-max-new-tokens", str(config.agent_max_new_tokens),
            "--receiver-max-new-tokens", str(config.receiver_max_new_tokens),
            "--prefix-tokens-per-agent", str(config.prefix_tokens_per_agent),
            "--variants", "statebridge", "text", "no_comm",
        ]
        if config.device:
            command.extend(["--device", config.device])
        subprocess.run(command, check=True)
        result_path = raw_output.with_suffix(".jsonl")
        return [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
