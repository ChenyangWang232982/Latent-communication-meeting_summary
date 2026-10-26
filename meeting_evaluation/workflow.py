"""Adapter for batched receiver-conditioned KV evaluation."""

from __future__ import annotations

from .records import MeetingRecord
from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner


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
        result = runner.run(record.id, record.context, variants=("routed_kv", "full_kv", "selected_text", "specialist_text", "natural_text"))
        result["gold_answer"] = record.answer
        results.append(result)
    return results
