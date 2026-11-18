"""Keep incomplete generations inspectable without scoring or stopping them."""
import os
import hashlib
import json
from datetime import datetime, timezone

from .storage import atomic_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class GenerationTrace:
    STAGES = {"routed_kv_receiver", "specialist_text_sender", "specialist_text_receiver"}

    def __init__(self, directory, index, sample_id, log):
        self.directory = directory / "live" / f"{index:05d}"
        self.sample_id = sample_id
        self.log = log
        self.warned = set()
        self.inputs = {}
        self.pending = {}

    def input(self, event):
        stage = event["stage"]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush(stage)
        self.inputs[stage] = event
        value = {**event,
                 "source_token_sha256": fingerprint(event["prefix_token_ids"]),
                 "suffix_token_sha256": fingerprint(event["suffix_token_ids"]),
                 "selected_position_sha256": fingerprint(event["selected_positions"]),
                 "source_positions": len(event["prefix_token_ids"]),
                 "retained_positions": len(event["selected_positions"])}
        atomic_json(self.directory / f"{stage}.input.json", value)
        (self.directory / f"{stage}.tokens.jsonl").write_text("", encoding="utf-8")

    def token(self, stage, logits, token_id, step):
        """Observe actual forward logits; no resampling or text re-tokenization."""
        import torch
        vector = logits[0].detach()
        values, indices = vector.topk(min(8, vector.numel()))
        top_values, top_ids = values.cpu().tolist(), indices.cpu().tolist()
        eos = self.inputs[stage]["eos_token_ids"]
        eos_values = vector.index_select(0, torch.tensor(eos, device=vector.device, dtype=torch.long)).cpu().tolist()
        entry = {"step": step, "token_id": token_id,
                 "top_logits": [{"token_id": index, "logit": value} for index, value in zip(top_ids, top_values)],
                 "top1_top2_margin": top_values[0] - top_values[1] if len(top_values) > 1 else None,
                 "eos_logits": [{"token_id": index, "logit": value} for index, value in zip(eos, eos_values)]}
        self.pending.setdefault(stage, []).append(json.dumps(entry, allow_nan=False) + "\n")
        if len(self.pending[stage]) >= 16 or token_id in eos:
            self.flush(stage)

    def flush(self, stage=None):
        for name in ([stage] if stage else list(self.pending)):
            entries = self.pending.get(name, [])
            if entries:
                with (self.directory / f"{name}.tokens.jsonl").open("a", encoding="utf-8") as handle:
                    handle.writelines(entries)
                entries.clear()

    def __call__(self, event):
        stage = event["stage"]
        if stage not in self.STAGES and not stage.startswith(('chunk_', 'text_chunk_')):
            raise ValueError(f"Unknown generation stage: {stage}")
        if event.get("event") == "input":
            self.input(event)
            return
        self.flush(stage)
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{stage}.partial.txt"
        temporary = destination.with_suffix(".tmp")
        try:
            temporary.write_text(event["text"], encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        status = {key: value for key, value in event.items() if key != "text"}
        status.update(sample_id=self.sample_id, updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.directory / f"{stage}.status.json", status)
        if event["repeated_tail"] and stage not in self.warned:
            self.warned.add(stage)
            self.log(f"Repeated token pattern detected in {stage}; generation continues. Inspect {destination}")
