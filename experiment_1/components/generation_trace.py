"""Keep incomplete generations inspectable without scoring or stopping them."""
import os
from datetime import datetime, timezone

from .storage import atomic_json


class GenerationTrace:
    STAGES = {"routed_kv_receiver", "specialist_text_sender", "specialist_text_receiver"}

    def __init__(self, directory, index, sample_id, log):
        self.directory = directory / "live" / f"{index:05d}"
        self.sample_id = sample_id
        self.log = log
        self.warned = set()

    def __call__(self, event):
        stage = event["stage"]
        if stage not in self.STAGES:
            raise ValueError(f"Unknown generation stage: {stage}")
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
