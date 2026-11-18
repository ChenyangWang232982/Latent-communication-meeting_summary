"""Atomic per-sample checkpoints and strict resume checks."""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunStore:
    def __init__(self, output_dir, resume, manifest):
        if resume:
            self.directory = Path(resume)
            old = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
            # A commit change alone is harmless when the actual source files are identical.
            for key in ("schema_version", "benchmark", "review", "variants", "config", "input_sha256", "sample_ids", "code_sha256", "environment"):
                if old[key] != manifest[key]:
                    raise ValueError(f"Cannot resume: {key} differs from the saved manifest. Start a new run.")
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.directory = Path(output_dir) / stamp
            self.directory.mkdir(parents=True, exist_ok=False)
            atomic_json(self.directory / "manifest.json", manifest)

    def path(self, index):
        return self.directory / "samples" / f"{index:05d}.json"

    def load(self, index, sample_id):
        path = self.path(index)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["id"] != sample_id:
            raise ValueError("Saved sample ID does not match the manifest")
        return value

    def save(self, index, sample):
        if sample["status"] == "completed":
            folder = self.directory / "summaries" / f"{index:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            documents = {**sample["result"]["answers"], "reference": sample["reference"]}
            for name, handoff in sample['result'].get('handoffs', {}).items():
                documents[f'{name}_notes'] = handoff
            for name, detail in sample['result'].get('chunk_specialists_by_variant', {}).items():
                atomic_json(folder / f'{name}.specialists.json', detail)
                for stage, thought in detail.get('thinking_outputs', {}).items():
                    documents[f'{name}.{stage}.thinking'] = thought
            for name, content in documents.items():
                path = folder / f"{name}.md"
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    temporary.write_text(content + "\n", encoding="utf-8")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        # The JSON is the completion marker and is committed after all readable outputs.
        atomic_json(self.path(index), sample)
