"""Freeze a small original-meeting input shared by workflow and direct baseline."""
import argparse
import json
from pathlib import Path

from meeting_evaluation.records import load_records
from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json


def prepare(input_path, output, *, num_samples=3, skip=3):
    if num_samples < 1 or skip < 0:
        raise ValueError('num_samples must be positive; skip must be non-negative')
    records = load_records('jsonl', input_path)[skip:skip+num_samples]
    if not records or any(not r.answer.strip() for r in records):
        raise ValueError('Original meeting records with nonempty reference summaries are required')
    if output.exists():
        raise FileExistsError(f'Frozen input already exists: {output}; reuse it or choose a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='\n') as handle:
        for r in records:
            handle.write(json.dumps(dict(id=r.id, transcript=r.context, summary=r.answer), ensure_ascii=False)+'\n')
    atomic_json(output.with_suffix(output.suffix+'.report.json'), dict(input=str(input_path.resolve()),
        input_sha256=sha256(input_path.read_bytes()), output_sha256=sha256(output.read_bytes()),
        skip=skip, requested_samples=num_samples, selected_samples=len(records), sample_ids=[r.id for r in records],
        note='Original source/reference unchanged. Both generation commands must use this same frozen file. '
             'Skipping records is positional, not a guarantee these meetings are held out from other experiments.'))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--skip', type=int, default=3, help='Skip the first three development sources in the existing AMI input')
    args = parser.parse_args()
    print('Prepared:', prepare(args.input, args.output, num_samples=args.num_samples, skip=args.skip))
