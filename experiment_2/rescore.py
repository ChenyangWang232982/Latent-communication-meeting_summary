"""Evaluate saved direct summaries without loading the generation model."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from meeting_evaluation.metrics import reference_metrics, source_overlap_support
from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json
from experiment_1.components.semantic_quality import add_bertscore
from .components.config import VARIANT
from .components.reporting import write_reports


def load_samples(run_dir):
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    completed,failed=[],[]
    fingerprints={}
    for index, sample_id in enumerate(manifest['sample_ids']):
        path=run_dir/'samples'/f'{index:05d}.json'
        sample=json.loads(path.read_text(encoding='utf-8'))
        if sample['id']!=sample_id:
            raise ValueError(f'Sample ID mismatch: {path}')
        fingerprints[path.relative_to(run_dir).as_posix()]=sha256(path.read_bytes())
        (completed if sample['status']=='completed' else failed).append(sample)
    return manifest,completed,failed,fingerprints


def rescore(run_dir, *, bertscore_model=None, layers=12, device='cpu', output_dir=None):
    run_dir=Path(run_dir).resolve()
    manifest,completed,failed,fingerprints=load_samples(run_dir)
    if not completed:
        raise ValueError('No completed direct summaries to score')
    for s in completed:
        s['quality']={VARIANT:{**reference_metrics(s['result']['answers'][VARIANT],s['reference']),
                              **source_overlap_support(s['result']['answers'][VARIANT],s['source'])}}
    semantic=add_bertscore(completed,bertscore_model,layers,device,variants=(VARIANT,)) if bertscore_model else None
    folder=Path(output_dir).resolve() if output_dir else run_dir/'evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if folder==run_dir or run_dir.is_relative_to(folder):
        raise ValueError('Use a separate evaluation directory')
    folder.mkdir(parents=True,exist_ok=False)
    write_reports(folder,completed,failed,len(manifest['sample_ids']))
    atomic_json(folder/'evaluation_manifest.json',dict(source_run=str(run_dir),
        source_manifest_sha256=sha256((run_dir/'manifest.json').read_bytes()),sample_sha256=fingerprints,
        semantic_evaluation=semantic,note='No generation rerun; resources reused from original checkpoints.'))
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--bertscore-model')
    parser.add_argument('--bertscore-layers',type=int,default=12)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    print('Reports:',rescore(args.run_dir,bertscore_model=args.bertscore_model,layers=args.bertscore_layers,
                            device=args.device,output_dir=args.output_dir))
