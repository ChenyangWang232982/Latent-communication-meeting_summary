"""Run the AMI100 workflow, direct, random, and scoring commands sequentially."""
import argparse
import importlib.util
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json
from meeting_evaluation.records import load_records
from .components.task_queue import SerialQueue, queue_lock, run_command, now

PROJECT = Path(__file__).resolve().parents[1]
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
SNAPSHOT_CODE = (
    'import json,sys; from pathlib import Path; from huggingface_hub import snapshot_download; '
    'p=snapshot_download(sys.argv[1],revision=sys.argv[2],allow_patterns=["*.json","*.safetensors","*.txt","*.jinja"]); '
    'out=Path(sys.argv[3]); out.parent.mkdir(parents=True,exist_ok=True); '
    'out.write_text(json.dumps(dict(path=p,repo=sys.argv[1],revision=sys.argv[2])),encoding="utf-8")')


def run(args, *, executor=run_command):
    if args.resume:
        root = args.resume.resolve()
        state = json.loads((root/'queue.json').read_text(encoding='utf-8'))
        config = state['config']
        print('Resuming saved queue configuration; execution settings come from queue.json.', flush=True)
    else:
        records = load_records('jsonl', args.input)
        if len(records) < args.num_samples:
            raise ValueError(f'Requested {args.num_samples} AMI samples, input has only {len(records)}')
        records = records[:args.num_samples]
        if any(not r.answer.strip() for r in records):
            raise ValueError('AMI100 quality evaluation requires original reference summaries')
        seeds = list(dict.fromkeys(args.random_seeds))
        payload = args.input.read_bytes()
        config = dict(input_source=str(args.input.resolve()), input_sha256=sha256(payload),
            num_samples=args.num_samples, model=args.model, model_revision=args.model_revision,
            device=args.device, embedding_model=args.embedding_model, random_seeds=seeds,
            bertscore_model=args.bertscore_model, bertscore_layers=args.bertscore_layers)
        root = args.output_dir.resolve()/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        root.mkdir(parents=True, exist_ok=False)
        frozen_input = root/'input.jsonl'
        frozen_input.write_bytes(payload)
        config['input'] = str(frozen_input)
        names = ['00_model', '01_workflow', '02_direct',
                 *[f'03_random_seed{s}' for s in seeds], '04_evaluation']
        state = dict(status='pending', created_at_utc=now(), config=config,
            tasks={name:dict(status='pending') for name in names})
        atomic_json(root/'queue.json', state)
    print('Task queue:', root, flush=True)
    with queue_lock(root):
        q = SerialQueue(root, state, cwd=PROJECT, executor=executor)
        try:
            if sha256(Path(config['input']).read_bytes()) != config['input_sha256']:
                raise ValueError('Input data changed since this queue was created')
            state.update(status='running')
            state.pop('error', None)
            state.pop('finished_at_utc', None)
            q.save()
            py = [sys.executable, '-u']
            model_task = q.step('00_model', 'model', lambda out:
                [*py, '-c', SNAPSHOT_CODE, config['model'], config['model_revision'], str(out/'model.json')])
            model_path = json.loads((model_task/'model.json').read_text(encoding='utf-8'))['path']
            generation = ['--model', model_path, '--device', config['device'], '--seed', '12',
                '--summary-seed', '42', '--summary-temperature', '0.3', '--summary-top-p', '0.9', '--summary-thinking']
            data = ['--input', config['input'], '--benchmark', 'ami', '--num-samples', str(config['num_samples'])]
            workflow = q.step('01_workflow', 'workflow', lambda out:
                [*py, '-m', 'experiment_3', *data, '--output-dir', str(out),
                 '--conditions', 'baseline', 'semantic', *generation,
                 '--role-budget-ratio', '0.25', '--speaker-neighbor-blocks', '1',
                 '--vector-routing', '--embedding-model', config['embedding_model'],
                 '--supplement-blocks', '2', '--exclude-neighbor-blocks', '1'])
            direct = q.step('02_direct', 'direct', lambda out:
                [*py, '-m', 'experiment_2', *data, '--output-dir', str(out), *generation])
            random_runs = []
            for seed in config['random_seeds']:
                random_runs.append(q.step(f'03_random_seed{seed}', 'random', lambda out, seed=seed:
                    [*py, '-m', 'experiment_3.random_control', '--source-run', str(workflow),
                     '--output-dir', str(out), '--num-samples', str(config['num_samples']),
                     '--random-seed', str(seed), '--device', config['device']]))
            evaluation = q.step('04_evaluation', 'evaluation', lambda out:
                [*py, '-m', 'experiment_3.evaluate', '--run-dir', str(workflow), '--direct-run', str(direct),
                 '--random-runs', *map(str, random_runs), '--output-dir', str(out),
                 '--bertscore-model', config['bertscore_model'],
                 '--bertscore-layers', str(config['bertscore_layers']), '--device', config['device']])
            state.update(status='completed_with_failures' if any(t['status']=='completed_with_failures'
                for t in state['tasks'].values()) else 'completed', evaluation=str(evaluation), finished_at_utc=now())
            q.save()
            print(f'\n[queue] Finished: {state["status"]}\nScores: {evaluation/"summary.txt"}', flush=True)
        except BaseException as exc:
            if state['status'] not in ('failed', 'interrupted'):
                state['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                state['error'] = str(exc)
                q.save()
            print(f'[queue] Stopped: {exc}\nResume with: python -m experiment_3.task_queue --resume {root}', flush=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
    return root, state['status'] not in ('completed', 'completed_with_failures')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('input/experiment_1_unfiltered_cap100/ami.jsonl'))
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=Path('test_result/ami100_task_queue'))
    parser.add_argument('--resume', type=Path, help='Existing queue timestamp directory; inherits saved settings')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--model-revision', default=MODEL_REVISION)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--random-seeds', type=int, nargs='+', default=[7])
    parser.add_argument('--bertscore-model', default='allenai/longformer-base-4096')
    parser.add_argument('--bertscore-layers', type=int, default=12)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    if importlib.util.find_spec('bert_score') is None:
        parser.error('Install scoring dependencies before starting: python -m pip install -r experiment_1/requirements-quality.txt')
    def stop_on_term(signum, frame):
        raise KeyboardInterrupt('Task queue received SIGTERM')
    signal.signal(signal.SIGTERM, stop_on_term)
    folder, failed = run(args)
    raise SystemExit(1 if failed else 0)
