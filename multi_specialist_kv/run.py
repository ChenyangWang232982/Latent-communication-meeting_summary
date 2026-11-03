"""Run the experimental four-specialist KV meeting workflow."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner
from vector.cli import add_arguments, config_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--role-budget-ratio', type=float, default=0.25)
    parser.add_argument('--speaker-neighbor-blocks', type=int, default=1)
    parser.add_argument('--span-tokens', type=int, default=32)
    parser.add_argument('--query-batch-tokens', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    transcript = args.transcript.read_text(encoding='utf-8-sig').strip()
    if not transcript:
        parser.error('Transcript is empty')
    def log(message):
        print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)
    config = ReceiverKVConfig(model=args.model, device=args.device, role_budget_ratio=args.role_budget_ratio,
                              speaker_neighbor_blocks=args.speaker_neighbor_blocks, span_tokens=args.span_tokens,
                              query_batch_tokens=args.query_batch_tokens, seed=args.seed, **config_kwargs(args))
    runner = ReceiverKVRunner(config, progress=log)
    result = runner.run(args.transcript.stem, transcript, variants=('multi_specialist_kv',))
    report = result['variants']['multi_specialist_kv']
    report.update(model=args.model, source_positions=result['source_positions'],
                  source_cache_bytes=result['source_cache_bytes'], prefill_seconds=result['prefill_seconds'],
                  selection_seconds=result['selection_seconds'], vector_routing=result['vector_routing'], seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result['answers']['multi_specialist_kv'] + '\n', encoding='utf-8')
    specialist_path = args.output.with_name(args.output.stem + '.specialists.json')
    metrics_path = args.output.with_name(args.output.stem + '.metrics.json')
    specialist_path.write_text(json.dumps(result['specialist_handoffs'], ensure_ascii=False, indent=2), encoding='utf-8')
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'Summary: {args.output}; specialists: {specialist_path}; metrics: {metrics_path}')
    if report['output_cap_hit']:
        log('Warning: final summary reached the model context limit')
    if any(not role['valid_json'] for role in report['roles'].values()):
        log('Warning: one or more specialist handoffs were not valid JSON; inspect the specialist file')


if __name__ == '__main__':
    main()
