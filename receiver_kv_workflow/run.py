"""Summarize a transcript with global selected context and chronological specialists."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from .cli import add_workflow_arguments, model_config
from .engine import ReceiverKVRunner


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, help='Primary final summary; other variants use sibling files')
    parser.add_argument('--output-dir', type=Path, default=Path('output/receiver_kv'))
    parser.add_argument('--metrics-dir', type=Path, default=Path('test_result/receiver_kv'))
    parser.add_argument('--routing-audit', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['chunk_kv', 'chunk_text'], default=['chunk_kv'])
    return parser.parse_args()


def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


def main():
    args = arguments()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = args.output.parent if args.output else args.output_dir / stamp
    metrics_dir = args.metrics_dir / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    primary = args.variants[0]
    base = args.output.with_suffix('') if args.output else output_dir / primary

    def live(event):
        stage = event['stage']
        base.with_name(f'{base.name}.{stage}.partial.txt').write_text(event['text'], encoding='utf-8')
        base.with_name(f'{base.name}.{stage}.status.json').write_text(json.dumps(
            {k: event[k] for k in ('tokens', 'finished', 'stop_reason', 'repeated_tail') if k in event},
            indent=2), encoding='utf-8')

    transcript = args.transcript.read_text(encoding='utf-8-sig').strip()
    config = model_config(args)
    runner = ReceiverKVRunner(config, progress=log)
    result = runner.run(args.transcript.stem, transcript, variants=tuple(args.variants),
                        routing_audit=args.routing_audit, on_generation=live)
    from meeting_evaluation.metrics import rouge_l_f1
    from dataclasses import asdict
    reference = args.reference.read_text(encoding='utf-8-sig') if args.reference else ''
    for variant, answer in result['answers'].items():
        path = (args.output if variant == primary else args.output.with_name(
            f'{args.output.stem}.{variant}{args.output.suffix or ".txt"}')) if args.output else output_dir / f'{variant}.txt'
        path.write_text(answer + '\n', encoding='utf-8')
        artifact_base = path.with_suffix('')
        detail = result['chunk_specialists_by_variant'][variant]
        artifact_base.with_name(f'{artifact_base.name}.specialists.json').write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding='utf-8')
        for stage, reasoning in detail['thinking_outputs'].items():
            artifact_base.with_name(f'{artifact_base.name}.{stage}.thinking.txt').write_text(reasoning, encoding='utf-8')
        result['variants'][variant]['rouge_l_f1'] = rouge_l_f1(answer, reference) if reference else None
        log(f'Summary: {path}')
    if args.routing_audit:
        from .routing_audit import render_audit
        base.with_name(f'{base.name}.audit.txt').write_text(render_audit(result['routing_audit']), encoding='utf-8')
        base.with_name(f'{base.name}.audit.json').write_text(json.dumps(result['routing_audit'], ensure_ascii=False, indent=2), encoding='utf-8')
    report = {k: v for k, v in result.items() if k not in ('answers', 'handoffs', 'chunk_specialists', 'chunk_specialists_by_variant', 'routing_audit')}
    report.update(model=args.model, source_file=str(args.transcript), settings=asdict(config))
    (metrics_dir / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ['Global selected context + original-part specialists',
            'variant | ROUGE-L | pipeline sec | all-stage input/output token events | final output tokens']
    for variant, measurement in result['variants'].items():
        rows.append(f"{variant} | {measurement['rouge_l_f1']} | {measurement['pipeline_seconds_excluding_model_load']:.2f} | "
                    f"{sum(measurement['token_accounting'].values())} | {measurement['generated_tokens']}")
        for stage, values in result['chunk_specialists_by_variant'][variant]['stages'].items():
            rows.append(f"  {stage}: prompt={values['prompt_tokens']} | output={values['output_tokens']} | "
                        f"source KV positions={values['source_kv_positions']} | seconds={values['seconds']:.2f}")
    rows.append('Model-token events, KV tensor bytes and UTF-8 bytes are different units; no network traffic or billing is measured.')
    (metrics_dir / 'metrics.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    log(f'Metrics: {metrics_dir}')


if __name__ == '__main__':
    main()
