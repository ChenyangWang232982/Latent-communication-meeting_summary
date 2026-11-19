"""Prepare AMI, ICSI and Committee with one tokenizer load; no shell loop required."""
import argparse
from pathlib import Path

from .prepare_qmsum import prepare
from .token_filter import TranscriptTokenFilter, add_token_arguments, load_token_filter


DOMAINS = {'ami': 'Product', 'icsi': 'Academic', 'qmsum_committee': 'Committee'}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('external/QMSum'))
    parser.add_argument('--output-dir', type=Path, default=Path('input/experiment_1_unfiltered_cap100'))
    parser.add_argument('--domains', nargs='+', choices=list(DOMAINS), default=list(DOMAINS))
    parser.add_argument('--max-records', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--overwrite', action='store_true')
    add_token_arguments(parser)
    return parser.parse_args()


def run(args):
    tokenizer = None
    results = {}
    for name in args.domains:
        settings = argparse.Namespace(**vars(args), input=args.input_root / 'data' / DOMAINS[name] / 'all',
                                      output=args.output_dir / f'{name}.jsonl')
        try:
            if settings.output.exists() and not args.overwrite:
                raise FileExistsError(f'{settings.output} exists; use --overwrite to rebuild or --domains to select other domains.')
            if tokenizer is None:
                tokenizer = load_token_filter(args).tokenizer
            # Share weights/files, not counters: every domain gets its own audit statistics.
            counter = TranscriptTokenFilter(tokenizer, model=args.tokenizer_model,
                minimum=args.min_transcript_tokens, maximum=args.max_transcript_tokens)
            results[name] = dict(status='completed', **prepare(settings, counter))
        except Exception as exc:
            results[name] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'{name}: {results[name]["error"]}', flush=True)
    print('\nDataset | status | records / error', flush=True)
    for name, result in results.items():
        print(f"{name} | {result['status']} | {result.get('records_written', result.get('error'))}", flush=True)
    return results


def main():
    results = run(arguments())
    if any(result['status'] == 'failed' for result in results.values()):
        raise SystemExit('Some domains failed; successful exports were preserved. See the errors above.')


if __name__ == '__main__':
    main()
