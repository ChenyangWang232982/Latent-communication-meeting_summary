"""Prepare AMI, ICSI, Committee and diarized MeetingBank with one tokenizer load."""
import argparse
from pathlib import Path

from .prepare_qmsum import prepare as prepare_qmsum
from .prepare_meetingbank import DEFAULT_DATASET, prepare as prepare_meetingbank
from .token_filter import TranscriptTokenFilter, add_token_arguments, load_token_filter


DOMAINS = {'ami': 'Product', 'icsi': 'Academic', 'qmsum_committee': 'Committee'}


def arguments(default_domains=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('external/QMSum'), help='QMSum root')
    parser.add_argument('--meetingbank-root', type=Path, default=Path('external/MeetingBank'))
    parser.add_argument('--split', default='test', help='MeetingBank split')
    parser.add_argument('--output-dir', type=Path, default=Path('input/experiment_1_unfiltered_cap100'))
    parser.add_argument('--domains', nargs='+', choices=[*DOMAINS, 'meetingbank'],
                        default=default_domains if default_domains is not None else [*DOMAINS, 'meetingbank'])
    parser.add_argument('--max-records', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--overwrite', action='store_true')
    add_token_arguments(parser)
    return parser.parse_args()


def run(args):
    tokenizer = None
    results = {}
    for name in args.domains:
        settings = argparse.Namespace(**vars(args), output=args.output_dir / f'{name}.jsonl')
        try:
            if settings.output.exists() and not args.overwrite:
                raise FileExistsError(f'{settings.output} exists; use --overwrite to rebuild or --domains to select other domains.')
            if name == 'meetingbank':
                settings.input = args.meetingbank_root
                settings.dataset, settings.cache_dir, settings.require_speakers = DEFAULT_DATASET, None, True
                export = prepare_meetingbank
            else:
                settings.input = args.input_root / 'data' / DOMAINS[name] / 'all'
                export = prepare_qmsum
            if tokenizer is None:
                tokenizer = load_token_filter(args).tokenizer
            # Tokenizer is shared; independent counters preserve per-dataset statistics.
            counter = TranscriptTokenFilter(tokenizer, model=args.tokenizer_model,
                minimum=args.min_transcript_tokens, maximum=args.max_transcript_tokens)
            results[name] = dict(status='completed', **export(settings, counter))
        except Exception as exc:
            results[name] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'{name}: {results[name]["error"]}', flush=True)
    print('\nDataset | status | records / error', flush=True)
    for name, result in results.items():
        print(f"{name} | {result['status']} | {result.get('records_written', result.get('error'))}", flush=True)
    return results


def main(default_domains=None):
    results = run(arguments(default_domains))
    if any(result['status'] == 'failed' for result in results.values()):
        raise SystemExit('Some datasets failed; successful exports were preserved. See the errors above.')


if __name__ == '__main__':
    main()
