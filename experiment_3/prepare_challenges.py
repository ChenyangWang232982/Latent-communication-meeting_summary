"""Freeze twenty new endpoint challenges on distinct original AMI meetings."""
import argparse
from pathlib import Path

from experiment_2.challenge_samples.prepare import create
from .challenge_cases import CASES


def prepare(input_path, output_dir, *, skip=6):
    return create(output_dir, input_path=input_path, additions=CASES, skip=skip,
                  version='ami_endpoint_expanded_v1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('input/ami_endpoint_challenge_20_v1'))
    parser.add_argument('--skip', type=int, default=6,
                        help='Skip the three old challenge sources and three original-meeting pilot sources')
    args = parser.parse_args()
    if args.skip < 0:
        parser.error('--skip must be non-negative')
    print('Prepared:', prepare(args.input, args.output_dir, skip=args.skip))
