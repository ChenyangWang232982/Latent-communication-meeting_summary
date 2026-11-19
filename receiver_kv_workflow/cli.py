"""Shared model and specialist settings for CLI, experiments and streaming."""
import argparse
from dataclasses import fields

from .engine import ReceiverKVConfig


def add_workflow_arguments(parser):
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--role-budget-ratio', type=float, default=0.25)
    parser.add_argument('--speaker-neighbor-blocks', type=int, default=1)
    parser.add_argument('--span-tokens', type=int, default=32)
    parser.add_argument('--query-batch-tokens', type=int, default=8)
    parser.add_argument('--fp32-lm-head', action='store_true', help='Optional numerical control; default model precision otherwise')
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Final synthesis repetition penalty')
    parser.add_argument('--specialist-overlap', type=float, default=0.25)
    parser.add_argument('--specialist-temperature', type=float, default=0.3)
    parser.add_argument('--specialist-top-p', type=float, default=0.9)
    parser.add_argument('--specialist-repetition-penalty', type=float, default=1.1)
    parser.add_argument('--summary-thinking', action=argparse.BooleanOptionalAction, default=True,
                        help='Thinking only at final synthesis, identically in both variants')
    parser.add_argument('--summary-seed', type=int, default=42)
    parser.add_argument('--summary-temperature', type=float, default=0.3)
    parser.add_argument('--summary-top-p', type=float, default=0.9)
    from vector.cli import add_arguments
    add_arguments(parser)


def model_config(args):
    defaults = ReceiverKVConfig()
    return ReceiverKVConfig(**{field.name: getattr(args, field.name, getattr(defaults, field.name))
                               for field in fields(defaults)})
