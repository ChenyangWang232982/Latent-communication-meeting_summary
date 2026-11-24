import argparse
from dataclasses import dataclass
from pathlib import Path

from experiment_1.components.config import reproducibility, sha256

VARIANT = 'direct_text'


@dataclass(frozen=True)
class DirectConfig:
    model: str = 'Qwen/Qwen3-8B'
    device: str = 'cuda'
    seed: int = 12
    summary_seed: int = 42
    summary_temperature: float = 0.3
    summary_top_p: float = 0.9
    summary_thinking: bool = True
    top_k: int = 0
    repetition_penalty: float = 1.0
    fp32_lm_head: bool = False


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='Experiment 2: full-transcript single-agent summary; no selectors or specialists.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--benchmark', choices=('ami', 'icsi', 'qmsum_committee', 'meetingbank'), default='ami')
    parser.add_argument('--output-dir', type=Path, default=Path('test_result/experiment_2'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--num-samples', type=int, default=3, help='First N input records; 0 means all')
    parser.add_argument('--model', default=DirectConfig.model)
    parser.add_argument('--device', default=DirectConfig.device)
    parser.add_argument('--seed', type=int, default=12, help='Model initialization seed; generation uses summary-seed')
    parser.add_argument('--summary-seed', type=int, default=42)
    parser.add_argument('--summary-temperature', type=float, default=0.3)
    parser.add_argument('--summary-top-p', type=float, default=0.9)
    parser.add_argument('--summary-thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--fp32-lm-head', action='store_true')
    args = parser.parse_args(argv)
    if args.num_samples < 0:
        parser.error('--num-samples must be non-negative')
    return args


def model_config(args):
    return DirectConfig(**{key: getattr(args, key) for key in DirectConfig.__dataclass_fields__})


def manifest_for(args, records, config):
    manifest = reproducibility(args, records, config)
    root = Path(__file__).resolve().parents[1]
    manifest['code_sha256'].update({f'experiment_2/{p.relative_to(root).as_posix()}': sha256(p.read_bytes())
                                    for p in sorted(root.rglob('*.py'))})
    manifest.update(variants=[VARIANT], workflow='full_transcript_single_agent',
                    note='Complete source text in one fresh prompt. Normal autoregressive KV is internal, not inter-agent communication.')
    return manifest
