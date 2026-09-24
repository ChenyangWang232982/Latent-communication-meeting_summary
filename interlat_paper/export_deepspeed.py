"""Convert a ZeRO checkpoint into the consolidated checkpoint used by evaluate.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing DeepSpeed tag 'best'")
    parser.add_argument("--metadata", type=Path, required=True, help="best_metadata.pt written by train.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except ImportError as error:
        raise RuntimeError("Install deepspeed to export a ZeRO checkpoint.") from error
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(args.checkpoint_dir), tag="best")
    actor = {name.removeprefix("actor."): value for name, value in state_dict.items() if name.startswith("actor.")}
    adapter = {name.removeprefix("adapter."): value for name, value in state_dict.items() if name.startswith("adapter.")}
    metadata = torch.load(args.metadata, map_location="cpu", weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "actor": actor, "adapter": adapter}, args.output)
    print(f"saved consolidated checkpoint: {args.output}")


if __name__ == "__main__":
    main()
