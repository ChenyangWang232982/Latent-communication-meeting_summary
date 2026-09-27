"""Run LaTER-style Sender reasoning followed by training-free StateBridge transfer.

The LaTER repository is intentionally an external dependency.  This keeps its
custom latent/explcit switching implementation unmodified, while this module
adapts the Sender prompt and its final explicit handoff to QASPER.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .bridge import StateBridge
from .run import (
    answer_variant,
    decode,
    load_records,
    remove_thinking_tokens,
    trim_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="QASPER JSONL file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--later-root",
        type=Path,
        required=True,
        help="Checkout of https://github.com/TioeAre/LaTER",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=8, help="0 means all records")
    parser.add_argument("--source-max-tokens", type=int, default=4096)
    parser.add_argument("--later-max-new-tokens", type=int, default=512)
    parser.add_argument("--later-max-steps", type=int, default=32)
    parser.add_argument("--later-latent-tokens", type=int, default=64)
    parser.add_argument("--later-explicit-tokens", type=int, default=96)
    parser.add_argument("--later-entropy-threshold", type=float, default=1.2)
    parser.add_argument("--later-temperature", type=float, default=0.7)
    parser.add_argument("--later-top-p", type=float, default=0.95)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=64, help="StateBridge K")
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["statebridge", "text", "no_comm"],
        choices=["statebridge", "text", "raw_hidden", "random_prefix", "no_comm"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_later(later_root: Path, device: str):
    """Import the official training-free implementation from an external checkout."""
    root = later_root.resolve()
    if not (root / "models.py").is_file() or not (root / "methods" / "latent_switch.py").is_file():
        raise FileNotFoundError(
            f"--later-root must be a LaTER checkout containing models.py and methods/: {root}"
        )
    sys.path.insert(0, str(root))
    try:
        models = importlib.import_module("models")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Could not import LaTER. Install its dependencies from the supplied checkout, "
            "for example: pip install -r external/LaTER/requirements.txt"
        ) from error

    later_args = SimpleNamespace(
        latent_space_realign=False,
        enable_prefix_caching=False,
        method="latent_switch",
        device2=device,
        use_second_HF_model=False,
    )
    return models.ModelWrapper, later_args


def sender_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Reason privately. Then state a concise factual handoff for another agent. "
                "Use only the supplied paper evidence and include the exact answer."
            ),
        },
        {
            "role": "user",
            "content": f"Evidence:\n{context}\n\nQuestion: {question}",
        },
    ]


@torch.inference_mode()
def states_for_handoff(model, tokenizer, handoff: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover final-layer states for only the explicit handoff tokens."""
    encoded = tokenizer(handoff, return_tensors="pt", add_special_tokens=False).to(device)
    if encoded.input_ids.size(1) < 2:
        raise RuntimeError("LaTER produced no usable explicit handoff tokens.")
    outputs = model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        output_hidden_states=True,
    )
    return outputs.hidden_states[-1][0], encoded.input_ids[0]


@torch.inference_mode()
def later_sender_message(wrapper, record: dict[str, Any], args: argparse.Namespace):
    """Run LaTER's official latent/explcit switch on a QASPER-adapted prompt."""
    tokenizer = wrapper.tokenizer
    context = trim_context(tokenizer, record["context"], args.source_max_tokens)
    _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
        [sender_messages(context, record["question"])], add_generation_prompt=True
    )
    (
        generated_texts,
        _,
        type_masks,
        all_generated_ids,
        _,
        _,
        _,
    ) = wrapper.generate_step_reasoning_batch(
        input_ids,
        attention_mask,
        max_steps=args.later_max_steps,
        max_new_tokens=args.later_max_new_tokens,
        entropy_threshold=args.later_entropy_threshold,
        latent_tokens_limit=args.later_latent_tokens,
        explicit_tokens_limit=args.later_explicit_tokens,
        temperature=args.later_temperature,
        top_p=args.later_top_p,
    )

    # In the official method type 0 is an internal latent step and type 1 is
    # explicit text.  Only the latter can be a clean handoff for StateBridge.
    explicit_ids = [
        int(token_id)
        for token_id, token_type in zip(all_generated_ids[0], type_masks[0])
        if int(token_type) == 1
    ]
    if not explicit_ids:
        # A run may finish entirely in latent mode.  Keep it observable rather
        # than silently pretending it produced a factual message.
        raise RuntimeError(
            "LaTER ended without explicit tokens; increase --later-max-new-tokens "
            "or lower --later-latent-tokens."
        )
    handoff_ids = torch.tensor([explicit_ids], device=wrapper.model.device, dtype=torch.long)
    states, _ = states_for_handoff(wrapper.model, tokenizer, decode(tokenizer, handoff_ids), args.device)
    handoff_ids, states = remove_thinking_tokens(tokenizer, handoff_ids, states)
    if handoff_ids.size(1) < 2:
        raise RuntimeError("LaTER's explicit handoff contained fewer than two usable tokens.")
    count = min(args.prefix_tokens, states.size(0), handoff_ids.size(1))
    return (
        decode(tokenizer, handoff_ids),
        states[-count:],
        handoff_ids[0, -count:],
        generated_texts[0],
        len(explicit_ids),
    )


def main() -> None:
    args = parse_args()
    if args.prefix_tokens < 2:
        raise ValueError("--prefix-tokens must be at least 2")
    torch.manual_seed(args.seed)
    model_wrapper_cls, later_args = load_later(args.later_root, args.device)
    wrapper = model_wrapper_cls(args.model, torch.device(args.device), args=later_args)
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    bridge = StateBridge(
        model.get_input_embeddings().weight,
        regularization=args.regularization,
        snap_ratio=args.snap_ratio,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    records = load_records(args.data, args.num_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jsonl, args.output.open("w", encoding="utf-8") as report:
        report.write(f"LaTER-style Sender + StateBridge QASPER run: {datetime.now().isoformat(timespec='seconds')}\n")
        report.write(
            f"model={args.model}, K={args.prefix_tokens}, later_root={args.later_root}, "
            f"variants={','.join(args.variants)}\n\n"
        )
        for index, record in enumerate(records, start=1):
            handoff, states, token_ids, trace, explicit_count = later_sender_message(wrapper, record, args)
            answers = {
                variant: answer_variant(model, tokenizer, bridge, record, handoff, states, token_ids, variant, args)
                for variant in args.variants
            }
            result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "later_explicit_handoff": handoff,
                "later_debug_trace": trace,
                "later_explicit_token_count": explicit_count,
                "statebridge_prefix_tokens": int(states.size(0)),
                "answers": answers,
            }
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\n")
            report.write(f"Question:\n{record['question']}\n\n")
            report.write(f"LaTER explicit handoff (debug only):\n{handoff}\n\n")
            report.write(f"LaTER explicit token count: {explicit_count}\n")
            report.write(f"StateBridge prefix token count: {states.size(0)}\n\n")
            for variant, answer in answers.items():
                report.write(f"{variant} answer:\n{answer}\n\n")
            report.write(f"Gold answer:\n{record['answer']}\n\n")
            report.flush()
            print(f"completed {index}/{len(records)}: {record['id']}", flush=True)


if __name__ == "__main__":
    main()
