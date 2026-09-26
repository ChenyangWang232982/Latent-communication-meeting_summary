"""Compare multiple StateBridge prefix lengths on one fixed QASPER sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bridge import StateBridge
from .run import answer_variant, load_records, sender_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based line index in the JSONL file")
    parser.add_argument("--k-values", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    parser.add_argument("--source-max-tokens", type=int, default=4096)
    parser.add_argument("--sender-max-new-tokens", type=int, default=256)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=128)
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.k_values or min(args.k_values) < 2:
        raise ValueError("--k-values must contain integers of at least 2")
    records = load_records(args.data, 0)
    if not 0 <= args.sample_index < len(records):
        raise IndexError(f"--sample-index must be between 0 and {len(records) - 1}")
    record = records[args.sample_index]

    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    bridge = StateBridge(
        model.get_input_embeddings().weight,
        regularization=args.regularization,
        snap_ratio=args.snap_ratio,
        vocab_chunk_size=args.vocab_chunk_size,
    )

    # Generate once, then slice the same latent message for every K.
    args.prefix_tokens = max(args.k_values)
    plan, states, token_ids = sender_message(model, tokenizer, record, args)
    text_answer = answer_variant(model, tokenizer, bridge, record, plan, states, token_ids, "text", args)
    no_comm_answer = answer_variant(model, tokenizer, bridge, record, plan, states, token_ids, "no_comm", args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as report:
        report.write(f"StateBridge K sweep\nmodel={args.model}, id={record['id']}\n\n")
        report.write(f"Question:\n{record['question']}\n\n")
        report.write(f"Sender message (debug only):\n{plan}\n\n")
        report.write(f"Available final message states: {states.size(0)}\n\n")
        report.write(f"Text-message control:\n{text_answer}\n\n")
        report.write(f"No-communication control:\n{no_comm_answer}\n\n")
        for requested_k in args.k_values:
            actual_k = min(requested_k, states.size(0))
            suffix_states = states[-actual_k:]
            suffix_ids = token_ids[-actual_k:]
            answer = answer_variant(
                model, tokenizer, bridge, record, plan, suffix_states, suffix_ids, "statebridge", args
            )
            report.write(f"StateBridge answer (K={requested_k}, actual={actual_k}):\n{answer}\n\n")
        report.write(f"Gold answer:\n{record['answer']}\n")
    print(f"wrote K sweep: {args.output}", flush=True)


if __name__ == "__main__":
    main()
