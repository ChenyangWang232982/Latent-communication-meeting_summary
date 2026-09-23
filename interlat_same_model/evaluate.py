"""Generate answers from Interlat latent messages without Sender text access."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlat_same_model.data import HiddenStateDataset
from interlat_same_model.latent import InterlatReceiver
from interlat_same_model.train import receiver_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.parent / "tokenizer")
    receiver = AutoModelForCausalLM.from_pretrained(checkpoint["model_name"], torch_dtype=dtype, low_cpu_mem_usage=True)
    receiver.resize_token_embeddings(len(tokenizer))
    receiver.config.pad_token_id = tokenizer.pad_token_id
    if checkpoint["unfreeze_receiver"]:
        receiver.load_state_dict(checkpoint["receiver"])
    bop_id, eop_id = tokenizer.convert_tokens_to_ids("<bop>"), tokenizer.convert_tokens_to_ids("<eop>")
    receiver.get_input_embeddings().weight.data[[bop_id, eop_id]] = checkpoint["boundary_embeddings"].to(receiver.dtype)
    model = InterlatReceiver(
        receiver, checkpoint["source_hidden_size"], checkpoint["num_heads"],
        checkpoint["compressed_latent_len"] or None, checkpoint["plan_similarity_weight"], checkpoint["random_contrast_weight"],
    ).to(device=device, dtype=dtype).eval()
    model.adapter.load_state_dict(checkpoint["adapter"])
    if model.compressor:
        model.compressor.load_state_dict(checkpoint["compressor"])
    rows = HiddenStateDataset(args.hidden_data, args.num_samples)
    blocks = []
    for index, record in enumerate(rows, start=1):
        prompt = receiver_prompt(tokenizer, record["question"], device)
        generated = model.generate(
            sender_states=record["sender_states"].unsqueeze(0).to(device=device, dtype=dtype),
            bop_id=bop_id, eop_id=eop_id, prompt_ids=prompt, max_new_tokens=args.max_new_tokens,
        )
        answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        blocks.append(
            f"{'=' * 80}\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n"
            f"Sender internal plan (not transmitted as text):\n{record['sender_draft']}\n\n"
            f"Interlat receiver answer:\n{answer}\n\nGold answer:\n{record['answer']}"
        )
        print(f"processed {index}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"saved output: {args.output}")


if __name__ == "__main__":
    main()
