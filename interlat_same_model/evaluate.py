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
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also generate question-only and Sender-plan-as-text control answers.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def baseline_user_prompt(question: str, sender_draft: str | None = None) -> str:
    if sender_draft is None:
        return (
            "Answer the question concisely. Do not invent details that are not "
            f"supported by the question.\n\nQuestion: {question}"
        )
    return (
        "Answer the question using only the internal plan below. Return only a "
        "concise answer and do not mention the plan.\n\n"
        f"Internal plan:\n{sender_draft}\n\nQuestion: {question}"
    )


@torch.inference_mode()
def generate_text_control(receiver, tokenizer, user_prompt: str, device: str, max_new_tokens: int) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    output = receiver.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0, inputs.input_ids.size(1):], skip_special_tokens=True).strip()


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
    model = InterlatReceiver(
        receiver, checkpoint["source_hidden_size"], checkpoint["num_heads"],
        checkpoint["compressed_latent_len"] or None, checkpoint["plan_similarity_weight"], checkpoint["random_contrast_weight"],
    ).to(device=device, dtype=dtype).eval()
    model.boundary_embeddings.data.copy_(checkpoint["boundary_embeddings"].to(device=device, dtype=dtype))
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
        controls = ""
        if args.include_baselines:
            question_only = generate_text_control(
                receiver,
                tokenizer,
                baseline_user_prompt(record["question"]),
                device,
                args.max_new_tokens,
            )
            text_plan = generate_text_control(
                receiver,
                tokenizer,
                baseline_user_prompt(record["question"], record["sender_draft"]),
                device,
                args.max_new_tokens,
            )
            controls = (
                f"\n\nQuestion-only baseline (no context, no Sender message):\n{question_only}"
                f"\n\nText-plan upper bound (Sender plan transmitted as text):\n{text_plan}"
            )
        blocks.append(
            f"{'=' * 80}\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n"
            f"Sender internal plan (not transmitted as text):\n{record['sender_draft']}\n\n"
            f"Interlat receiver answer (latent-only):\n{answer}{controls}\n\nGold answer:\n{record['answer']}"
        )
        print(f"processed {index}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"saved output: {args.output}")


if __name__ == "__main__":
    main()
