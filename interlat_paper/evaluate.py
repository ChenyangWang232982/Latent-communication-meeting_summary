"""Evaluate a paper-stage Interlat checkpoint with latent-only and text controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlat_paper.model import InterlatActor
from interlat_paper.train import draft_ids, prompt_parts
from interlat_same_model.data import HiddenStateDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt or exported DeepSpeed checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--include-text-control", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def generate_text_message(model, prompt_prefix, assistant_prefix, plan, bop_id, eop_id, max_new_tokens):
    embedding = model.actor.get_input_embeddings()
    bop, eop = model._boundary_embeddings(bop_id, eop_id, prompt_prefix.size(0))
    inputs = torch.cat((embedding(prompt_prefix), bop, embedding(plan), eop, embedding(assistant_prefix)), dim=1)
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    return model.actor.generate(
        inputs_embeds=inputs,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=model.actor.config.pad_token_id,
    )


def main():
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "actor" not in checkpoint:
        raise ValueError("Checkpoint has no consolidated Actor weights. Run export_deepspeed first for a ZeRO checkpoint.")
    metadata = checkpoint["metadata"]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.parent / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    actor = AutoModelForCausalLM.from_pretrained(metadata["actor_model"], torch_dtype=dtype, low_cpu_mem_usage=True)
    actor.resize_token_embeddings(len(tokenizer))
    actor.config.pad_token_id = tokenizer.pad_token_id
    model = InterlatActor(actor, metadata["source_hidden_size"], metadata["num_heads"])
    model.actor.load_state_dict(checkpoint["actor"])
    model.adapter.load_state_dict(checkpoint["adapter"])
    model = model.to(device=device, dtype=dtype).eval()
    records = HiddenStateDataset(args.hidden_data, args.num_samples)
    blocks = []
    for number, record in enumerate(records, start=1):
        prompt_prefix, assistant_prefix = prompt_parts(tokenizer, record["question"], device, metadata["max_prompt_tokens"])
        states = record["sender_states"].unsqueeze(0).to(device=device, dtype=dtype)
        latent = model.generate(
            prompt_prefix, assistant_prefix, states, metadata["bop_id"], metadata["eop_id"], args.max_new_tokens
        )
        latent_answer = tokenizer.decode(latent[0], skip_special_tokens=True).strip()
        controls = ""
        if args.include_text_control:
            plan = draft_ids(tokenizer, record["sender_draft"], device, metadata["max_plan_tokens"])
            text = generate_text_message(
                model, prompt_prefix, assistant_prefix, plan, metadata["bop_id"], metadata["eop_id"], args.max_new_tokens
            )
            controls = "\n\nText-message control:\n" + tokenizer.decode(text[0], skip_special_tokens=True).strip()
        blocks.append(
            f"{'=' * 80}\nSample {number}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n"
            f"Sender plan (debug only; not sent as text):\n{record['sender_draft']}\n\n"
            f"Interlat latent-only answer:\n{latent_answer}{controls}\n\nGold answer:\n{record['answer']}"
        )
        print(f"processed {number}/{len(records)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"saved output: {args.output}")


if __name__ == "__main__":
    main()
