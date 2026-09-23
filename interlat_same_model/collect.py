"""Collect Sender last-layer generation states from long-context JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_last_generation_states(hidden_states):
    states = [layers[-1][:, -1:, :] for layers in hidden_states if layers]
    if not states:
        raise RuntimeError("Sender generated no hidden states")
    return torch.cat(states, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="JSONL with context, question, answer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--source-max-tokens", type=int, default=4096)
    parser.add_argument("--sender-max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sender = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    records = []
    with args.data.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            sender_prompt = (
                "Read the context and create a concise factual reasoning plan for answering "
                "the question. Preserve useful evidence but do not write the final answer.\n\n"
                f"Context:\n{row['context']}\n\nQuestion: {row['question']}"
            )
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": sender_prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=args.source_max_tokens).to(device)
            prompt_length = inputs.input_ids.size(1)
            with torch.inference_mode():
                output = sender.generate(
                    **inputs, max_new_tokens=args.sender_max_new_tokens, do_sample=False, num_beams=1,
                    return_dict_in_generate=True, output_hidden_states=True, pad_token_id=tokenizer.pad_token_id,
                )
            records.append({
                "id": row.get("id", str(len(records))),
                "question": row["question"],
                "answer": row["answer"],
                "sender_draft": tokenizer.decode(output.sequences[0, prompt_length:], skip_special_tokens=True).strip(),
                "sender_states": collect_last_generation_states(output.hidden_states).squeeze(0).cpu().to(torch.float16),
            })
            print(f"collected {len(records)}", flush=True)
            if args.limit and len(records) >= args.limit:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, args.output)
    print(f"saved {len(records)} records: {args.output}")


if __name__ == "__main__":
    main()
