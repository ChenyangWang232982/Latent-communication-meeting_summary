"""Run a same-model, training-free StateBridge QASPER experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bridge import StateBridge, random_embedding_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="QASPER JSONL file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-samples", type=int, default=8, help="0 means all records")
    parser.add_argument("--source-max-tokens", type=int, default=4096)
    parser.add_argument("--sender-max-new-tokens", type=int, default=256)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=64, help="StateBridge K")
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["statebridge", "text", "raw_hidden", "random_prefix", "no_comm"],
        choices=["statebridge", "text", "raw_hidden", "random_prefix", "no_comm"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def trim_context(tokenizer, context: str, max_tokens: int) -> str:
    ids = tokenizer(context, add_special_tokens=False).input_ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nAnswer:"


@torch.inference_mode()
def generate_from_ids(model, tokenizer, input_ids, attention_mask, max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output[:, input_ids.size(1) :]
    return output, generated_ids


@torch.inference_mode()
def generate_from_prefix(model, tokenizer, prompt_ids, prefix, max_new_tokens: int) -> torch.Tensor:
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(prompt_ids)
    inputs_embeds = torch.cat([prefix.unsqueeze(0), prompt_embeddings], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    output = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Transformers returns generated tokens alone for some inputs_embeds paths,
    # and full sequences for others. Keep only the generated suffix when present.
    return output[:, inputs_embeds.size(1) :] if output.size(1) > inputs_embeds.size(1) else output


def decode(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()


@torch.inference_mode()
def sender_message(model, tokenizer, record: dict[str, Any], args: argparse.Namespace):
    evidence = trim_context(tokenizer, record["context"], args.source_max_tokens)
    user = (
        f"Evidence:\n{evidence}\n\nQuestion: {record['question']}\n\n"
        "Write a short factual handoff for another agent. Include the exact answer and the "
        "evidence needed to justify it. Do not add unsupported details."
    )
    prompt = chat_prompt(tokenizer, "You extract factual answers from research papers.", user)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.source_max_tokens).to(args.device)
    full_ids, message_ids = generate_from_ids(
        model, tokenizer, encoded.input_ids, encoded.attention_mask, args.sender_max_new_tokens
    )
    if message_ids.size(1) < 2:
        raise RuntimeError("Sender produced fewer than two tokens; cannot construct a StateBridge prefix.")
    outputs = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), output_hidden_states=True)
    message_states = outputs.hidden_states[-1][:, -message_ids.size(1) :, :][0]
    count = min(args.prefix_tokens, message_ids.size(1))
    return decode(tokenizer, message_ids), message_states[-count:], message_ids[0, -count:]


def receiver_prompt(tokenizer, question: str, handoff: str | None = None) -> str:
    user = f"Question: {question}\n\nAnswer the question concisely and factually."
    if handoff is not None:
        user += f"\n\nSender handoff:\n{handoff}"
    return chat_prompt(tokenizer, "You answer research-paper questions using the provided handoff.", user)


@torch.inference_mode()
def answer_variant(model, tokenizer, bridge, record, plan, states, token_ids, variant, args):
    if variant == "text":
        prompt = receiver_prompt(tokenizer, record["question"], plan)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        _, answer_ids = generate_from_ids(
            model, tokenizer, encoded.input_ids, encoded.attention_mask, args.receiver_max_new_tokens
        )
        return decode(tokenizer, answer_ids)

    prompt = receiver_prompt(tokenizer, record["question"])
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    if variant == "no_comm":
        _, answer_ids = generate_from_ids(
            model, tokenizer, encoded.input_ids, encoded.attention_mask, args.receiver_max_new_tokens
        )
        return decode(tokenizer, answer_ids)
    if variant == "statebridge":
        prefix = bridge.align(states, token_ids)
    elif variant == "raw_hidden":
        prefix = states
    elif variant == "random_prefix":
        prefix = random_embedding_prefix(model.get_input_embeddings().weight, states.size(0))
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return decode(tokenizer, generate_from_prefix(model, tokenizer, encoded.input_ids, prefix, args.receiver_max_new_tokens))


def main() -> None:
    args = parse_args()
    if args.prefix_tokens < 2:
        raise ValueError("--prefix-tokens must be at least 2")
    torch.manual_seed(args.seed)
    records = load_records(args.data, args.num_samples)
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jsonl, args.output.open("w", encoding="utf-8") as report:
        report.write(f"StateBridge QASPER run: {datetime.now().isoformat(timespec='seconds')}\n")
        report.write(f"model={args.model}, K={args.prefix_tokens}, variants={','.join(args.variants)}\n\n")
        for index, record in enumerate(records, start=1):
            plan, states, token_ids = sender_message(model, tokenizer, record, args)
            answers = {
                variant: answer_variant(model, tokenizer, bridge, record, plan, states, token_ids, variant, args)
                for variant in args.variants
            }
            result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "sender_message": plan,
                "prefix_tokens": int(states.size(0)),
                "answers": answers,
            }
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\n")
            report.write(f"Question:\n{record['question']}\n\nSender message (debug only):\n{plan}\n\n")
            report.write(f"StateBridge prefix token count: {states.size(0)}\n\n")
            for variant, answer in answers.items():
                report.write(f"{variant} answer:\n{answer}\n\n")
            report.write(f"Gold answer:\n{record['answer']}\n\n")
            report.flush()
            print(f"completed {index}/{len(records)}: {record['id']}", flush=True)


if __name__ == "__main__":
    main()
