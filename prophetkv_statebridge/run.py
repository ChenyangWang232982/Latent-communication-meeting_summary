"""Run a query-aware long-context selection plus StateBridge QASPER baseline.

This is a practical integration scaffold, not a claim of paper-exact
ProphetKV KV-cache fusion.  It measures the useful first question: can a
query-conditioned subset of a long source preserve the facts that specialist
agents need before we invest in a custom cache-reuse serving backend?
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from statebridge_qasper.bridge import StateBridge
from statebridge_qasper.run import (
    chat_prompt,
    decode,
    generate_from_ids,
    generate_from_prefix,
    load_records,
    remove_thinking_tokens,
)

from .selection import Chunk, chunk_document, select_chunks


ROLE_INSTRUCTIONS = {
    "facts": "Extract the facts that directly answer the question.",
    "decisions": "Extract decisions, commitments, and chosen approaches relevant to the question.",
    "actions": "Extract actions, owners, deadlines, and follow-ups relevant to the question.",
    "risks": "Extract unresolved issues, caveats, and risks relevant to the question.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL with id, context, question, and answer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=1, help="0 means all records")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=64)
    parser.add_argument("--top-chunks", type=int, default=6)
    parser.add_argument("--roles", nargs="+", choices=sorted(ROLE_INSTRUCTIONS), default=["facts", "decisions", "actions", "risks"])
    parser.add_argument("--agent-max-new-tokens", type=int, default=160)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=16)
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--strict-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each specialist handoff to tie its conclusion to selected chunk IDs.",
    )
    parser.add_argument("--variants", nargs="+", choices=["statebridge", "text", "no_comm"], default=["statebridge", "text", "no_comm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def evidence_text(selected: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[chunk {chunk.chunk_id}, score={score:.3f}]\n{chunk.text}" for chunk, score in selected)


@torch.inference_mode()
def specialist_message(model, tokenizer, record: dict[str, Any], role: str, selected, args: argparse.Namespace):
    evidence_instruction = (
        "First give one or more short exact supporting phrases with their [chunk N] IDs, then give "
        "the conclusion. Do not claim that evidence is absent unless none of the selected excerpts states it."
        if args.strict_evidence
        else "Write the conclusion directly."
    )
    user = (
        f"Question: {record['question']}\n\n"
        f"Selected source excerpts:\n{evidence_text(selected)}\n\n"
        f"{ROLE_INSTRUCTIONS[role]} Write a concise factual handoff for a summary agent. "
        f"Only use the excerpts. {evidence_instruction} State missing information explicitly rather than guessing."
    )
    prompt = chat_prompt(
        tokenizer,
        "You are a careful long-meeting evidence extraction specialist.",
        user,
        args.enable_thinking,
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
    full_ids, message_ids = generate_from_ids(
        model, tokenizer, encoded.input_ids, encoded.attention_mask, args.agent_max_new_tokens
    )
    if message_ids.size(1) < 2:
        raise RuntimeError(f"{role} specialist produced fewer than two tokens")
    outputs = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), output_hidden_states=True)
    states = outputs.hidden_states[-1][:, -message_ids.size(1) :, :][0]
    message_ids, states = remove_thinking_tokens(tokenizer, message_ids, states)
    count = min(args.prefix_tokens_per_agent, message_ids.size(1))
    return decode(tokenizer, message_ids), states[-count:], message_ids[0, -count:]


def receiver_prompt(tokenizer, question: str, enable_thinking: bool, handoffs: str | None = None) -> str:
    user = f"Question: {question}\n\nGive a concise factual answer."
    if handoffs is not None:
        user += f"\n\nSpecialist handoffs:\n{handoffs}"
    return chat_prompt(
        tokenizer,
        "You aggregate specialist findings into an evidence-grounded answer. Do not invent facts.",
        user,
        enable_thinking,
    )


@torch.inference_mode()
def answer_variant(model, tokenizer, bridge, record, reports, states, token_ids, variant, args):
    if variant == "text":
        prompt = receiver_prompt(tokenizer, record["question"], args.enable_thinking, reports)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        _, answer_ids = generate_from_ids(model, tokenizer, encoded.input_ids, encoded.attention_mask, args.receiver_max_new_tokens)
        return decode(tokenizer, answer_ids)

    prompt = receiver_prompt(tokenizer, record["question"], args.enable_thinking)
    encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
    if variant == "no_comm":
        _, answer_ids = generate_from_ids(model, tokenizer, encoded.input_ids, encoded.attention_mask, args.receiver_max_new_tokens)
        return decode(tokenizer, answer_ids)
    if variant != "statebridge":
        raise ValueError(f"Unsupported variant: {variant}")
    prefix = bridge.align(states, token_ids)
    return decode(tokenizer, generate_from_prefix(model, tokenizer, encoded.input_ids, prefix, args.receiver_max_new_tokens))


def main() -> None:
    args = parse_args()
    if args.prefix_tokens_per_agent < 2:
        raise ValueError("--prefix-tokens-per-agent must be at least 2")
    torch.manual_seed(args.seed)
    records = load_records(args.data, args.num_samples)
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    bridge = StateBridge(model.get_input_embeddings().weight, regularization=args.regularization, snap_ratio=args.snap_ratio, vocab_chunk_size=args.vocab_chunk_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as jsonl, args.output.open("w", encoding="utf-8") as report:
        report.write(f"Query-aware StateBridge run: {datetime.now().isoformat(timespec='seconds')}\n")
        report.write(f"model={args.model}, chunk_tokens={args.chunk_tokens}, top_chunks={args.top_chunks}, roles={','.join(args.roles)}\n")
        report.write("Note: query-aware selection baseline; not paper-exact ProphetKV KV fusion.\n\n")
        for index, record in enumerate(records, start=1):
            chunks = chunk_document(tokenizer, record["context"], chunk_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap_tokens)
            specialists = []
            all_states, all_ids = [], []
            for role in args.roles:
                selection_query = record["question"] if role == "facts" else f"{record['question']} {ROLE_INSTRUCTIONS[role]}"
                selected = select_chunks(chunks, selection_query, args.top_chunks)
                message, states, token_ids = specialist_message(model, tokenizer, record, role, selected, args)
                specialists.append({"role": role, "message": message, "selected": [{"chunk_id": chunk.chunk_id, "score": score} for chunk, score in selected]})
                all_states.append(states)
                all_ids.append(token_ids)
            reports = "\n\n".join(f"[{item['role']}]\n{item['message']}" for item in specialists)
            states, token_ids = torch.cat(all_states), torch.cat(all_ids)
            answers = {variant: answer_variant(model, tokenizer, bridge, record, reports, states, token_ids, variant, args) for variant in args.variants}
            result = {"id": record["id"], "question": record["question"], "gold_answer": record.get("answer", ""), "chunks": len(chunks), "specialists": specialists, "prefix_tokens": int(states.size(0)), "answers": answers}
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            report.write("=" * 80 + f"\nSample {index}\nId: {record['id']}\n\nQuestion:\n{record['question']}\n\n")
            report.write(f"Source chunks: {len(chunks)}; StateBridge prefix tokens: {states.size(0)}\n\n")
            for item in specialists:
                selected = ", ".join(f"{entry['chunk_id']} ({entry['score']:.3f})" for entry in item["selected"])
                report.write(f"{item['role']} selected chunks: {selected}\n{item['role']} handoff (debug only):\n{item['message']}\n\n")
            for variant, answer in answers.items():
                report.write(f"{variant} answer:\n{answer}\n\n")
            report.write(f"Gold answer:\n{record.get('answer', '')}\n\n")
            report.flush()
            print(f"completed {index}/{len(records)}: {record['id']}", flush=True)


if __name__ == "__main__":
    main()
