"""python -m receiver_kv_workflow.run --transcript input/meeting1.txt"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

from .prompts import ROLE_QUERIES, prompt_parts
from .selection import select_spans, union_positions


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional human summary; English ROUGE-L only")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", type=Path, default=Path("output/receiver_kv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("test_result/receiver_kv"))
    parser.add_argument("--role-budget-ratio", type=float, default=0.15,
                        help="Source positions per role; union can reach four times this ratio")
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--query-batch-tokens", type=int, default=8,
                        help="Attention scoring batch only, NOT source truncation/chunk summaries")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Explicit final output reserve; report flags when exhausted")
    parser.add_argument("--variants", nargs="+", choices=["routed_kv", "full_kv", "selected_text"],
                        default=["routed_kv", "full_kv", "selected_text"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0 < args.role_budget_ratio <= 1:
        parser.error("--role-budget-ratio must be in (0, 1]")
    if min(args.span_tokens, args.query_batch_tokens, args.max_new_tokens) < 1:
        parser.error("Token budgets must be positive")
    return args


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = arguments()
    transcript = args.transcript.read_text(encoding="utf-8-sig").strip()
    if not transcript:
        raise ValueError("Transcript is empty")
    reference = args.reference.read_text(encoding="utf-8-sig") if args.reference else ""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from meeting_evaluation.metrics import rouge_l_f1
    from .cache import forward_tokens, generate, select_cache, size_bytes
    from .scoring import query_scores

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != "qwen3" or getattr(config, "use_sliding_window", False):
        raise ValueError("This experimental cache adapter supports dense full-attention Qwen3 only")
    if any(x != "full_attention" for x in (getattr(config, "layer_types", None) or [])):
        raise ValueError("Sliding/hybrid caches require a separate position-aware adapter")
    rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if rope_parameters.get("rope_type", rope_parameters.get("type", "default")) != "default":
        raise ValueError("Scaled/dynamic RoPE is not validated by this workflow")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prefix, suffix, protected = prompt_parts(tokenizer, transcript)
    queries = {role: tokenizer.encode(query, add_special_tokens=False)
               for role, query in ROLE_QUERIES.items()}
    limit = config.max_position_embeddings
    required = len(prefix) + max(len(suffix) + args.max_new_tokens,
                                 max(map(len, queries.values())))
    if required > limit:
        raise ValueError(f"The context requires {required} positions but the model supports {limit}. "
                         "No transcript was truncated. KV selection cannot undo prefill overflow.")
    budget = max(len(protected), math.ceil(len(prefix) * args.role_budget_ratio))
    log(f"source={len(prefix)} positions; context={limit}; role budget={budget}; loading model")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def tensor(ids):
        return torch.tensor([ids], device=device, dtype=torch.long)

    with torch.inference_mode():
        sync(torch, device)
        start = time.perf_counter()
        log("full source prefill started")
        output = forward_tokens(model, tensor(prefix), None, 0, logits_to_keep=1)
        source = output.past_key_values
        del output
        sync(torch, device)
        prefill_seconds = time.perf_counter() - start
        source_bytes = size_bytes(source)
        log(f"prefill completed in {prefill_seconds:.2f}s; {source_bytes / 2**20:.1f} MiB")

        messages, role_metrics = [], {}
        selection_seconds = 0.0
        if any(v != "full_kv" for v in args.variants):
            start = time.perf_counter()
            for role, query in queries.items():
                log(f"selector {role} started ({len(messages)}/4 complete)")
                scores = query_scores(model, source, tensor(query), query_batch=args.query_batch_tokens)
                selected = select_spans(scores, budget, span_size=args.span_tokens, protected=protected)
                messages.append(selected)
                role_metrics[role] = {"query_tokens": len(query), "selected_positions": len(selected)}
                log(f"selector {role} finished: {len(selected)} positions")
            sync(torch, device)
            selection_seconds = time.perf_counter() - start
        selected = union_positions(messages)
        report = {
            "method": "Receiver-conditioned KV routing, CacheBack-inspired, NOT paper reproduction",
            "model": args.model, "source_file": str(args.transcript), "seed": args.seed,
            "source_positions": len(prefix), "context_limit": limit,
            "source_cache_bytes": source_bytes, "prefill_seconds": prefill_seconds,
            "selection_seconds": selection_seconds, "roles": role_metrics,
            "settings": {"role_budget_ratio": args.role_budget_ratio, "span_tokens": args.span_tokens,
                         "query_batch_tokens": args.query_batch_tokens, "max_new_tokens": args.max_new_tokens},
            "variants": {},
            "notes": ["Positions are not bytes, billed tokens or end-to-end compute savings.",
                      "selected_text is selected-source text, NOT generated specialist handoffs.",
                      "ROUGE-L is lexical reference overlap, not factual accuracy; no reference means N/A.",
                      "Queries are selector instructions, not generated latent reasoning.",
                      "Source prefill processes every token; selection subsequently drops evidence positions.",
                      "The full immutable source cache remains resident for controls and selectors; smaller receiver KV is not a claim of lower process peak memory.",
                      "Logical payload bytes are accounting, not measured network traffic."],
        }
        answers = {}
        for variant in dict.fromkeys(args.variants):
            log(f"receiver {variant} started")
            sync(torch, device)
            start = time.perf_counter()
            ids = list(range(len(prefix))) if variant == "full_kv" else selected
            if variant == "selected_text":
                # Diagnostic: identical selected evidence but rebuilt causal context.
                output = forward_tokens(model, tensor([prefix[i] for i in ids]), None, 0, logits_to_keep=1)
                incoming = output.past_key_values
                del output
                absolute_start = len(ids)
                payload_bytes = len(ids) * 4  # explicit int32 token-ID wire representation
            else:
                incoming = select_cache(source, ids)
                absolute_start = len(prefix)
                payload_bytes = size_bytes(incoming)
            incoming_bytes = size_bytes(incoming)
            answer, generated, cap_hit = generate(
                model, tokenizer, incoming, tensor(suffix), absolute_start=absolute_start,
                max_new_tokens=args.max_new_tokens, context_limit=limit,
            )
            del incoming
            sync(torch, device)
            receiver_seconds = time.perf_counter() - start
            answers[variant] = answer
            report["variants"][variant] = {
                "incoming_positions": len(ids), "incoming_cache_bytes": incoming_bytes,
                "logical_payload_bytes": payload_bytes,
                "positions_vs_full_percent": 100 * (len(ids) - len(prefix)) / len(prefix),
                "cache_bytes_vs_full_percent": 100 * (incoming_bytes - source_bytes) / source_bytes,
                "receiver_seconds": receiver_seconds,
                "pipeline_seconds_excluding_model_load": prefill_seconds + receiver_seconds + (
                    0 if variant == "full_kv" else selection_seconds),
                "generated_tokens": generated, "output_cap_hit": cap_hit,
                "rouge_l_f1": rouge_l_f1(answer, reference) if reference else None,
            }
            log(f"receiver {variant} completed: {generated} output tokens; cap_hit={cap_hit}")
        # Token positions are audit metadata; do not write private source passages into metrics.
        report["selected_source_positions"] = selected
        report["peak_cuda_allocated_bytes_excluding_load"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir, metrics_dir = args.output_dir / stamp, args.metrics_dir / stamp
    output_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    for variant, answer in answers.items():
        (output_dir / f"{variant}.txt").write_text(answer + "\n", encoding="utf-8")
    (metrics_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = ["Receiver-conditioned KV routing (experimental)",
            "variant | ROUGE-L | input positions | KV MiB | positions vs full | total seconds | output tokens | cap-hit"]
    for variant, values in report["variants"].items():
        score = "N/A" if values["rouge_l_f1"] is None else f"{values['rouge_l_f1']:.4f}"
        rows.append(f"{variant} | {score} | {values['incoming_positions']} | "
                    f"{values['incoming_cache_bytes']/2**20:.1f} | {values['positions_vs_full_percent']:+.1f}% | "
                    f"{values['pipeline_seconds_excluding_model_load']:.2f} | "
                    f"{values['generated_tokens']} | {values['output_cap_hit']}")
    rows.extend(["", *report["notes"]])
    (metrics_dir / "metrics.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(f"Summaries: {output_dir}; metrics only: {metrics_dir}")


if __name__ == "__main__":
    main()
