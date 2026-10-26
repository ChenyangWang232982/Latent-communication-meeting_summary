"""python -m receiver_kv_workflow.run --transcript input/meeting1.txt"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from .engine import ReceiverKVConfig, ReceiverKVRunner


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional human summary; English ROUGE-L only")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output", type=Path,
                        help="Write routed_kv directly to this file; controls use sibling files with the same stem")
    parser.add_argument("--output-dir", type=Path, default=Path("output/receiver_kv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("test_result/receiver_kv"))
    parser.add_argument("--role-budget-ratio", type=float, default=0.15,
                        help="Source positions per role; union can reach four times this ratio")
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--selection-strategy", choices=["fixed", "fixed_pooled", "sentence"], default="fixed")
    parser.add_argument("--routing-audit", action="store_true", help="Write separate source-coordinate audit files; never sent to the receiver")
    parser.add_argument("--query-batch-tokens", type=int, default=8,
                        help="Attention scoring batch only, NOT source truncation/chunk summaries")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Explicit final output reserve; report flags when exhausted")
    parser.add_argument("--timeline", action="store_true",
                        help="Also write a chronological event timeline with one approximate time per event")
    parser.add_argument("--variants", nargs="+", choices=["routed_kv", "full_kv", "selected_text", "specialist_text", "natural_text"],
                        default=["routed_kv", "full_kv", "selected_text", "specialist_text", "natural_text"])
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


def main():
    args = arguments()
    transcript = args.transcript.read_text(encoding="utf-8-sig").strip()
    if not transcript:
        raise ValueError("Transcript is empty")
    reference = args.reference.read_text(encoding="utf-8-sig") if args.reference else ""
    from meeting_evaluation.metrics import rouge_l_f1
    runner = ReceiverKVRunner(ReceiverKVConfig(
        model=args.model, device=args.device, role_budget_ratio=args.role_budget_ratio,
        span_tokens=args.span_tokens, query_batch_tokens=args.query_batch_tokens,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        selection_strategy=args.selection_strategy,
    ), progress=log)
    result = runner.run(args.transcript.stem, transcript, variants=tuple(args.variants), timeline=args.timeline,
                        routing_audit=args.routing_audit)
    answers = result["answers"]
    report = {
        "method": "Receiver-conditioned KV routing, CacheBack-inspired, NOT paper reproduction",
        "model": args.model, "source_file": str(args.transcript), "seed": args.seed,
        "source_positions": result["source_positions"], "context_limit": result["context_limit"],
        "source_cache_bytes": result["source_cache_bytes"], "prefill_seconds": result["prefill_seconds"],
        "selection_seconds": result["selection_seconds"], "roles": result["roles"],
        "settings": {"role_budget_ratio": args.role_budget_ratio, "span_tokens": args.span_tokens,
                     "query_batch_tokens": args.query_batch_tokens, "max_new_tokens": args.max_new_tokens,
                     "timeline": args.timeline, "selection_strategy": args.selection_strategy},
        "baseline_selected_positions": result["baseline_selected_positions"],
        "variants": result["variants"],
        "notes": ["Positions are not bytes, billed tokens or end-to-end compute savings.",
                  "selected_text is retrieved selected-source text; specialist_text is generated multi-role handoffs; natural_text is one sender-to-one receiver natural-language communication.",
                  "ROUGE-L is lexical reference overlap, not factual accuracy; no reference means N/A.",
                  "Queries are selector instructions, not generated latent reasoning.",
                  "Source prefill processes every token; selection subsequently drops evidence positions.",
                  "The full immutable source cache remains resident for controls and selectors; smaller receiver KV is not a claim of lower process peak memory.",
                  "Logical payload bytes are accounting, not measured network traffic."],
    }
    for variant, answer in answers.items():
        report["variants"][variant]["rouge_l_f1"] = rouge_l_f1(answer, reference) if reference else None
    report["selected_source_positions"] = result["selected_source_positions"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir, metrics_dir = args.output_dir / stamp, args.metrics_dir / stamp
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    primary_variant = "routed_kv" if "routed_kv" in answers else next(iter(answers))
    if args.routing_audit:
        from .routing_audit import render_audit
        audit_base = args.output.with_suffix("") if args.output else output_dir / "routing"
        audit_text = audit_base.with_name(audit_base.name + ".audit.txt")
        audit_json = audit_base.with_name(audit_base.name + ".audit.json")
        audit_text.write_text(render_audit(result["routing_audit"]), encoding="utf-8")
        audit_json.write_text(json.dumps(result["routing_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Routing audit: {audit_text}")
    for variant, answer in answers.items():
        if args.output:
            path = args.output if variant == primary_variant else args.output.with_name(
                f"{args.output.stem}.{variant}{args.output.suffix or '.txt'}"
            )
        else:
            path = output_dir / f"{variant}.txt"
        path.write_text(answer + "\n", encoding="utf-8")
    if result["timeline"] is not None:
        timeline_path = args.output.with_name(f"{args.output.stem}.timeline{args.output.suffix or '.txt'}") if args.output else output_dir / "timeline.txt"
        timeline_path.write_text(result["timeline"] + "\n", encoding="utf-8")
    (metrics_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = ["Receiver-conditioned KV routing (experimental)",
            "variant | ROUGE-L | input positions | KV MiB | positions vs full | total seconds | output tokens | cap-hit"]
    for variant, values in report["variants"].items():
        score = "N/A" if values["rouge_l_f1"] is None else f"{values['rouge_l_f1']:.4f}"
        position_delta = values["positions_vs_full_percent"]
        position_delta_text = "N/A" if position_delta is None else f"{position_delta:+.1f}%"
        rows.append(f"{variant} | {score} | {values['incoming_positions']} | "
                    f"{values['incoming_cache_bytes']/2**20:.1f} | {position_delta_text} | "
                    f"{values['pipeline_seconds_excluding_model_load']:.2f} | "
                    f"{values['generated_tokens']} | {values['output_cap_hit']}")
    rows.extend(["", *report["notes"]])
    (metrics_dir / "metrics.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_location = args.output.parent if args.output else output_dir
    log(f"Summaries: {summary_location}; metrics only: {metrics_dir}")


if __name__ == "__main__":
    main()
