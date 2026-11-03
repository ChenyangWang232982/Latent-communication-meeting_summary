"""python -m receiver_kv_workflow.run --transcript input/meeting1.txt"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from .engine import ReceiverKVConfig, ReceiverKVRunner


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    from vector.cli import add_arguments
    add_arguments(parser)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional human summary; English ROUGE-L only")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output", type=Path,
                        help="Write routed_kv directly to this file; specialist_text uses a sibling file")
    parser.add_argument("--output-dir", type=Path, default=Path("output/receiver_kv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("test_result/receiver_kv"))
    parser.add_argument("--role-budget-ratio", type=float, default=0.15,
                        help="Source positions per role; union and neighbor expansion can exceed this ratio")
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--speaker-neighbor-blocks", type=int, default=1,
                        help="For speaker selection, retain this many whole preceding and following speaker blocks")
    parser.add_argument("--routing-audit", action="store_true", help="Write separate source-coordinate audit files; never sent to the receiver")
    parser.add_argument("--review", action="store_true", help="Review routed_kv summary using its selected source KV")
    parser.add_argument("--review-debug", action="store_true", help="Compare the same editing prompt with and without selected source KV; requires --review")
    parser.add_argument("--query-batch-tokens", type=int, default=8,
                        help="Attention scoring batch only, NOT source truncation/chunk summaries")
    parser.add_argument("--variants", nargs="+", choices=["routed_kv", "specialist_text"],
                        default=["routed_kv", "specialist_text"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking for the final summary receiver")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for model inference (default: 7)")
    args = parser.parse_args()
    if not 0 < args.role_budget_ratio <= 1:
        parser.error("--role-budget-ratio must be in (0, 1]")
    if min(args.span_tokens, args.query_batch_tokens) < 1 or args.speaker_neighbor_blocks < 0:
        parser.error("Token budgets must be positive")
    if args.review and 'routed_kv' not in args.variants:
        parser.error('--review requires --variants to include routed_kv')
    if args.review_debug and not args.review:
        parser.error('--review-debug requires --review')
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
    from vector.cli import config_kwargs
    runner = ReceiverKVRunner(ReceiverKVConfig(
        model=args.model, device=args.device, role_budget_ratio=args.role_budget_ratio,
        span_tokens=args.span_tokens, query_batch_tokens=args.query_batch_tokens,
        seed=args.seed, enable_thinking=args.enable_thinking,
        speaker_neighbor_blocks=args.speaker_neighbor_blocks,
        **config_kwargs(args),
    ), progress=log)
    result = runner.run(args.transcript.stem, transcript, variants=tuple(args.variants),
                        routing_audit=args.routing_audit, review=args.review, review_debug=args.review_debug)
    answers = result["answers"]
    review_report = None
    if result['review'] is not None:
        review_report = {k: v for k, v in result['review'].items() if k not in ('answer', 'debug')}
        debug = result['review'].get('debug')
        if debug is not None:
            review_report['debug'] = {k: v for k, v in debug.items() if k != 'conditions'}
            if 'conditions' in debug:
                review_report['debug']['conditions'] = {
                    name: {k: v for k, v in values.items() if k != 'answer'}
                    for name, values in debug['conditions'].items()
                }
    report = {
        "method": "Receiver-conditioned KV routing, CacheBack-inspired, NOT paper reproduction",
        "model": args.model, "source_file": str(args.transcript), "seed": args.seed,
        "source_positions": result["source_positions"], "context_limit": result["context_limit"],
        "source_cache_bytes": result["source_cache_bytes"], "prefill_seconds": result["prefill_seconds"],
        "selection_seconds": result["selection_seconds"], "roles": result["roles"],
        "vector_routing": result["vector_routing"],
        "settings": {"role_budget_ratio": args.role_budget_ratio, "span_tokens": args.span_tokens,
                     "query_batch_tokens": args.query_batch_tokens,
                     "selection_strategy": "speaker",
                     "enable_thinking": args.enable_thinking,
                     "review": args.review,
                     "speaker_neighbor_blocks": args.speaker_neighbor_blocks},
        "baseline_selected_positions": result["baseline_selected_positions"],
        "variants": result["variants"],
        "review": review_report,
        "notes": ["Positions are not bytes, billed tokens or end-to-end compute savings.",
                  "routed_kv and specialist_text use the identical selected source positions; only the specialist-to-receiver communication representation differs.",
                  "ROUGE-L is lexical reference overlap, not factual accuracy; no reference means N/A.",
                  "Queries are selector instructions, not generated latent reasoning.",
                  "Source prefill processes every token; selection subsequently drops evidence positions.",
                  "The full immutable source cache remains resident for controls and selectors; smaller receiver KV is not a claim of lower process peak memory.",
                  "Logical payload bytes are accounting, not measured network traffic."],
    }
    for variant, answer in answers.items():
        report["variants"][variant]["rouge_l_f1"] = rouge_l_f1(answer, reference) if reference else None
    report["selected_source_positions"] = result["selected_source_positions"]
    report["seed_selected_positions"] = result["seed_selected_positions"]

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
    if result['review'] is not None and result['review']['status'] == 'completed':
        review_path = (args.output.with_name(f"{args.output.stem}.reviewed{args.output.suffix or '.txt'}")
                       if args.output else output_dir / 'routed_kv.reviewed.txt')
        review_path.write_text(result['review']['answer'] + '\n', encoding='utf-8')
        log(f"Reviewed summary: {review_path}")
        debug = result['review'].get('debug', {})
        for name, values in debug.get('conditions', {}).items():
            path = review_path.with_name(f"{review_path.stem}.debug_{name}{review_path.suffix}")
            path.write_text(values['answer'] + '\n', encoding='utf-8')
            log(f"Review diagnostic {name}: {path}")
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
        counts = values['generation_counts']
        rows.append(f"  thinking tokens={counts['thinking_tokens']} | answer tokens={counts['answer_tokens']} | "
                    f"delimiter tokens={counts['delimiter_tokens']} | terminal tokens={counts['terminal_tokens']} | "
                    f"thinking complete={counts['thinking_complete']}")
    rows.extend(["", *report["notes"]])
    if report['review'] is not None:
        review = report['review']
        rows.append(f"Review: status={review['status']}")
        if review['status'] == 'completed':
            rows.append(f"Review extra: prompt tokens={review['prompt_tokens']} | output tokens={review['output_tokens']} | "
                        f"seconds={review['seconds']:.2f} | source KV positions={review['source_kv_positions']} | "
                        f"cap-hit={review['output_cap_hit']} | draft cap-hit={review['draft_output_cap_hit']}")
            rows.append(f"Review comparison: same text={review['same_text_as_draft']} | "
                        f"same generated tokens={review['same_generated_tokens_as_draft']}")
            rows.append(f"Draft prompt token SHA-256: {review['draft_prompt_token_sha256']}")
            rows.append(f"Review prompt token SHA-256: {review['prompt_token_sha256']}")
            rows.append(f"Draft generated token SHA-256: {review['draft_generated_token_sha256']}")
            rows.append(f"Review generated token SHA-256: {review['generation_counts']['generated_token_sha256']}")
            draft_counts = report['variants']['routed_kv']['generation_counts']
            review_counts = review['generation_counts']
            rows.append(f"Draft actual forward input SHA-256: {draft_counts['forward_input_token_sha256']}")
            rows.append(f"Review actual forward input SHA-256: {review_counts['forward_input_token_sha256']}")
            rows.append(f"Draft first-step logits SHA-256: {draft_counts['first_step_logits_sha256']} | "
                        f"top token id: {draft_counts['first_step_top_token']}")
            rows.append(f"Review first-step logits SHA-256: {review_counts['first_step_logits_sha256']} | "
                        f"top token id: {review_counts['first_step_top_token']}")
            debug = review.get('debug')
            if debug and debug['status'] == 'completed':
                rows.append(f"Review diagnostic: identical suffix and absolute positions; selected KV present only in with_kv. "
                            f"Same text between conditions={debug['same_text_between_conditions']} | "
                            f"prompt tokens={debug['prompt_tokens']} | source KV positions={debug['source_kv_positions']}")
                for name, values in debug['conditions'].items():
                    rows.append(f"  {name}: same as draft={values['same_text_as_draft']} | "
                                f"tokens={values['output_tokens']} | cap-hit={values['output_cap_hit']} | "
                                f"seconds={values['seconds']:.2f} | "
                                f"prompt SHA-256={values['generation_counts']['forward_input_token_sha256']} | "
                                f"output SHA-256={values['generation_counts']['generated_token_sha256']}")
            elif debug:
                rows.append(f"Review diagnostic skipped: {debug['reason']}")
        else:
            rows.append(f"Review skipped: {review['reason']}")
    if result["vector_routing"] is not None:
        vector = result["vector_routing"]
        rows.append(f"Embedding input tokens (separate encoder): {vector['embedding_input_tokens']}; "
                    f"embedding/ranking seconds: {vector['seconds']:.3f}; "
                    "included in selection/pipeline time; model load excluded.")
    (metrics_dir / "metrics.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_location = args.output.parent if args.output else output_dir
    log(f"Summaries: {summary_location}; metrics only: {metrics_dir}")


if __name__ == "__main__":
    main()
