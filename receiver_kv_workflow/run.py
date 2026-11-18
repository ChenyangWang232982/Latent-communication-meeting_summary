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
                        help="Write the final routed_kv summary to this file; specialist_text uses a sibling file")
    parser.add_argument("--output-dir", type=Path, default=Path("output/receiver_kv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("test_result/receiver_kv"))
    parser.add_argument("--role-budget-ratio", type=float, default=0.15,
                        help="Source positions per role; union and neighbor expansion can exceed this ratio")
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--speaker-neighbor-blocks", type=int, default=1,
                        help="For speaker selection, retain this many whole preceding and following speaker blocks")
    parser.add_argument("--routing-audit", action="store_true", help="Write separate source-coordinate audit files; never sent to the receiver")
    parser.add_argument("--review", action="store_true", help="Also save the routed_kv draft and review change log")
    parser.add_argument('--chunk-specialists', action='store_true', help='Independent specialists over selected KV windows')
    parser.add_argument('--summary-thinking', action='store_true', help='Enable thinking only for final chunk_summary')
    parser.add_argument('--specialist-parts', type=int, default=8, help='Target original transcript part size: full transcript tokens divided by this number')
    parser.add_argument('--specialist-overlap', type=float, default=0.25, help='Approximate overlap at complete speaker-block boundaries')
    parser.add_argument('--specialist-temperature', type=float, default=0.3)
    parser.add_argument('--specialist-top-p', type=float, default=0.9)
    parser.add_argument('--specialist-repetition-penalty', type=float, default=1.1)
    parser.add_argument("--query-batch-tokens", type=int, default=8,
                        help="Attention scoring batch only, NOT source truncation/chunk summaries")
    parser.add_argument("--variants", nargs="+", choices=["routed_kv", "specialist_text"],
                        default=["routed_kv"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7, help="Seed for source routing and summary generation (default: 7)")
    parser.add_argument("--review-seed", type=int, default=7, help="Seed reset before review generation (default: 7)")
    parser.add_argument("--review-temperature", type=float, default=0.0, help="Review sampling temperature; 0 uses greedy decoding (default: 0)")
    parser.add_argument("--review-top-p", type=float, default=0.9, help="Review nucleus sampling threshold (default: 0.9)")
    args = parser.parse_args()
    if args.chunk_specialists and args.variants != ['routed_kv']:
        parser.error('--chunk-specialists currently supports only --variants routed_kv')
    if args.summary_thinking and not args.chunk_specialists:
        parser.error('--summary-thinking requires --chunk-specialists')
    if args.review and 'routed_kv' not in args.variants:
        parser.error('--review requires --variants to include routed_kv')
    return args


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def main():
    args = arguments()
    transcript = args.transcript.read_text(encoding="utf-8-sig").strip()
    reference = args.reference.read_text(encoding="utf-8-sig") if args.reference else ""
    from meeting_evaluation.metrics import rouge_l_f1
    from vector.cli import config_kwargs
    # Preserve live window reports on interruption or context exhaustion.
    chunk_base = args.output.with_suffix('') if args.output else args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / 'routed_kv'
    if args.chunk_specialists:
        log(f"Live specialist output prefix: {chunk_base}")
    def live_chunks(event):
        if args.chunk_specialists and event.get('text') is not None:
            chunk_base.parent.mkdir(parents=True, exist_ok=True)
            stage = event['stage']
            chunk_base.with_name(chunk_base.name + f'.{stage}.partial.txt').write_text(event['text'], encoding='utf-8')
            chunk_base.with_name(chunk_base.name + f'.{stage}.status.json').write_text(
                json.dumps({k: event[k] for k in ('tokens', 'finished', 'stop_reason', 'repeated_tail') if k in event}, indent=2), encoding='utf-8')
    runner = ReceiverKVRunner(ReceiverKVConfig(
        model=args.model, device=args.device, role_budget_ratio=args.role_budget_ratio,
        span_tokens=args.span_tokens, query_batch_tokens=args.query_batch_tokens,
        seed=args.seed, review_seed=args.review_seed, review_temperature=args.review_temperature,
        review_top_p=args.review_top_p,
        chunk_specialists=args.chunk_specialists, specialist_parts=args.specialist_parts,
        summary_thinking=args.summary_thinking,
        specialist_overlap=args.specialist_overlap,
        specialist_temperature=args.specialist_temperature, specialist_top_p=args.specialist_top_p,
        specialist_repetition_penalty=args.specialist_repetition_penalty,
        speaker_neighbor_blocks=args.speaker_neighbor_blocks,
        **config_kwargs(args),
    ), progress=log)
    result = runner.run(args.transcript.stem, transcript, variants=tuple(args.variants),
                        routing_audit=args.routing_audit, review="routed_kv" in args.variants and not args.chunk_specialists,
                        on_generation=live_chunks if args.chunk_specialists else None)
    answers = result["answers"]
    review_report = ({k: v for k, v in result['review'].items() if k not in ('answer', 'changes')}
                     if result['review'] is not None else None)
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
                     "review": "routed_kv" in args.variants and not args.chunk_specialists,
                     "chunk_specialists": args.chunk_specialists, "save_review_details": args.review,
                     "specialist_parts": args.specialist_parts, "specialist_overlap": args.specialist_overlap,
                     "summary_thinking": args.summary_thinking,
                     "specialist_temperature": args.specialist_temperature, "specialist_top_p": args.specialist_top_p,
                     "specialist_repetition_penalty": args.specialist_repetition_penalty,
                     "review_seed": args.review_seed if "routed_kv" in args.variants else None,
                     "speaker_neighbor_blocks": args.speaker_neighbor_blocks},
        "baseline_selected_positions": result["baseline_selected_positions"],
        "variants": result["variants"],
        "review": review_report,
        "chunk_specialists": ({k: v for k, v in result['chunk_specialists'].items() if k not in ('reports', 'windows', 'thinking_outputs')}
                  if result['chunk_specialists'] else None),
        "notes": ["Positions are not bytes, billed tokens or end-to-end compute savings.",
                  "routed_kv and specialist_text use the identical selected source positions; only the specialist-to-receiver communication representation differs.",
                  "ROUGE-L is lexical reference overlap, not factual accuracy; no reference means N/A.",
                  "Queries are selector instructions, not generated latent reasoning.",
                  "Source prefill processes every token; selection subsequently drops evidence positions.",
                  "The full immutable source cache remains resident for controls and selectors; smaller receiver KV is not a claim of lower process peak memory.",
                  "Logical payload bytes are accounting, not measured network traffic."],
    }
    if args.chunk_specialists:
        report['method'] = 'Selected-KV window specialists with source-grounded synthesis'
        report['notes'][1] = 'Local reports are natural language; original selected source context is delivered as KV. This is a hybrid workflow.'
    for variant, answer in answers.items():
        if variant == 'routed_kv' and result['review'] and result['review']['status'] == 'completed':
            answer = result['review']['answer']
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
    if result['chunk_specialists']:
        chunk_base = args.output.with_suffix('') if args.output else output_dir/'routed_kv'
        reports_path = chunk_base.with_name(chunk_base.name + '.specialists.json')
        reports_path.write_text(json.dumps(result['chunk_specialists'], ensure_ascii=False, indent=2), encoding='utf-8')
        for stage, reasoning in result['chunk_specialists']['thinking_outputs'].items():
            chunk_base.with_name(chunk_base.name + f'.{stage}.thinking.txt').write_text(reasoning, encoding='utf-8')
        log(f'Window reports and KV positions: {reports_path}')
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
        review = result['review'] if variant == 'routed_kv' else None
        final_answer = review['answer'] if review and review['status'] == 'completed' else answer
        path.write_text(final_answer + "\n", encoding="utf-8")
        log(f"{'Reviewed summary' if review and review['status'] == 'completed' else 'Summary'}: {path}")
        if review and review['status'] != 'completed':
            log(f"Review skipped: {review['reason']} The output contains the draft.")
        if args.review and review and review['status'] == 'completed':
            draft_path = path.with_name(f"{path.stem}.draft{path.suffix or '.txt'}")
            changes_path = path.with_name(f"{path.stem}.reviewed.changes{path.suffix or '.txt'}")
            draft_path.write_text(answer + '\n', encoding='utf-8')
            changes_path.write_text(review['changes'], encoding='utf-8')
            log(f"Draft summary: {draft_path}; review change log: {changes_path}")
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
        rows.append(f"  answer tokens={counts['answer_tokens']} | terminal tokens={counts['terminal_tokens']}")
    rows.extend(["", *report["notes"]])
    if result['chunk_specialists']:
        rows.append('Independent window specialists; source prefill once, no extra review pass.')
        for stage, values in result['chunk_specialists']['stages'].items():
            rows.append(f"{stage}: prompt={values['prompt_tokens']} | output={values['output_tokens']} | "
                        f"source KV positions={values['source_kv_positions']} | seconds={values['seconds']:.2f}")
            if values['thinking_enabled']:
                rows.append('  generation counts: ' + json.dumps(values['generation_counts']))
        accounting = result['variants']['routed_kv']['token_accounting']
        rows.append(f"Logical decoder token events (all stages): {sum(accounting.values())}")
    if report['review'] is not None:
        review = report['review']
        rows.append(f"Review: status={review['status']}")
        if review['status'] == 'completed':
            rows.append(f"Review extra: prompt tokens={review['prompt_tokens']} | output tokens={review['output_tokens']} | "
                        f"seconds={review['seconds']:.2f} | source KV positions={review['source_kv_positions']} | "
                        f"cap-hit={review['output_cap_hit']} | draft cap-hit={review['draft_output_cap_hit']}")
            rows.append(f"Generation seeds: summary={args.seed} | review={review['seed']} | "
                        f"summary decoding=greedy | review decoding={'sampling' if review['temperature'] > 0 else 'greedy'} | "
                        f"review temperature={review['temperature']} | review top_p={review['top_p']}")
            rows.append(f"Review comparison: same text as draft={review['same_text_as_draft']}")
            rows.append(f"Organization separate reasons present: {review['review_reasons_present']}")
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
