"""Run a one-specialist-to-one-receiver StateBridge K ablation on MeetingBank."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .metrics import rouge_l_f1, score_results
from .records import load_meetingbank


DEFAULT_SPECIALIST_ROLE = "facts"
MEETING_ROLES = ("facts", "decisions", "actions", "discussion", "risks", "questions", "next_steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MeetingBank JSON or JSONL input.")
    parser.add_argument("--output-dir", type=Path, default=Path("meeting_evaluation/output"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--role", choices=MEETING_ROLES, default=DEFAULT_SPECIALIST_ROLE)
    parser.add_argument("--budgets", nargs="+", type=int, default=[48, 52, 56, 60, 64, 68, 72, 76])
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--source-max-tokens", type=int, default=8192)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--backfill-rounds", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=192)
    parser.add_argument("--device")
    return parser.parse_args()


def run_directory(output_dir: Path) -> Path:
    run_id = os.environ.get("MEETING_EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M")
    directory = output_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_budget(records_path: Path, output_path: Path, args: argparse.Namespace, budget: int) -> list[dict]:
    command = [
        sys.executable,
        "-m",
        "prophetkv_statebridge.run_prefix_cache",
        "--meeting-data",
        str(records_path),
        "--output",
        str(output_path),
        "--model",
        args.model,
        "--num-samples",
        "0",
        "--source-max-tokens",
        str(args.source_max_tokens),
        "--prefill-chunk-tokens",
        str(args.prefill_chunk_tokens),
        "--backfill-rounds",
        str(args.backfill_rounds),
        "--agent-max-new-tokens",
        str(args.agent_max_new_tokens),
        "--receiver-max-new-tokens",
        str(args.receiver_max_new_tokens),
        "--prefix-tokens-per-agent",
        str(budget),
        "--roles",
        args.role,
        "--variants",
        "statebridge",
        "text",
        "no_comm",
    ]
    if args.device:
        command.extend(["--device", args.device])
    subprocess.run(command, check=True)
    return [
        json.loads(line)
        for line in output_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_line_chart(path: Path, budgets: list[int], results_by_budget: dict[str, list[dict]]) -> None:
    """Write accuracy and savings curves against the text-token communication baseline."""
    colors = ("#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#65a30d", "#be185d")
    accuracy_series: dict[str, list[tuple[float, float]]] = {}
    savings_series: dict[str, list[tuple[float, float]]] = {}
    for budget in budgets:
        accuracy_points = []
        savings_points = []
        for result in results_by_budget[str(budget)]:
            text_tokens = float(result["communication"]["text_handoff_tokens"])
            rouge = rouge_l_f1(result["answers"]["statebridge"], result.get("gold_answer", ""))
            if rouge is not None:
                accuracy_points.append((text_tokens, rouge))
            savings = -float(result["communication"]["latent_minus_text_percent"])
            savings_points.append((text_tokens, savings))
        accuracy_series[f"K={budget}"] = sorted(accuracy_points)
        savings_series[f"K={budget}"] = sorted(savings_points)

    all_x = [x for series in savings_series.values() for x, _ in series]
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max:
        x_min, x_max = max(0.0, x_min - 1.0), x_max + 1.0
    all_savings = [value for series in savings_series.values() for _, value in series]
    saving_min, saving_max = min(all_savings), max(all_savings)
    padding = max(1.0, (saving_max - saving_min) * 0.1)
    saving_min, saving_max = saving_min - padding, saving_max + padding

    width, height = 1200, 900
    left, right, chart_width = 100, 70, 1030

    def panel_svg(y_top: int, title: str, series: dict[str, list[tuple[float, float]]], y_min: float, y_max: float) -> list[str]:
        panel_height = 250
        parts = [
            f'<text x="{left}" y="{y_top - 24}" class="panel-title">{html.escape(title)}</text>',
            f'<line x1="{left}" y1="{y_top}" x2="{left}" y2="{y_top + panel_height}" class="axis"/>',
            f'<line x1="{left}" y1="{y_top + panel_height}" x2="{left + chart_width}" y2="{y_top + panel_height}" class="axis"/>',
        ]
        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4
            y = y_top + panel_height - panel_height * tick / 4
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
        for tick in range(5):
            value = x_min + (x_max - x_min) * tick / 4
            x = left + chart_width * tick / 4
            parts.append(f'<line x1="{x:.1f}" y1="{y_top + panel_height}" x2="{x:.1f}" y2="{y_top + panel_height + 6}" class="axis"/>')
            parts.append(f'<text x="{x:.1f}" y="{y_top + panel_height + 27}" text-anchor="middle" class="tick">{value:.0f}</text>')
        for index, (label, points) in enumerate(series.items()):
            color = colors[index % len(colors)]
            rendered = [
                (
                    left + (x - x_min) * chart_width / (x_max - x_min),
                    y_top + panel_height - (y - y_min) * panel_height / max(1e-9, y_max - y_min),
                )
                for x, y in points
            ]
            if len(rendered) >= 2:
                point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in rendered)
                parts.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in rendered:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            legend_x = left + (index % 4) * 240
            legend_y = y_top - 54 + (index // 4) * 22
            parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{legend_x + 32}" y="{legend_y + 5}" class="legend">{html.escape(label)}</text>')
        return parts

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>.title{font:700 25px Arial,sans-serif;fill:#172033}.panel-title{font:700 18px Arial,sans-serif;fill:#172033}.tick{font:13px Arial,sans-serif;fill:#4b5563}.legend{font:14px Arial,sans-serif;fill:#172033}.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#dbe3ef;stroke-width:1}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="100" y="42" class="title">MeetingBank: One Specialist to One Receiver K Ablation</text>',
        '<text x="100" y="848" class="tick">Natural-language handoff token baseline per sample</text>',
        '<text x="100" y="874" class="tick">Each colored curve is one fixed K. Upper: StateBridge ROUGE-L. Lower: token saving versus text handoff.</text>',
    ]
    svg.extend(panel_svg(110, "Final-summary accuracy (ROUGE-L)", accuracy_series, 0.0, 1.0))
    svg.extend(panel_svg(550, "Communication saving versus text handoff (%)", savings_series, saving_min, saving_max))
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] < 2:
        raise ValueError("--budgets must contain K values of at least 2.")

    records = load_meetingbank(args.input)
    if args.num_samples and len(records) < args.num_samples:
        raise ValueError(f"Requested {args.num_samples} records but input contains only {len(records)}.")
    selected = records[: args.num_samples] if args.num_samples else records
    if not selected:
        raise ValueError("No MeetingBank records selected.")
    source_by_id = {record.id: record.context for record in selected}

    metrics_by_budget: dict[str, dict] = {}
    results_by_budget: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="meeting_single_agent_k_") as temp_dir:
        temp = Path(temp_dir)
        normalized_input = temp / "records.jsonl"
        with normalized_input.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

        for budget in budgets:
            print(f"=== MeetingBank one-agent K={budget} ===", flush=True)
            results = run_budget(normalized_input, temp / f"k{budget}.txt", args, budget)
            results_by_budget[str(budget)] = results
            metrics_by_budget[str(budget)] = score_results(results, source_by_id)

    payload = {
        "benchmark": "meetingbank_single_specialist_k_ablation",
        "samples": len(selected),
        "configuration": {
            "model": args.model,
            "single_specialist_role": args.role,
            "source_max_tokens": args.source_max_tokens,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "backfill_rounds": args.backfill_rounds,
            "agent_max_new_tokens": args.agent_max_new_tokens,
            "receiver_max_new_tokens": args.receiver_max_new_tokens,
            "budgets": budgets,
        },
        "metrics_by_budget": metrics_by_budget,
        "note": (
            "This is a controlled one-specialist-to-one-receiver experiment. The selected specialist is the only sender; "
            "K is its full StateBridge state budget. Text and no-communication are controls generated under the same setup."
        ),
    }

    directory = run_directory(args.output_dir)
    text_path = directory / "meetingbank_single_agent_k_ablation.txt"
    json_path = directory / "meetingbank_single_agent_k_ablation.json"
    chart_path = directory / "meetingbank_single_agent_k_ablation.svg"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_line_chart(chart_path, budgets, results_by_budget)

    lines = [
        "MeetingBank Single-Agent StateBridge K Ablation",
        f"Samples: {len(selected)}",
        f"Sender: {args.role} specialist",
        "Topology: one specialist -> one receiver",
        "",
        "K | StateBridge ROUGE-L | StateBridge source-support | text ROUGE-L | no_comm ROUGE-L | latent states | text handoff tokens | latent-text | saving vs text",
    ]
    for budget in budgets:
        metrics = metrics_by_budget[str(budget)]
        quality = metrics["quality"]
        efficiency = metrics["efficiency"]
        saving = efficiency["latent_minus_text_percent"]
        lines.append(
            f"{budget} | "
            f"{metric_text(quality['statebridge']['rouge_l_f1'])} | "
            f"{quality['statebridge']['supported_claim_rate']:.4f} | "
            f"{metric_text(quality['text']['rouge_l_f1'])} | "
            f"{metric_text(quality['no_comm']['rouge_l_f1'])} | "
            f"{efficiency['latent_state_tokens']} | "
            f"{efficiency['text_handoff_tokens']} | "
            f"{efficiency['latent_minus_text_tokens']:+d} | "
            f"{'N/A' if saving is None else f'{-saving:+.1f}%'}"
        )
    lines += [
        "",
        "Saving vs text is positive when StateBridge uses fewer token-equivalent communication units.",
        "ROUGE-L and source-support are evaluated on final summaries; source-support is a lexical proxy, not an LLM or human MESA score.",
        payload["note"],
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"K-ablation report: {text_path}")
    print(f"Machine-readable K-ablation metrics: {json_path}")
    print(f"Line chart: {chart_path}")


if __name__ == "__main__":
    main()
