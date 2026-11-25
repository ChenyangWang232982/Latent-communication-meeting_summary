#!/usr/bin/env bash
# Run from the repository root with latent-agents activated.
set -euo pipefail

SOURCE=${1:-input/experiment_1_unfiltered_cap100/ami.jsonl}
DATASET=${2:-input/ami_endpoint_challenge_20_v1}
MODEL=${MODEL:-Qwen/Qwen3-8B}
ROOT="test_result/experiment_3_challenge20/$(date +%Y%m%d_%H%M%S)_$$"

if [[ ! -f "$DATASET/manifest.json" ]]; then
  python -m experiment_3.prepare_challenges \
    --input "$SOURCE" --output-dir "$DATASET" --skip 6
fi

COMMON=(--model "$MODEL" --device cuda --seed 12 --summary-seed 42
        --summary-temperature 0.3 --summary-top-p 0.9 --summary-thinking)

python -m experiment_3 \
  --input "$DATASET/meetings.jsonl" --benchmark ami \
  --output-dir "$ROOT/workflow" --num-samples 0 \
  --conditions baseline semantic \
  --role-budget-ratio 0.25 --speaker-neighbor-blocks 1 --vector-routing \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --supplement-blocks 2 --exclude-neighbor-blocks 1 "${COMMON[@]}"

# This existing endpoint runner accepts transcripts without reference summaries.
# experiment_2's ordinary benchmark entry point requires references instead.
python -m experiment_2.challenge_samples.run \
  --dataset "$DATASET" --condition direct_text \
  --output-dir "$ROOT/direct" "${COMMON[@]}"

WORKFLOW_RUN=$(find "$ROOT/workflow" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
DIRECT_RUN=$(find "$ROOT/direct" -mindepth 2 -maxdepth 2 -type d -name direct_text | sort | tail -n 1)

python -m experiment_3.challenge_report \
  --dataset "$DATASET" --workflow-run "$WORKFLOW_RUN" \
  --direct-run "$DIRECT_RUN" --output-dir "$ROOT/report"

cat "$ROOT/report/coverage.txt"
printf '\nResults: %s\nReview sheet: %s\n' "$ROOT" "$ROOT/report/target_review.csv"
