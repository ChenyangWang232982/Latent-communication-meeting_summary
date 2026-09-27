# Query-Aware StateBridge Baseline

This module deploys the practical first stage of a `ProphetKV + StateBridge`
meeting workflow:

```text
long transcript -> query-aware chunk selection -> specialist extraction
                -> StateBridge continuous handoff -> final aggregation
```

It is deliberately labelled **ProphetKV-inspired**, not an exact ProphetKV
implementation. ProphetKV's contribution is query-driven *KV-cache reuse and
selective recomputation* in a modified serving path. The public project does
not provide that cache-fusion backend here, and Hugging Face `past_key_values`
cannot be safely concatenated across independently prefixed chunks. This
module therefore verifies the task-level premise first: whether query-aware
selection preserves the meeting facts each specialist needs.

`statebridge` is still a genuine same-model, training-free StateBridge handoff:
only final hidden states move from specialist to aggregator; no specialist text
is given to that variant. `text` and `no_comm` remain mandatory controls.

## QASPER smoke run

```bash
python -m prophetkv_statebridge.run \
  --data data/qasper_latent/validation.jsonl \
  --output prophetkv_statebridge/output/qasper_smoke.txt \
  --num-samples 1 \
  --model Qwen/Qwen3-4B \
  --chunk-tokens 512 \
  --chunk-overlap-tokens 64 \
  --top-chunks 6 \
  --prefix-tokens-per-agent 16
```

The default four specialists are `facts`, `decisions`, `actions`, and `risks`.
Each uses the question to select its own four excerpts. Their last 16 states
are concatenated, so the default StateBridge prefix has at most 64 states.

For a cheaper QASPER factual-answer check, use only the direct-answer
specialist:

```bash
python -m prophetkv_statebridge.run \
  --data data/qasper_latent/validation.jsonl \
  --output prophetkv_statebridge/output/qasper_facts_only.txt \
  --num-samples 8 \
  --roles facts \
  --top-chunks 6 \
  --prefix-tokens-per-agent 160
```

## What to compare

- `statebridge`: selected specialist information as aligned hidden states.
- `text`: the same specialist information as natural-language text.
- `no_comm`: question only.

The report records selected chunk IDs and scores, so factual misses can be
separated into retrieval failures and StateBridge-transfer failures. Only after
the selected-context pipeline has good evidence coverage is it worthwhile to
add a true ProphetKV-compatible serving backend for KV reuse and selective
recomputation.

By default, each specialist must cite short source phrases with `[chunk N]`
identifiers before reaching a conclusion (`--strict-evidence`). This makes the
handoff more auditable and reduces unsupported claims. Use
`--no-strict-evidence` only for an ablation.

## Exact cross-agent Prefix KV cache

`run_prefix_cache.py` adds a separate, real KV-cache baseline for repeated
analysis of the same source. It pre-fills one complete transcript prefix once,
then each specialist continues from a clone of that immutable `past_key_values`
cache with only its role-specific instruction appended.

```bash
python -m prophetkv_statebridge.run_prefix_cache \
  --data data/qasper_latent/validation.jsonl \
  --output prophetkv_statebridge/output/qasper_prefix_cache_smoke.txt \
  --num-samples 1 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 4096 \
  --roles facts \
  --agent-max-new-tokens 256 \
  --prefix-tokens-per-agent 256
```

The report includes the shared-prefix token count, one-time prefill time, and
KV-cache size. It is an **exact prefix-reuse** baseline: it tokenizes every
normal chat prompt first, finds their exact common prefix, and caches only that
shared prefix. It is not ProphetKV's selective recomputation, and it does not
attempt unsafe concatenation of independently created caches.

To construct the same shared cache as chronological updates (`KV1 -> KV1-2 ->
KV1-2-3`), add `--prefill-chunk-tokens 1024`. Each new 1024-token block is
forwarded with the previous block's `past_key_values`; separate chunk caches
are never manually concatenated.

`--backfill-rounds N` adds exactly `N` deterministic review passes for every
role. Each pass appends a query-ranked local transcript packet to a clone of
the shared meeting cache and asks the role to revise its previous handoff.
The number of passes is a command-line experiment parameter, not an Agent
decision. Use `--backfill-chunk-tokens` and `--backfill-neighbor-chunks` to
control the local reread evidence.

For a real meeting, use all four roles. Keep each role's handoff concise (for
example `--agent-max-new-tokens 96 --prefix-tokens-per-agent 96`) so that the
StateBridge aggregator receives the useful facts rather than a long repeated
transcript.

## Meeting transcript run

The same runner accepts one UTF-8 transcript directly. It produces a meeting
summary with `Decisions`, `Action items`, and `Risks and open questions` for
each communication condition.

```bash
python -m prophetkv_statebridge.run_prefix_cache \
  --transcript meeting_latent_workflow/input/example.txt \
  --output prophetkv_statebridge/output/example_prefix_cache_summary.txt \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 8192 \
  --prefill-chunk-tokens 1024 \
  --backfill-rounds 2 \
  --backfill-chunk-tokens 256 \
  --backfill-neighbor-chunks 1 \
  --roles facts decisions actions risks \
  --agent-max-new-tokens 128 \
  --prefix-tokens-per-agent 128
```

The long transcript is prefetched once; all four roles reuse its exact common
chat-prefix KV cache. The `text` condition is the natural-language handoff
control, `statebridge` passes only aligned continuous states, and `no_comm`
shows what the aggregator produces without specialist evidence.
