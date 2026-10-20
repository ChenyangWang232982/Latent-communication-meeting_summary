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
  --output output/qasper_smoke.txt \
  --num-samples 1 \
  --model Qwen/Qwen3-4B \
  --chunk-tokens 512 \
  --chunk-overlap-tokens 64 \
  --top-chunks 6 \
  --prefix-tokens-per-agent 16
```

For research-QA input, the default four specialists are `facts`, `decisions`,
`actions`, and `risks`. For `--transcript` meeting mode, the runner defaults to
the following non-overlapping four-category schema:

- `decisions`: explicitly accepted or approved outcomes only.
- `actions`: explicit assignments, commitments, owners, or deadlines.
- `open_issues`: unresolved questions, risks, blockers, disagreements, and
  uncertainty.
- `context`: important background, proposals, and discussion that does not
  belong to another category.

This separation prevents common meeting-minute errors such as treating "we
need to discuss finance" as a decision, and prevents the same item from being
repeated as a fact, question, risk, and next step.
Each uses the question to select its own four excerpts. Their last 16 states
are concatenated, so the default StateBridge prefix has at most 64 states.

For a cheaper QASPER factual-answer check, use only the direct-answer
specialist:

```bash
python -m prophetkv_statebridge.run \
  --data data/qasper_latent/validation.jsonl \
  --output output/qasper_facts_only.txt \
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
  --output output/qasper_prefix_cache_smoke.txt \
  --num-samples 1 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 4096 \
  --roles facts \
  --prefix-tokens-per-agent 256
```

The report includes the shared-prefix token count, one-time prefill time, and
KV-cache size. It is an **exact prefix-reuse** baseline: it tokenizes every
normal chat prompt first, finds their exact common prefix, and caches only that
shared prefix. It is not ProphetKV's selective recomputation, and it does not
attempt unsafe concatenation of independently created caches.

Every `run_prefix_cache.py` report also records a specialist-to-receiver
communication comparison. `text_handoff_tokens` is the tokenizer count of the
full natural-language handoff passed to the text receiver; `latent_state_tokens`
is the number of StateBridge states passed to the latent receiver. The reported
`latent-text` delta uses `latent - text`: a negative value means latent saved
token-equivalent communication units, while a positive value means it used more.
Add `--communication-metrics-only` to skip final-answer generation and omit
all handoff text from the report; this is the preferred mode for a pure
communication-cost sweep.

For a dedicated communication-only test entry point, use
`python -m prophetkv_statebridge.communication_benchmark` with the same
arguments as `run_prefix_cache.py`. It always suppresses handoff, review, and
answer content in its report while measuring the generated specialist-to-receiver
communication.

For a transcript longer than one model context, use document blocks instead of
raising `--source-max-tokens`. Each document block gets its own exact shared
Prefix KV cache. For a meeting split into two or more blocks, the runner first
uses StateBridge to produce one structured block summary per block, then uses a
second StateBridge handoff to pass those chronological block summaries to the
final merger. Independent KV caches are never concatenated. For example:

```bash
python -m prophetkv_statebridge.run_prefix_cache \
  --transcript input/my_long_meeting.txt \
  --output output/my_long_meeting.txt \
  --document-chunk-overlap-tokens 512 \
  --prefix-tokens-per-agent 64 \
  --block-summary-prefix-tokens 64 \
  --variants statebridge
```

The meeting runner reads the loaded model's context limit and automatically
uses 75% of it as the document-block budget. The `text` control passes the
same block summaries to the merger as natural language. The report counts both
communication hops: specialist-to-block and block-to-final-merger.

For an alignment ablation, keep the same transcript, model, seed, K, and
evidence policy, changing only `--alignment-scope joint` (the original
baseline) to `--alignment-scope per-message`. The latter aligns each
specialist's states separately before concatenation and may take longer.
The report lists each specialist's handoff length, selected states, and
truncated tokens so omissions can be checked against the K budget.

In meeting mode with `--evidence-policy off`, specialists mark the status of
each handoff item. The receiver reports conflicting statuses as unresolved
instead of silently omitting them. Each specialist still sends its final K
states; K is not shared across specialists.

## Fact-Centric Evidence Grounding

With `--evidence-policy strict`, every meeting specialist must emit a small JSON claim list with a
literal `evidence_quote` from its source block. The runner rejects claims whose
quotation is absent from the block or whose decision/action evidence lacks the
required commitment cue. The same model then verifies whether each remaining
quote explicitly supports its claim. Only `supported` claims are re-encoded as
the StateBridge handoff; unsupported and uncertain claims do not reach the
block receiver or final merger. Use `--evidence-policy off` for the raw-handoff
ablation; that setting improves recall but is not evidence-verified.

For a real meeting, use the default four meeting roles. Generation continues until
the model emits an end token or reaches the remaining context capacity. The
final receiver outputs the same four stable sections.
Use `--debug-progress` to see each document block's prefill and specialist
completion counts, followed by block-summary and final-answer stages. Progress
lines are printed immediately; the report still contains the finished result.

## Meeting transcript run

The same runner accepts one UTF-8 transcript directly. It produces the four
meeting-summary sections above for each communication condition.

For multiple meeting records in JSONL (each containing `id`, `context`,
`question`, and optional `answer`), use `--meeting-data` instead of `--data`.
This preserves reference answers while enabling the meeting roles and receiver:

```bash
python -m prophetkv_statebridge.run_prefix_cache \
  --meeting-data input/meetingbank_test.jsonl \
  --output output/meetingbank.txt
```

```bash
python -m prophetkv_statebridge.run_prefix_cache \
  --transcript input/my_meeting.txt \
  --output output/my_meeting_statebridge.txt \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 8192 \
  --prefix-tokens-per-agent 128
```

For a single block, the transcript is prefetched once and all seven meeting
roles reuse its exact common chat-prefix KV cache. For multiple document
blocks, this occurs once per block; each block then produces a StateBridge
summary before the final chronological merge. The `text` condition is the
natural-language handoff control, `statebridge` passes only aligned continuous
states, and `no_comm` shows what the aggregator produces without specialist
evidence.
