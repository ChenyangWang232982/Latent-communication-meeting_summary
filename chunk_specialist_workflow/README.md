# Semantic-supplemented window specialists

This package contains shared chronological windowing, stage generation and
thinking extraction. It is used by production and experiment controls.

Production flow:

```text
transcript -> one full source prefill -> attention/optional vector selection
 -> chronological selected-context windows -> V3 distant evidence supplementation
 -> independent specialists -> ordered text reports
 -> final synthesis with global selected context and reports
```

Window count: max(1, floor(raw_transcript_tokens/1024 + .5)). Complete consecutive
speaker blocks are balanced across windows, with .25 backward overlap by default.
Whole turns remain intact. Window count cannot exceed available blocks.

SemanticSummaryRunner prepares augmented windows using vector/supplement.py:
up to two related already-selected distant blocks per window, excluding adjacent
blocks within radius 1. Original positions and order are preserved. Instructions
identify primary and supplementary block IDs. Similarity and recency alone do not
establish agreement or a revised decision. Dropped global evidence is not restored.

Each local stage receives only its augmented window. Final synthesis receives
all global selected evidence and ordered local reports. Local stages do not think;
final thinking is enabled by default. No separate review agent is used.

KV and text modes use the same evidence coordinates and stage settings. The KV
path retains full-source contextual and positional history; re-encoded text does
not. This is a workflow comparison rather than a pure representation intervention.
The full immutable source cache stays resident; lower peak memory is not promised.

Production entry: python -m receiver_kv_workflow.run. Details and output formats:
[receiver workflow](../receiver_kv_workflow/README.md).
Unaugmented windows remain an explicit experimental control only.

```bash
python -m unittest discover -s chunk_specialist_workflow/tests
```
