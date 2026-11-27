# Semantic-supplemented meeting summarization

The production CLI and meeting_stream use SemanticSummaryRunner. Each specialist
receives its primary selected-context window plus up to two related distant
selected blocks (V3), excluding neighbors within one block. Positions retain
source order. Final synthesis receives all selected context and ordered reports.

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --vector-routing \
  --variants chunk_kv
```

Post-selection supplementation is always active. --vector-routing controls
initial selection only; without it, selected blocks are encoded separately.
KV is the default communication; --variants chunk_kv chunk_text also compares
verbatim text using the same supplemented windows. Neither runs an additional
unaugmented baseline. Historical controls remain in experiment_1/2/3.

Defaults: role budget .25, speaker neighbors 1, local overlap .25, local sampling
.3/.9 with repetition penalty 1.1. Final synthesis uses sampling .3/.9 and thinking;
--no-summary-thinking disables final thinking. Local stages remain non-thinking.

Outputs: final summary, .specialists.json (windows, supplementation matches,
reports, stage costs and retrieval metadata), separate thinking and live partial
files. Metrics are in test_result/receiver_kv/<timestamp>. --routing-audit adds
source-coordinate evidence audits. Retrieval time is included in pipeline time;
encoder tokens are recorded separately from decoder token events.

The original ReceiverKVRunner engine provides shared loading, selection and cache
operations to production and experimental controls. It is not a retired directory.
Production code does not import experiment packages. Experimental compatibility
imports forward V3 retrieval to vector/supplement.py.

Dense full-attention standard-RoPE Qwen3 only. Source prefill happens once; full KV
remains resident. Each generation uses an independent writable cache copy.
No silent truncation or arbitrary output cap. This is hybrid source-KV/report-text
communication, not a complete CacheBack reproduction.
