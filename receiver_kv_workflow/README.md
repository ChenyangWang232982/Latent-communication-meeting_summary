# Global selected context + transcript-part specialists

The active workflow uses attention and optional semantic routing to select source
context, then specialists analyze independent chronological windows of only that
selected context. Final synthesis combines ordered local reports with all selected
evidence. The old direct-summary and single generated-handoff
branches and separate review agent have been removed.

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --vector-routing \
  --variants chunk_kv
```

Use `--variants chunk_kv chunk_text` to generate both controls, or only `chunk_text`.
Both use the same selected window coordinates, instructions and stage decoding
settings. The text path renders each local window and final selected evidence
as verbatim text rather than transferring KV. Reports remain natural language.

Defaults: source budget per role 0.25, speaker neighbors +/-1, part size targeting
automatic window count `max(1, floor(full_transcript_tokens/1024 + 0.5))`,
speaker-boundary preservation with 0.25 overlap, local sampling 0.3/0.9 and penalty1.1,
final sampling 0.3/0.9 with thinking on. `--summary-*` controls final synthesis;
`--no-summary-thinking` disables its reasoning. Use `--help` for all settings.

`--output` writes the primary final answer; the second variant uses a sibling
file. Per-variant `.specialists.json`, separate thinking text, live partial/status
files and `test_result/receiver_kv/<timestamp>/metrics.*` retain provenance/costs.
`--routing-audit` writes source-coordinate audits, not extra model instructions.

Dense full-attention standard-RoPE Qwen3 is supported. Source prefill happens once
and full KV remains resident. Selected rows preserve original absolute positions;
each stage uses an independent writable cache copy. No silent source truncation
or arbitrary output cap; context exhaustion is an error. This is hybrid source-KV
and report-text communication, not a reproduction claim for CacheBack.

See [specialist architecture](../chunk_specialist_workflow/README.md) and
[paired experiment](../experiment_1/README.md).
