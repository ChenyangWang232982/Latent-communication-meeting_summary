# ???? / On hold

????????????????????????????????????
The experimental module is disconnected from the main workflow; the commands below are historical.

# Two-stage decision evolution

现在使用两阶段决策分析：先通过已选 KV 和摘要生成决策清单，再用清单向全文
发言块检索关联证据，合并去重后一次性生成所有决策的变化。决策分析固定增加
两次大模型生成，不再对每个候选主题分别生成。摘要与时间线流程不变。

清单是自然语言中间态；原文证据仍通过 KV 传递。保留原来的已选位置并补入
检索到的块，因此这一步可能增加 KV 覆盖范围，不能承诺进一步压缩。
JSON 审计中可检查清单、匹配分数及新增位置；实际质量和速度仍需实测。

`--grouped-decisions` now performs **two additional LLM generations**, replacing
one generation per semantic group. It requires `--vector-routing`, implies decision
 evolution, and leaves the summary and timeline paths unchanged.

1. Original selected source KV plus the final summary produce a short decision/proposal checklist.
2. Embed checklist items and match against the reused embeddings of **all** speaker blocks.
   Keep every block above `--decision-similarity` (default 0.75, experimental), or the
   closest block if no block reaches it. The audit labels this fallback explicitly.
   Expand by `--speaker-neighbor-blocks`, union with the original selected positions,
   deduplicate, and sort in source order. No source text is silently dropped to meet a budget.
3. The merged original-source KV, summary and checklist produce one joint decision-evolution
   report. Similarity is only retrieval; the model checks proposal identity and final status.
   It may add supported decisions missing from the checklist. Neither completeness nor
   correct interpretation is guaranteed.

The checklist is a natural-language intermediate, not latent communication. Evidence is
supplied through original KV slices, preserving original positions. Source KV is computed
once. There is no fixed checklist count or per-topic generation loop. This reduces call
count, not necessarily retained KV size; evidence union may approach the full transcript.
Long outputs still take time and model context capacity remains a limit. Insufficient
input capacity raises an error rather than truncating evidence or the checklist.

## Linux

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/vector/meeting1_two_stage.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --embedding-device cuda \
  --grouped-decisions \
  --variants routed_kv \
  --timeline \
  --routing-audit
```

The final report is `.decisions.txt`. JSON audit records the checklist, matching scores,
expanded block IDs, final KV positions and newly added positions. The normal text audit
still describes summary routing; use JSON to inspect decision retrieval.
`metrics.json` includes both stages' input/output tokens, time, cap hits and additional
checklist embedding tokens (separate encoder). These costs are additional to summary
pipeline metrics. Source embedding cost is already counted in vector routing.

For the original one-pass baseline, replace `--grouped-decisions` with
`--decision-evolution`. Compare coverage of decisions and subsequent revisions, final
status, factuality and total runtime on the same meetings; reduced calls alone do not
prove better output quality.
