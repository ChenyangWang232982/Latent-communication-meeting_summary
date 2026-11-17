# Topic specialist workflow / 动态话题工作流

入口名称仍为 `grounded_specialist_workflow`。流程改为：

```text
全文 + 发言块编号 → 一次原文 KV prefill → 全文话题发现
                                            ↓
                         每话题 selector：注意力 + 语义匹配
                                            ↓
                      独立 KV（多个相关片段按原文顺序排列）
                                            ↓
                     固定并发数的 specialist 队列 → 结构化报告
                                            ↓
                                    最终摘要 agent
```

Topic discovery reads the complete source KV and returns topic titles/scopes in
first-appearance order. Each topic becomes a selector query, not a fixed action or
decision role. Topic and source-block embeddings are computed in one encoder call;
source vectors are reused. Attention and topic similarity rank complete speaker
blocks; neighbor expansion follows. Topics may share blocks. All selected positions
retain original order and absolute positions within the annotated source prompt.

`[B0]`, `[B1]`, ... markers are added after speaker headers before prefill, so the
model can identify block order. Original words remain intact. This changes prompt
tokens relative to the baseline and must be accounted for in comparisons.

No fixed topic count. The discovery model determines topic granularity and can
still miss or split topics; there is no recursive topic creation or review stage.
Each specialist reports discussion, decisions, actions, open issues and final
status for ONE topic. Reports are held in memory and saved, then merged in discovery
order regardless of completion order. Final synthesis receives structured text.
This is a hybrid workflow, not KV-only communication or independently verified output.

## 使用 / Run

```bash
conda activate latent-agents
python -m grounded_specialist_workflow \
  --transcript input/meeting1.txt \
  --output output/topic_specialists/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --topic-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --specialist-workers 2 \
  --seed 12 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repetition-penalty 1.1
```

- `--topic-budget-ratio`: initial positions per topic; neighbor expansion can exceed it.
  Reuses the baseline config's internal `role_budget_ratio` field. The old CLI
  `--role-budget-ratio` has been replaced in this entry point only.
- `--specialist-workers`: parallel topic tasks, not the number of topics. One shared
  model, independent writable KV and RNG per task; more workers need more memory.
- `--routing-recency-weight`: optional initial ranking position bias, default 0.
  Later statements do not automatically override earlier decisions.
- Generation uses default model precision and no thinking. No fixed output-item
  or token cap; EOS/model context ends generation. Oversize context raises an error.

Selectors execute before parallel generation because attention scoring temporarily
changes the model backend. This keeps the shared model safe from concurrent backend
changes. Worker threads may not yield GPU speedup; measure wall time.

## 输出 / Outputs

`--output` is the final summary. `<stem>.artifacts/<timestamp>/` contains:

- `topics.json`, `topic_discovery.txt`: discovered topics and raw discovery output.
- `source_blocks.json`, `routing.json`: source text/coordinates and per-topic selections.
- `specialist_topic_*.txt`, `report_topic_*.json`: raw/structured topic analyses.
- `topic_reports.json`: all topic reports in discovery order, ready for synthesis.
- `summary.input.txt`, `summary.txt`, `*.partial.txt`: prompts/results and live output.
- `config.json`, `stages.json`, `metrics.json`, `status.json`: settings and accounting.

There are N+2 generation calls for N topics: discovery, N specialists, synthesis.
All are included in decoder token totals; embedding tokens are separate. Concurrent
stage times overlap; use pipeline wall time, excluding model loading. Source block
IDs in reports indicate supplied context, not verified claim-level citations.
Only basic JSON parsing and model-context checks remain; no content verdict filter.

```bash
python -m unittest discover -s grounded_specialist_workflow/tests
```

Tests cover variable topic counts, ordered KV selections/reports, cache and RNG
isolation, block markers, accounting and failure audit. Actual GPU quality requires
running the example command. Existing baseline/Experiment 1 entry points are separate.
