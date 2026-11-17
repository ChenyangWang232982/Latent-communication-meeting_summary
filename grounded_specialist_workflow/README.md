# Grounded specialist workflow / 分类别 KV 与证据检索工作流

This experimental workflow has its own CLI. The existing `receiver_kv_workflow`
and Experiment 1 remain the comparison baseline. This is a workflow change, not
a representation-only intervention, and is not an external AI evaluation service.

## 流程 / Flow

1. 全文一次 prefill，建立不可变原文 KV；按连续发言人分块。
2. 全文发言块只向量化一次，复用 Qwen3-Embedding。
3. context、decisions、actions、open_issues 各自查询注意力、各自语义排序、各自预算和邻居扩张。
   coordination 查询并入 actions。角色之间允许选中同一块，不取全角色并集后统一发送。
4. 各 specialist 从原文 KV 提取独立可写副本，输出结构化条目。没有固定条目数量。
5. 每条结论检索**全部原文块**，加入前后邻居并去重，按原会议顺序呈现证据。
6. 按类别合并核查调用，每条都有独立检索结果、判定、理由和证据块 ID。
7. 最后一个 agent 对结构化结果去重、归位，生成四章节摘要。

Each specialist receives selected **source KV**, with original absolute positions.
It does not receive other specialists' generated KV. Cached hidden representations
were computed with preceding source context, so separate selected positions do not
imply strictly isolated knowledge. Grounding receives **verbatim retrieved text +
structured claims**; synthesis receives **structured checked text**. This is a
hybrid communication workflow. Model checks remain fallible, not factuality scores.

## Run / 使用

Use the existing `latent-agents` environment and downloaded Qwen3 embedding model.
No additional dependencies or API keys. Input requires line-start speaker labels,
as in `Alice: ...` or `Alice  01:23` (same parser as the baseline).

```bash
conda activate latent-agents
python -m grounded_specialist_workflow \
  --transcript input/meeting1.txt \
  --output output/grounded_specialists/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --specialist-workers 2 \
  --evidence-top-k 4 \
  --evidence-candidate-pool 12 \
  --evidence-neighbors 1 \
  --evidence-recency-weight 0.1 \
  --routing-recency-weight 0 \
  --seed 12
```

`0.1` is a trial value, not an established optimum. First compare it with `0` on
the same meeting and settings. Default time weights are both zero. Decoding defaults
are temperature 0.7, top_p 0.8, top_k 20, generated-only repetition_penalty 1.1;
all are configurable. Thinking is disabled. No extra output length or item-count
cap: EOS or the loaded model's context boundary ends generation.

Model precision follows the existing loader: CUDA BF16 when supported, otherwise
CUDA FP16, or CPU FP32. The output head uses the same loaded model precision.

## Position weighting / 位置加权

Embeddings themselves are unchanged. Each block has a relative **block order**
in [0,1], not elapsed meeting time. Evidence retrieval first selects the semantic
candidate pool, always retains the best semantic anchor, then ranks the remaining
candidates by `cosine + evidence_recency_weight * relative_position`. Neighbors
are added afterwards. Final evidence is sorted chronologically, not by relevance.
Time weighting cannot introduce blocks outside the candidate pool (except their
explicit neighbors). It does not establish entailment or identify final decisions.
There is no claim of exhaustive recall: top-k retrieval can miss later reversals.

`--routing-recency-weight` independently adds a position preference to the initial
role block ranking; default 0 because a blanket late-meeting preference can remove
important early context. Vary this separately from evidence weighting.

## Execution and outputs / 执行与输出

`--specialist-workers 1..4` controls concurrent tasks sharing one model. Each has
its own KV and seeded random generator. Default 1 is the lower-memory option;
2 or 4 enables concurrency. GPU kernels can still serialize; speedup is not
guaranteed. More workers require more live KV memory. Selectors run sequentially
before specialist execution because they temporarily change the attention backend.

`--output` contains the final summary. Sibling `<stem>.artifacts/<UTC timestamp>/`
contains configuration, source blocks, per-role positions/scores, raw specialist
outputs, structured claims, retrieved candidates and chronological evidence,
grounding prompts/results, summary input, stage metrics and status. Partial decoded
text is saved during long generations. Block IDs are zero-based and stable within
the run. Treat artifacts as containing the full original meeting text.

Each grounding result accounts for its input claim with a verdict:
supported, revised, uncertain, or unsupported. Uncertain claims are retained with
qualifications; clearly unsupported claims stay in the audit but are excluded from
the final synthesis input. This is a model judgment, not an exact quote filter.
JSON shape/claim IDs/citation bounds are checked to prevent silent data loss;
invalid transport or context exhaustion fails explicitly with raw output saved.
No automatic retries, fabricated fallback or silent truncation. Very large evidence
packets may exceed context; reduce retrieval pool usage/top-k/neighbors or use a
compatible longer-context model. Reducing retrieval changes evidence coverage.

Metrics count every decoder prompt/output event, shared prefill/probes once, and
encoder tokens separately. Grounding text is encoded again and counts as decoder
input. Concurrent stage durations overlap; pipeline wall time is the end-to-end
measure. Model loading is excluded. No process peak-memory measurement is claimed.

## Validation

```bash
python -m unittest discover -s grounded_specialist_workflow/tests
```

Tests cover independent role routes, full-source retrieval beyond selected KV,
recency reranking, neighbor deduplication, claim/citation integrity and failure
audits. Real GPU generation and quality require running the command above.
