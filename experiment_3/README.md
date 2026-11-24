# Experiment 3: semantic supplementation of selected KV windows

当前检索版本为 **`relative_diverse_v3`**。V1/V2 说明保留作历史记录；新运行默认采用
下面的逐锚点候选与多样性策略。主工作流的全局 selector 不变。

## V3：逐锚点候选、相对优势与多样性

V2 诊断发现三个开发案例的前后证据，双向独立检索都排第 1，但窗口统一排名分别降到
3/7、7/5、31/32，未进入前两名。V3 因此改变名额分配，而不是更换 embedding 或扩充名额。

1. 在原有邻居、空内容、同内容排除之后，每个非零信息权重的窗口块独立检索。
2. 每个锚点只提出一个最佳候选；比较的是不同候选内容，避免重复文本人为压低优势。
3. 计算：`relative_advantage = (best_score - runner_up_score) / best_score`。
   分数仍是 V2 的信息加权余弦。只有一个可用候选时 runner-up 为 0；并列最佳优势为 0，
   不产生明确提名。不再让绝对余弦较高的话题直接占满名额。
4. `base_priority = relative_advantage * sqrt(anchor_information * candidate_information)`。
   同一候选被多个锚点提出时，只保留优先级最高的提名。
5. 贪心选择：`selection_score = base_priority * (1 - redundancy)`。
   redundancy 是与已选匹配的候选相似度、锚点相似度的正值最大值；首个匹配为 0。
   完全冗余的匹配不再加入。它是连续多样性惩罚，不是固定话题数量或聚类阈值。
6. 最多补两个不同内容块，仍去重并保持原始 KV 位置。可能少于两个。

这是无需训练的启发式；相对优势不是置信概率，也不能保证稀疏话题都获得名额。
没有针对首尾位置、负责人、授权等词语或挑战目标的特殊规则。三条案例已用于开发，
后续质量验证需要保留其他未用于调参的数据。

### Linux：先只测 V3 检索和排名

```bash
python -m experiment_3.retrieve \
  --run-dir test_result/experiment_3_endpoint_3/20261125_052753_701837 \
  --output-dir test_result/experiment_3_retrieval_v3 \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --supplement-blocks 2 \
  --exclude-neighbor-blocks 1 \
  --answer-key input/ami_endpoint_challenge_3/answer_key.json
```

`matches.txt` 新增相对优势、多样性惩罚和最终 selection score。
诊断保留 `absolute_v2_rank` 对照旧排序，新增 `nominated`、`window_selection_score`、
`relative_advantage`、`redundancy`。`score` / `window_score` 仍是原信息加权证据分数，
不能和新的 selection score 混用。候选未被锚点提出时记录
`no_distinctive_anchor_nomination`，被多样性完全抑制时记录 `diversity_redundant`。
确认候选有用后，原 `python -m experiment_3` 生成入口也会使用同一 V3 排序。

## 目的 / Purpose

检验将远距离相关证据补充到局部 specialist，是否帮助保留条件、指代和决策变化。
固定一次全局选择，比较 `baseline` 与 `semantic`；不训练模型，不恢复被丢弃的 KV。

Test whether local access to related distant evidence improves detail preservation.
Both conditions share one immutable source cache and the exact same selected positions.

## 方法 / Method

1. 原流程 prefill、五类查询、speaker 完整块选择、邻居扩张，仅执行一次。
2. 原有时间窗口和 overlap 不变。`baseline` 只读取窗口 KV。
3. `semantic` 在全局已选块中检索：每个候选与窗口内各块比较，按信息加权分数排序。
   排除窗口内块，以及距离任一窗口块 ID 不超过 `--exclude-neighbor-blocks` 的块。
   默认取排名最高的最多 2 个不同内容块；没有调优的相似度阈值，并不保证候选确实相关。
4. 只合并选中位置，去重并按原位置排序。保留原 RoPE 位置，不把补充块改成连续位置。
5. 两组都使用相同的补充证据解释规则、相同种子和解码设置；baseline 的补充列表为空。
   因此 baseline 多了来源元数据提示，与旧实验的原始输出不保证逐字相同。
6. 最终汇总都读取原来的全局 selected KV 和各自的局部报告。最终证据集合不扩大。
7. 偶数样本先 baseline，奇数先 semantic；这是简单平衡顺序，不是随机化。

`--vector-routing` controls the original **selection** stage for both conditions.
Post-selection semantic supplementation always runs. When selection already computed
the block embeddings, they are reused on CPU; otherwise only selected blocks are encoded once.
The embedding model stays resident in both conditions. Both generation paths use sampling
with the shared existing seeds/configuration; no new output-token cap or reviewer is added.

### V2：重复短发言修复

V1 单纯取最大余弦相似度，实测被重复的短发言与转录注释占满。
V2 不设词语黑名单，而在每场会议的已选块上计算内容区分度：

```text
idf(t) = log((N + 1) / (含该词的块数 + 1))
mass(block) = sum(idf(t) for 每个不同词)
information(block) = mass / (mass + 全部块的平均 mass)
score(candidate, anchor) = max(0, cosine) * sqrt(information(candidate) * information(anchor))
```

候选取其与窗口内各 anchor 的最高加权分数。speaker header 与通用括号式转录注释
不参与内容统计，空内容不补充；与窗口内容完全相同的词序不补充；一次窗口内相同
候选内容只补一次。词序不同或含不同否定词的发言不会因为词集合相同而被合并。
余弦值裁剪到 [-1,1]，防止浮点舍入使其略大于 1。

这是无需训练的排序启发式，可能偏向内容较多的块，不等价于真值、显著性或最终状态。
它修复了已观察到的短句占位问题，但新检索和摘要质量仍需实测。全局 selector 不变。

### 先用旧结果只检查检索（不跑 Qwen3-8B）

```bash
python -m experiment_3.retrieve \
  --run-dir test_result/experiment_3_endpoint_3/20261125_052753_701837 \
  --output-dir test_result/experiment_3_retrieval_v2 \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --supplement-blocks 2 \
  --exclude-neighbor-blocks 1
```

读取旧审计保存的块原文、精确选择位置、原窗口，仅执行 embedding 和新排序。
不会重跑全文 prefill、selector 或生成，不读取挑战答案作为检索输入。
每条样本输出 `matches.txt`（匹配双方原文、分数）与 `retrieval.json`（位置、元数据）。
旧版未保存向量，故此入口需要重新编码块文本；审计字符片段与在线 tokenizer.decode
在边界可能略有差异。不要把本入口耗时当成在线复用向量的耗时。
先检查相关发言是否真正连上，再用下面原有成对命令重新生成，output-dir 改成新目录。

### 目标关联的离线排名诊断

V2 已修复短发言占位，但三个挑战目标仍未入选。因此先诊断，不继续调整检索规则。
加 `--answer-key` 后，在普通检索结束后，才将注释字符范围映射到 speaker blocks：

```bash
python -m experiment_3.retrieve \
  --run-dir test_result/experiment_3_endpoint_3/20261125_052753_701837 \
  --output-dir test_result/experiment_3_retrieval_diagnostic \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --supplement-blocks 2 \
  --exclude-neighbor-blocks 1 \
  --answer-key input/ami_endpoint_challenge_3/answer_key.json
```

`--run-dir` 仍指向原始成对生成结果，不能指向仅检索的 V2 目录。
每个样本新增 `target_diagnostics.txt` / `.json`，检查双向关联而不假设目标一定是首尾块：

| 字段 | 含义 |
|---|---|
| `raw_anchor_rank` | 以该目标发言为锚点，正确关联按原始余弦的排名 |
| `weighted_anchor_rank` | 同一候选池按信息加权分数的排名 |
| `raw_top_candidates` / `weighted_top_candidates` | 两种排名前五的分数和原文 |
| `window_candidate_rank` | 在实际窗口所有锚点竞争后的排名，应用生产排除规则 |
| `window_distinct_rank` | 再去除相同候选内容后的排名，实际取前 K 项 |
| `exclusion_reasons` | 已在窗口内、邻居排除、相同内容、空内容或无正分等 |
| `actually_supplemented` | 普通检索是否实际补入该目标块 |

排名从 1 开始；null 表示不在该候选池，不赋零排名。独立锚点排名的候选池是所有
已选远距离块，未先过滤短句，便于观察原始语义与信息加权的区别；实际窗口排名则复用
生产检索的同一个函数。目标未被全局选中时明确报告，不把缺失证据当成 embedding 错误。

只用答案文件里的样本标识、原文 hash 和字符范围，不用答案内容或期望结论加权。
原文不匹配时跳过该样本的诊断并记录原因。目标注释可用于开发诊断，不能据此反复调参
后仍将这些案例称为未见测试。此步骤不改变匹配结果、KV 选择或生成工作流。

**边界：**语义组局部上下文更多，本实验不是等量证据对照。相似不等于赞成、真值或重要性。
它不能找回未选中的发言，也不能证明模型内部注意力被改善。
来源块 ID 只是提示元数据，不是在 KV 内插入的文字锚点；不要把生成的块引用当可靠标签。
原始位置保序不能保证模型正确理解所有跨块关系。下一步若有收益，应补等量非语义证据对照。

## Linux：已有三条挑战样本

```bash
conda activate latent-agents
cd /root/autodl-tmp/Latent-communication-meeting_summary

python -m experiment_3 \
  --input input/ami_endpoint_challenge_3/meetings.jsonl \
  --output-dir test_result/experiment_3_endpoint_3 \
  --num-samples 3 \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --supplement-blocks 2 \
  --exclude-neighbor-blocks 1 \
  --seed 12 \
  --summary-seed 42
```

若文件不存在，先按 `experiment_2/challenge_samples/README.md` 准备；已有目录不需要重建。
这里的答案只在生成后用于覆盖检查，不进入提示词。旧会议参考摘要不适用于人工插入的决策，
因此不自动为挑战样本计算 ROUGE/BERTScore，也不把目标正确性自动判为通过。

## Linux：meeting1 或真实数据

```bash
python -m experiment_3 \
  --input input/meeting1.txt \
  --output-dir output/experiment_3_meeting1 \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --vector-routing \
  --seed 12 --summary-seed 42
```

将 `--input` 换成标准 `id/transcript/summary` JSONL 可运行真实会议。默认前三条，
`--num-samples 0` 为全部。不加 `--vector-routing` 可固定 attention-only 的全局选择，
此时 embedding 仅服务于后选取的局部补充，两组全局选择仍相同。

## 输出 / Outputs

每次生成独立时间戳目录，不覆盖旧结果。样本目录用序号，原 ID 在 `source.json`。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 配置、输入文件 SHA-256、样本顺序 |
| `status.json` | 完成对数与失败数 |
| `metrics.csv` | 各组逻辑 token、独立运行估算耗时、KV 位置事件和字节 |
| `manual_review.csv` | 空白人工判定表，不预填模型正确性 |
| `00000/source.json` | 原文、参考摘要与原 ID |
| `00000/selection.json` | 两组共用的精确选中位置及 hash、routing audit |
| `00000/baseline.summary.txt` | 无语义补充的最终摘要 |
| `00000/semantic.summary.txt` | 补充远距离相关块后的摘要 |
| `00000/summary.diff.txt` | 两组最终摘要的文本差异，不是正确性评分 |
| `00000/*.specialists.json` | 窗口、primary/supplement block IDs、匹配分数、报告与各阶段成本 |
| `00000/*.metrics.json` | 单组成本 |
| `00000/target_coverage.json` | 若同目录有匹配的挑战答案：端点保留及同窗覆盖 |
| `00000/*.partial.txt` | 实时生成内容 |
| `00000/failure.json` | 若失败，记录原因；不赋零质量分 |

成本中完整 prefill/selection 在每个独立工作流估算中计一次；实际成对运行只做一次。
baseline 不计补充检索开销，semantic 计入该开销；这是从共享执行拆出的估算，
不是两次独立测量。embedding 输入事件与生成模型 token 分开。
KV payload 是逐阶段逻辑张量字节，不是网络流量；重复访问 KV 也仍有注意力计算。
没有测量各组独立显存峰值。不要用这里推断 baseline 不加载 embedding 时的部署显存。

人工检查：先看目标在两组是否被选中，再看是否同窗，最后看局部报告与最终摘要。
记录负责人、条件、数量、范围与决策状态错误，不能只判断有没有提到主题。

## CPU tests (no models downloaded)

```bash
python -B -m unittest experiment_3.tests.test_supplement experiment_3.tests.test_diagnostics
```
