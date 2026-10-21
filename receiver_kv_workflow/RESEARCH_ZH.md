# 2025–2026 方法调查与本项目判断

调查时间：2026-10-21。优先原论文，而不是根据综述描述推定实现。
以下 2025 年早期方法属于你指定的 2025–2026 范围，但不严格属于最近 12 个月。

## 候选方法

| 方法 | 年份 | 对当前问题的意义 | 选择 |
|---|---|---|---|
| [KVCOMM](https://arxiv.org/abs/2510.12872) | 2025 | 用在线 anchor 修正不同前缀下共享内容的 KV offset，避免重复 prefill；training-free | 当前相同前缀已经能精确复用，暂不引入其额外复杂度 |
| [LatentMAS](https://arxiv.org/abs/2511.20639) | 2025/2026 | 生成 latent thoughts，用 KV 工作记忆协作；无需训练 | 可作为纯潜空间推理对照，但完整 KV 在长会议中膨胀，会议事实覆盖尚需验证 |
| [StateBridge](https://arxiv.org/abs/2608.13317) | 2026 | 最终层 hidden states 对齐到输入 embedding 空间；无需训练 | 保留现有实现作为基线，不把 reasoning/QA 的收益外推为会议保证 |
| [CacheBack](https://arxiv.org/html/2609.32046v1) | 2026 | 接收需求参与状态选择，针对长私有证据通信；training-free | 最值得借鉴的选取原则，但本项目不是完整复现 |
| [CacheFuse](https://arxiv.org/abs/2610.16165) | 2026 | 无训练、异构模型 KV 插值融合 | 当前同一 Qwen3 权重不需要解决异构问题；也没有长会议收益保证 |
| [KVzip](https://arxiv.org/abs/2505.23416) | 2025 | 通过上下文重建评估 KV 重要性，适合后续不同查询复用 | 若会议会被反复提问，值得作为独立缓存压缩基线 |
| [ChunkKV](https://arxiv.org/abs/2502.00299) | 2025 | 连续片段级压缩，避免只考虑孤立 token | 借鉴邻接片段保留，不等于实现其完整算法 |
| [LCF](https://arxiv.org/abs/2605.22863) / C2C | 2026 | 学习跨模型 KV 转换/压缩 adapter | 冻结主模型不等于无需训练；不符合当前严格要求 |

## 一个重要核查结果

CacheBack 附录 C.3 明确：论文实验发送选中 continuous rows，再在接收端重新 prefill；
并没有评估需要额外处理位置问题的直接 KV 转移。因此不能把本项目的 KV index_select
标成“CacheBack 复现”。这里借鉴其 receiver-conditioned attention scoring，并单独测试
直接 KV 传递所需的源缓存同一性、位置保留及因果 mask。

原论文还含 latent thought steps 等设置，新实现没有复制这部分。
新代码的角色是证据选择器，不是 LatentMAS 的 latent reasoning agents。

## 为什么暂时不继续强化 tail hidden-state 压缩

旧链条大致为：原文 → specialist 改写/选择 → 选取 handoff 的部分 states →
StateBridge 对齐 → receiver 再总结。遗漏可能发生在任何一层：

- 专家生成时漏掉事实，后面的通信再强也无法补回。
- 长 handoff 的前段事实不能因为末尾 states 有因果上下文，就保证可完整解码。
- 固定 K 与每个专家的信息量并不一致。
- 对齐后的连续前缀是否被接收方有效使用，也需要与文本对照。
- 证据验证误删与 receiver 的分类冲突是另外的问题，不能全部归咎于 K。

新方案绕开前两次摘要式改写：直接从全文表征选择证据位置，最终仅生成一次摘要。
这只是可检验的假设，不表示 token attention 就等同事实重要性。

## 两段通信怎么选

**前段**：保留一次 exact prefill，加入角色需求驱动的 KV 选择。
不能为了听起来更像 latent communication，就把本来精确的 prefix reuse 换成
近似 cross-prefix KV 校准。计算复用与通信路由可以同时存在。

**后段**：首版选择原文 KV 并集直接交给 receiver，不继续把所选证据压成 96 个
hidden states。保留原始多层 K/V，代价是每位置字节数更大。

这里有意不保留“每个专家必须生成一份结论”的约束。如果这个约束是论文研究问题
不可缺少的一部分，应把本流程叫做独立的 evidence-routing baseline，而不是声称
它与旧多专家推理拓扑完全相同。

## 长会议特别需要警惕

[HalluKV / Anchoring the Cache](https://aclanthology.org/2026.acl-long.1542/)
专门研究 KV 压缩对长上下文摘要幻觉的影响，发现激进压缩可能加剧幻觉；其方法在
解码时处理 retrieval heads 的生成 KV，以更关注来源。它提醒我们：缓存更小不是
事实性更好。本实现没有加入 HalluKV，也不声称获得它的效果。

会议摘要不像单问题 QA，必须覆盖多个主题、晚出现的修订、责任归属和否定信息。
角色查询可以增加覆盖，但仍可能偏好显眼主题，遗漏仅出现一次的重要安排。
连续 span 能缓和碎片化，不解决所有跨段依赖。

## 必须做的实验

1. 相同输入运行 full_kv、routed_kv、selected_text，固定模型与输出预算。
2. 选取比例 0.15 / 0.25 / 1.0；1.0 应与 full_kv 等价。
3. 再与旧 StateBridge 和旧 text handoff 比较，但说明拓扑已经改变。
4. 不仅算 ROUGE：人工标注若干关键决定/行动的覆盖、支持与矛盾，特别检查人名、
   日期、否定、提议到决定的时间变化。不要把词汇重合当成事实验证。
5. 对每个样本记录耗时、生成 token、receiver 位置、KV 字节、峰值显存和输出截断。
6. 使用多会议独立样本，报告逐样本差和置信区间；不要反复挑一个会议调参后称泛化。
7. 以后可加入无关 query / 错误会议 cache 的审计，验证结果确实依赖通信证据。

没有参考摘要时，本程序输出 N/A，不制造精确度分数。现阶段的交付是独立可测试的
实验实现，不是已经证明优于 StateBridge 的研究结论。

## 文件职责

- `prompts.py`：通用会议角色需求和最终摘要指令，无会议人名/案例硬编码。
- `scoring.py`：查询注意力均值与集中度分数，分批计算，保持源缓存不变。
- `selection.py`：连续 span 预算与按源位置去重。
- `cache.py`：相同源 KV 选择、位置保持、最终生成。
- `run.py`：输入检查、三个对照、分离内容和指标输出。
- `test_workflow.py`：算法、模型缓存与 CLI smoke tests。
