# 扩展首尾关联挑战集 / Expanded endpoint challenges

20 条人工事项分别插入 20 场不同的真实 AMI 会议。原会议全文保持不变，新增事项只在首尾
各出现一段，中间不插入提示。结尾用相关表达回指开头；不在生成要求中点名该事项。
这些是合成诊断案例，不是原始 AMI 事实，也不是人工写的完整参考摘要。

默认跳过当前 AMI 输入前 6 条：前三条旧挑战来源，以及后三条真实会议试跑来源。
固定取第 7–26 条，不按模型结果筛选。准备时保存真实 ID、源 hash、插入证据字符范围和独立答案。
“新”指这些新增事项未参与此前 V3 调参，不代表基础会议未被其他实验使用，也不代表官方测试划分。

## 设计

| 类型 | 数量 | 检查重点 |
|---|---:|---|
| revision | 4 | 包装、安装、取件日期、房间的明确修订 |
| owner_change | 4 | 最终唯一负责人；交接不扩大到相邻任务 |
| condition | 4 | 多条件未全部满足、条件已满足但日期未定、报价超过上限 |
| scope | 4 | 局部借用、学校展示、展览样机、翻译测试不能泛化到主产品 |
| no_change | 4 | 较晚的建议、协助意愿、低价报价不自动推翻已有决定 |

每条同时有状态和细节要求，包含数量、范围、条件或负责人。部分事项已经明确批准，部分仍被
条件阻挡，另一些保留早先决定；不能一律用“尚未决定”或“以最后发言为准”获得正确答案。
这些自然语言场景仅存在于测试数据定义，检索和生成算法没有按案例关键词加权。

不过首尾放置、常见 AMI 角色、同一人工编写者和场景模板都会限制代表性。
新增事项可能与原会议整体摘要的重要性竞争；遗漏应单独记录，不等同于虚构。
这套数据不能证明普遍解决长文本注意力问题。看完结果后再改规则，后续结果就属于开发结果。

## 可选：用原摘要粗看参考相似度，不重跑工作流

默认挑战核对仍使用上述预设目标。若仅希望粗看原会议主题的摘要相似度，可另外对已生成的
60 份完整摘要计算 ROUGE 和 BERTScore P/R/F1。此模式**只用原始 AMI 摘要**，不追加挑战答案，
不改写候选摘要。脚本会核对去掉首尾插入行后的原会议全文与参考记录是否完全一致。

```bash
python -m experiment_3.rescore_challenges \
  --run-dir test_result/experiment_3_challenge20/20261126_001225_109957 \
  --references input/experiment_1_unfiltered_cap100/ami.jsonl \
  --bertscore-model allenai/longformer-base-4096 \
  --bertscore-layers 12 \
  --device cuda
```

只加载评分 encoder，不加载 Qwen、不重新生成摘要、不调用 AI judge。缺少依赖时先运行
`pip install -r experiment_1/requirements-quality.txt`。输出位于本批目录的
`original_reference_evaluation/<时间戳>/`，包括 `summary.md`、`summary.txt`、`summary.json`、
`per_sample.csv`、`references.json` 及评分溯源信息。

此模式是**探索性原主题参考相似度**：原摘要没有插入事项，正确写出新增内容也可能降低 precision
或 F1，所以不把分数当作挑战事实精度、正确率或新增事项覆盖率。BERTScore P 是语义匹配精度，
不是“事实正确的句子比例”。不静默截断；不使用过短的目标答案来评整篇纪要。

## 运行（Linux，项目根目录）

先同步当前代码，再激活服务器环境：

```bash
conda activate latent-agents
cd /root/autodl-tmp/Latent-communication-meeting_summary
bash experiment_3/run_challenges.sh
```

脚本会：

1. 若冻结数据不存在，生成 `input/ami_endpoint_challenge_20_v1/`；已存在则复用。
2. 跑原时间窗口 `baseline` 与 V3 `semantic`，共享同一次全局选择，交替先后顺序。
3. 跑 `direct_text` 全文单模型，使用相同最终采样参数。此处复用接受无参考摘要的挑战入口，
   **不是**要求参考摘要的普通 `experiment_2` 入口。
4. 导出 `report/coverage.txt` 和 `report/target_review.csv`。

指定其他 AMI 源路径或新数据目录：

```bash
bash experiment_3/run_challenges.sh \
  input/experiment_1_unfiltered_cap100/ami.jsonl \
  input/ami_endpoint_challenge_20_v1
```

手动只准备数据：

```bash
python -m experiment_3.prepare_challenges \
  --input input/experiment_1_unfiltered_cap100/ami.jsonl \
  --output-dir input/ami_endpoint_challenge_20_v1 \
  --skip 6
```

目录已有时不要反复准备；文件冻结且不覆盖。生成模型不读取 `ANSWER_KEY.md` 或期望答案，
只接收 `meetings.jsonl`。答案只在生成之后用于证据覆盖和人工导出。
原参考摘要保存在答案中用于溯源，不用于评价修改后的会议。

## 输出和判定

每次在 `test_result/experiment_3_challenge20/<时间戳>/` 下创建：

| 路径 | 内容 |
|---|---|
| workflow/<时间戳>/00000/{baseline,semantic}.summary.txt | 两种窗口工作流最终摘要 |
| workflow/<时间戳>/00000/*.specialists.json | 各窗口证据和局部报告 |
| workflow/<时间戳>/00000/target_coverage.json | 实际首尾位置保留与同窗覆盖 |
| direct/<时间戳>/direct_text/00000_<ID>/summary.txt | 全文单模型摘要 |
| report/coverage.txt | 覆盖命中数、可评估数、完成数；不是摘要正确率 |
| report/target_review.csv | 20×3 行空白目标核对表，含答案与输出路径 |

核对表 `verdict` 手动填：

| 值 | 含义 |
|---|---|
| correct | 最终状态及该事项的负责人、数量、条件、范围等关键细节全部正确 |
| partial | 核心状态正确，但遗漏必要细节；没有相反断言 |
| incorrect | 存在错误状态、错误负责人、扩大范围、虚构条件满足等相反断言 |
| omitted | 未提及目标事项 |
| 空白 | 尚未审阅，不算通过 |

严格目标正确率 = correct / 已审阅成功输出数；另外报告 partial、incorrect、omitted 和失败/缺失数。
建议按类型各报 4 条，并同时比较 baseline–semantic 和 semantic–direct；不把 60 行当 60 个独立会议。
可先隐藏 condition 列逐条核对，但文风可能暴露方法，不能声称严格盲评或多人一致性。

证据同窗覆盖按 [METRICS.md](METRICS.md) 计算。20 条时至少 16 条覆盖达到本项目 80% 目标；
只有全部 20 条可评估时才能直接使用 16/20 口径。此门槛不是事实正确率门槛。
本脚本不自动评判最终摘要是否正确，也不调用 AI judge，不给挑战集算 ROUGE/BERTScore。
覆盖统计读现成输出，无需再跑生成；重新导出时使用新报告目录，避免覆盖手动判定。

这是三组全工作流比较，V3 局部证据更多，并非等证据量或等成本对照。若 V3 优势成立，
仍需结合真实会议自动评估、开销与失败率决定是否替换默认工作流。
