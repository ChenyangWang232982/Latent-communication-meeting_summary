# Experiment 1: Data Analysis and Thesis Research Questions

Prepared on 21 November 2026. Data source: `D:\Download\experiment_1_auto1024_cap100_seed12`.

BERTScore update: 23 November 2026. Added scores come from the user-provided server terminal output; the underlying per-record scoring files have not yet been imported locally.

This report analyzes saved outputs and rescoring summaries without regenerating summaries or changing the source data or workflow. Quality, efficiency, and length-group statistics include only records for which both variants completed. Failed records are reported for sample accounting and excluded from paired means. BERTScore has now been added; pending human ratings are not treated as zero.

## 1. Main findings

295 records were planned and 276 pairs completed, a completion rate of 93.56%. Weighted by successful records, mean logical decoder token events fell from 43,129.6 to 20,227.3, a **53.10%** reduction. Mean pipeline latency fell from 180.84 to 162.30 seconds, a **10.25%** reduction. Mean ROUGE-L was **0.10256 / 0.10288 (KV / text)**, showing a small observed difference.

After rescoring, BERTScore F1 differences (KV−text) across the four datasets are −0.0011, +0.0007, −0.0002, and −0.0023. Weighting the displayed four-decimal means by sample counts gives approximately **0.8296 / 0.8307 (KV / text)** overall, a difference of approximately **−0.0011**. This is descriptive evidence of similar reference semantic similarity, not proof of equal factual accuracy or non-inferiority.

| Evidence | Supported interpretation | Unsupported inference |
| --- | --- | --- |
| Four datasets | KV reduces logical token events by approximately 50–55% in each dataset | The same reduction in network traffic, FLOPs, or billing |
| Paired ROUGE-L intervals | All four intervals include zero; no consistent directional advantage is established | Proven equivalence or equal factual accuracy |
| Semantic assistance | Enabled in both variants as a shared setting | Embedding has been shown here to solve attention failures |
| Length groups | Longer groups generally show larger token reductions | Length is the sole cause or meetings of unlimited length are supported |
| BERTScore / human ratings | BERTScore summaries added with small F1 differences; manual checking not yet conducted | Proven semantic equivalence or equal factual accuracy |

## 2. Provenance, protocol, and integrity checks

| Dataset | Run directory relative to data root | Planned | Completed | Excluded | Completion |
| --- | --- | --- | --- | --- | --- |
| AMI | `ami/20261119_083659_646986` | 100 | 100 | 0 | 100.00% |
| ICSI | `icsi/20261119_174529_255779` | 59 | 49 | 10 | 83.05% |
| QMSum Committee | `qmsum_committee/20261120_020952_214447` | 36 | 31 | 5 | 86.11% |
| MeetingBank | `meetingbank/20261120_073025_968960` | 100 | 96 | 4 | 96.00% |
| Total | — | 295 | 276 | 19 | 93.56% |

Each dataset is capped at 100 records; ICSI and Committee use all 59 and 36 records available in these preprocessed inputs. The unit is an input record, not necessarily an independent full meeting: MeetingBank records may represent different agenda items from the same meeting. AMI, ICSI, and Committee are the domains used by the current QMSum preprocessing and should not be described as unrelated dataset origins.

Inputs include each run’s `manifest.json`, `runtime.json`, `summary.json`, `tokens.json`, `per_sample.csv`, `repetition.json`, and all 295 `samples/*.json` checkpoints. Checks passed for manifest ID order, identical local window positions/counts and synthesis selection across all 276 pairs, token component arithmetic, and agreement between per-record calculations and reported ROUGE means/token totals. Configuration, code fingerprints, Git commit, and runtime match across the four runs.

The originally supplied directory did not include BERTScore rescoring artifacts under `evaluation/`. Dataset-level P/R/F1 summaries have now been added from user-provided server output. Rating fields in `human_review.csv` remain unfilled; manual checking of a controlled challenge set is an agreed future plan, not an implemented evaluation. Earlier three-record pilots, the retired routed_kv baseline, and the global-KV-plus-raw-part experiment are not mixed into these results.

## 3. Workflow and reproducible settings

Shared pipeline: full-source prefill → five selector attention queries → semantic assistance → complete speaker-block selection and neighbor union → chronological selected-context windows → local specialist reports → final synthesis over global selected evidence and ordered reports. Consecutive turns from the same speaker form a block. Specialists share model weights and execute sequentially in this implementation; these measurements do not establish parallel speedup.

| Setting | Recorded value |
| --- | --- |
| Generation model | Qwen/Qwen3-8B |
| Embedding model | models/Qwen3-Embedding-0.6B |
| vector_routing / vector_weight | true / 0.3 |
| role_budget_ratio / speaker_neighbor_blocks | 0.25 / 1 |
| Target window count | max(1, floor(n / 1024 + 0.5)) |
| Definition of n | Raw full-transcript tokenizer tokens, including speaker/time labels but excluding chat templates |
| specialist_overlap | 0.25 |
| specialist seed / temperature / top_p / repetition_penalty | 12 + window index / 0.3 / 0.9 / 1.1 |
| summary seed / temperature / top_p / repetition_penalty | 42 / 0.3 / 0.9 / 1.0 |
| Thinking mode | Disabled for local specialists; enabled for final synthesis |
| Decoding | Sampling rather than greedy decoding; top_k=0 |
| Separate review | Disabled |
| fp32_lm_head | false |
| GPU | NVIDIA GeForce RTX 4080 SUPER; 31.48 GiB reported |
| Python / PyTorch / Transformers | 3.12.3 / 2.8.0+cu128 / 5.16.1 |
| CUDA / tokenizers / safetensors | 12.8 / 0.23.2 / 0.8.0 |
| Git commit | `3d4d9e7bb57230a578e972c0a1c032d160a834b0` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |

Windows balance complete selected speaker blocks. The 1024-token scale determines the target specialist count; it does not guarantee exactly 1024 tokens per window. Actual count is capped by available selected blocks. The 25% backward overlap includes complete blocks from the preceding base window without adding specialists. role_budget_ratio is a selection-stage budget parameter, not the final retained fraction.

| Path | Local specialist input | Final synthesis input |
| --- | --- | --- |
| chunk_kv | Source KV for the selected window plus stage instructions | All selected source KV plus ordered local natural-language reports |
| chunk_text | Verbatim evidence for the same window plus corresponding stage instructions | All selected verbatim evidence plus ordered local natural-language reports |

Local stages receive neither additional global selected KV nor an additional raw transcript part. Both variants produce local natural-language reports, making the system a hybrid communication workflow. The text control re-prefills verbatim evidence; it is not the retired single generated-handoff baseline. Full-source contextualized KV retains original history and positional information, whereas re-encoded text has a different contextual history. This is a controlled workflow comparison, not a pure causal intervention that changes only tensor versus string representation.

## 4. Metrics, aggregation, and interpretation

| Metric | Definition | Interpretation boundary |
| --- | --- | --- |
| ROUGE-1 / ROUGE-2 F1 | Unigram/bigram overlap with the reference | Lexical similarity, not fact checking |
| ROUGE-L P/R/F1 | Longest common subsequence over whole-summary token sequences; not ROUGE-Lsum | Lowercased ASCII alphanumeric tokens, no stemming; not directly interchangeable with other ROUGE implementations |
| BERTScore P/R/F1 | Contextual-vector matching of generated and reference summaries using one encoder; rescoring summaries now available | Semantic similarity is not factual accuracy; encoder/rescaling configurations must match |
| source-support proxy | A candidate sentence is lexically supported if at least 45% of its distinct terms appear in the full-source vocabulary | Does not reliably check negation, owners, dates, or decision revisions; not MESA or human factuality |
| Logical decoder token events | Source prefill + probes + local prompts/outputs + synthesis prompt/output | Reused KV is not counted again as fresh text input; not attention FLOPs or billing |
| Pipeline seconds | Includes prefill, selection/embedding, specialists, and final generation; excludes model loading | Fixed KV-then-text execution on one GPU may introduce order effects |
| CUDA peak | Per-variant PyTorch allocated peak including models and resident full-source KV | Excludes reserved memory, non-PyTorch allocations, and shared prefill/selection peaks |
| Exact duplicate bullet | Fraction of samples with duplicate normalized bullet lines in final summaries/local reports | Not semantic redundancy; sender counts refer to concatenated local reports |
| Actual retention | Unique selected source positions / all source positions | Source positions include protected template positions; not word compression |
| Logical payload | Sum of stage KV tensor bytes or evidence UTF-8 bytes plus report bytes | Different representations are not equivalent byte measures; no actual network transfer measured |

Quality metrics are calculated per record and macro-averaged over completed pairs. Pooled results weight the 276 records equally; a separate dataset-balanced average weights the four datasets equally. Savings are 100 × (1 − KV total / text total), not the average of per-record percentage savings. Output-token accounting includes final reasoning, delimiters, and EOS; displayed/scored summaries exclude reasoning text. Embedding encoder tokens are reported separately and are not interchangeable with Qwen decoder billing or compute.

## 5. Summary-quality results

| Dataset | n | Variant | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | L precision | L recall | Support proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | 100 | chunk_kv | 0.1905 | 0.0447 | 0.1058 | 0.0633 | 0.3420 | 0.9171 |
| AMI | 100 | chunk_text | 0.1849 | 0.0465 | 0.1037 | 0.0615 | 0.3514 | 0.9292 |
| ICSI | 49 | chunk_kv | 0.1414 | 0.0243 | 0.0790 | 0.0456 | 0.3116 | 0.9690 |
| ICSI | 49 | chunk_text | 0.1392 | 0.0230 | 0.0785 | 0.0452 | 0.3177 | 0.9684 |
| QMSum Committee | 31 | chunk_kv | 0.1824 | 0.0500 | 0.1084 | 0.0648 | 0.3524 | 0.9957 |
| QMSum Committee | 31 | chunk_text | 0.1792 | 0.0500 | 0.1069 | 0.0638 | 0.3597 | 0.9960 |
| MeetingBank | 96 | chunk_kv | 0.1519 | 0.0683 | 0.1093 | 0.0657 | 0.4810 | 0.9311 |
| MeetingBank | 96 | chunk_text | 0.1595 | 0.0725 | 0.1131 | 0.0691 | 0.4761 | 0.9327 |

| Aggregation | Variant | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Support proxy |
| --- | --- | --- | --- | --- | --- |
| Pooled, 276 pairs | chunk_kv | 0.1675 | 0.0499 | 0.1026 | 0.9400 |
| Pooled, 276 pairs | chunk_text | 0.1673 | 0.0518 | 0.1029 | 0.9449 |
| Four-dataset balanced | chunk_kv | 0.1666 | 0.0468 | 0.1006 | 0.9532 |
| Four-dataset balanced | chunk_text | 0.1657 | 0.0480 | 0.1006 | 0.9566 |

KV has slightly higher ROUGE-L in AMI, ICSI, and Committee, and slightly lower ROUGE-L in MeetingBank. Neither communication path dominates all quality measures. Absolute scores should not be used alone to rank datasets because reference length, domain, and input-record granularity differ.

| Dataset | Mean reference words | Mean KV words | Mean text words | Mean KV/reference ratio | Mean text/reference ratio |
| --- | --- | --- | --- | --- | --- |
| AMI | 112.4 | 602.2 | 634.0 | 5.70 | 5.99 |
| ICSI | 95.9 | 648.1 | 670.0 | 7.17 | 7.47 |
| QMSum Committee | 140.0 | 744.3 | 778.0 | 5.71 | 6.02 |
| MeetingBank | 56.6 | 430.5 | 429.8 | 10.46 | 10.49 |

Lengths use the ASCII-token convention of the lexical scorer, not Qwen tokenizer tokens. The mean length ratio is the average of per-record ratios, not the ratio of two means. Outputs substantially exceed reference length, affecting precision and F1. This explains part of the score behavior but does not establish that every deficit is caused by length or excuse irrelevant content and omissions.

### 5.1 BERTScore rescoring results

Scoring configuration: `allenai/longformer-base-4096`, `num_layers=12`, `idf=False`, `rescale_with_baseline=False`, `batch_size=1`, `use_fast_tokenizer=False`, CUDA. The comparison is between final summaries and reference summaries, not full meeting transcripts. The implementation rejects overlong scoring inputs rather than silently truncating them. Evaluation costs are excluded from original workflow token, latency, and memory measurements.

| Dataset | Completed pairs | Variant | BERTScore P | BERTScore R | BERTScore F1 |
| --- | --- | --- | --- | --- | --- |
| AMI | 100 | chunk_kv | 0.7747 | 0.8887 | 0.8277 |
| AMI | 100 | chunk_text | 0.7760 | 0.8895 | 0.8288 |
| ICSI | 49 | chunk_kv | 0.7743 | 0.8873 | 0.8269 |
| ICSI | 49 | chunk_text | 0.7729 | 0.8875 | 0.8262 |
| QMSum Committee | 31 | chunk_kv | 0.7924 | 0.8939 | 0.8400 |
| QMSum Committee | 31 | chunk_text | 0.7938 | 0.8927 | 0.8402 |
| MeetingBank | 96 | chunk_kv | 0.7772 | 0.8906 | 0.8297 |
| MeetingBank | 96 | chunk_text | 0.7806 | 0.8915 | 0.8320 |
| Pooled estimate | 276 | chunk_kv | ≈0.7775 | ≈0.8897 | ≈0.8296 |
| Pooled estimate | 276 | chunk_text | ≈0.7790 | ≈0.8902 | ≈0.8307 |

| Dataset | ΔP (KV−text) | ΔR (KV−text) | ΔF1 (KV−text) |
| --- | --- | --- | --- |
| AMI | −0.0013 | −0.0008 | −0.0011 |
| ICSI | +0.0014 | −0.0002 | +0.0007 |
| QMSum Committee | −0.0014 | +0.0012 | −0.0002 |
| MeetingBank | −0.0034 | −0.0009 | −0.0023 |
| Pooled estimate | ≈−0.001563 | ≈−0.000504 | ≈−0.001097 |

Pooled estimates use `Σ(n_dataset × dataset_mean) / 276`. Dataset means were supplied as four-decimal terminal output, so pooled values and differences are approximate rather than exports at original precision. Pooled F1 averages per-record F1; it must not be replaced by the harmonic mean of the displayed average P/R. Before further rounding, the F1 estimates from the supplied means are approximately 0.829635 and 0.830732.

Interpretation: KV has slightly higher F1 in ICSI and slightly lower F1 in the other three datasets; absolute differences do not exceed 0.0023 at the displayed precision. Together with the similar overall ROUGE-L, this supports a descriptive quality–efficiency trade-off: fewer logical token events and lower mean latency accompany small reference semantic-similarity differences. P below R is consistent with longer generated summaries, but P/R alone does not establish the cause, and BERTScore recall is not important-fact coverage.

Per-record BERTScore results and paired intervals from this rescore are not yet available locally. No new statistical-significance, equivalence, or non-inferiority conclusion is drawn, and confidence intervals are not constructed from the eight dataset means. The server report’s `paired_quality_differences.bertscore_f1` in `evaluation/<timestamp>/summary.json` can be inspected without regenerating summaries.

| Dataset | Server scoring directory relative to data root |
| --- | --- |
| AMI | `ami/20261119_083659_646986/evaluation/20261122_222254_692521` |
| ICSI | `icsi/20261119_174529_255779/evaluation/20261122_222329_027442` |
| QMSum Committee | `qmsum_committee/20261120_020952_214447/evaluation/20261122_222355_822880` |
| MeetingBank | `meetingbank/20261120_073025_968960/evaluation/20261122_222426_256504` |

## 6. Paired differences and uncertainty

| Dataset | n | Mean ΔROUGE-L (KV−text) | 95% bootstrap interval | KV higher / tied / lower |
| --- | --- | --- | --- | --- |
| AMI | 100 | 0.002113 | [-0.000073, 0.004227] | 53 / 0 / 47 |
| ICSI | 49 | 0.000473 | [-0.001920, 0.003107] | 25 / 0 / 24 |
| QMSum Committee | 31 | 0.001454 | [-0.004067, 0.006464] | 20 / 0 / 11 |
| MeetingBank | 96 | -0.003842 | [-0.009045, 0.001363] | 44 / 0 / 52 |
| Pooled: stratified by dataset | 276 | -0.000323 | [-0.002547, 0.001767] | 142 / 0 / 134 |

Dataset-level intervals are taken directly from the saved reports: 2,000 paired record-level bootstrap repetitions, seed 7. The pooled interval is newly calculated using 10,000 paired bootstrap repetitions stratified by dataset with successful sample counts held fixed, NumPy default_rng(7), and percentile endpoints. Resampling units are records rather than meeting clusters; intervals may be too narrow if records share meetings.

| Pooled statistic | Estimate | New 95% stratified bootstrap interval |
| --- | --- | --- |
| ΔROUGE-L (KV−text) | -0.000323 | [-0.002547, 0.001767] |
| Logical token reduction | 53.10% | [52.84%, 53.36%] |
| Pipeline latency reduction | 10.25% | [9.29%, 11.22%] |

All ROUGE-L intervals include zero. A defensible statement is that observed reference overlap is similar under this sample and metric, not that exact equivalence or lossless factual accuracy is proven. Equivalence/non-inferiority requires a prespecified margin and corresponding test. Token and latency intervals describe record-composition uncertainty, not variation across generation seeds, GPUs, models, or prompts.

## 7. Logical token consumption and its components

| Dataset | Variant | Source prefill | Probes | Local prompt | Local output | Synthesis prompt | Synthesis output | Total input | Total output | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | chunk_kv | 9,344.0 | 124.0 | 1,514.6 | 2,986.9 | 3,870.9 | 1,565.3 | 14,853.5 | 4,552.2 | 19,405.8 |
| AMI | chunk_text | 9,344.0 | 124.0 | 13,250.9 | 3,186.7 | 13,084.2 | 1,727.2 | 35,803.1 | 4,913.9 | 40,717.0 |
| ICSI | chunk_kv | 17,784.5 | 124.0 | 2,894.7 | 5,510.8 | 6,745.9 | 1,575.9 | 27,549.0 | 7,086.7 | 34,635.7 |
| ICSI | chunk_text | 17,784.5 | 124.0 | 26,284.2 | 6,024.4 | 24,760.2 | 1,695.8 | 68,952.9 | 7,720.2 | 76,673.2 |
| QMSum Committee | chunk_kv | 16,642.0 | 124.0 | 2,700.7 | 5,573.5 | 6,744.4 | 1,711.2 | 26,211.1 | 7,284.7 | 33,495.8 |
| QMSum Committee | chunk_text | 16,642.0 | 124.0 | 24,448.7 | 6,623.8 | 23,292.2 | 1,781.7 | 64,507.0 | 8,405.5 | 72,912.5 |
| MeetingBank | chunk_kv | 3,910.6 | 124.0 | 657.6 | 1,411.0 | 2,041.7 | 1,299.3 | 6,734.0 | 2,710.2 | 9,444.2 |
| MeetingBank | chunk_text | 3,910.6 | 124.0 | 6,094.2 | 1,591.5 | 5,876.2 | 1,307.6 | 16,004.9 | 2,899.1 | 18,904.1 |

| Aggregation | Variant | Mean total input | Mean total output | Mean total tokens | Sum total tokens |
| --- | --- | --- | --- | --- | --- |
| 276 successful pairs | chunk_kv | 15,558.9 | 4,668.4 | 20,227.3 | 5,582,738 |
| 276 successful pairs | chunk_text | 38,026.1 | 5,103.5 | 43,129.6 | 11,903,767 |

| Dataset | KV total/record | Text total/record | Reduction/record | Reduction |
| --- | --- | --- | --- | --- |
| AMI | 19,405.8 | 40,717.0 | 21,311.3 | 52.34% |
| ICSI | 34,635.7 | 76,673.2 | 42,037.5 | 54.83% |
| QMSum Committee | 33,495.8 | 72,912.5 | 39,416.7 | 54.06% |
| MeetingBank | 9,444.2 | 18,904.1 | 9,459.9 | 50.04% |
| Pooled | 20,227.3 | 43,129.6 | 22,902.3 | 53.10% |

The main accounting difference is avoiding repeated fresh text-token input for source evidence in local and synthesis stages. Local reports and final answers are still generated in natural language, and KV attention, cache copying, and full-source prefill remain costly. Thus, the 53.10% logical-token reduction is not a 53.10% compute reduction or end-to-end speedup. Prefill/probes are charged once to each standalone workflow; adding both table columns does not represent actual paired-run GPU work because shared preparation is reused.

| Dataset | Separate embedding encoder tokens/record | Embedding/ranking seconds/record |
| --- | --- | --- |
| AMI | 9,826.0 | 2.246 |
| ICSI | 18,444.5 | 2.982 |
| QMSum Committee | 16,768.0 | 1.019 |
| MeetingBank | 3,899.9 | 0.362 |

Embedding time is already included in pipeline time and should not be added again. Both paths use the shared semantic-routing result.

## 8. Latency and memory

| Dataset | Variant | Mean sec | Median sec | P95 sec | Local specialist sec | Final synthesis sec | CUDA peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | chunk_kv | 159.19 | 160.80 | 225.65 | 96.57 | 56.84 | 20,467.7 |
| AMI | chunk_text | 174.96 | 174.08 | 253.86 | 104.49 | 64.39 | 21,260.4 |
| ICSI | chunk_kv | 249.27 | 250.65 | 324.93 | 168.43 | 70.98 | 24,062.7 |
| ICSI | chunk_text | 277.92 | 276.87 | 350.60 | 187.71 | 79.57 | 25,280.0 |
| QMSum Committee | chunk_kv | 248.84 | 240.52 | 344.57 | 167.08 | 74.43 | 23,505.3 |
| QMSum Committee | chunk_text | 291.74 | 273.40 | 439.37 | 202.22 | 81.39 | 24,763.0 |
| MeetingBank | chunk_kv | 93.22 | 72.29 | 238.37 | 46.02 | 45.00 | 18,446.9 |
| MeetingBank | chunk_text | 101.60 | 71.99 | 293.93 | 53.48 | 45.78 | 18,760.9 |

| Dataset | Latency reduction | Speedup text/KV | Peak-memory difference MiB | Peak reduction |
| --- | --- | --- | --- | --- |
| AMI | 9.02% | 1.099× | 792.7 | 3.73% |
| ICSI | 10.31% | 1.115× | 1,217.3 | 4.82% |
| QMSum Committee | 14.70% | 1.172× | 1,257.6 | 5.08% |
| MeetingBank | 8.25% | 1.090× | 314.0 | 1.67% |

Under the fixed-order, same-device protocol, KV has lower mean latency in all four datasets. Memory savings are much smaller than logical-token savings because the full source cache and models remain resident. MiB=2^20 bytes and GiB=2^30 bytes; these are not whole-process nvidia-smi peaks. Model loading is excluded. Local-report length, final reasoning, and attention length all influence latency, so token counts and seconds are not proportional.

## 9. Input-length distribution and grouped results

| Dataset | All records: min/median/max tokens | Completed: min/median/max tokens | Mean target/actual specialists |
| --- | --- | --- | --- |
| AMI | 1,537 / 9,488 / 20,479 | 1,537 / 9,488 / 20,479 | 9.08 / 9.08 |
| ICSI | 7,822 / 18,539 / 35,694 | 7,822 / 18,012 / 26,597 | 17.29 / 17.29 |
| QMSum Committee | 10,706 / 16,700 / 31,084 | 10,706 / 16,199 / 26,175 | 16.13 / 16.13 |
| MeetingBank | 244 / 2,046 / 82,061 | 244 / 1,824 / 27,315 | 3.95 / 3.95 |

Length bins are left-closed and right-open: [0,4000), [4000,8000), [8000,12000), [12000,16000), and [16000,+∞). Counts use raw full-transcript tokenizer tokens, not prefill size or generated-summary length.

| Dataset | Length group | Planned | Completed | Excluded | KV R-L | Text R-L | KV sec | Text sec | KV tokens | Text tokens | Token reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | 0-4000 | 9 | 9 | 0 | 0.1132 | 0.1093 | 82.4 | 87.0 | 8,092.9 | 14,962.3 | 45.91% |
| AMI | 4000-8000 | 31 | 31 | 0 | 0.1052 | 0.1042 | 122.0 | 135.5 | 14,016.6 | 28,609.7 | 51.01% |
| AMI | 8000-12000 | 38 | 38 | 0 | 0.1061 | 0.1044 | 171.5 | 186.8 | 21,192.7 | 44,857.0 | 52.76% |
| AMI | 12000-16000 | 20 | 20 | 0 | 0.1033 | 0.0990 | 215.5 | 239.2 | 27,556.2 | 58,959.5 | 53.26% |
| AMI | 16000+ | 2 | 2 | 0 | 0.1036 | 0.1056 | 285.1 | 315.4 | 38,390.0 | 83,192.5 | 53.85% |
| ICSI | 0-4000 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ICSI | 4000-8000 | 1 | 1 | 0 | 0.0707 | 0.0676 | 131.6 | 140.4 | 16,808.0 | 34,873.0 | 51.80% |
| ICSI | 8000-12000 | 5 | 5 | 0 | 0.0748 | 0.0756 | 166.4 | 185.9 | 21,969.2 | 47,998.2 | 54.23% |
| ICSI | 12000-16000 | 9 | 9 | 0 | 0.0760 | 0.0784 | 213.4 | 242.4 | 28,277.3 | 61,850.6 | 54.28% |
| ICSI | 16000+ | 44 | 34 | 10 | 0.0806 | 0.0793 | 274.4 | 304.9 | 38,705.9 | 86,043.1 | 55.02% |
| QMSum Committee | 0-4000 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| QMSum Committee | 4000-8000 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| QMSum Committee | 8000-12000 | 6 | 6 | 0 | 0.1239 | 0.1287 | 187.9 | 208.3 | 23,992.3 | 49,751.5 | 51.78% |
| QMSum Committee | 12000-16000 | 9 | 9 | 0 | 0.1128 | 0.1082 | 213.2 | 246.8 | 28,122.4 | 61,133.9 | 54.00% |
| QMSum Committee | 16000+ | 21 | 16 | 5 | 0.1001 | 0.0980 | 291.8 | 348.3 | 40,082.1 | 88,223.4 | 54.57% |
| MeetingBank | 0-4000 | 69 | 67 | 2 | 0.1283 | 0.1338 | 59.8 | 59.9 | 4,741.4 | 7,867.7 | 39.73% |
| MeetingBank | 4000-8000 | 17 | 17 | 0 | 0.0644 | 0.0628 | 123.7 | 135.5 | 13,472.1 | 28,092.1 | 52.04% |
| MeetingBank | 8000-12000 | 3 | 3 | 0 | 0.0432 | 0.0393 | 172.3 | 219.3 | 20,231.3 | 44,904.7 | 54.95% |
| MeetingBank | 12000-16000 | 5 | 5 | 0 | 0.0889 | 0.0953 | 224.9 | 261.6 | 27,491.2 | 60,760.8 | 54.76% |
| MeetingBank | 16000+ | 6 | 4 | 2 | 0.0571 | 0.0579 | 300.3 | 367.3 | 40,447.8 | 92,893.5 | 56.46% |

| Pooled length group | Planned | Completed | Excluded | KV R-L | Text R-L | Token reduction | Latency reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-4000 | 78 | 76 | 2 | 0.1265 | 0.1309 | 40.99% | 1.09% |
| 4000-8000 | 49 | 49 | 0 | 0.0904 | 0.0891 | 51.38% | 9.49% |
| 8000-12000 | 52 | 52 | 0 | 0.1015 | 0.1007 | 52.91% | 9.49% |
| 12000-16000 | 43 | 43 | 0 | 0.0979 | 0.0962 | 53.81% | 11.63% |
| 16000+ | 73 | 56 | 17 | 0.0853 | 0.0841 | 54.96% | 12.58% |

Among completed pairs, pooled logical-token reduction rises from about 41% below 4k to about 55% at 16k+, while latency reduction rises from about 1% to about 13%. This is consistent with higher repeated-encoding cost for longer evidence, but cannot be attributed solely to length: MeetingBank dominates the shortest group, while ICSI/Committee dominate the longest. Interpret within-dataset groups first. Small cells, such as AMI 16k+ with only two pairs, do not support strong claims. No length-bin ROUGE result alone demonstrates that attention failures have been solved.

## 10. Evidence retention, repetition, and logical payload

| Dataset | Mean retained fraction | Min/max | KV sender duplicate | Text sender duplicate | KV final duplicate | Text final duplicate |
| --- | --- | --- | --- | --- | --- | --- |
| AMI | 94.65% | 72.87% / 99.79% | 7/100 | 4/100 | 9/100 | 11/100 |
| ICSI | 98.11% | 94.52% / 100.00% | 12/49 | 13/49 | 8/49 | 6/49 |
| QMSum Committee | 92.52% | 84.39% / 99.18% | 5/31 | 4/31 | 2/31 | 3/31 |
| MeetingBank | 91.34% | 31.68% / 100.00% | 11/96 | 15/96 | 0/96 | 1/96 |

Exact duplicate bullets occur in 19/276 KV final summaries and 21/276 text final summaries. Corresponding concatenated local-report counts are 35/276 and 36/276. These are descriptive counts, not a full measure of semantic redundancy or evidence of a statistically established difference. Handoff/receiver cap-hit counts are zero in both paths for completed pairs; excluded failures are not implied to have terminated normally.

Although role budget=0.25, the final retained fraction is approximately 91–98% because multiple selectors, complete blocks, and neighbor expansion are combined. This is the observed protocol; final context compression must not be labeled as 25% retention. The results mainly support contextual-cache reuse rather than independently demonstrating aggressive evidence filtering. A semantic-routing ablation must report retained fractions to avoid attributing differences in evidence volume to semantic scoring.

| Dataset | KV logical payload GiB/record | Text logical payload KiB/record |
| --- | --- | --- |
| AMI | 2.811 | 98.2 |
| ICSI | 5.586 | 172.5 |
| QMSum Committee | 5.087 | 199.5 |
| MeetingBank | 1.239 | 46.4 |

Payload sums logical deliveries across stages and can count the same KV row repeatedly. It is neither unique resident memory nor measured transmission. KV tensors are much larger than text, so the token table cannot justify a network-bandwidth saving. The measured implementation primarily concerns same-model local cache reuse; cross-machine benefits require separate measurement.

## 11. Excluded-record accounting

Following the chosen analysis policy, failures are not rerun and do not enter successful-pair quality or efficiency means. The following provides transparent record accounting only; no zero scores are assigned, and paired checkpoints are not used to infer complete per-variant success rates.

| Exclusion reason | Records |
| --- | --- |
| CUDA OOM | 8 |
| Source exceeds model context | 1 |
| Stage prompt leaves no generation room | 6 |
| Context exhausted before EOS | 4 |
| Total | 19 |

There are 10 excluded ICSI records, five Committee records, four MeetingBank records, and none in AMI. Long-input failures favor feasible lengths in the successful set, so the 16k+ results do not generalize unconditionally to arbitrary input lengths. Two MeetingBank generation-exhaustion failures have only 549 and 3,378 input tokens, showing that not every failure is caused by an excessively long source. No additional repair experiment is performed in this report.

## 12. Three refined research questions

Retain the submitted title: **Design and Evaluation of a Semantic-Assisted Latent Communication Workflow for Long-Form Meeting Summarization**. The three questions address communication, semantic evidence selection, and workflow structure separately, avoiding attribution of every improvement to latent communication.

### RQ1 — Communication mechanism

How does KV-cache-based latent communication affect summary quality, logical token consumption, execution latency, and memory usage compared with natural-language evidence transfer in long-form meeting summarization workflows?

The current experiment directly supplies the main evidence for this question. “How does” avoids presupposing superior quality or efficiency. Define the text control as verbatim natural-language transfer of the same selected evidence. Conclusions remain workflow-level because contextualized cache histories differ.

### RQ2 — Semantic routing

How does semantic-similarity-assisted evidence selection, compared with attention-only routing, affect important-information coverage and summary quality across meeting lengths, and what additional cost does it introduce?

Avoid phrasing this as “solving attention problems.” Embedding assistance changes routing scores and available evidence rather than directly repairing the generator’s internal attention mechanism. Both current paths enable it, so RQ2 remains unanswered. Measure coverage using a small source-based annotation or a defined important-fact checklist; ROUGE recall is not complete factual coverage.

### RQ3 — Workflow structure

Given the same selected evidence and communication mechanism, how does a chronological chunk-specialist–synthesis workflow compare with single-agent summarization in preserving important details and final decision states, reducing redundancy, and balancing quality against efficiency?

This is the natural third question after the move from one-step summarization to chunk specialists. It tests the value of local analysis followed by synthesis without assuming superiority over a single agent. No matched single-agent condition exists in the current data; historical runs with different windows or prompts cannot substitute for it. A separate timeline/decision module is unnecessary: evaluate whether the final summary preserves the last supported decision state.

| Question | Suggested contrast | Hold fixed | Primary measures | Status |
| --- | --- | --- | --- | --- |
| RQ1 | chunk_kv vs chunk_text | Same evidence, windows, model, stage instructions, and decoding settings | ROUGE, BERTScore, human quality; tokens, latency, memory | 276 pairs; BERTScore summaries added, manual diagnostic evaluation pending |
| RQ2 | Attention-only vs attention + semantic | Fix chunk_kv, budget, neighbors, window rule; report actual retention | Important-information coverage, ROUGE/BERTScore, embedding overhead, retention | Ablation not yet run |
| RQ3 | Single agent vs chunk specialists + synthesis | Same selected KV, model, final format and final decoding; report added generation cost | Factuality/final state, coverage, redundancy, tokens, latency | Matched baseline not yet run |

To attribute RQ2 effects to selection quality, check comparable final evidence volume as well as identical seed budgets; near-full retention may mask semantic assistance. First report the natural-expansion ablation and, if needed, add a matched-retention contrast. RQ3 inherently adds calls and outputs; report that cost instead of claiming identical compute budgets. A full 2×2×2 design is not necessary initially: run focused one-factor contrasts around the fixed KV workflow.

## 13. Thesis-ready interpretation and remaining work

**Suggested wording for Results/Discussion:**

> Across 276 successfully completed pairs from four meeting domains/datasets, the KV-cache-based chunked workflow reduced mean logical decoder token events by 53.10% and mean pipeline latency by 10.25% relative to verbatim text transfer. Mean ROUGE-L was 0.10256 versus 0.10288, with a paired interval spanning zero. This indicates similar observed reference overlap in this experiment, without establishing semantic equivalence or lossless factual accuracy. Benefits primarily arise from reusing contextualized source caches and are more pronounced in longer-input groups. Findings are conditional on successful records, the current model, and the single-device execution protocol.

> After BERTScore rescoring, the sample-weighted overall F1 estimated from the displayed dataset means is approximately 0.8296 for KV and 0.8307 for text, a difference of approximately −0.0011. This further supports similar observed reference semantic similarity. Statistical inference using per-record BERTScore intervals remains pending, and semantic similarity is not interpreted as factual accuracy.

| Ready to write | Pending evidence or qualification |
| --- | --- |
| Methods, implementation, and protocol | Fully specify both paths, sampling, final thinking, and windowing |
| Experiment 1 ROUGE, BERTScore, and efficiency results | BERTScore summaries added; import original-precision scores and paired intervals later, without regeneration |
| Small-scale human-analysis design | Randomly select five pairs per dataset: 20 pairs/40 summaries; hide method labels; rate factuality, coverage, conciseness and major errors |
| Single-rater limitation | Author-only evaluation is acceptable if disclosed; do not report inter-rater agreement |
| Semantic-assistance contribution | Requires the RQ2 contrast; not inferred from Experiment 1 |
| Chunk-specialist contribution | Requires the RQ3 matched single-agent baseline |

Shared limitations include one generation-seed configuration, one model/GPU, fixed variant order, complete-case analysis, domain/length confounding, output/reference format and length mismatch, an explicitly defined custom ROUGE implementation, and the absence of current human factuality ratings. Do not combine the RQs into a weighted overall score; present quality, resource cost, and limitations separately. Historical FRAME, topic-specialist, and global-KV iterations can motivate design choices, but are not formal superiority evidence without matched protocols.

## Appendix A. Excluded-record index

| Dataset | Record ID | Raw tokens | Category |
| --- | --- | --- | --- |
| ICSI | 13 | 30,368 | Stage prompt leaves no generation room |
| ICSI | 16 | 35,694 | Stage prompt leaves no generation room |
| ICSI | 17 | 26,086 | CUDA OOM |
| ICSI | 18 | 33,769 | Stage prompt leaves no generation room |
| ICSI | 27 | 29,229 | Context exhausted before EOS |
| ICSI | 29 | 27,839 | CUDA OOM |
| ICSI | 30 | 25,522 | CUDA OOM |
| ICSI | 36 | 27,213 | CUDA OOM |
| ICSI | 56 | 25,036 | CUDA OOM |
| ICSI | 58 | 30,010 | Context exhausted before EOS |
| QMSum Committee | 1 | 31,084 | Stage prompt leaves no generation room |
| QMSum Committee | 3 | 30,279 | Stage prompt leaves no generation room |
| QMSum Committee | 8 | 26,780 | CUDA OOM |
| QMSum Committee | 9 | 25,991 | CUDA OOM |
| QMSum Committee | 10 | 26,935 | CUDA OOM |
| MeetingBank | LongBeachCC_03192019_19-0260 | 549 | Context exhausted before EOS |
| MeetingBank | LongBeachCC_04022019_19-0262 | 82,061 | Source exceeds model context |
| MeetingBank | BostonCC_06082022_2022-0726 | 3,378 | Context exhausted before EOS |
| MeetingBank | LongBeachCC_08102021_21-0765 | 33,092 | Stage prompt leaves no generation room |

## Appendix B. Core artifacts and offline rescoring

- **AMI**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/manifest.json).

- **ICSI**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/manifest.json).

- **QMSum Committee**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/manifest.json).

- **MeetingBank**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/manifest.json).

The following server command is retained for reproducibility. All four datasets have completed the current rescore; updating this report does not require rerunning it. It reads saved summaries rather than rerunning the workflow:

```bash
for benchmark in ami icsi qmsum_committee meetingbank; do
  RUN_DIR=$(find "test_result/experiment_1_auto1024_cap100_seed12/$benchmark" \
    -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

  python -m experiment_1.rescore \
    --run-dir "$RUN_DIR" \
    --bertscore-model allenai/longformer-base-4096 \
    --bertscore-layers 12 \
    --device cuda
done
```

Record the scoring model, layer count, and evaluation configuration. Any encoder length limit should be explicit rather than silently truncating inputs. Importing `evaluation/<timestamp>/` later will allow replacement of BERTScore estimates based on four-decimal terminal values with original-precision results and paired intervals; original token/resource measurements need not be regenerated. Original ROUGE/resource differences and percentages use unrounded data; precision limits for newly added BERTScore values are documented in Section 5.1.
