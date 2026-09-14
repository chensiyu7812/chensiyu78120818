# v1 审稿意见补充结果

日期：2026-09-14。结论：已补齐能可靠完成的免费统计、方法定义和 48 状态双顺序质量复评。历史需求盲审全部返回，但有引用核验失败，作为带明确限制的探索性材料报告。这些补充没有证明学习型 PM 优于便宜固定配置，支持的贡献范围更加明确。

本轮新增 API 费用上界 **$1.1634704**，包含被隔离的第一版诊断请求和全部重试；加此前记录的约 $6.3874351，累计约 **$7.5509055**。未修改原 PM、生成回复、主表或冻结指标。没有后台请求仍在运行。

## 已补什么

| 网页版建议 | 本轮处理 | 结果 / 限制 |
|---|---|---|
| Rule 规则、阈值、训练量纲、inventory、空可行集 | 按实现补全文档和英文 LaTeX | 可直接用于方法与附录 |
| PM−Context Only 配对区间 | 从旧统计文件提取六维结果 | Support、Personalization、Memory Appropriateness 的逐项 CI 均为负；原逐状态旧分数未恢复，不能声称独立重算 |
| 成本破平与风险筛选作用 | 重放 204 外部状态和 1,728 OOF 开发 cards | 外部成本破平 0 次，风险筛选改变选择 0 次；开发风险改变 97 次 |
| 资源节省 CI、三 seed 检查 | 复制为可移植脚本并重算，纳入本材料包 | 新 OFF 输入减少 20.70% / 22.03%，三个 seed 节省方向一致 |
| ME+R0 统一比较与换序 | 恢复已有报告及找到的代码，补清证据层级 | 已有 204 状态两候选评价和 40 状态双序 probe；本轮未重复收费，原始 score 文件未恢复 |
| 新重跑候选顺序敏感性 | 实际完成 48 状态、96 次评分、576 条逐候选观测 | 点估计对顺序有变化，不能宣布顺序等效或完全鲁棒 |
| 当前轮次是否需要历史 | 对全部 204 状态做回复盲的探索性标注，另加 12 技术控制 | 全部返回；首次 35 条引用不合格，重试后仍有 25 条；不冒充金标准 |
| 独立 history-required 生成 probe | 本轮未另造任务或生成回复 | 如需主张保留必需历史能力，仍须独立设计并验证证据可检索性和评分灵敏度 |

## 1. PM 对 Context Only：应放到正文的已有结果

旧 GPT-4o 六候选评价，204 状态。以下使用状态均值差及 scenario-cluster CI（34 个 scenario，每个 6 状态）。旧 user-cluster CI 是等用户权重，不能混作同一估计量；完整 CSV 同时保留两种口径。各区间未做多重比较联合校正。

| 指标 | PM−Context Only | 95% scenario CI |
|---|---:|---:|
| Emotional Support | −0.0882 | [−0.1667, −0.0147] |
| Personalization | −0.0980 | [−0.1569, −0.0392] |
| Memory Appropriateness | −0.0539 | [−0.0931, −0.0147] |
| Factual Grounding | −0.0098 | [−0.0343, +0.0147] |
| Temporal Consistency | −0.0098 | [−0.0294, +0.0098] |
| Non-Intrusiveness | −0.0245 | [−0.0539, +0.0049] |

这项比较原本已经计算，缺的是充分呈现。既有证据没有显示结构化资源注入优于 Context Only。相对 Rule / Fixed 的资源减少，不能扩大解释成 PM 在这些外部轮次上整体最划算。

文件：[逐项结果](offline/pm_vs_context_only.csv)、[来源与口径](offline/pm_vs_context_only_existing_cis.json)。

## 2. 成本项和风险项究竟起了什么作用

外部完整状态哈希与原计划逐项匹配，PM 和 Rule 的 204 次选择全部复现。

| 项目 | 外部 204 状态 | 开发 OOF 1,728 cards |
|---|---:|---:|
| 最高回复分并列 | 0 | 0 |
| 成本项改变最终动作 | 0 | 0 |
| 风险筛选删除过候选的状态 | 200 | 170 |
| 移除风险筛选后最终选择改变 | 0 | 97 |
| omission / strategy 阈值删除动作 | 0 / 0 | 0 / 0 |
| fallback | 0 | 0 |
| RS 开启 | 192/204 | 1,287/1,728 |

外部所有状态均有完整 16 个合法动作，最终选择与直接取 response ranker 最高分相同。因此在这 204 个状态上，不能将节省归因于实证起作用的成本破平或风险约束。开发 OOF 重放使用最终 consensus 阈值，是机制诊断，不是新的独立性能估计，也不等于每折旧 selection 的汇总。

文件：[外部汇总](offline/external_selector_summary.json)、[开发汇总](offline/development_selector_summary.json)、[逐动作预测](offline/external_selector_replay.jsonl)。

## 3. 新重跑的实际换序结果

抽样不看分数：每个 seed×turn 分层 8 状态，共 48，覆盖 18 用户、34 场景。保留原来的六个回复、评分维度、judge snapshot 和提示内容；同批次分别评原顺序及严格反序。所有正式采用的 96 次请求通过完整性与传输对应校验。

以下为 **Overall**，不是旧 V4 的 Emotional Support 别名。CI 按用户聚类 10,000 次、种子 20260914，保持状态均值；完整文件也包含 scenario 聚类敏感性和其余六个指标。

| 比较 | 原顺序分差 | 反序分差 | 双序平均分差 | 双序平均的 95% 用户 CI |
|---|---:|---:|---:|---:|
| PM OFF−Rule OFF | +0.104 | −0.042 | +0.031 | [−0.061, +0.170] |
| PM OFF−Fixed OFF | +0.146 | 0.000 | +0.073 | [−0.051, +0.225] |
| PM ON−Rule ON | −0.063 | +0.021 | −0.021 | [−0.092, +0.050] |
| PM ON−Fixed ON | −0.063 | 0.000 | −0.031 | [−0.109, +0.044] |

OFF 两个比较的“反序分差−原序分差”均为 −0.146；对应用户 CI 分别为 [−0.326, 0.000] 和 [−0.333, +0.040]。结果显示点估计的顺序敏感性，不能解释为已通过等效或稳定排名检验；双序平均分差仍小且区间较宽。该 48 状态结果作为单独诊断，不替换 204 状态主表，也不验证旧 GPT-4o 六策略表。

文件：[全部逐候选分数](results/quality_scores.csv)、[所有差值和区间](results/order_intervals.csv)、[各方法均值](results/order_score_means.csv)。

## 4. 历史需求与遗漏：补充有用，但仍有缺口

首次完整返回的 204 个标签中：history helpful 169、not needed 35、necessary 0、uncertain 0。这里 necessary 的定义很严格：回答明确回忆问题或未解决的历史指代时，缺少当前窗口外的特定事实就无法给出具体回答。不能把 0 necessary 写成“历史没有用”。

标注时不提供回复、方法、分数或资源计数，当前输入使用生成器可见的最近 8 条消息，排除 evaluator-only current_topic；仍保留用于审查的已授权历史。6 个明确回忆问题控制及 6 个信息已复述控制全部通过。

但首次 35 条历史引用无法在历史字符串中逐字定位；严格按冻结规则重试后，25 个状态仍没有引用合格的结果。首次全部标签、核验标志及重试后的可用状态均已保留，不删除困难状态，不通过挑选重试结果调整首次比例。因此，这只是探索性 LLM 标注，不是已验证的“记忆需求比例”。

134 个 helpful 标签首次提供了逐字有效的历史引用；其中 102 个引用能在部署 memory backend 中字面定位，在实际选取的 PM OFF / Rule OFF / Fixed OFF memory 中可定位的分别为 14 / 18 / 18。这个检查只验证引用的字面覆盖，不能当作语义召回率或已证实的充分证据。它说明还不能假定“全开 Fixed 一定拿到了 judge 认为有帮助的历史”，继而宣称 PM 保留了同等的历史利用能力。

新 Table III 已有独立 omission 评分：OFF 的 PM / Rule / Fixed 分别有 7 / 10 / 5 个 clear omission，ON 为 9 / 4 / 6，分母均为 204。但其 judge 的 authorized context 包括 current_topic 的情绪、身体状态等评价信息。因此该指标不是纯跨会话历史召回分数，也不能把所有遗漏都归因于 PM 没有打开某种 memory。

文件：[所有首次标签与核验标志](results/history_need_labels.csv)、[控制结果](results/history_controls.json)、[信息需求汇总](results/history_summary.json)、[已有遗漏指标检查](results/existing_omission_measure_audit.json)。

## 5. 已有 ME+R0 诊断怎么用

本地 origin/main 的旧报告记录：204 状态下，PM−ME+R0 的 Support 为 −0.172，所报 CI [−0.250, −0.093]。另有 40 状态、80 次 forced-swap，双序平均 Support 差 −0.150，单位 bootstrap CI [−0.250, −0.050]。已有反向证据应在实验部分呈现，不必为获得另一种结论重复跑同一实验。

本次恢复了报告和找到的脚本，但没找到旧原始 score/judgment 文件，不能独立重算。还有一个复现性问题：报告称“顺序不一致都归 tie”，可恢复代码却在一边 tie、另一边有胜者时采用胜者，只对相反胜者记冲突。因此不宜把其“order disagreement=1”解读为全部两次标签只有一次不同，也不能称其所有偏好票数已被本轮核验。

这不自动推翻所报双序平均分差；它要求把数字的来源层级、两候选协议和合并规则说清楚。材料已保存在 [已有诊断证据](offline/existing_cost_matched_evidence/PROVENANCE.json)。

## 6. 论文落地材料

- [中文方法和措辞说明](../../analysis/reviewer_supplement_20260914/METHODS_AND_CLAIMS_CN.md)：Rule、9 inventories、开发量纲、风险目标、空集合、ME 构造、评分名称、主张边界。
- [英文 LaTeX 段落和表格](../../analysis/reviewer_supplement_20260914/PAPER_INSERTS_EN.tex)：可按标注位置加入论文。
- [英文补充材料 PDF 预览](paper_supplement_preview.pdf)：已编译检查，便于通读；不是对最新 Overleaf 稿的直接修改。
- [资源区间与三 seed 统计](offline/resources/resource_cis.json)、[seed 检查](offline/resources/seed_checks.json)。旧 1,020-turn 均值和 204 状态区间分开标注。

论文现在最稳妥的主张是：一个可部署的检索前来源选择器，在受控合成开发中表现出可学习信号，且在固定外部输入上相对指定高资源结构化策略减少输入。当前证据仍未证明它优于 Context Only、ME+R0，或保留了经过独立验证的必需历史能力。无需为这篇冻结版本增加新模块；若未来要主张后两项，就应另设有可检索证据和已验证评分的 history-required probe。

## 执行与复核

第一版新增请求在反序时连带排序了 JSON 字段，并让历史需求 judge 看到多于生成器的会话窗口，不符合本轮目标；该版结果和 $0.6140534 费用全部保留，未混入正式质量结果。第二版保留候选之外的提示字节顺序，对齐最近 8 条消息，重新固定请求。两版原请求、返回、费用 ledger、冻结哈希都在本目录。

正式补充 freeze：`9b18325b0194661fd12de02900b16d718152383d8f1f264d6f2512314c374912`。质量 96 次全部有效；历史引用剩余 25 条明确标为未通过，未额外放宽阈值或继续收费重试。

[费用与完成记录](results/receipt.json)、[分析文件校验和](results/analysis_manifest.json)。在 `project` 目录下，用本轮虚拟环境的 Python、`PYTHONNOUSERSITE=1 PYTHONPATH=src` 执行 `analysis/reviewer_supplement_20260914/verify_supplement.py` 即可离线复核。执行或重算均不要求重新提交 API。
