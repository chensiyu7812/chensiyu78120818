# PM V1.5：Strategy Bank V3 扩展方案

> 2026-07-29 更新：本方案中的五族/50 卡已降级为 top-down safety reference。
> 全八类 train-only 归纳审计证明 raw wording 不形成稳定 cluster，现进入隐藏原标签的
> open coding。完成归纳 codebook 并与现有 50 卡盲对齐之前，不继续扩卡，也不把 50
> 卡称为数据浓缩结果。活跃证据见
> `docs/PM_V1_5_ESCONV_INDUCTIVE_STRATEGY_AUDIT_ZH.md`。

## 决策

接受 `80–100` 作为候选卡库的目标范围，但不把 80 当作必须凑够的数字。当前五张卡
保留为机制基线；它们不能单独承担正式 Strategy-RAG 主张。旧 11,590 张 raw-response
卡保留为来源池和历史 baseline，不直接恢复为正式卡库。

当前五个安全策略族每族目标为 8–10 个真正不同的 support submove，共 40–50 张核心
卡。其余约 30–50 张只能是会改变适用边界或生成行为的 execution/context variant。
仅改变“考试、失恋、工作”主题，或仅改变 emotion/problem 标签，不能新建一张卡。

## 为什么不是自动聚类出 100 张

对当前 6,391 条合格来源进行了 outcome-blind 的文本多样性诊断。五个策略族分别有
586–1,723 条来源、420–662 个独立对话，来源数量足够。但对 raw supporter wording 做
TF-IDF 聚类时，`k=5,8,10,12,16,20` 的 silhouette 几乎都在 `-0.10–0.04`。

这说明原始文本主要混合了话题、措辞和对话细节，没有自然呈现出“十个可靠策略
亚型”。纯聚类会把 topic cluster 错写成 technique card，不能作为卡片定义依据。

可复跑结果：
`outputs/pm_v1_5_strategy_bank_v3_diversity_profile_v1.json`

## 什么才算一张不同的卡

一张卡必须有独立的：

- support move；
- when to use；
- when not to use；
- phase/mode/goal；
- burden；
- risk constraints；
- 会实际改变 generator 行为的 prompt guidance。

同一个 move 在不同 topic 或 emotion 上使用，通常只增加 metadata 标签，不复制卡片。
只有当情境会改变允许的行为，例如“低负担只反映、不追问”与“探索阶段允许一个澄清
问题”，才可以成为不同卡。

## 来源与防作弊

卡片只允许使用已经通过来源排除的 6,391 条 ESConv train lineage：

- 不使用 support-need packet 来源；
- 不使用 test/external；
- 不使用 judge 分数、生成回复结果或 PM action outcome；
- conversation-level survey 只能描述来源，不能选择“好卡”；
- 原始 supporter response 只用于归纳和核查，不进入 generator；
- 每张卡至少需要 20 个独立来源对话支持。

## 资格流程

1. 用透明 support-move taxonomy 定义每族 8–10 个 submove。
2. 将来源 turn outcome-blind 地映射到 submove，达不到来源门槛的合并或删除。
3. 对真正改变边界/行为的 submove 增加最多必要数量的情境变体。
4. 做 normalized/semantic 去重；高相似卡必须人工解释为何不能合并。
5. 所有 surviving cards 做一次简短人工内容审核；高风险/模糊卡和固定 20% 随机样本
   再做第二审核。
6. 冻结卡库后才做 retrieval relevance 和 R0/Top-1/Top-k 响应实验。

## 与当前人评的关系

第一批五卡 R0/RS 人评继续作为“策略提示能够产生 uptake”的机制证据。第二批 12 对
暂时暂停，不把旧五卡 treatment 的确认结果冒充为新卡库证据。V3 卡库冻结后重新生成
最小确认样本，避免重复人评和 treatment 漂移。

## 当前执行状态

已按五族各 10 个 submove 构造出第一批 50 张 core cards：

`outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1/strategy_cards_v3_core_candidate.jsonl`

这些定义遵守 ESConv 父级策略边界，并采用“distinct / understandable /
identifiable”的策略扩展原则。族内 card-description BGE cosine 最大值为
`0.83–0.88`，没有超过 `0.90` 的明显重复候选。

同时用 BGE-small 对 6,391 条来源 response 做了 outcome-blind 的弱映射。只有 31/50
张达到预设的 20 个高置信独立来源对话门槛；固定样例检查显示，多个 mapping 虽有较高
cosine 但功能上明显配错。因此：

- BGE mapping 不作 submove gold；
- `31/50` 不作卡片晋级或淘汰结论；
- ESConv 仍能提供 family-level provenance；
- 细粒度 submove 定义须由内容审核决定，来源映射只作失败诊断；
- 在 core definitions 冻结前不增加 30–50 张变体。

已生成 50 卡定义与每卡 3 个固定来源样例的分离审核页面：

`outputs/pm_v1_5_strategy_bank_v3_core_review_candidate_v1/human_review.html`

审核页面不展示 problem/emotion、survey、judge、生成结果、test 或 external outcome。
卡定义质量与来源例子 fit 分开填写；来源配错不能自动否定一张理论上清楚、安全的卡。

## 跨域覆盖与统一 RAG

当前 50 张卡在五种 support mode、三种 dialogue phase 和五种 goal 上均有结构覆盖，
因此可以先停止扩卡，完成审核与检索资格赛。卡片数不再自动向 80–100 推进；只有审核
发现真实的 decision-boundary 空缺时才增加卡。

主实验的 train/calibration/internal/ESConv/EvoEmo 必须绑定同一 catalog、filters、
query、retriever、Top-k、prompt 和 generator。不同状态检索到不同卡是正常的；为
不同 external domain 更换卡库会改变 RS treatment，只能作为未来单独的
domain-adaptation 消融。

50 张卡目前只能称为“跨域通用安全技术核心候选库”。人审和冻结检索栈上的
outcome-blind opportunity/relevance audit 通过后，才能称为 V1.5 合格 Strategy
Bank。完整边界见：

`docs/PM_V1_5_SHARED_STRATEGY_RAG_AND_COVERAGE_ZH.md`
