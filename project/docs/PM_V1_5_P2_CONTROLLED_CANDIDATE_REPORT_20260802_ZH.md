# PM V1.5 P2 受控候选池报告（2026-08-02）

## 结论

P2 的候选构造层已经完成，且可以进入唯一一次 H1 Candidate/Gold 总审核；尚未产生 gold，
尚未训练 PM，也不能把机械通过写成“RS/MP/MS/ME 已学会”。

正式候选池采用：

> 透明受控 synthetic state construction → 共享 memory compiler / Strategy Bank →
> 正式 typed exact Rank-1 retrieval → 独立 H1 opportunity adjudication。

它不再让一个 LLM 同时负责自然语言、时间因果、四组件候选结构和标签方向。私有构造矩阵
只保证训练环境中存在需要检验的正例、硬负例、无候选和联合动作；H1 可以推翻任何构造意图。

## 为什么停止整批 LLM 原始历史生成

有界开发 smoke 的证据一致：

| 版本 | 生成器/修复 | 付费尝试 | 静态接受 | 结论 |
|---|---|---:|---:|---|
| V5 | gpt-4o-mini + 严格 validator | 31 | 10 | 大量违反联合约束 |
| V6 | 换 gpt-4o | 15 | 1 | 更大模型没有解决构造耦合 |
| V7 | 加 exact topic bridge prompt | 4 | 1 | prompt 仍不能稳定执行四套合同 |
| V8 | deterministic realizer + gpt-4o-mini surface | 13 | 13 | 0 reject，根因修复成立 |

反复失败集中在：当前消息被复制进过去对话；语义相关但没有正式 lexical/content retriever
所需的两个共同内容词；MS/ME 从同一历史编译却被要求物理上互相矛盾；RS 明确许可短语与 cue
parser 不一致。它们是数据构造责任，不是 logistic、BAAI 或 PM 学习能力的证据。

正式 P2 不需要为无关 distractor 再支付 256 次 API。受控构造的局限会在论文中明确：它证明
最小、可审计的资源机会学习环境，不代表自然对话分布；自然泛化必须由 fresh confirmation、
sealed internal、ESConv 和 EvoEmo 在冻结后检验。

## 全量机械结果

- 256 states / 256 independent synthetic users；
- FIT 128、fresh confirmation 64、sealed internal 64；
- 每个 split 16 个 construction actions 等量；
- 1024 个四组件 feature rows；
- exact Rank-1 binding rate = 1.0；
- construction realization rate = 1.0；
- response/outcome、H1 gold、external lockbox 读取均为 0；
- MP candidate present/absent = 191/65；
- MS = 247/9；ME = 247/9；RS = 128/128；
- memory exact surface 现在保存实际 rank-1 的 `candidate_age_sessions`，不再用 Top-k 年龄统计代替。

机械 preflight 产物明确标记为不能直接训练或评测。训练身份必须等待 H1 gold 与第6.4节静态门。

## H1 已冻结

主包：256 state bundles、1024 component judgments。

- 211 个 `CANDIDATE_ABSENT` 由合同确定为 OFF；
- 813 个 present exact Rank-1 需主审判断 subtype 与 on/off/abstain；
- reviewer 看不到 split、construction mode/action、候选 ID、rank score、model features 或下一条回复；
- 第二评审 overlap 为 64 states：FIT 32、confirmation 16、sealed 16；每个 construction action
  固定抽 FIT 2 / confirmation 1 / sealed 1，选择不读取任何评审结果。

H1 的任务不是心理治疗质量评分，而是证据与资源机会审核：owner/时间是否正确、是否真正增量、
是否匹配当前目标、是否冗余/冲突、功能是否允许。ME-on 仅允许过去动作/选择加观察结果或机制。

## 当前唯一下一步

1. 完成 H1 主审；
2. 由独立评审完成冻结的 64-state overlap；
3. 只裁决分歧与 abstain/flag；
4. 绑定 gold 到同一 candidate ID/text hash；
5. 执行 IAA、split、near-duplicate、nuisance/template shortcut、feature collision 和 action-cell 审计；
6. 全部通过后才进入 P3 四个 L2 logistic heads。

不允许：重新生成有利样本、用 construction intent 补 gold、把自动 judge 当标签、根据 H1 结果改
confirmation/test 文本、下载更大 embedding、重开 Strategy Bank 或再拆十几条小人评包。

## 证据位置

- 私有构造矩阵：`data/pm_v1_5_final_candidate_first_v8/private/`
- 零 API raw preflight：`outputs/pm_v1_5_p2_zero_api_mechanical_raw_v9/`
- age-bound exact Rank-1：`outputs/pm_v1_5_p2_zero_api_mechanical_rank1_v9_age_bound/`
- H1 主包与 overlap：`outputs/pm_v1_5_final_h1_candidate_gold_v1_candidate/`
- 当前机器执行合同：`data/pm_v1_5_contracts/final_execution_plan_v3.json`（V2 仅为历史）
