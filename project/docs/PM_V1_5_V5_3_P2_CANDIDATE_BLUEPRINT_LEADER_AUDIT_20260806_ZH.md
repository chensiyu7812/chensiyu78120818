# V5.3 P2 候选蓝图 Leader 审计（2026-08-06）

## 结论

Worker W7（commit `78dbbf2`）证明了四项静态绑定没有漂移：6-card RS Bank、MS BGE-M3 snapshot、ME production lexical+typed-tier exact Rank-1/no-Rank2，以及静态 release identity `v53static_d45cfb6508f90030460da7b6` 均被真实调用。82 个 state ID 无重复，候选 owner/time 也没有在本轮暴露出跨用户或未来会话错误。

但 W7 **不能直接进入 P2-READY**。它是可复跑的开发蓝图，不是正式 paired-effect 训练蓝图。Leader 零 API 审计状态为：

`REPAIR_REQUIRED_BEFORE_P2_READY`

机器报告：`outputs/pm_v1_5_v5_3_p2_candidate_blueprint_audit_v1/report.json`

复跑脚本：`scripts/v1_5/75l_audit_v5_3_p2_candidate_blueprint_v1_5.py`

## 为什么不能直接生成

### 1. MP 的正例没有“隐藏增量”

16 条 MP 正例把存储的 preference/profile 内容再次写进当前用户消息；其中 14/16 已被同一特征实现判为 `current_redundant=True`。8 条 profile 正例的 `profile_goal_needs_advice_or_arrangement` 又全部为 false。

这不是 MP head 分不出来，而是题面把“需要检索命中”和“资源必须带来当前未明说的增量”混成一件事。若现在生成 ON/OFF，MP ON 要么只复述用户刚说的话，要么因强制使用无增量 profile 拖累 Step2，最终再次把 generator 的空转写进 Step1 标签。

修复不是再调词面阈值，而是：MP 使用同用户的结构化候选池；current turn 只表达需要什么响应动作或现实安排，不逐字复述 field value/preference；Step1 再判断该结构化事实是否会改变回复。

### 2. MS 没有真正测到 BGE 的外部难点

16/16 MS state 的候选目录都只有一条。BGE 在单候选池中不承担排序任务；同时 W7 没保存真实 `top1_semantic_relevance`，只保存 lexical descriptor。

因此“MS 100%有候选”不能解释成 BGE 同栈覆盖通过。W7R 必须使用严格过去、同用户、真实多候选密度的目录，并保存每个 Top-k 候选的 semantic score、Rank-1/Rank-2 margin 和 lineage。

### 3. ME 把构造答案放进了 Step1 特征

32/32 ME state 的 `step1_features` 包含 `rank1_matches_intended_target`。这是构造者知道的 target-gold，不是运行时 PM 能观察到的输入。它可保留在 `audit_only`，但不得进入 head feature matrix。

ME 的检索方法本身不重新搜索：仍使用已冻结 production exact Rank-1，编译失败即 unavailable，不晋升 Rank-2。

### 4. `current_goal` 目前是机器标签

82/82 state 的 `current_goal` 是 `positive`、`goal_mismatch`、`continuity_request`、subdomain 或 family 名，而不是从可见对话得到的自然目标表面。若 runner 把它作为输入，PM 可以学习构造标签而不是用户需要。

W7R 必须同时保存：

- `natural_current_goal`：只由可见对话得到、可给 PM/Step2；
- `construction_condition`：仅审计、永不进入模型。

### 5. 独立数据单位仍不足

82 state 表面上不少，但每个 head 只有 7–8 个 counterfactual group：MP=8、MS=8、ME=8、RS=7。多个 state 是同一模板的变体，例如 ME explicit-decline 与 goal-mismatch 两族的跨主题句式相似度均约 0.90，MS continuity-positive 约 0.87。

这不足以同时支持 fit、内容独立 confirmation 和 whole-family holdout。这里不设一个随意的固定 N；修复标准是：每个 head 的 fit 与 confirmation 均有真实不同的用户/内容组和 ON/OFF effect，且独立组数足以拟合该 head 的自由参数而不靠复制模板。实际 N 在 W7R 审计后、任何 paired outcome 前冻结。

### 6. 正式 runner 所需 lineage 不完整

79/79 单组件行缺少显式 candidate version、current surface hash 和 candidate surface hash；24/24 MP 行没有独立的 `field_type/field_value/version`；6 条 RS 把 `selection_mode` 错放进 `exact_rank1_subtype`；3 条 interaction state 只有候选 ID/available 摘要，没有完整 surface、feature、lineage，不能直接驱动公平六臂 runner。

## 同源检查

审计比较了 447 个内部可见 surface 与 48,912 个 EvoEmo/ES-MemEval/ESConv 外部 surface：

- exact overlap：0；
- normalized 8-gram：1；
- 展开后是普通话语片段，不含外部用户、事件、answer、evidence 或资源内容；已按 hash 留痕，判为非内容同源；
- 未裁决的内容性碰撞：0。

因此“RAG 同源作弊”不是当前阻断原因。

## 下一轮唯一修复：W7R

Worker 只重建数据面，不生成回复、不训练 head、不读取任何 quality/risk/outcome、不改 static retriever/Bank/executor：

1. MP 改为结构化候选池，正例 current turn 不复述候选；输出 field type/value/owner/subtype/version。
2. MS 改成真实多候选严格过去同用户目录，保存真实 BGE Top-k score/margin。
3. ME 只把 intended-target match 留在 audit-only；运行时特征中删除。
4. RS 将 atomic move/card subtype 与 `selection_mode` 分开。
5. 所有组件输出 natural goal、construction-only condition、surface hashes、candidate lineage。
6. 增加真实内容独立 group，不复制现有模板；interaction 至少覆盖 pairwise 与多组件情形，并输出完整的每组件 surface/feature/lineage。无需把 16 动作做成等频。

W7R 完成后 Worker 停止。Leader 再做一次同一审计；只有审计通过，才冻结 split/N、刷新三份权威事实源和唯一 P2 release，随后实现 formal runner 并进入一次 P3 paired generation。

## 两个 Codex 的先后顺序

```text
现在：Leader提交W7审计与W7R规范
  -> Worker仅执行W7R、提交并停止
  -> Leader重跑完整性/捷径/同源/split可行性审计
  -> 若数据面通过：Leader冻结真实N与split，刷新release identity
  -> Leader实现并dry-run唯一formal runner（零API）
  -> 同一runner物化预算/identity，等待用户付费授权
  -> 一次P3 paired generation
```

在 W7R 交付前，Leader 不提前实现绑定旧 schema 的正式 runner；Worker 也不进入 W8、生成、人评或训练。这个等待是数据依赖，不是新的性能“门”。
