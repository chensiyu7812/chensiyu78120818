# PM V1.5 V5.3 P2R：Leader 物化与静态审计报告

日期：2026-08-07
阶段：P2R 候选与可学习输入蓝图完成；等待独立 Worker 复核；尚未冻结 split、尚未生成 paired effect、尚未训练 PM。

## 一、结论

W7R 修复了若干局部题面错误，但仍不是可直接训练的数据。根本缺口是：它没有把“纵向用户历史、真实 Rank-1 候选、运行时可见特征、反事实效应标签、用户簇级切分”分成不同层。本轮用新的 P2R 蓝图替代该方法层，而不是继续在 W7R 上补模板。

新蓝图第一轮静态审计确认：868 个组件状态、381 个反事实组、44 个用户簇；未来/当前会话泄漏、gold/结果特征泄漏、split 泄漏、外部 exact/规范化 8-gram 同源、跨用户逐字重复均为 0；12 个多组件状态均具备 MP+MS+ME+RS 四类候选。随后 W9 独立复核发现，候选“存在”不等于候选“话题正确”：12 个交互中有 4 个的 MS 或 ME Rank-1 来自其他话题。因此本蓝图现状应记为 **STATIC_BLOCKED**，不能冻结 split 或进入联合效应生成；其余单组件结构结论不受影响。

## 二、从三个外部实验反推的训练要求

### ESConv

ESConv 主要测当前回合的支持策略、请求适配和低负担边界。它要求 RS 的机会判断来自自然支持对话，而不是“请执行某张卡”的人工命令句；同时要求 M0+R0 是真实可选动作，不能默认必须注入资源。

### EvoEmo

EvoEmo 提供多会话、多人和不断演化的用户状态。它要求：

- 候选池只含当前状态之前的会话，并保留 owner、session、版本和来源；
- MP 区分稳定 profile、回复偏好及失效版本，不能靠用户在当前话语里复述隐藏值才识别适用性；
- MS 在完整严格过去目录中检索，而非单候选题；
- ME 面对高密度同主题干扰项，并执行 exact Rank-1 compile-or-off，不能 Rank-1 失败后偷升 Rank-2；
- 同一用户的多个状态必须留在同一 split，避免把“记住这个用户”误当泛化；
- Step2 必须吸收并自然使用授权证据，而非 V5.2 的机械尾部拼接。

### ES-MemEval

ES-MemEval 主要检验跨会话事实、时间、冲突、拒答和 QA 检索。它要求完整历史目录、严格过去边界、owner/time/version 可追溯以及无证据时 abstain。它不能单独证明 MP/ME/RS 的回复边际价值，因此 QA 客观指标与 response QRC 分开报告。

因此，过去训练数据真正缺的是纵向目录深度、每态竞争候选、跨用户簇泛化、自然语义变化和真实 ON/OFF 效应，而不是再增加十几条孤立人评题。

## 三、本轮实现

### 1. 四层严格分责

- `longitudinal catalog`：存 owner、session、版本、subtype 和原始证据；
- `current state`：只存当前可见对话、当前目标/边界及运行时元数据；
- `Rank-1 lineage`：记录真实候选池、真实选择器和 exact Rank-1；
- `Step1 model input`：只允许推理时可见的低容量特征，不允许 state condition、gold、质量、risk 或 generator outcome。

`counterfactual_group_id` 与 `split_group_key` 已分开：前者配对同一构念条件，后者把同一用户全部状态绑定在同一 split。

### 2. 候选链路

- MP：按结构化 `field_type/topic scope/active version` 选择；回复偏好可在用户没有复述偏好时仍作为候选。当前消息是否含隐藏 profile 值不再是候选资格前提。
- MS：使用 BGE-M3 对完整严格过去候选池排序；增加 exact-text embedding cache，仅优化重复编码成本，不缓存排序答案或标签。
- ME：沿用 production typed lexical Rank-1，并坚持 Rank-1 compile-or-off；没有把 W6 中未胜出的 BGE/hybrid 切成默认。
- RS：使用冻结的 6-card Bank；候选层只处理机械安全/明确拒绝，语义价值留给 Step1。

### 3. 运行时贡献特征

四组件均恢复了候选存在、相关性、增量性、冗余性、边界、owner/time/version、候选角色等透明特征。ME 没有为了删除答案泄漏而把合法运行时特征一起删空。

这些特征描述“候选是什么、当前情况是什么”，不包含“打开以后质量是否更好”。后者必须来自下一阶段同状态 paired ON/OFF 效应。

## 四、规模与审计结果

- 纵向目录：20 名内部用户，每人 14–33 个会话；464 条资源；
- 当前状态：868；反事实组：381；用户簇：44；
- 组件状态：MP 322、MS 243、ME 243、RS 60；
- 四组件 interaction：12/12 均具备 MP+MS+ME+RS 候选，但 W9 复核确认其中 4/12 的 MS/ME 实际 Rank-1 话题错配；只有 8/12 可继续作为交互开发样本；
- 候选池规模：每态 15–31 条可用历史资源；
- 外部源表面：48,912；与内部当前状态 exact overlap=0，规范化 8-gram overlap=0；
- 跨用户逐字相同当前题面=0；保留的 12 组重复仅是同一 interaction 的四个组件视图；
- future/current candidate=0；gold/model-input leak=0；缺失运行时特征=0；
- 相关单测：37/37 通过；API 调用=0；未读取生成回复、质量或 risk 结果。

## 五、必须如实保留的能力边界

透明低容量观察器并不覆盖全部自然语义：

- MP 正向主信号 166/167；
- MS 正向连续性显式信号 26/89；
- ME 正向行动邀请 28/89，明确拒绝 52/77；
- RS 正向卡片前提 30/36。

未被透明规则捕获的状态没有改成负例，而是保留为 `semantic-extension/effect-only probes`。正式训练应同时报告：

1. 透明低容量 PM 能否超过 always-off、fixed-high 和 transparent-rule；
2. 增加 BAAI/BGE 语义表示后，跨自然表达和跨用户簇泛化是否改善。

这正好限定论文主张：V1.5 的目标是一个可审计、有限容量、能学到有用资源开关的 PM；不声称它独立解决完整自然语言理解。

## 六、尚未完成与固定顺序

1. Worker 对本轮产物做独立、只读审计；
2. Leader 根据审计结果只修实现/构造 bug，然后冻结用户簇级 split 和 release identity；
3. 物化同状态 ON/OFF effect 计划，锁定候选、generator、seed、提示和基线；
4. 获得付费授权后一次生成，不用结果反向修改题目；
5. 独立评估质量、grounding risk、资源做功和 token/cost，并据此生成 Step1 worth-opening 标签；
6. 分别训练透明 head 与 BGE 增强 head，做用户簇/语义族分组验证；
7. 比较 always-off、fixed-high、transparent-rule、cost-matched-fixed、learned-PM-full、learned-PM-conservative，以及必要的旧 V1.0/原始 session RAG 次表；
8. 只有内部确认完成后，才在 ESConv、EvoEmo、ES-MemEval 上运行同一冻结栈。

本轮没有冻结“神奇及格线”。V5.3 的最低有意义结果定义仍是：learned PM 相对人工透明规则表现出可学习增量；相对 fixed-high 明显降低资源与 token 成本及 grounding risk；相对 always-off 的质量不出现不可接受下降。点估计、区间、用户簇稳定性和失败边界同时报告。

## 七、W9 独立复核后的更正

W9（commit `86c164c`）独立复现了全部主要结构数字，同时纠正了两个判断：

1. 原审计的 MP `candidate_family_alignment` 把 49 条正确的偏好候选和 12 条交互 taxonomy 差异误算成 mismatch；修正后真正的 MP 跨字段错配为 16/322，而不是 77/322。
2. 交互候选完整性必须同时检查“存在、严格过去、owner/version 正确、实际 Rank-1 话题/功能一致”。旧蓝图只检查了存在性，因而漏掉 4/12 的跨话题 MS/ME。

这两点现已进入 `79l` 的机器审计。旧蓝图保留为开发证据，不作为正式 P2 effect 数据；正式数据合同改为先生成每动作 24 条交互蓝图，再在任何回复/结果产生前按静态条件筛选前 20 条合格项，最终形成 16 动作 × 20 = 320 条交互效应状态。若某动作不足 20 条，只能新增内容，不能降低条件或把 Rank-2 偷升为 Rank-1。
