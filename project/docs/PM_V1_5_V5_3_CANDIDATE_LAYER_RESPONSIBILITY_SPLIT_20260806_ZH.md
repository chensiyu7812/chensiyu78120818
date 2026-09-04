# 统一候选层分责：机械hard-off vs 语义Step1特征（2026-08-06）

## 目标

只允许机械、可由代码直接证明的事实触发hard-off：candidate absent、wrong owner、
future/stale invalid、明确当前拒绝、明确重复/已执行。像"INVITES_ACTION"、
"explicit_advice_welcome"、"RS opportunity"这类语义观察，只能作为Step1的输入特征，
不能继续替Step1做"要不要考虑这个候选"的最终裁决。UNKNOWN不得自动等于OFF。

## 审计结果：MP/MS/ME已经合规，RS是真正需要改的地方

直接读`discover_final_typed_memory_candidates()`（`v1_5_candidate_discovery.py:491`）：
MP/MS/ME的候选发现**唯一的hard-off条件是`match_level(item) > 0.0`**（词面/内容重叠地板，
机械事实）加精确文本去重。`explicit_advice_welcome`/`current_action_invitation`/
`continuity_request`全部只出现在下游`contribution_slot`函数的**输出特征**里
（`me_contribution_slots`/`mp_contribution_slots`/`ms_contribution_slots`，
`v1_5_v5_3_contribution_slot_features.py`），从未在候选发现层被用来清空候选池。
**这三个组件不需要改。**

RS是唯一的例外，而且是两层问题：

1. `QualifiedStrategyRAG.retrieve()`（`v1_5_strategy_rag_runtime.py:266`）：调用
   `eligible_moves()`，只要没有任何atomic move满足语义条件（不只是`active_high_stakes`），
   状态就是`off_no_explicit_opportunity`，候选列表直接为空——Step1从未有机会看到任何候选；
2. `effect_study_rank_applicable_cards()`（`v1_5_strategy_rag_repair.py:382`）：用
   `advice_welcome`/`listen_only`/`no_probing`这些语义判断，在排序前先筛掉不匹配的
   strategy family，同样是候选发现层的语义硬过滤，不是Step1特征。

## 修复：新增模块，不改已有冻结调用点

`src/metacom_pm/v1_5_v5_3_candidate_layer_responsibility.py`（新文件，加法式修改，
不修改`eligible_moves`/`rank_applicable_v4_cards`/`effect_study_*`任何一个既有调用点）：

- `rs_mechanical_candidate_pool()`：候选池只在两种情况清空——`active_high_stakes`
  （安全信号，见下方说明）和`explicit_stop`（属于"明确当前拒绝"这一允许分类）；
  某个move因为"上一轮已执行"被排除时，**只丢掉那一个move，不清空整个候选池**
  （对应"明确重复/已执行"这一允许分类）。其余情况下，全部6个move始终作为候选返回，
  每个候选上都附带语义特征（`explicit_advice_welcome`、`listen_only`、
  `one_low_risk_step_available`等），供Step1或transparent-rule使用，不再由这层替
  Step1决定"有没有机会"。
- `rs_transparent_rule_top1()`：一个明确标注为"transparent-rule baseline，不是learned
  Step1替代品"的排序函数，复用`eligible_moves()`原来编码的语义优先级（open expression→
  focused clarification→paraphrase→grounded validation→one low-risk step），但作用
  在**始终非空**的候选池上做排序偏好，不是清空候选池的过滤器。

8个新单元测试全部通过（`tests/test_v1_5_v5_3_candidate_layer_responsibility.py`），
覆盖：无信号时候选池仍完整返回6个（对应之前的`off_no_explicit_opportunity`情形）、
安全信号仍机械hard-off、显式拒绝仍机械hard-off、单个已执行move被单独排除而非清空全池、
语义特征正确附加到全部候选、transparent-rule排序在有/无信号时的正确行为。

## 一个明确标注、留给leader定的例外：安全信号是否算"机械"

`active_high_stakes`（自杀/自伤/暴力等信号）在这次修复里**仍然保留为机械hard-off**，
没有降级成特征——虽然计划给出的五类允许hard-off事实里没有明确写"安全重定向"这一条。
理由：这个项目在其他地方（`v1_5_strategy_rag_runtime.py`自己的`retrieve()`、
`effect_study_observable_flags()`的`ordinary_rag_hard_off`）一贯把`active_high_stakes`
当成独立于"RS机会判断"之外的安全早退机制处理，这次沿用这个惯例，而不是擅自改变。
**这是一个需要leader明确确认的设计决定，不是本次单方面定案。**

## 待leader决定、本次未擅自处理的事项

1. RS到底该用6卡`v1_5_strategy_rag_runtime`系统还是50核心/100执行卡的
   `v1_5_strategy_rag_v4`+`v1_5_strategy_rag_repair`系统，这次新增的
   `rs_mechanical_candidate_pool()`只覆盖了6卡系统，另一套系统的对应修复需要leader
   先定基调再做，避免做两遍或做错方向；
2. 安全信号是否应该从"机械hard-off"降级为"高优先级特征"，还是保持现状，需要明确决定；
3. 这次新模块是**加法式**的，还没有接入任何真实生成/训练管线——是否要把它换成真正的
   candidate discovery入口，属于P2范围的后续决定，不在本轮自行执行。
