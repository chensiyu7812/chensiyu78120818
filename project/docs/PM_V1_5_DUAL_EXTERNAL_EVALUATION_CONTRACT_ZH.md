# PM v1.5 双外部评测与可见状态支持冻结合同

状态：`SUPERSEDED_AS_ACTIVE_ROUTE_BY_MINIMUM_PUBLISHABLE_PROTOCOL`

> 2026-07-28 起，V1.5 活跃执行以
> `PM_V1_5_MINIMUM_PUBLISHABLE_PROTOCOL_ZH.md` 为准。本文保留双外部设计背景，
> 但“一次性消费”的代码锁、全双域同时通过和完整复杂门不再自动阻塞快速 V1.5。
> 研究纪律改为非破坏性数据角色：evaluation outcome 若参与方法选择，该批数据自动降为
> development；同版本复现/恢复允许重跑。另已确认旧 ESConv adapter 错把 corpus-level
> `situation` 暴露为 session summary，旧 2,831 条状态不得进入新训练/评测；V2 adapter
> 在 ESConv 单会话中固定 `current_session_summary=""`。

## 1. 研究问题与唯一允许的主张

同一个在合成 development states 上训练并冻结的 PM，在不读取检索条目、
不针对外部 test 重新训练或调阈值的条件下，能否：

1. 在 ESConv 单会话环境中选择性开启 Strategy RAG，并在保持即时回复质量
   的同时减少不必要的策略检索；
2. 在 EvoEmo/ES-MemEval-derived 多会话环境中选择 MP/MS/ME/Strategy，改善
   质量—风险—成本权衡。

两项是并列外部实验，不合并为一个总分。只有两项都通过各自的冻结门，才
允许使用“跨单会话策略与多会话记忆环境的资源调度”这一较宽表述。任一项
失败时，结论必须缩窄到实际通过的环境。

本研究不主张真实理解用户、真实情绪改善、临床有效性、在线 RL/POMDP 或
真实长期部署效果。

## 2. 数据角色

### 2.1 合成 development data

- 作用：训练、calibration、一次性 internal test；不作为外部有效性证据。
- 内容：使用非重合 ESConv-train dialogue 仅作风格启发；不复制 EvoEmo
  topic、session、措辞或 outcome。
- 可见状态：history 以 2/4/6/8 turns 反平衡；session summary 以
  present/absent 反平衡；两者由 user/case 的 outcome-blind 设计决定。

### 2.2 ESConv

- 使用当前 1,300-dialogue expanded release。
- 单会话 PM 只看当前用户文本与之前可见对话；dataset `situation` 只作离线 provenance，
  不得进入 PM feature、query 或 generator prompt，`current_session_summary` 固定为空。
- 当前 split 是 seed=13 的稳定哈希、dialogue-level 自定义 70/15/15，原始
  计数为 934/186/180；剔除 84 个 EvoEmo 血缘源后为 875/172/169。
- 该 split 不是原始 1,053-dialogue 版本的官方 split，论文和产物不得称其
  为 official split。
- 875 个 non-overlap train dialogues 是可用 development seed pool；其中
  52 个实际 development seed source 从 Strategy Bank 中按 dialogue ID 排除。
- Strategy Bank 只能来自其余 train dialogues；validation/test 均不得进入
  bank、seed、rule tuning 或 learned-PM tuning。
- test 的科学单位是 dialogue；turn 只作为 dialogue 内重复观测。
- 不在已经建立 Bank、seed 与 overlap lineage 后临时改成 8/1/1。当前冻结
  70/15/15 不是因为它优于所有比例，而是因为 dialogue-level assignment 在读取
  turn outcome 前已确定、non-overlap test 仍有 169 个独立 dialogue；事后重切会
  重新定义 held-out universe 并使既有 52-seed/823-source lineage 失效。
- test 中原有 2,275 个 supporter turn。主分析纳入全部拥有至少 2 个 prior-history
  turn 的决策点，不按 gold response、gold strategy 或 judge outcome 抽样；共排除
  163 个零/一历史的早期 turn，保留 2,112 states、169 dialogues。
- 对 3/5/7 或超过 8 条的 history，只保留最近且不超过原历史的最大
  2/4/6/8-turn window；不 padding、不生成不存在的对话，也不挑选 early/late
  “好看轮次”。

### 2.3 EvoEmo / ES-MemEval

- EvoEmo 与 ESConv 有已知语义/生成血缘，因此称 development-informed
  external transfer 或 ES-MemEval-derived evaluation，不称 pristine test。
- 若声称运行 ES-MemEval official task，必须逐项遵循其输入、输出、ground
  truth 和指标；自定义 dialogue-generation 评测不能自动继承 benchmark 名称。
- primary bootstrap cluster 是 user；scenario 仅作 sensitivity。

## 3. 同一个 PM 的硬定义

训练完成后一次性冻结：checkpoint、feature builder、BAAI snapshot/runtime、
Step-0、transparent rule、选择阈值、generator/retrieval/prompt/judge 与成本定义。

ESConv 只施加环境合法动作 mask：`M0+R0`、`M0+RS`。MP/MS/ME 在单会话
环境中结构性不可用。不得重新训练、校准或根据 ESConv test 结果选择算法。

旧 `src/metacom_pm/esconv.py` 的 `PMModel/LearnedPMPolicy` 路径不是 V1.5
主实验。V1.5 必须通过 `src/metacom_pm/esconv_v1_5.py`，使用 PMV2Model、
同一 BAAI/Step-0 合同和训练报告绑定的 checkpoint。

## 4. 可见状态支持而非 benchmark 仿写

训练与外部不要求文字分布相同，只要求外部 observable state 落在训练支持
范围，并报告剩余 shift：

- history length；
- summary presence；
- current-user token range；
- BAAI centroid/radius 与 semantic OOD；
- Step-0 source/strategy similarity；
- inventory count/age/token；
- severe-OOD/no-feasible fallback。

外部 outcome 不得用于扩大支持范围、调 threshold 或重训。EvoEmo 的具体
文本、topic、related session 和标注不得进入 development generation。

## 5. ESConv 指标层级

主比较：learned PM、同 Step-0 transparent rule、always R0、always RS。

主 outcome：blind pairwise immediate-response quality。成功不是要求 PM 在
raw quality 上击败所有 fixed，而是：

1. 分别相对 always R0 与 always RS 的 normalized quality 非劣（冻结 margin
   0.02）；不得看完 test 后只挑较弱 fixed；且
2. 相对 always RS 的 Strategy 调用/input-token 成本严格更低；
3. learned-vs-rule utility 与 CI 必须单独报告，不用 fixed 比较替代机制比较。

guardrails：unsupported personal-memory claim、过度建议/策略误用、fallback。
Gold strategy Recall@k、BLEU/ROUGE 等只作诊断/appendix；单一 gold response
不是唯一正确回复，不能替代主要质量判断。

为避免四种 policy 重复生成/重复评分同一个物理 prompt，ESConv 每个 state 只
生成两个唯一 treatment：R0 与 RS。learned/rule/fixed policy 只是在这两个冻结
outcome 上作选择；主 judge 每 state 只盲评一次 orientation-balanced R0-vs-RS，
再按各 policy 的预先冻结 action 映射计算配对结果。alias 不被伪装成独立样本。

ESConv memory-call rate 为零只是 action contract 检查，不能作为长期记忆能力
证据。

## 6. EvoEmo 指标层级

主比较：learned PM、同 Step-0 transparent rule、cost-matched fixed、ME+R0、
structured high-resource fixed。

主门：相对 high-resource fixed 质量非劣且 observed input tokens 更低；相对
cost-matched fixed 和 transparent rule 的 utility 分别报告。长期个性化、
时序一致性、冲突/陈旧记忆、非侵入性和 abstention 为记忆相关结果与风险
guardrails。

## 7. 顺序与不可回看规则

1. 完成 observable-state generation contract 的新 compatibility pilot；
2. 生成 52 users，完成 actual-468 QA；
3. sweep、双家族 development labels、train/calibration；
4. 冻结全部候选；一次性消费 internal test；
5. 建立 V1.5 ESConv test artifacts 与无 API policy preflight；
6. 在同一 study freeze 下批准 ESConv 与 EvoEmo 外部阶段；
7. 可以先执行 ESConv，但看见 ESConv test outcome 后不得修改 PM 再执行
   EvoEmo。任何修改都使两套外部身份一起失效。

ESConv test content 已是公开数据，因此这里不能声称研究者从未见过 test。可信性
来自：split/eligible-turn rule outcome-free、PM/阈值先冻结、policy preflight 不读
`audit_only` gold 字段、同一 study freeze 同时锁定 ESConv 与 EvoEmo、看见任一
外部结果后禁止再修改另一项。论文必须称 frozen public-benchmark evaluation，
不能称秘密 held-out challenge。

## 9. 划分与外部依据

- ESConv 官方仓库与原论文描述的是原始 1,053-dialogue corpus；仓库后来公布的
  扩展数据为 1,300 dialogues。因此本研究对 1,300 条做的 seed=13 哈希划分只能
  称自定义可复现划分，不能借用“官方 split”名称：
  https://github.com/thu-coai/Emotional-Support-Conversation
- ESConv 原论文定义了 seeker/supporter emotional-support conversation 与策略标注，
  支持把它用于单会话 Strategy 调度与即时回复质量，而不支持长期记忆主张：
  https://arxiv.org/abs/2106.01144
- ES-MemEval 是长期记忆 benchmark；其结果与协议不能由“使用 EvoEmo 数据”自动
  继承。若我们的执行单元或生成路径不同，必须称 ES-MemEval-derived：
  https://arxiv.org/abs/2602.01885

## 8. 人评边界

第一篇允许使用经过独立 controls 验证、与 development judge 隔离的双家族
LLM judge 作为回复质量 proxy。若不做人工抽查，必须在 limitation 明示；若
增加人工工作，只允许预先冻结的小规模盲法 audit，不得用于选模型或调阈值。
