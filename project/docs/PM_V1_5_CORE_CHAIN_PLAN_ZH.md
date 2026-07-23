# PM-v1.5 快速会议版：研究合同与执行顺序

> **2026-07-23 权威更新：** 本文是 PM-v1.5 后续执行的唯一活跃主路线，只回答“研究要
> 证明什么、现在到哪、下一步按什么顺序做”。问题编号、根因和关闭状态只在
> `PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md` 维护；旧 V1 postmortem 为只读历史证据。
> 不得在本文或其他文件再建立第二套活跃问题清单。
>
> 此前拟定的双域监督路线已按预注册停止规则关闭。719-state ESConv auxiliary
> generation 已完成，但其单动作绝对量表在 train 上为 `NOT_SUPPORTED`；随后独立的
> 24-state balanced-order pairwise pilot 虽 96/96 调用完整，安全偏好的有效非 tie 率仅
> 0.1667，低于冻结下限 0.25，故结论为
> `ESCONV_AUXILIARY_PAIRWISE_INSTRUMENT_NOT_SUPPORTED`。不得继续打开 calibration 或
> internal、调 prompt/阈值或换 judge 直到通过，也不得把 auxiliary 数据用于监督训练。
> 正式 PM-v1.5 回到 52-user/468-state 纵向单域训练；双域入口保留为未消费的研究基础设施，
> 不是本篇的正式路线。正式外部评测仍是两项互补而非同义的试验：ESConv test 仅比较
> 固定 `M0+R0/M0+RS` 的即时回复质量与 Strategy 机制，不主张 learned PM 在 ESConv
> 路由；EvoEmo/ES-MemEval-derived 才检验冻结纵向 PM 的 memory/strategy
> 质量—风险—成本权衡。Hybrid retrieval 已依据
> calibration 与 ESConv validation 的合法证据作出 `NOT_ADOPTED` 决定，正式链路继续
> 使用 lexical-only；它不是待办事项。并发只允许作为保持 call plan、prompt、模型、
> judge、重试和统计单位不变的执行层优化，具体边界见第 4.3 节。

> **2026-07-18 审查修复提示：** 下一次正式运行的方法定义已由
> `PM_V1_5_PROTOCOL_REPAIR_CONTRACT_ZH.md` 取代。新合同引入正式、受限且计费的
> source-level Step-0，区分 requested/attempted/realized action，并重组机制、固定策略和
> 外部效率三层 gate。V8.2 已真实消费并 fail-closed；中央配置现已固定为 release 状态，
> 但 approval manifest 仍为空，因而任何付费调用继续 fail-closed。旧 V8.3 dry-run 已失效；
> V8.4 已真实执行并在 transport/schema 层 PASS，但 exact surface 复核发现
> readiness 未进入两个 Strategy case 的可见话语，且旧自动审核没有绑定付费九例；
> 因而 V8.4 已归档为 `CONSUMED_PASS_SEMANTICALLY_SUPERSEDED`。V8.5 两次真实调用
> 都因 provider 的通用 role list 以 user 结尾而 fail-closed，identity 已消费。V8.6 将历史
> 改成 schema-level `user_text → assistant_text` exchanges，由本地 compiler 保证角色顺序，
> 已真实 PASS：9/9 accepted、12 次物理尝试、0 fallback；3 次 initial failure 均为
> `unique_current_user_text`，没有复现 role-order bug。随后绑定该 exact attestation 的
> V3 双家族语义审核真实执行：DeepSeek 首次调用成功，Gemini 的 OpenAI-compatible
> `response_format` 请求在第二次物理调用返回 terminal HTTP 400，identity 已消费并
> fail-closed。该失败不能证明 rubric schema 不受支持；V4 已将 Google judge 冻结为官方
> native `generateContent + responseJsonSchema` transport，保留相同模型、严格 schema 和
> 本地 Pydantic 验证，并将 Gemini 排为首个物理调用以最小化再次不兼容的花费。由于
> V8.6 的历史 attestation 绑定了整份配置和共享 API 代码，不能在新代码上重新盖章复用；
> V8.7 已改为只绑定 generator-relevant config projection，今后的 judge-only 修改不会再
> 误伤上游 pilot。V8.7 进一步把 pilot 预算上限冻结进 YAML，消除了依赖隐藏 CLI
> 参数而产生多个 cost identity 的歧义；identity `920807ec…84e8` 已真实执行并
> `CONSUMED_PASS`：9/9 accepted、12 attempts、3 次 bounded repair、0 fallback，
> 18,288 input / 1,964 output tokens，约 `$0.0039216`。当前没有 active approval；
> V4 native-Gemini semantic review 已真实消费并在测量有效性上 fail-closed：120 个
> logical calls 最终成功，但 targeted controls 0/24。旧单字段 v1 因本地 Gemini usage
> parser 漏分项停止；v2 验证了精确 token 记账，却只得到 7/14 内容正确，并在第 15 个
> logical call 的首次 malformed JSON 上按旧合同立即终止，并非耗尽重试。balanced v3
> 随后真实消费两次：Gemini PASS；DeepSeek verdict 与逐字 quote 正确，却把 memory text
> 引到 `memory_id`，原始 surface 还带 `We` 前缀，旧 client 又静默截取 JSON。当前 v4
> 将 record metadata 移出 citable evidence，并把 citation integrity 与 verdict accuracy
> 分开记录。v4.0 `acb07969…f7caf` 因短时 provider availability 仅完成 1/16 endpoint calls，
> 已消费且无法提供测量结论。v4.1 将 transport 与 malformed-output 计数拆开，显式审计
> 确定性 JSON wrapper 规范化，单项不可用时继续冻结矩阵。fresh 16-call dry-run 已双目录
> 复现，identity `22265947…936d`，最大 160 physical / `$0.037195`；
> 中央 approval 为空。本诊断不是 formal gate，不能授权
> training、V5、52 users，也不支持真实用户需求识别/真实改善主张。
> 本文保留为 2026-07-17 版本的历史设计
> 背景，与新合同冲突时以新合同为准。

更新时间：2026-07-23
历史收口状态（截至 2026-07-19）：V8.11.1 已真实 `CONSUMED_PASS`：9/9 accepted、10 次物理调用、
1 次普通 content repair、0 transport retry、0 fallback；9 例有 8 个不同 current turn，
唯一重复组是同一 user/family/split 的合法反事实对。identity `6876e2d3…25ed` 永久
禁止复用，approval map 与 pending map 均为空。其 artifact 中“full generation 前需要
independent human semantic review”的 scope 句是旧 V4 流程的历史描述，不是当前执行门；
权威顺序是正式 468-state corpus 生成后执行 actual-468 structured QA v3，并在 7,488-action
sweep 前 fail-closed。为保留已通过 attestation，不回写或重签历史 artifact。

历史状态（仅描述当时，不是当前快照）：免费代码与 fail-closed 链路已搭建；历史整包 generation compatibility
transport pilot 已失效，V8.1 逐例试运行在 9 个 case 中有 2 个话题 lint 失败。当前合同已改为
surface-only 逐 case 生成和每 case 最多一次预预算 repair；该历史状态后来已被第 4.1 节
列出的正式 corpus/sweep 进展取代，不能视为当前待办或论文 efficacy 结果。任何 dry-run、
脚手架测试或旧 V1 诊断都不能写成论文结果。

## 1. V1.5 到底要支持什么主张

V1.5 的正式主张分成两层，不能混写。

### 1.1 外部回复层主张

冻结后的 EvoEmo 评测只在以下两项同时成立时输出 `SUPPORTED`：

1. PM-v1.5 相对 `best_fixed`（代码中的条件名；语义上是固定
   `MPMSME+RS` structured-high-resource policy）的冻结 quality composite
   非劣效：按 `user_id` 聚类的配对 bootstrap 95% CI 下界不低于 `-0.02`；
2. 同一配对比较中，PM-v1.5 的 `observed_input_tokens` 差值 95% CI 上界严格
   小于 0。

判定由 `src/metacom_pm/v1_5_external_claims.py` 自动写入
`claim_assessment.json`。外部 judging 运行成功只代表数据完整，绝不自动代表主张
成立；任一条件失败都必须报告 `NOT_SUPPORTED`。

quality composite 不是只靠 emotional support 和 personalization 两项。主分析冻结为六个
独立维度：emotional support、personalization、memory appropriateness、factual
grounding、temporal consistency、non-intrusiveness；judge 不输出 `overall`，最终
composite 由冻结权重确定性计算。GPT-4o 对全部 192 个正式 unit 做主质量判断；Claude
只对预冻结的 54-unit（每用户 3 个）分层样本做敏感性分析，不与 GPT 分数取中位数、
也不混入主效应估计。Claude 结果无论同向或反向都必须报告。这是 V1.5 的成本—证据
折中：主矩阵每 unit 只有一个 judge，不能写成“全量双家族复核”。batched prompt 也可能
产生 contrast/context bias；循环位置平衡、双 order transport pilot 和 Claude 分层敏感性
能检查但不能彻底消除该局限，论文 limitations 必须明示。

evidence/resource-use risk 不进入上述质量—成本主张。两家 judge 对预冻结的 36-unit
（每用户 2 个、覆盖 turn 3/8）分层样本做独立 batched audit。只有相对 `best_fixed` 的
用户聚类 CI 上界不超过冻结的 [0,1] 风险 margin 0.05 时，才允许写“预注册分层审计未
发现 PM 增加证据误用问题”；这不是总体安全性、临床风险或真实部署风险结论。

V1.5 当前没有可支持确认性 latency 优势的交错测量设计。因此 latency 只能作为
描述性诊断：脚本 25 会从同一 scored unit matrix 生成
`latency_diagnostic.json`，汇总各条件的 mean/median/P95 和 PM-minus-baseline 配对
点差，并硬编码 `confirmatory_latency_claim_allowed: false`。论文不能写成“显著降低
latency”。若会议稿必须把 latency 放入正式主张，需要另加冻结的 interleaved
latency 实验；不能拿顺序运行的 wall-clock 时间补写结论。

ESConv 是与上述 EvoEmo 评测并列、而不是混在一起的第二项外部实验。它只允许检验：

1. 同一冻结 PM 在单会话、跨会话 memory 结构性不可用时，能否选择性地在
   `M0+R0` 与 `M0+RS` 之间切换；
2. 相对 always-R0 与 always-RS，冻结的即时回复质量是否分别非劣；
3. 相对 always-RS，Strategy 调用和实际 generator input-token 成本是否更低；
4. learned PM 与读取相同 Step-0 的 transparent rule 的 utility 差异。

ESConv 不支持 memory 能力或长期个性化主张。ESConv 与 EvoEmo 的分数、样本和
bootstrap cluster 不得合并；只有两项分别通过自身冻结门，才允许使用“跨单会话策略与
多会话记忆环境的资源调度”这一较宽表述。

### 1.2 路由层主张

“调用较合适的资源”由两个互补、但分开封存和报告的 internal-test 支持：

- 纵向合成域：每个 state 都实际生成并判断全部 16 个合法 action，而不是用启发式
  标签猜结果；它支持 MP/MS/ME/RS 的细粒度路由结论；
- ESConv auxiliary 域：719 个真实单会话 state 只生成和判断两个合法 action
  `M0+R0/M0+RS`；它为同一 PM 补充“没有跨会话 memory 时如何开关 Strategy”的
  训练支持，不能反过来支持 memory 结论；
- PM 与在 calibration split 选出的 same-token cost-matched fixed policy 做用户聚类
  配对比较；内部 advantage gate 不通过时禁止进入外部生成；
- 冻结报告给出 regret、oracle-hit-rate、memory-source precision/recall/F1、M0
  判断、RS 判断、quality-acceptable rate 和 excess observed cost；ESConv auxiliary
  另行给出 R0/RS regret、选择率、质量—风险—成本差异和 OOD/fallback，不与纵向指标
  汇总成一个分数。

这些是“在本研究的合成纵向状态与 ESConv 单会话状态、以及模型 judge 定义下的决策
质量”，不是人工确认的资源正确性，更不是临床、真实世界或用户获益主张。

### 1.3 明确不主张

- 不声称临床改善、真实用户改善、安全有效性或现实世界资源选择正确；
- 不声称做过人工评测、人工校准或人工 Strategy Bank 审批；
- 不把自动语义审核写成人评替代的等价证据；
- 不把 forced-swap canary 写成 PM efficacy 证据。

## 2. 训练与测试必须统一的机制

以下内容在 development、内部测试和外部评测中由配置、attestation 与 study freeze
共同锁定：

| 项目 | V1.5 冻结值/纪律 |
|---|---|
| Strategy Bank | `data/strategy/strategy_cards_v1_5.jsonl`，11,590 张，来自 823 个 ESConv 对话；正式 52 个 development seed 来源已逐实例排除，8 个策略家族均保留 |
| 泄漏排除 | `escN -> esconv_N` 确定性映射与既有 Jaccard 规则取并集；已移除 `0539/0585/1212` 的 26 张卡 |
| development 域 | 纵向合成域 52 users/468 states/16 actions；ESConv auxiliary 域 52 个 bank-disjoint train dialogues/719 states/2 actions；同一 PM 联合训练但分域加权、分域 gate |
| supporter prompt/cap | 同一 `SupporterGenerationContract` 与 `response_mechanism_contract`；同一 generator endpoint/model、system prompt、temperature、output cap、evidence compiler 与 finish-reason 规则 |
| memory/strategy retrieval | 同一 canonical lexical-only 实现、query builder、top-k、minimum-score、token budget 和 source 定义；Hybrid 诊断已 `NOT_ADOPTED`，不得进入任何正式 consumer |
| Evidence Filter | 全链路关闭；PM-v1.5 是纯 pre-retrieval router |
| Step-0/state semantic input | 同一 section-aware bounded visible-state text/vector；Step-0 与 state embedding 的 query hash 必须相同，禁止 tokenizer 隐式截断 |
| requested/attempted/realized action | 全链路分开记录；outcome、cost、label lineage 必须绑定真实 realized evidence，不得把 requested action 直接当作已执行 action |
| prompt-equivalence alias | 相同 state 下实际 generator prompt 完全相同的 action 只允许一次物理生成/判断；结果可映射给 alias，但 requested-action cost、类别大小和 lineage 必须保留 |
| fixed seeker | V3 独立 sidecar/目录；provider cap 保持 300，原始响应与 finish reason 完整留账；正式轨迹只接收不超过 60 个空白分词的完整表面文本，超长或 provider `length` 只能确定性选取界内最长完整句前缀，禁止中句截断；先过同合同小 pilot，再生成 102 条冻结轨迹 |
| 配置与 checkpoint | `configs/pm_v1_5.yaml`、`pm_v1_5.joblib` 及两个 fixed checkpoint 独立命名和哈希 |
| 外部单元 | 同一 EvoEmo、simulator、3 个 robustness seeds、turn 3/8；freeze 后不可改变 |

不同用户拥有不同的 memory 内容是研究对象本身，不属于配置漂移。需要统一的是 memory
构造/检索机制，而不是强迫 train/test 存储相同文本。

RAG card 与测试文本不重合是必要条件，但不是单独充分条件。还必须同时保证：来源级
排除、split 用户隔离、近重复审计、同一检索合同、测试前冻结以及测试后不调参。当前
bank 的剩余 turn-level 命中是通用寒暄/共情短句，应保留审计报告并在方法中说明，不能
宣称“字面零重合”。最新 v2 turn-level audit 把每条命中反查到 card 的
`source_dialogue_id`：13 条均为跨来源通用短句，确定性同源命中为 0。

`v1_5_create_freeze.py` 会重新核对上述 bank/seed/audit 的内容哈希、84 个来源排除、
875 条 train seed、两个 development 域、response mechanism、development 与 external 的
retrieval 参数，并要求纵向 sweep/judging 覆盖 468 states × 16 actions = 7,488
action labels，ESConv auxiliary 覆盖 719 states × 2 actions = 1,438 action labels。
任一旧 bank、旧 seed、pilot 矩阵、generator/prompt/compiler 或 retrieval 漂移都会在
external 付费生成前失败。

## 3. 数据与调参纪律

1. 两个 development 域各自的 `train` 只拟合模型；ESConv auxiliary 的 52 个 dialogue
   是按冻结 manifest 顺序确定的 bank-disjoint seed，不是 outcome-aware 随机样本；
2. 两个域各自的 `calibration` 只选择允许的 selector/uncertainty 超参数和 comparator；
   ESConv validation/test 不进入训练、阈值选择或 OOD threshold 拟合；
3. 联合训练必须采用冻结的 domain→dialogue/user→state→action/alias 分层权重：两个域
   先等权，再在域内按独立 dialogue/user、state 和 action 等权；不得让 719-state 域按
   原始行数淹没 216 个纵向 train states，也不得把 prompt-equivalent alias 当作额外样本；
4. 两个 `internal_test` 都在模型、阈值和 comparator 冻结后各自只查看一次，用于各域
   reportability 与 decision-quality；不能用一个域通过抵消另一个域失败。看过后再改
   模型，就必须更换对应的 held-out dialogue/user；
5. 只有 training report 为 `COMPLETE`、两个内部域的 learned-routing advantage 已按
   各自合同验证、
   decision-quality 与 fixed-baseline 产物完整时才能创建 freeze；
6. freeze 后才允许正式 external generation；外部结果出来后不得回头改模型、margin、
   composite、baseline 或筛样规则；
7. EvoEmo 应称为“冻结后的 development-informed external evaluation”，不包装成完全
   pristine 的临床外部验证。

## 4. 唯一允许的执行顺序

每个付费阶段都必须先运行 `--dry-run`，保存完整 call plan，核对预算，并由用户明确
接受当前 `cost_estimate_sha256`（fixed seeker 使用对应的 dry-run acceptance hash）。
`--run --overwrite` 被禁止；发生一次已花费的失败后不能用覆盖文件伪装成同一确认性运行。

| 顺序 | 动作 | API 调用规模 | 进入下一步的条件 |
|---:|---|---:|---|
| 0 | clean bank、split manifest、875 条 clean seed、overlap audits | 0 | 已完成且后续只认其 SHA |
| 0.5 | 专用 Python 3.13.2 venv 中运行 `v1_5/19_preflight_semantic_runtime_v1_5.py` | 0 | exact package/device/dtype/user-site 与 3×384 canary `PASS`；真实 BGE 长上下文 strict `PMV2State` 构造、Step-0/state 文本与向量 hash 相等、implicit truncation=0；通用 readiness 20-case 结果只报告；compiler-owned 12 个 readiness surface 必须 12/12 被 frozen BGE 识别 |
| 1 | `v1_5/20a_run_generation_compatibility_pilot_v1_5.py` | 成功路径 9；18 个 content attempts；上限 54 个 physical attempts | observable-state-support v18 + fresh V8.11.1 必须 PASS；同一 user/family/split 最多两对完全相同的 current turn、每组最多 2，9 例至少 7 个独立表达，跨 user/family/split 与人工 nonce 均禁止；两对上限逐例执行；每个 content attempt 各有最多 3 个 transport slots，最终 fallback 必须为 0 |
| 2（历史失败） | `v1_5_run_automated_semantic_review.py` V4 | 已真实执行 120 logical / 144 physical | native transport 成功，但测量工具 targeted controls 0/24，永久 `CONSUMED_FAILED_CLOSED`；不得重跑或作为上游 PASS |
| 2.1（已消费校准） | `v1_5/20c_prepare_v4_single_field_diagnostic_v1_5.py` + `20d_run_v4_single_field_diagnostic_v1_5.py` | V4.2 实际 8 logical / 12 physical | 8/8 完成；确定性=1.0、verdict=.875、citation=.875。Gemini 一次 citation section 不足；DeepSeek 一次把 `explore_first` 错当成必须出现“ready”字样。identity 永久 consumed；结果只用于修订正式测量合同 |
| 2.2（已落实为正式合同） | actual-468 structured QA v3 | 0（合同/测试） | 4 个确定性字段代码硬验；仅 context grounding、advice readiness 进入原子双家族 panel；readiness 有明确操作定义；引用只报告。不得把分歧改写成 gold 或据此调样本/阈值 |
| 3 | `v1_5/20_generate_pm_v2_development_data_v1_5.py` | 成功路径 468；最多 936 个 content attempts；每个 content attempt 最多 3 个独立 transport slots，physical 硬上限 2,808 | exact V8.11.1 attestation 的路径、raw/internal SHA 与 contract SHA 进入 cost identity；52 users × 9 个逐例 surface，每例最多一次 content repair，transport retry 不消耗 repair；允许的同文反事实由 history/catalog 区分且逐 bundle/split 审计；468 states 完整并反平衡 |
| 3.5 | V8.19.2 actual-468 structured QA lineage + 冻结 post-hoc instrument qualification | 完整矩阵仍为 1,880 logical calls；只允许按 call-key、prompt hash、旧 ledger hash 和新 evaluator corpus hash 继承未变化结果；新增调用数与 physical/cost 上限只认两次独立 dry-run | 原 gate **永久保留 FAIL**，不得改写为 PASS。25/25 缺陷修复有 canonical 重编译与 attestation；4 个 deterministic fields 全部 PASS；真实 packet 双家族一致 `not_supported` 为 0。仅允许以独立状态 `QUALIFIED_DATA_CORPUS_WITH_DISCLOSED_INSTRUMENT_LIMITATIONS` 放行 development sweep：冻结披露 1 个负控一致漏检、1 个 provider 截断、316 个按预注册 panel policy 保留的分歧及 report-only citation 完整率；此后不再改 prompt/control/阈值/数据。论文必须称其为 post-hoc instrument qualification，不能称原 gate PASS 或 held-out confirmation |
| 3.6 | `v1_5/20b_run_step0_shortcut_audit_v1_5.py` | 0 | 完整 468 states 上的单阈值和 train-only user-group 多变量 probe 均未达到冻结的 near-oracle 上限；报告与数据 attestation 内容寻址绑定 |
| 3.7 | `v1_5/20b_preflight_rule_grid_v1_5.py` | 0 | 只读 train/calibration states、不读 outcome/internal；候选至少形成 2 种 state-level policy mapping，且最大 pairwise disagreement 不低于冻结下限；报告在 sweep/训练前绑定 |
| 3.8 | `v1_5/12b_build_esconv_auxiliary_v1_5.py` 及三 split 无 API preflight | 0 | 冻结 52 个 bank-disjoint ESConv train dialogue、719 states（318/170/231）和仅 `M0+R0/M0+RS` 合法动作；不读 ESConv gold response/strategy/outcome；held-out ESConv test 不被读取 |
| 3.9 | ESConv auxiliary 两动作 generation + train-only measurement | generation 1,438 outcomes 已完成；absolute train 636 pairs 已测；pairwise pilot 96 calls 已测 | generation 三 split 完整，但 absolute instrument 与 pairwise replacement 均 `NOT_SUPPORTED`；按预注册停止规则，calibration/internal judging 不运行，auxiliary 不进入训练。其产物只作诊断与第一篇局限证据 |
| 4 | `v1_5/06_run_action_sweep_v1_5.py --v1-5-full-sweep-scope` | 7,488 logical action outcomes；prompt-equivalent actions 允许共享一次物理生成，但必须物化全部 7,488 行 | 验证 actual-468 的 exact PASS **或**上述独立、内容寻址的 post-hoc qualification（二者不得混称）、Step-0 shortcut 与 response-mechanism attestation；真正 `scope=full`；每 state × 16 requested actions 完整，alias/cost lineage 可审计 |
| 5 | `v1_5/21_judge_pm_v2_action_sweep_v1_5.py` | 合计仍为 7,488 × quality/risk × 2 judge families = 29,952 logical calls；先显式 `train_calibration`（5,184 outcomes/20,736 calls），另以 `sealed_internal_test`（2,304/9,216）生成 opaque holdout | train/calibration requested-action labels 完整且 judge-health gates PASS；internal runner 不计算 outcome aggregate，只在完整矩阵后立即 seal，并在 candidate/阈值/comparator 冻结后由 one-shot ledger 消费；禁止用原先的 all-split runner 先汇总 internal 再补 seal |
| 6 | `v1_5/22_train_pm_v2_v1_5.py` 纵向单域训练 | 0 | 入口现场重验 exact runtime；只用 longitudinal train/calibration 选择模型与冻结阈值/comparator；candidate 冻结后才一次性消费 longitudinal internal-test。`22a` 双域入口保留但由两个 auxiliary `NOT_SUPPORTED` freeze fail-closed，本篇不执行 |
| 7 | `v1_5/23_build_decision_quality_report_v1_5.py` 与 `29_prepare_fixed_baselines_v1_5.py` | 0 | 两份报告均与 checkpoint/training report SHA 一致 |
| 7.5 | `v1_5/12_build_esconv_test_v1_5.py` + fixed-condition preflight | 0 | 自定义 70/15/15 split 的 169 个 non-overlap test dialogues 全保留；2,275 supporter turns 中按 outcome-free history-support rule 保留 2,112；只构建固定 `M0+R0/M0+RS` 条件，不把 longitudinal PM/rule 的 ESConv policy choice 当正式 condition，不读 gold response/strategy |
| 8 | `v1_5/15a_build_evoemo_fixed_tracks_v1_5.py` V3 bounded-surface pilot → formal | pilot 为 2 tracks / 20 logical calls；formal 为 102 tracks / 1,020 logical calls；physical/cost 上限各以独立 dry-run 为准 | pilot 与 formal 使用同一 V3 prompt、模型、surface selector 和 transport 合同；原始 provider 输出不删除，正式 track 每轮 `<=60` words、完整句边界、`mid_sentence_truncation_count=0`，102 条轨迹完整后才可进入 freeze |
| 9 | `v1_5_create_freeze.py` | 0 | 重新验证步骤 3–8 的内容寻址链；同时绑定 ESConv build/policy artifacts 与 EvoEmo fixed tracks；任一外部结果出现后不得修改 PM 再跑另一外部环境 |
| 9.5 | V1.5 ESConv fixed-condition 两动作 sweep + orientation-balanced blind R0-vs-RS judging | 2,112 × 2 唯一 generation outcomes；judge 规模以独立 dry-run 为准 | 只比较 always-R0 与 always-RS；dialogue-cluster CI；报告回复质量、风险、Strategy 调用与 input cost。不得把纵向 PM 在 ESConv 上的选择写成正式 learned condition，不得声称长期记忆 |
| 10 | `v1_5/24...` 生成 learned、cost-matched-fixed、ME+R0；`24a...` 生成 4 个 reference baselines | 当前单元合同为 204 × 7 = 1,428 calls；最终以各 dry-run 为准 | 7 条 condition 的同一 frozen unit matrix 完整；learned/rule external artifact 自包含 runtime lineage 与 development/external score comparison，禁止外部调阈值 |
| 11 | `v1_5/30_eval_forced_swap_canary_v1_5.py` | 12 units × 2 orders × 2 families = 48 | schema、顺序稳健性和跨家族方向敏感性 PASS；12 units 从主评测排除 |
| 12 | `v1_5/36_run_external_batched_schema_order_pilot_v1_5.py` | 3 units × 2 schemas × 2 orders × 2 families = 24 | 使用 canary 排序后的前三个单元；schema 必须全通过，mean/max absolute order delta 还必须低于冻结阈值；只作 transport diagnostic |
| 13 | `v1_5/25_eval_pm_v2_external_v1_5.py` | 192 GPT 主质量 + 54 Claude 敏感性 + 36×2 risk audit = 318；以 dry-run 为准 | 七个 condition 全部进入每个 batched call；完整后单独生成 `SUPPORTED`/`NOT_SUPPORTED`、有边界的 risk audit 与非确认性 latency diagnostic |

V1.5 不运行 PM-v2.2 的 180-generation/360-judge compatibility pilot；这是快速通道的
明确范围缩减。代价是证据强度低于 V2.2，但不能用把全量 sweep 标成 “pilot” 的方式
绕过：V1.5 的 sweep 现在必须诚实记录为 `full`。

旧版“约 41,390 次物理调用”的总数已失效：它既没有包含 719-state auxiliary 域，也把
logical labels 和可安全共享的 prompt-equivalent physical calls 混在了一起。今后每个阶段
分别报告 logical scientific units、unique physical calls、最大 physical attempts、预计/硬
上限费用和实际 token；不得再用一个总数字掩盖这些区别。V1.5 的“快”主要是省掉人工
流程和 V2.2 的额外兼容性/复核层，不代表它是几十次调用的小实验；若时间窗口承受不了
正式矩阵，应另立、重新命名 pilot，不能事后把不完整矩阵称作 V1.5 正式结果。外部付费
judge 仍采用冻结的 V1.5 scorer，不恢复 V1 的旧评测链。

### 4.1 截至 2026-07-23 的真实进度

- 52-user/468-state 纵向 development corpus 已完成；actual-468 首轮审计暴露了
  measurement wording ambiguity 和 25 个真实 context defects。25 个状态已通过窄字段
  repair overlay canonical 重编译并逐项解决（双家族一致拒绝由 14 降为
  0）。审计仍诚实保留 1 个 control 宽松漏检与 1 个不可恢复的截断调用，因此不能把
  测量工具写成“无缺陷 PASS”；按预注册停止规则将其作为已披露的 instrument limitation，
  不再反复调 prompt/control 追求全绿。独立的 post-hoc qualification 已在 V8.19.2
  内容寻址冻结为 `QUALIFIED_DATA_CORPUS_WITH_DISCLOSED_INSTRUMENT_LIMITATIONS`；原 gate
  永久保留 `FAIL`。sweep、judging 与 freeze 已能 fail-closed 地传播二者而不混称。
- 719-state ESConv auxiliary 输入已构建完成：52 个与 Strategy Bank 来源零重合的
  dialogues，split 为 train 318 states/24 dialogues、calibration 170/12、
  internal-test 231/16。完整 generation 已执行收官：train 636/636、calibration 340/340、
  internal-test 462/462 outcomes，三 split 均 `CONSUMED_PASS`，合计真实费用约 `$0.1342`。
  bounded transport 在 1,438 次真实调用中恢复了 124 次 429 和 1 次 5xx，零终止性失败。
  这证明生成链可用，不代表 judging labels 或 PM 已完成。
- auxiliary train judging 的 636/636 judge-pair 矩阵已经完整恢复，但原 quality gate 暴露
  三类不同问题：memory 结构性不可用造成的 N/A 维度被误判为常数缺陷、稀疏零值造成的
  假 duplicate/correlation、以及 Gemini/DeepSeek 在真正适用维度上的近常数或近重复行为。
  第一类已由 action applicability 合同纠正；后两类、risk-head applicability mask、
  joint-reliability 的诊断角色、统一 `median±MAD` conservative utility 和正式 attestation
  已实现并通过针对性测试；零 API 重聚合后 train gate 仍为 `NOT_SUPPORTED`，主要剩余
  问题是适用 response 维度低 MAD coverage 和 family-specific 近常数行为。新出现的
  `strategy_overuse`/`strategy_omission=-1` 还包含 action 适用性与互斥构念混合的二阶 gate
  bug，不能当作新的 judge failure 直接定案。独立探索复算还显示两个 judge 对 R0/RS
  nominal composite 偏好仅约 44% 同向，须进入正式 train-only measurement report。
  calibration judging 继续暂停；internal-test judging 即使生成 labels 也只能立即密封，
  禁止在 candidate/threshold/comparator 冻结前查看聚合。
- `PMV2Model`、routing-objective、transparent-rule、train-group CV 与 calibration grid
  已实现 domain→dialogue/user→state→action/alias 等权；双域输入验证器、零 API preflight
  和两个彼此独立的 sealed-holdout/consumption-ledger 原语也已实现。新的正式入口
  `22a_train_pm_v2_dual_domain_v1_5.py` 负责联合装载、分域 label audit、分域 OOD/uncertainty
  报告、纵向与 ESConv 各自 comparator/internal gate、联合 candidate freeze 和“一域失败即
  NOT_SUPPORTED”。但 absolute 与 pairwise auxiliary instruments 均已真实
  `NOT_SUPPORTED`，所以该双域入口现在必须 fail-closed，不能在本篇消费。正式路线使用
  `22_train_pm_v2_v1_5.py` 训练纵向单域 PM；这不是事后挑简单结果，而是预注册停止规则
  对测量工具失败的既定分支。两套入口目前都尚未真实训练。
- `catalog_embedding` 向 legacy runtime 泄漏且无法从落盘 state 重建的问题已修复；V8.19.2
  零 API 重编译后 468/468 runtime lineage PASS，且 states/evaluator contexts/memory backend/
  bundles 相对 V8.19.1 字节不变，因此不需要重跑 actual-468 judge。Step-0 shortcut audit 与
  transparent rule-grid preflight 已在 V8.19.2 上零 API 实跑 PASS，并由输入
  hash/attestation 绑定。旧的单次尝试 dry-run identity `faf13c51…f93c69` 已因正式
  transport execution contract 而失效，不得批准。新的 longitudinal full action sweep
  两次独立 dry-run 已在不同目录逐字节一致：7,488 logical calls、最多 29,952 physical
  attempts，logical estimate `$2.36494155`、最坏上限 `$9.4597662`，cost identity
  `a4a1a94f…ee913`，call-plan SHA `3f473200…f75cc`，transport contract
  `0af7d371…40fdc`，budget gate PASS。该 identity 随后真实执行：7,487/7,488 outcomes
  成功，8155 physical attempts（666 个 429、2 个 5xx 失败尝试，其余成功），成功调用
  usage 为 3,761,153 input + 394,280 output tokens，约 `$0.80074095`；唯一剩余调用在
  429/429/429/503 后耗尽冻结的 4-attempt cap，原阶段诚实保留为 `INCOMPLETE`。不得原地
  扩展旧 identity，也不得重跑 7,487 个成功调用。exact-plan carry-forward 已在两个独立
  目录逐字节复现：完整 call-plan SHA 仍为 `3f473200…f75cc`，继承 7,487 条、仅剩 1 条，
  最多 4 次新 physical attempts，logical estimate `$0.00023535`、最坏上限 `$0.0009414`，
  fresh identity `526c0ac6…2fbe`。该 continuation 已获独立批准并真实 `CONSUMED_PASS`：唯一
  新调用首次成功，新增 207 input + 52 output tokens、费用 `$0.00006225`；最终输出
  7,488/7,488、零 failure，artifact attestation SHA `687cecc0…26c0`。原始 incomplete 与
  continuation 两个 identities 均永久禁止复用。旧正式 judging runner 曾是 29,952
  logical calls 中任一单次失败即终止、每 call 只有 1 个 physical slot；现已在不改变
  双 judge、quality/risk prompt、schema、seed、阈值和
  标签算法的前提下，单立 development-judging execution transport contract：每 logical
  call 最多 4 个 ledger-visible physical attempts，只重试 429/408/5xx/timeout 与有界
  provider-output 格式噪声；孤立的已知 provider failure 继续矩阵，连续 5 个同类失败熔断，
  未完整矩阵固定为 `NONREPORTABLE_INCOMPLETE_MATRIX`。fresh continuation 只可从 call plan
  逐字节相同的旧目录继承 `SUCCEEDED` 行，旧 ledger SHA 进入新 cost identity，避免一条
  terminal failure 迫使约三万条成功判断全部重跑。完整 7,488 outcomes 到齐后发现原
  all-split runner 会在 seal 前把 internal-test 纳入 quality/reliability 聚合，因此其
  identity `1b73b0cf…c8877` 永久失效。正式 runner 已拆成两个显式 scopes：
  `train_calibration` 只评 5,184 outcomes；`sealed_internal_test` 只生成 2,304 outcomes
  的标签并立即密封，禁止预先计算 outcome aggregate。train/calibration 两次 dry-run 已
  逐字节一致：20,736 logical calls、最多 82,944 physical attempts、单次成功保守估算
  `$8.50905808`、最坏上限 `$34.03623232`、identity `d07f2441…3b52`。internal-test
  独立 scope 也已两次 dry-run 一致：2,304 outcomes、9,216 logical calls、最多 36,864
  physical attempts、单次成功保守估算 `$3.80874806`、最坏上限 `$15.23499224`、identity
  `24c46e31…a388`；它只能生成 opaque bundle 并立即 seal，直到纵向
  candidate/阈值/comparator 冻结后由 one-shot ledger 消费。
  纵向训练、纵向 internal-test 的正式
  消费、study freeze、正式 ESConv external 和 EvoEmo external 均未开始。任何“模型已经
  训练/内部测试已经通过/外部结果已经得到”的说法都不真实。
- memory item canonical builder、chunk 边界与 digest/consumer binding 已完成并由测试
  保护；正式链路只有一个 canonical memory contract。
- Hybrid retrieval 已完成合法 calibration/validation 诊断并正式 `NOT_ADOPTED`：Strategy
  在 ESConv validation 上没有改善且点估计略低，Memory 只在小样本 ME precision 上有
  单项信号。正式链路继续 lexical-only；不得再安排 Hybrid Part 4 或让其拖住主线。

### 4.2 当前可并行的工作

auxiliary 测量路线已经按两个独立 `NOT_SUPPORTED` freeze 关闭，不再与纵向 judging
并行消费。当前唯一训练测量线是 longitudinal train/calibration judging。

同时可在本地并行完成：

- longitudinal-only preflight、训练报告/论文表格的无 outcome 骨架；
- deterministic shard/merge 的执行层实现与小型机制 pilot；
- BGE runtime、input SHA、weight/ESS、seal 与 attestation 的只读核验；
- fixed-seeker formal 和两项外部 runner 的零 API dry-run 准备，但不能提前执行或读取
  external outcome。

longitudinal train/calibration labels 完整后才能训练并冻结 candidate。只有 longitudinal
internal-test 会在 candidate、threshold、rule、fixed frontier 与 external matrix 全部冻结后
一次性消费；auxiliary internal-test 保持未判定、未打开。不能为了“并行”提前打开
internal-test，或根据一个外部结果修改 PM 后再跑另一个外部环境。

### 4.3 安全并发与去重加速合同

并发只改变墙钟时间，不得改变科学问题、prompt、模型、temperature、seed、重试语义、
judge 家族、call plan 或统计单位。启用前必须满足：

- `PersistentAttemptLedger` 的 reserve/finish、attempt 序号、预算检查和 fsync 写入必须
  线程/进程安全；也可使用确定性 shard 独立 ledger 后进行 exact-key、hash-bound merge。
  网络等待和 backoff 必须发生在锁外；每个 worker 使用独立 client；
- call plan 先完整冻结，输出按 call key 确定性排序。dry-run identity 必须绑定并发协议、
  worker 数、provider-specific semaphore/rate limiter、retry/backoff 和 circuit-breaker；
- Gemini、DeepSeek official、NVIDIA generator 分别限流。初始 pilot 只允许保守并发：
  NVIDIA 4–6 workers，Gemini 4–8，DeepSeek official 4–8；实际额度或 429/5xx 指标更差时
  自动降并发，而不是放松 schema 或换 judge；
- 单条 terminal failure 应被完整记账并使阶段成为 `INCOMPLETE_NO_GATE_DECISION`，而不是
  丢失其他已成功调用；新 identity 的 continuation 只能按旧 ledger hash 与剩余 call keys
  续跑，不得原地复用已消费 identity；
- 先用 24-generation 与 96-judge 量级的公开机制 pilot 验证：无重复计费、无丢行、预算
  不超限、串行/并发产物键集合一致、失败可恢复。通过后才允许用于正式批次。

物理调用复用只允许针对完全相同的实际输入。相同 state 下，若多个 requested actions 的
generator prompt（包含检索 evidence、compiler 输出和生成参数）逐字节相同，可生成一次并
向 alias 物化；judge 侧只有在 response、evidence、rubric、schema、judge endpoint/model
全部相同且 equivalence hash 相等时才可判断一次。每个 requested action 的 action/cost
标签继续保留，训练按 equivalence class size 逆权重，不能把 alias 当作独立证据。

纯本地检索也允许同一 sweep invocation 内的确定性只读缓存：同一 observable query 对
同一冻结 Strategy Bank/top-k/score-floor 的结果只计算一次，再为该 state 的多个 RS
requested actions 返回副本。缓存不得跨 contract identity 持久化，也不得改变检索顺序、
evidence、attempt 行、prompt 或 action lineage；必须有回归测试证明 16 个逻辑 action 仍
全部物化。该缓存只消除对 11,590 张卡的重复词法扫描，不构成 Hybrid、向量检索或方法变更。

禁止用以下方式“提速”：删掉 DeepSeek/Gemini 任一家、合并 quality 与 risk rubric、减少
state/action、抽样替代正式矩阵、跨 ESConv/EvoEmo 合并结果、调低 validator，或把 provider
失败当作语义 PASS。按保守并发和等价复用，若 provider 稳定，余下正式链路可由纯串行的
约 3–5 天压缩到约 1–2 天；这是工程预算，不是保证，也不能写入论文 efficacy 结果。
截至本次更新，该并发合同仍是 `DESIGN_FROZEN_NOT_IMPLEMENTED`；不得仅通过 CLI 提高
worker 数。与并发不同，longitudinal sweep 已实现并冻结**串行的** bounded-transport
韧性：科学 treatment 不变，每 logical call 最多 4 个 ledger-visible physical attempts，
只重试 429/408/5xx/timeout，孤立 terminal failure 不立即杀死矩阵，连续 5 个同类失败触发
circuit breaker。该能力减少偶发传输故障造成的整批作废，但不提供并行加速，也不把失败
调用当作成功。ESConv auxiliary judging 也已按同一原则收口，但其独立合同为每 logical
call 最多 10 个 physical attempts：dry-run 同时报告单次逻辑成本与 10-attempt 最坏上界，
预算门只按后者放行；孤立 provider-surface failure 可继续矩阵，连续 5 个同类失败熔断，
任一缺行均只产出 `NONREPORTABLE_INCOMPLETE_MATRIX`，不得生成可训练 labels。该变更不
影响已经执行或正在执行的 auxiliary generation；所有早于此合同的 full auxiliary-judging
dry-run identity 均因曾只计首个 attempt 而失效，必须等对应 generation 完成后重新计算。

#### 从当前阶段开始的加速优先级

当前关键路径已经从 generation 转为 longitudinal judging。加速必须按以下顺序实施，前一项不足时
才进入后一项：

1. **先消除不必要重跑。** 所有大矩阵使用 exact-plan continuation，只继承旧 ledger 中
   `SUCCEEDED` 且 call-key/input hash 完全一致的行；格式完整但仅触发本地长度上限的响应，
   只能走独立、只读、可审计 recovery schema，不能修改在线 schema 后给旧调用补签。
2. **把网络 stage 与零 API 工作并行，而不是把 holdout 混在一起。** longitudinal
   train/calibration judging 运行时，可同时在本地运行 BGE/preflight/report 和
   fixed-seeker promotion 的只读检查；auxiliary judging 已停止。internal-test 仍须等
   candidate/threshold/comparator 冻结后一次性消费，不能为了并行提前打开。
3. **实现确定性 provider sharding。** 优先把冻结 call plan 按
   `(provider_family, stable_call_key_hash % N)` 切为 4 个独立 shard；每 shard 顺序执行并
   使用独立 ledger，最后按 exact key 集合、无重复、预算与 artifact SHA fail-closed merge。
   这比在共享无锁 ledger 上直接开线程更容易审计。4-shard pilot 与串行键集合一致后，
   才考虑 Gemini 4–8、DeepSeek official 4–8 的保守并发；遇到 429/5xx 只降速，不改模型、
   prompt 或 schema。
4. **只做已证明语义等价的缓存/复用。** Strategy Bank 的同-query lexical 结果可在一次
   invocation 内缓存；judge/generator 只有完整 provider-visible payload hash 相同时才复用。
   逻辑 action 与成本标签仍逐行物化，不能用复用减少科学样本数。
5. **本地 GPU 只跑零 API 工作。** RTX A6000 48GB 可承担 BGE、HGB、bootstrap、preflight
   和报告；当前 hosted Llama generation 已完成，而 Gemini 没有可等价本地部署的权重、
   DeepSeek official judge 也不能在单张 A6000 上等价复现。当前版本中途换成本地
   Llama/Gemma/量化 DeepSeek 会改变 treatment，不能作为提速。未来 V1.6 可从一开始冻结
   本地 open generator/judge。

若只采用第 1–2 项，不需要改变科学合同；第 3 项需要新的 execution contract、机制 pilot
和 fresh dry-run identity，但不要求重生已经完成的 corpus/sweep。第 5 项对当前关键路径
没有净收益。磁盘当前接近满载，任何本地模型下载前还必须先完成独立的存储空间审计。

截至 2026-07-23，已记录以下 dry-run 与历史执行状态。generation compatibility
一行保留历史调用及预算哈希用于追溯，但该调用绑定旧配置，不能充当当前上游 gate：

| 阶段 | 当前 dry-run 上界 | 当前 hash |
|---|---:|---|
| generation compatibility | V8.4 transport/schema PASS 但语义性淘汰；V8.5 真实 FAIL；V8.6 真实 PASS：9/9 accepted、12 attempts、3 repairs、0 fallback、18,288 input / 1,942 output tokens、约 `$0.0039084`；没有 role-order failure | V8.6 attestation `15135ad9…7f2c`；identity `64a06993…ef85` 已消费并关闭，禁止复用 |
| 自动语义审核 | V4 native 测量合同失败；旧单字段 v1/v2/v3/v4.0 失败事实保留。v4.1 identity `22265947…936d` 已完成 16/16，20 physical attempts，花费约 `$0.0016565`；deterministic=1.0、semantic verdict=.75、citation=.875、joint=.6875，暴露了 source taxonomy、tri-state verdict 与 citation/outcome 混合问题 | v1/v2/v3/v4.0/v4.1 identities 均永久 consumed。v4.2 改为 8 code truths + 4 semantic items/8 calls；协议 `first-paper-scoped`，fresh dry-run/identity 尚待生成，绝非 formal gate |
| generation compatibility V8.7 | generator-relevant scoped lineage；预算从 YAML 独立 stage contract 唯一冻结为 18 attempts / `$0.018` / 4000 input tokens，拒绝冲突 CLI；真实结果 9/9 accepted、12 attempts、3 repairs（均为 `unique_current_user_text`）、0 fallback；18,288 input / 1,964 output tokens，约 `$0.0039216` | identity `920807ec…84e8` 与 attestation `09f1f90b…60c5` 已消费 PASS、禁止复用；当前无 active approval |
| generation compatibility V8.9 | observable-support v17；identity `e7b89550…600a` 真实执行时 context_only 成功，profile_needed 首次 HTTP 500；旧 runner 将每个 content attempt 的 transport cap 错设为 1，2 个 physical attempts 后停止，约 `$0.0003459` | `CONSUMED_FAILED_CLOSED_TRANSIENT_500`；不是内容/方法失败，identity 永久禁止复用 |
| generation compatibility V8.10 | transport-resilient identity `f518d8e3…eed4` 真实执行 4/9 后失败；6 physical、零 transport retry、约 `$0.0018943`；失败来自过严的全局 current-text 唯一门，而非 provider 波动 | `CONSUMED_FAILED_CLOSED`，永久禁止复用 |
| generation compatibility V8.11.1 | v18 controlled-counterfactual + V8.10 transport resilience；identity `6876e2d3…25ed` 已真实执行：9/9 accepted、10 physical、1 次普通 content repair、0 transport retry、0 fallback、8/9 unique current turns；attestation `dd96e1df…cfd7`、ledger `2f25e11e…3177` | `CONSUMED_PASS`；永久禁止复用。后续 compiler 修复使其 shared-code binding 正确失效 |
| 52-user generation 第一次尝试 | identity `85d1eda3…8788f` 真正执行到 user 1：9/9 surface 调用成功、0 retry/repair；随后固定 `health_routine_stress × MS semantic_decoy` 为 162 字符，超过本地 160 schema 而 fail-closed；约 `$0.004121`，users 2–52 未调用 | `CONSUMED_FAILED_CLOSED`；不是 provider/seed 内容错误；原 9-call ledger 不覆盖、不删除，可离线恢复 |
| generation compatibility V8.12 | 缩短 compiler-owned decoy；付费前穷举 216 个 family/source/role 模板，最大 155/160；identity `bebeb1b1…8564` 已真实 9/9 PASS：10 physical、1 content repair、0 transport retry、0 fallback，约 `$0.0031569`；attestation `d132e5ef…1c16` | `CONSUMED_PASS`，永久禁止复用 |
| 52-user resumable generation | exact V8.12 attestation + 原 9-call ledger 已在两目录恢复同一 user-1 bundle；fresh identity `799e1cce…a6a7`、binding `47e7d25f…e4e1`、plan `8ef7aeab…40c6`；剩余 51 users，459 success / 918 content / 2,754 new physical，上限 `$2.7235611` | `DRY_RUN_REPRODUCED_UNAPPROVED_NO_NEW_API`；必须在原 canonical 目录无 `--overwrite` 执行，禁止重生 user 1 |
| fixed seeker V2（历史、不可放行） | 102 tracks / 1,020 calls；实际只完成 5 tracks / 56 turns 后遇到 provider `length`；且 51/56 已成功表面文本超过原提示中的 60-word 意图 | 旧 acceptance `018c2c95…39353` 只作历史，不能进入 freeze |
| fixed seeker V3 bounded-surface pilot | 已真实 `PASS`：2/2 tracks、20/20 logical calls、20 physical attempts、0 transport retry、0 failure；1 次确定性选择完整句前缀，最大表面 55 words，`mid_sentence_truncation_count=0`；153,314 input / 1,148 output tokens，实付约 `$0.0236859`（批准上限 `$0.431109`） | identity `6dc86e09…f2c74` 已消费并永久禁止复用；call plan `a95e6667…0dda`；contract `b0118ef0…f74`；attestation `afa96655…2f01`。只认证 V3 小 pilot，不自动授权 102-track formal |

除明确标为历史真实调用的一行外，这些 dry-run 数字只证明当前计划可计算且未创建 API
client，不等于授权执行。正式预算仍必须核对当前 endpoint/pricing 与完整 hash；任何相关
代码、配置、bank 或 seed 改动都会使上述 hash 失效。action sweep、development judging、
external generation/canary/judging 要等上游真实产物出现后才能得到精确 dry-run，不能继续
使用经验估算。

表中的统一 `$3/M`、`$15/M` 只是 fail-closed 授权上界，不是论文指标，也不是钱包实付
预测；各 NVIDIA-hosted family 是否免费或受额度约束，应由各 provider 账户单独记载，
不能从 HTTP 500 或响应体缺字段这类传输/兼容性错误反推额度结论——`api.py` 只把
HTTP 429 当作限流信号单独处理，其余错误类型都不构成额度证据。论文中的 cost 主指标
始终是各生成条件真实记录的 `observed_input_tokens`，与这张 API 采购预算表是两件事。

## 5. 外部 canary 与主评测的关系

冻结时确定性选 12 个 unit：

- forced-swap canary 只验证 judge 能否对故意交换的成对响应保持 schema 成功、顺序稳定和
  基本方向敏感；
- batched schema/order pilot 使用其中排序后的前三个 unit，并真实覆盖 quality/risk 两种
  schema、两个 order 和 GPT/Claude 两家；除 schema 成功外，还用冻结的 mean/max
  absolute order delta 阈值作数值放行；
- 12 个 unit 全部从正式 efficacy/非劣效主集合排除；
- 新的 V1.5 batched scorer 不调用共享 PM-v2.2 pointwise evaluator；旧 pointwise scorer
  与 4-call smoke 仅保留为 legacy fallback，不是论文主链路。forced swap 仍不证明 PM
  efficacy；正式主张只由第 1.1 节冻结的主质量/成本配对 CI 合同决定。
- 这一小节曾经只用一个名不副实的布尔字段
  `require_forced_swap_or_human_check_for_key_claims: true` 表达上面这段话，容易被
  误读成"仍在用 PM-v2.2 原版 forced-swap efficacy 检验"。已改成结构化的
  `external_evaluation.key_claim_gate`（`require_v1_5_forced_swap_canary: true`、
  `canary_role: judge_sensitivity_not_pm_efficacy`、
  `efficacy_decision: frozen_paired_quality_and_cost_ci`、
  `shared_pmv22_key_claim_verification: false`），单一事实来源在
  `src/metacom_pm/v1_5_forced_swap_canary.py:KEY_CLAIM_GATE`；
  `scripts/v1_5_create_freeze.py`（写入 freeze 前）和
  `scripts/v1_5/25_eval_pm_v2_external_v1_5.py`（外部评测前）都会核对配置/freeze
  里的这段内容与该常量逐字段一致，不一致直接 fail-closed。
  另外，`claim_assessment.quality_noninferiority_margin: 0.02` 是在 quality_composite
  的 [0,1] 归一化尺度上取的（`(原始1-5分-1)/4`），换算回原始1-5量表约等于0.08分，
  写论文时按这个换算表述，不要直接说"0.02分"。

## 6. 当前实现与剩余工作

已实现：真实 `version: pm-v1.5`、独立配置/目录/checkpoint、clean bank/seed 实例级隔离、
EF 全链路关闭、requested/attempted/realized action、统一 bounded semantic input、memory
builder/digest binding、actual-468 修复与 qualification、完整 7,488-action sweep、719-state
auxiliary 双动作 generation、双家族 judging 基础设施、domain/alias weighting、sealed
internal holdout、fixed baselines、双外部 runner/freeze 合同、canary/batched scorer 和相应
fail-closed 测试。Hybrid 检索负结果已归档且与 production consumer 隔离。

剩余主线必须按第 4 节推进：

1. auxiliary absolute instrument 与 pairwise replacement 已先后得到
   `NOT_SUPPORTED`；pairwise 工程执行 96/96 完整，但 safety effective non-tie
   `0.1667 < 0.25`。冻结结论，不再打开 auxiliary calibration/internal，不生成训练
   labels，不调 prompt、judge、样本或阈值。719-state generation 与两次 pilot 仅作
   diagnostic/limitation 证据；
2. 完成 longitudinal train/calibration judging；缺行只允许 exact continuation/recovery，
   不重跑已成功行。冻结 applicability、sparse-zero、MAD utility、judge-health 与
   attestation 后，用纵向 train 选择模型族、calibration 冻结阈值、transparent rule、
   fixed frontier 与 comparator；
3. 用 `22_train_pm_v2_v1_5.py` 训练唯一 longitudinal candidate。不得把已经实现但未获
   measurement 支持的双域入口用于本篇正式结果；
4. candidate 完全冻结后，一次性生成/打开 longitudinal internal-test labels并消费唯一
   holdout ledger；ESConv auxiliary internal-test 保持未判定、未打开；
5. fixed-seeker V3 pilot 已 PASS，但正式 config/freeze/外部 consumers 仍是 V2 合同，
   当前 zero-API promotion preflight 因此明确 `BLOCKED`。先原子迁移 config、study
   freeze、PM generation、reference baseline 和 shared EvoEmo runner 的 stage/目录
   合同并补反向测试；再双 dry-run、独立批准并完整生成 V3 102 tracks，构建
   decision-quality/fixed-baseline reports，最后创建绑定同一
   response/retrieval/judge mechanism 的 study freeze；
6. 在同一 freeze 下执行 ESConv test 的即时回复质量/Strategy 开关评测，以及
   EvoEmo/ES-MemEval-derived 的纵向 memory/strategy 评测；
7. 只按各自预注册 gate 给出 `SUPPORTED`/`NOT_SUPPORTED`。若只支持透明 rule、只支持一个
   外部域或两个都不支持，也必须如实报告，不继续调方法直到全绿。

第一篇只报告冻结特征与 LLM-judged labels 下，监督式 pre-item-retrieval router 的
质量—风险—成本权衡。无人工金标签、无真实用户、HGB+BAAI 不等于语言理解、EvoEmo 与
ESConv 均有合成/语义血缘、以及 provider 可靠性都必须明确列为局限。一次只批准一个可
审计的付费 identity；可并行的独立 stage 也必须各有自己的目录、ledger、预算与收尾记录。
任一科学 gate 失败，应保留失败产物并按主张边界报告，而不是不断换模型、prompt、样本或
阈值直到通过。

执行加速只改变传输编排，不改变科学请求。纵向 train+calibration judging 当前冻结计划为
20,736 logical calls；已用 `sha256(physical_call_key) mod 4` 零 API 划分为
5,298/5,127/5,130/5,181 四个 disjoint shards，两次产物逐字节一致并通过 exact coverage
校验。正式启用前仍须补 shard-aware runner、每片独立 ledger/identity、hash-bound merge
和一个小 pilot；现有只读分片文件不能直接视为执行授权，也不能与旧全量 identity 混用。
