# PM-v1.5 快速会议版：研究合同与执行顺序

更新时间：2026-07-17
状态：免费代码与 fail-closed 链路已搭建；历史 1-call generation compatibility
transport pilot 曾结构性通过，但绑定的是旧版 `pm_v1_5.yaml` 哈希，当前配置下必须重跑，
不能视为仍然有效的上游 gate，更不是论文 efficacy 结果。其余正式链路尚未执行，也没有
真实 V1.5 主结果。任何 dry-run、脚手架测试或旧 V1 诊断都不能写成论文结果。

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

### 1.2 路由层主张

“调用较合适的资源”只由 synthetic internal-test 的完整反事实 action matrix 支持：

- 每个 state 都实际生成并判断全部 16 个合法 action，而不是用启发式标签猜结果；
- PM 与在 calibration split 选出的 same-token cost-matched fixed policy 做用户聚类
  配对比较；内部 advantage gate 不通过时禁止进入外部生成；
- 冻结报告给出 regret、oracle-hit-rate、memory-source precision/recall/F1、M0
  判断、RS 判断、quality-acceptable rate 和 excess observed cost。

这些是“在本研究的 synthetic 状态和模型 judge 定义下的决策质量”，不是人工确认的
资源正确性，更不是临床、真实世界或用户获益主张。

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
| Strategy Bank | `data/strategy/strategy_cards_v1_5.jsonl`，12,403 张 |
| 泄漏排除 | `escN -> esconv_N` 确定性映射与既有 Jaccard 规则取并集；已移除 `0539/0585/1212` 的 26 张卡 |
| supporter prompt/cap | 同一 `SupporterGenerationContract`；temperature 0；300 output-token cap；length/未知 finish reason 失败 |
| memory/strategy retrieval | 同一实现、top-k、minimum-score 和 source 定义 |
| Evidence Filter | 全链路关闭；PM-v1.5 是纯 pre-retrieval router |
| fixed seeker | 独立 `outputs/evoemo_fixed_tracks_v1_5/`；300 API cap；任何 truncation、缺轨或旧 V2.2 bundle 都拒绝 |
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
875 条 train seed、development 与 external 的三项 retrieval 参数，并要求 sweep/judging
确实覆盖 468 states × 16 actions = 7,488 outcomes/labels。任一旧 bank、旧 seed、pilot
矩阵或 retrieval 漂移都会在 external 付费生成前失败。

## 3. 数据与调参纪律

1. `train` 只拟合模型；
2. `calibration` 只选择 selector 超参数和 fixed comparator；
3. `internal_test` 在上述选择冻结后只查看一次，用于内部 reportability 与
   decision-quality；看过后再改模型，就必须更换新的 held-out 用户；
4. 只有 training report 为 `COMPLETE`、内部 learned-routing advantage 已验证、
   decision-quality 与 fixed-baseline 产物完整时才能创建 freeze；
5. freeze 后才允许正式 external generation；外部结果出来后不得回头改模型、margin、
   composite、baseline 或筛样规则；
6. EvoEmo 应称为“冻结后的 development-informed external evaluation”，不包装成完全
   pristine 的临床外部验证。

## 4. 唯一允许的执行顺序

每个付费阶段都必须先运行 `--dry-run`，保存完整 call plan，核对预算，并由用户明确
接受当前 `cost_estimate_sha256`（fixed seeker 使用对应的 dry-run acceptance hash）。
`--run --overwrite` 被禁止；发生一次已花费的失败后不能用覆盖文件伪装成同一确认性运行。

| 顺序 | 动作 | API 调用规模 | 进入下一步的条件 |
|---:|---|---:|---|
| 0 | clean bank、split manifest、875 条 clean seed、overlap audits | 0 | 已完成且后续只认其 SHA |
| 1 | `v1_5/20a_run_generation_compatibility_pilot_v1_5.py` | 1 | 最便宜的 transport stop-loss；schema、来源绑定、finish reason 全部 PASS |
| 2 | `v1_5_run_automated_semantic_review.py` | 66 | 27 真案例 + 6 positive controls × 2 家族（`training_judge_deepseek_flash`、`final_judge`；`training_judge_qwen122` 已按下方记录的协议修订移出面板），attested `PASS` |
| 3 | `v1_5/20_generate_pm_v2_development_data_v1_5.py` | 52 | 52 users / 468 states 完整，生成 attestation |
| 4 | `v1_5/06_run_action_sweep_v1_5.py --v1-5-full-sweep-scope` | 7,488 | 真正 `scope=full`；每 state × 16 action 完整；旧 compatibility-pilot/筛选分支拒绝 |
| 5 | `v1_5/21_judge_pm_v2_action_sweep_v1_5.py` | 29,952 | 7,488 × response/risk × 2 judge families；完整性与 judge-health gates PASS |
| 6 | `v1_5/22_train_pm_v2_v1_5.py` | 0 | 只用 train/calibration 调参；internal gate 为 `COMPLETE`，否则停止 |
| 7 | `v1_5/23_build_decision_quality_report_v1_5.py` 与 `29_prepare_fixed_baselines_v1_5.py` | 0 | 两份报告均与 checkpoint/training report SHA 一致 |
| 8 | `v1_5/15a_build_evoemo_fixed_tracks_v1_5.py` | 由 dry-run 给出；当前数据设计为 1,020 个 seeker turns | 完整、无 truncation、独立 V1.5 bundle |
| 9 | `v1_5_create_freeze.py` | 0 | 重新验证步骤 3–8 的内容寻址链，并锁定 external 合同 |
| 10 | `v1_5/24...` 生成 learned、cost-matched-fixed、ME+R0；`24a...` 生成 4 个 reference baselines | 当前单元合同为 204 × 7 = 1,428 calls；最终以各 dry-run 为准 | 7 条 condition 的同一 frozen unit matrix 完整 |
| 11 | `v1_5/30_eval_forced_swap_canary_v1_5.py` | 12 units × 2 orders × 2 families = 48 | schema、顺序稳健性和跨家族方向敏感性 PASS；12 units 从主评测排除 |
| 12 | `v1_5/36_run_external_batched_schema_order_pilot_v1_5.py` | 1 unit × 2 schemas × 2 orders × 2 families = 8 | 使用 canary 首个单元；七候选 structured schema 在 GPT/Claude 的 quality/risk 传输均 PASS；顺序差异只作 transport diagnostic |
| 13 | `v1_5/25_eval_pm_v2_external_v1_5.py` | 192 GPT 主质量 + 54 Claude 敏感性 + 36×2 risk audit = 318；以 dry-run 为准 | 七个 condition 全部进入每个 batched call；完整后单独生成 `SUPPORTED`/`NOT_SUPPORTED`、有边界的 risk audit 与非确认性 latency diagnostic |

V1.5 不运行 PM-v2.2 的 180-generation/360-judge compatibility pilot；这是快速通道的
明确范围缩减。代价是证据强度低于 V2.2，但不能用把全量 sweep 标成 “pilot” 的方式
绕过：V1.5 的 sweep 现在必须诚实记录为 `full`。

按当前冻结规模，上表从 compatibility pilot 到 external judging 合计最多约 40,381 个逻辑调用，
其中 29,952 个来自 development 双家族 judging。V1.5 的“快”主要是省掉人工流程和
V2.2 的额外兼容性/复核层，不代表它是几十次调用的小实验；若时间窗口承受不了这个
规模，应在付费前另立一个明确降级、重新命名的 pilot，不能事后把不完整矩阵称作 V1.5
正式结果。外部付费 judge 部分已经从旧 pointwise 设计的 5,376 calls 降为 318 calls；
不是恢复 V1 老评测链，而是在当前 freeze/policy-lock/attestation 之后接入独立的 V1.5
batched scorer。

截至 2026-07-17，已记录以下 dry-run 与历史执行状态。generation compatibility
一行保留历史调用及预算哈希用于追溯，但该调用绑定旧配置，不能充当当前上游 gate：

| 阶段 | 当前 dry-run 上界 | 当前 hash |
|---|---:|---|
| generation compatibility | 历史真实 1 call 曾结构性通过；绑定旧 config，当前必须重新 dry-run 并重跑；历史预算上界 `$0.00606165` | historical accepted `c9b48fe7…e08efed` |
| 自动语义审核 | 三次真实 `--run` 分别在 `training_judge_qwen122`（HTTP 500）、同一 Qwen 路由（响应体缺 `message.content`）、`training_judge_deepseek_flash`（HTTP 503）上失败：即同一 NVIDIA base URL 上的两个模型路由、三次瞬时故障；这支持“本次观测路径不稳定”，但不能证明 NVIDIA 整个共享网关或全部模型都不稳定。三次均非 429/402/403，不构成额度耗尽证据。现采用 `pm-v1.5-bounded-retry-v2`：每个逻辑调用最多3次物理尝试且每次先写 append-only ledger；仅对408/429/5xx/网络超时在总预算内重试，2xx缺字段最多只增加一次物理尝试；4xx、schema校验、stage postcondition失败和已解析但不利的评分均永久终止。重试资格完全从ledger恢复，重启不能重置终止状态或缺字段计数；旧/无分类失败fail-closed。删除了在当前“一次终止失败即整批退出”执行模型中实际不可触发的circuit breaker。66 个逻辑调用 × 最多3次物理尝试 = 198 次worst-case上限；按统一保守代理价 input `$3/M`、output `$15/M` 计 `$2.86866`（3倍于单次尝试的 `$0.95622`） | `54369b9c…adb3b0c`（v2重试合同的新dry-run；此前 v1重试哈希 `2ff9d94e…4baa6f27`、99-call 哈希 `757ac238…783067` 与66-call单尝试哈希 `12e79140…9bb19a5` 均已作废） |
| 52-user generation | 52 calls；`$0.31501515` | `955276b6…ae0739` |
| fixed seeker | 102 tracks / 1,020 calls；代理价上界 `$2.63391075` | acceptance `018c2c95…39353` |

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
- batched schema/order pilot 使用其中排序后的第一个 unit，并真实覆盖 quality/risk 两种
  schema、两个 order 和 GPT/Claude 两家；
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

已实现的免费部分包括：真实 `version: pm-v1.5`、独立配置/目录/checkpoint、clean
bank/seed、EF 全链路关闭、66-call 自动审核 runner、1-call generation pilot、完整数据
attestation、真实 full-sweep gate、两家族 judging、internal decision-quality、fixed
baseline 派生、无截断 fixed-track 验证、轻量 study freeze、12-unit canary、4-call external
legacy pointwise smoke、8-call batched schema/order pilot、318-call batched 主评测规划、
stratified risk audit、external claim assessment 以及对应的 fail-closed 测试。freeze 还会独立
重验 bank/seed lineage、development/external retrieval lock 和完整 468×16 链，不能只靠
目录名或某个阶段的 `COMPLETE` 字段放行。

尚未完成的是“实验结果”，不是继续堆脚手架：正式 automated review、模型训练、
freeze 和 external evaluation 都仍待执行。步骤 1 的历史 transport pilot 曾结构性
`PASS`，但它绑定的 `pm_v1_5.yaml` SHA-256 为 `372c95dd…579156`，当前配置为
`81c1d120…07762`；现有 fail-closed lineage 会拒绝该不一致，因此它必须在当前配置下
重新 dry-run 并执行，不能继续当作有效上游 gate。步骤 2 已完成当前 66-call dry-run，
尚未产生 `PASS` gate report。一次只批准一个付费阶段。若任一 gate 失败，应保留失败产物并停止，不得在
同一冻结协议下不断换模型/提示词直到通过。
