# PM V1 → V1.5_1 全局失效模式账本与不可回归合同

更新时间：2026-07-18
适用分支：`pm-v1.5_1` 及其后续修复分支
文档性质：历史复盘、研究有效性威胁账本、改动影响检查表；不是实验结果，也不替代冻结配置

## 0. 为什么需要这份文档

本项目过去最危险的问题很少是“某一行代码写错”。更常见的是：局部代码分别看起来合理，
但整条因果链中的数据、动作、检索、prompt、生成器、judge、比较器或论文文字并没有指向
同一个实验处理（treatment）。这种错位可以让测试全绿、均值漂亮，却仍然不能支持论文主张。

本文把 PM-v1 的负面结果和局限、当前重新训练版 V1.5 从创建到 V1.5_1 修复期间发现的
全部主要问题，以及修复本身引入的二阶风险放进同一份账本。今后任何修复都必须先回答：

1. 改动触碰了因果链的哪一层；
2. 上下游是否仍然是同一个 treatment；
3. 哪些旧 artifact、hash、approval、checkpoint 和结果因此失效；
4. 修复会不会改变 estimand、样本独立性、比较器或允许的论文主张；
5. 哪些门必须在付费前、outcome 前、internal-test 前或 external 前重跑。

本文与 `PM_V1_5_PROTOCOL_REPAIR_CONTRACT_ZH.md` 的关系是：

- 后者定义当前目标方法；
- 本文保存“为什么必须这样定义”以及所有已知不可回归条件；
- 本文不能单独授权 API、训练、internal-test 或 external；
- 若本文、配置、代码、release manifest 和 artifact index 对同一状态说法不一致，必须先停机
  对齐，不能任选一个最方便的版本继续运行。

## 1. 一页总判断

### 1.1 四类足以推翻论文结论的错误

只要出现下列任一类，结果即使数值有利也不能支持目标主张。

| 类别 | 典型错误 | 为什么是 claim-fatal |
|---|---|---|
| treatment 身份错位 | 训练、固定策略和外部评测使用不同 prompt、RAG、Evidence Filter、检索阈值或生成参数 | 比较的不是同一个方法在不同策略下的结果，差值无法归因给 PM |
| 决策边界泄漏 | PM 在 requested action 前看到 item-level 结果、oracle、judge label，或近似答案键的免费目录探针 | “路由能力”可能只是读取答案或人工模板 |
| holdout / evaluator 污染 | internal 可反复消费；final judge 参与开发；external 结果反向用于阈值、模型或样本筛选 | 置信区间和泛化主张失去未见数据含义 |
| comparator / claim 错位 | 只和昂贵 high-resource fixed 比省 token，却声称 learned routing 优于同预算 fixed/rule | 结果最多证明“更省”，不能证明“学会了状态条件路由” |

### 1.2 PM-v1 最重要的结论

PM-v1 不是“完全没有价值”，也没有证据表明最终负面结果主要由故意作弊造成。它可靠地
显示了：更多历史和更多 RAG 不必然提高单步回复质量，资源调度问题值得研究。但它没有
证明 learned PM 优于同预算 `ME+R0`，没有学会稳定 abstention，且其数据、测量、特征和
selection rule 不足以支持强条件路由主张。

PM-v1 的正式外部动作曾表现为 `M0=0%`、`RS≈96.4%`，并集中到 `MSE+RS`。同预算
`ME+R0` 的 forced-swap 结果仍更好，说明“动作空间有 16 个”不等于“PM 学会了 16 种
状态条件决策”。

### 1.3 当前 V1.5_1 的准确状态

当前 V1.5_1 已把多数已知结构性问题改成代码合同，但仍没有新的 52-user development
corpus、7,488-action sweep、训练 checkpoint、一次性 internal 结果、study freeze 或 external
主结果。因此只能说“方法具备重新检验条件”，不能说“PM 已成功”。

V8.2 generation compatibility pilot 已真实消费并 fail-closed：8 个物理尝试中 6 成功、2 失败，
失败原因是词面 advice-request 正则错误拒绝了语义有效的请求。原 approval/index 矛盾已按真实
账本闭环为 `CONSUMED_FAILED_CLOSED`；旧输出、旧 cost hash 和旧批准均不得重用。

当前生成协议仍为 9 个逐 case surface-only 请求、每 case 最多一次
预预算 repair、最终 fallback 必须为 0，但删除了词面意图硬标签，引入冻结本地语义编码器，
并把 Advice Readiness 与 Strategy RAG 边际价值做成独立因子。本轮又冻结了 exact runtime、
section-aware input 与训练/外部 lineage，因此 V8.3 dry-run identity
`758ae052...8cf3df2` 也已失效，只保留历史。当前没有 fresh identity、没有付费授权；状态仍是
`NO-RUN`，且不再是 V8.2 的 open reconciliation。

## 2. 全链路因果图与实验身份

每个 reportable 单元必须沿同一条内容寻址链传播：

```text
数据来源与实例血缘
  -> split / state / evaluator-only context
  -> pre-action Step-0 observation
  -> requested_action_id
  -> retrieval_attempts
  -> realized_action_id + realized evidence
  -> prompt_equivalence_id + exact supporter treatment
  -> generated response
  -> development/final judge treatment
  -> response/risk labels
  -> train-only model-family selection
  -> calibration-only selector/cost frontier
  -> frozen candidate + sealed internal bundle
  -> one-time internal Gate M / Gate F
  -> study freeze + external condition matrix
  -> external Gate E
  -> 与实际通过 gate 完全一致的论文文字
```

“同一套东西”不等于所有条件输入文本完全相同，而是除研究操纵外，所有影响 outcome 的
处理都相同或被明确建模。例如不同用户的 memory 内容本来就应不同；但 retriever、top-k、
minimum score、Evidence Filter 状态、supporter prompt、generator model/temperature/cap 和
finish-reason 规则不能因 condition 或 split 偷偷变化。

## 3. 严重度和状态词典

### 3.1 严重度

- `C0`：可使主要因果比较无效、逆转结论或使 holdout 不再成立；必须阻止付费/训练/报告。
- `C1`：可显著夸大性能、缩小不确定性或使机制主张超出证据；必须修复或降级主张。
- `C2`：工程、复现、成本或说明问题；通常不单独推翻结果，但可掩盖 C0/C1。
- `LIM`：当前单步 synthetic/LLM-judge 研究无法仅靠代码消除的永久边界。

### 3.2 状态

- `HISTORICAL_CLOSED`：历史问题已定位，相关旧结果不得迁移到新合同。
- `CODE_CLOSED_RUN_UNVERIFIED`：已有代码门和测试，但真实数据尚未证明会通过。
- `PENDING_RUN_EVIDENCE`：只能由未来冻结运行回答。
- `PERMANENT_LIMITATION`：必须在论文中披露，不能通过继续打补丁假装消失。
- `OPEN_RECONCILIATION`：当前 artifact 或单一事实源互相矛盾，必须先停机对齐。

“代码已修”永远不等于“科学问题已解决”。例如 shortcut audit 已实现，只说明我们有尺子；
只有实际 468-state 报告通过，才能说明本次数据没有触发该门。

## 4. PM-v1 全局问题账本

### 4.1 treatment、评测与比较器

| ID | 级别 | 问题 | 对主张的影响 | 当前处理 |
|---|---|---|---|---|
| V1-SYS-01 | C0 | V1 研发链历史上出现过 internal/external prompt、RAG、Evidence Filter、seeker world 或 baseline treatment 不一致 | 不能把差值归因给路由策略；不同机制的结果不能拼表 | 历史旧链不得复用；V1.5_1 用单一 supporter/retrieval/filter 合同和 freeze 绑定 |
| V1-SYS-02 | C0 | 早期 interactive comparison 允许不同 policy 产生不同 seeker 后续世界线 | 比较同时混入了 policy 和输入轨迹差异 | 后来改为固定 seeker tracks；当前只支持 fixed-input 单步因果比较 |
| V1-EVAL-01 | C0 | 10-turn bundle pairwise 出现强 AB/BA 顺序效应，四项 orientation consistency 均未过 0.80 | 816 次调用和约 12.27M prompt tokens 的该批结果只能作失败诊断 | `HISTORICAL_CLOSED`；不得作为 confirmatory result |
| V1-EVAL-02 | C1 | V4 multi-candidate absolute scoring 仍有 candidate-set/contrast effect | 不同 judge prompt 中的绝对 PM 分不能互相替换 | 只允许同一 prompt 内 paired delta；V1.5 增加位置平衡和数值 order pilot |
| V1-COMP-01 | C0 | 原主链缺少不可绕开的同预算 fixed；后补 `ME+R0` 才显示 PM 不占优 | 仅胜过 Full History/高资源 fixed 不能证明 learned routing | V1.5 Gate F 强制 cost-matched fixed 与 `ME+R0` 护栏 |
| V1-COMP-02 | C1 | 高资源 `best_fixed`、same-token fixed、Context Only、raw-session 和 full-history回答不同问题 | 把它们混成“PM 全面更优”会扩大主张 | V1.5 按 Gate M/F/E 和 secondary references 分层陈述 |

这里必须特别保留一个历史事实：V1 最终版本的许多直接泄漏和世界线问题后来已修复，
所以不能把 V1 的负面结果简单解释成“全是代码作弊”。V1 的有效负面结果主要说明当前
数据、观测、训练和 selection 不足；但任何产生于 treatment 不一致阶段的旧数字仍不可用。

### 4.2 数据、标签和统计单位

| ID | 级别 | 问题 | 对主张的影响 | 当前处理 |
|---|---|---|---|---|
| V1-DATA-01 | C1 | 1,728 cards 主要是 192 states × 9 inventory variants，只有 36 个 unique current texts | 行数虚大；模板和状态语义的有效独立样本很小 | V1.5 重新生成 52×9 states，并要求真实 corpus 审计 |
| V1-DATA-02 | C1 | user-fold 没有同时隔离 semantic family 和 normalized text | 内部泛化可由相同语义/句式模板支撑 | 新 split 和近重复/identity shortcut 审计；实际效果待运行 |
| V1-DATA-03 | C1 | 各 state 只有 2/4/8/16 个不等 action；完整 16-action state 很少 | action 与 inventory-rich 状态绑定，动作效果不可公平比较 | V1.5 每个 state 必须完整 sweep 16 actions |
| V1-DATA-04 | C1 | 没有均衡 context-only、memory-harmful、source-needed、strategy-helpful/harmful 等 regimes | M0/R0 即使在动作空间中，也未获得足够“应该赢”的监督 | V1.5 九 regime；但语义成立与否必须由 actual-corpus gate 证明 |
| V1-LABEL-01 | C0 | judge 的 `Overall` 与 Emotional Support 完全重合 | 所谓整体质量其实是 support，原主指标构念失效 | 删除 Overall；冻结六维 composite，并设 duplicate/correlation gate |
| V1-LABEL-02 | C1 | 单一 silver judge、风险维度低方差、verdict 与数值不一致 | PM 可能只学 judge 偏好，风险头接近常量 | 双 development family、逐维 MAD/constant/duplicate gate；无人评仍是 limitation |
| V1-STAT-01 | C1 | 同一 user/state/action 派生行高度相关，却容易被当作大 N | 置信区间过窄、模型选择过度乐观 | 以 user 为主要 bootstrap/block 单位；prompt alias 另行折权 |
| V1-RISK-01 | C1 | representative sample 与 stress sample 给出不同风险印象 | stress 率不能当 prevalence，普通抽样也不能证明安全 | 分层报告 representative/stress，风险主张限于 evidence/resource-use risk |

### 4.3 观测、模型、选择和检索

| ID | 级别 | 问题 | 对主张的影响 | 当前处理 |
|---|---|---|---|---|
| V1-OBS-01 | C1 | stable external PM 看不到实际 source relevance，只见文本和 metadata | 16 路选择在信息上可能不可辨识，理性退化为 action prior | V1.5 正式化有限 Step-0；无 Step-0 仅作消融 |
| V1-FEAT-01 | C1 | fitted TF-IDF 在 synthetic 覆盖 100%，EvoEmo token coverage 约 26.7% | semantic OOD 导致内部多样、外部 collapse | 固定表示和 Step-0；外部泛化仍需真实结果 |
| V1-MODEL-01 | C1 | Logistic/Ridge 线性头主要学习绝对分和 action prior，缺少可靠 uncertainty | 难以学习状态×动作交互，微小噪声决定 winner | V1.5 train-only 算法 family、bootstrap HGB、delta/residual/rank 候选 |
| V1-SEL-01 | C0 | `epsilon=0` 使成本只在完全同分时 tie-break | 名义 quality–risk–cost 实际是 quality winner-take-all | 新 utility 和 Gate M/F；禁止继续声称 V1 联合优化三者 |
| V1-SEL-02 | C0 | omission/strategy risk 阈值为 1.0，而预测被裁剪到 [0,1] | 两类门实际关闭，RS 的小质量优势即可常开 | 新逐维 UCB/相对风险门；效果待测 |
| V1-POLICY-01 | C1 | 外部 `M0=0%`、`RS≈96.4%`、`MSE+RS≈51.1%` | 没学会 abstention 或稳定策略关闭 | V1 的真实负面结果；新版本以多样性、M0/R0、最大 action share 为硬门 |
| V1-RETR-01 | C1 | source-level PM 配固定 top-k lexical retrieval，无 relevance threshold | 选中 source 就强塞弱相关 evidence；source 正确不等于 item 正确 | V1.5 明确只主张 source routing；realized evidence 和 zero-hit 分开记录 |
| V1-CREDIT-01 | C1 | 最终 outcome 混合 source 选择、item retrieval、generator 使用和 judge 偏好 | 低分无法唯一归因给 PM | 记录完整 lineage；不宣称 item-level 最优性 |

### 4.4 工程、范围与真实负面结果

| ID | 级别 | 问题 | 对主张的影响 | 当前处理 |
|---|---|---|---|---|
| V1-ENG-01 | C1 | 早期 M2b omission 可静默缺失，provenance/freeze 不完整 | 缺失风险标签仍可能被当完整样本 | 后续 fail-closed；旧未 attested 结果不可迁移 |
| V1-NEG-01 | 结果 | PM 未超过 Context Only、Session Retrieval 或同预算 `ME+R0` | learned routing advantage 未成立 | 必须诚实报告，不能靠新版本文字覆盖 |
| V1-SCOPE-01 | LIM | fixed-input 单步、turn 3/8、18 users、LLM judge | 不能证明长期情绪改善、信任或真实用户获益 | 永久披露；需要另一个闭环/真人研究 |
| V1-SCOPE-02 | LIM | PM-v1 是 supervised contextual router，不是 RL/POMDP | 不能声称长期 return、在线适应或 policy-induced distress change | 方法名和论文措辞永久收紧 |
| V1-SAFETY-01 | LIM | risk 是 evidence/resource-use risk，不是临床安全 | 不能写“心理健康安全”或“临床有效” | 使用限定术语并披露无临床验证 |

## 5. V1.5 → V1.5_1 修复时间线

| 阶段 | 主要发现 | 结论 |
|---|---|---|
| 初始会议版，PR #3 head `7ea2248` | free catalog probe、requested/realized 错位、internal 未密封、final judge 参与开发、CI 失败 | `NO-GO`；当时尚无正式结果需要撤回 |
| Step-0 方法争论 | 完全保留免费 probe 不合法；完全删除又可能让 16 动作不可辨识 | 选择正式、有限、可计费 source-level Step-0，并加入无 Step-0 消融 |
| 第一轮 V1.5_1 重构 `484e1e2` 附近 | Step-0、动作 lineage、judge 隔离、算法比较、三层 gate 已进入代码 | 仍发现 internal oracle audit、actual 468 语义门和中央 paid gate 等问题 |
| PR #4 head `b2333ac` | 上述主体修复；seed/Bank 共源确认并消除；chronology/order/internal seal 加固 | 仅 actual-corpus negative controls 仍不足 |
| `c5c38b6` | 12 fields × 2 controls、自然 corruption、静态扫描覆盖完成 | 又发现 semantic judge panel 可被 CLI 替换且未绑定 experiment config |
| `8910f4d` | endpoint alias/family/model/base URL/config hash 全绑定 | 方法代码审查通过，可进入逐阶段 dry-run |
| staged release `5fc2a88` 及后续 | 历史 pilot/approval 被记录；旧 config PASS 不能复用 | 继续遵守逐阶段 exact hash 授权 |
| V8 pilot | provider schema 用无研究意义的 `coverage_rationale<=180` 拒绝真实输出 | fail-closed；证明 provider 不应负责 evaluator rationale |
| V8.1 pilot | 正交 family 后仍有 2/9 surface fallback | 暴露 whole-bundle generation 的跨 case 污染/稳定性问题 |
| V8.2 真实 pilot | 8 次物理尝试、6 成功、2 失败；词面 advice-request gate 误杀语义有效输出 | `CONSUMED_FAILED_CLOSED`；approval/index 已按账本对齐 |
| 本轮修复前的 V8.3 工作树 | 冻结本地语义表示；Advice Readiness × Strategy Resource 正交；same-topic irrelevant decoy；pilot/formal 共用 V14 compiler | 该 dry-run 已因后续 runtime/input 修复失效；需全新 post-repair dry-run、审查和精确批准 |

## 6. V1.5_1 全局问题账本

### 6.1 方法身份、treatment parity 与论文对象

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-ID-01 | C1 | `pm-v1.5-supplemental` 与重新训练会议版都曾简称 V1.5 | 分支、release revision、checkpoint、freeze、结果完全分离 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ID-02 | C0 | development/sweep/external 可能分别声明 prompt、RAG、EF 或 generator 参数 | 单一 `supporter_generation_treatment`；所有 stage 内容寻址绑定 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ID-03 | C0 | V1.5 关闭 EF，但共享 metadata 曾写成 supervised filter | 行为与名称都必须写 `disabled_passthrough`；不得让审查者误认处理不同 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ID-04 | C0 | pilot 修成新生成协议，但 formal 52-user 一度仍保留旧 whole-bundle 路径 | pilot 与 formal 必须共享同一 contract/version/schema/prompt compiler | 当前已同步改为 casewise，`RUN_UNVERIFIED` |
| V15-ID-05 | C1 | V1.6/其他分叉与 `pm-v1.5_1` 同时存在，容易审错 branch/head | 每次审查和运行记录 branch、commit、dirty status、config SHA | 持续护栏 |
| V15-ID-06 | C1 | canary/forced-swap 的布尔名曾暗示 efficacy | 明确 `judge_sensitivity_not_pm_efficacy`；主效应由冻结 paired CI 决定 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ID-07 | C1 | 归一化 margin 0.02 容易写成原始量表 0.02 分 | 明确 [0,1] 的 0.02 ≈ 原 1–5 量表 0.08 | 持续论文护栏 |
| V15-ID-08 | C1 | fixed seeker 的“回复尽量不超过 60 tokens”一度可能被误作 provider output cap，造成截断或不同输入轨迹 | 60 只作内容指令；API cap 固定 300；只接受 complete finish reason，所有 condition 共享冻结 tracks | `CODE_CLOSED_RUN_UNVERIFIED` |

### 6.2 pre-action observation 与 shortcut

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-OBS-01 | C0 | 决策前读取 MP/MS/ME query-to-catalog similarity，却称 pure pre-retrieval，且不计成本 | 正式 Step-0，公开方法身份、固定表示、单独计量；no-Step0 作消融 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-02 | C0 | catalog embedding norm/mean/std 可成为 source/environment/template 指纹 | 从 PM-visible features 删除；仅保留任务相关有限标量 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-03 | C0 | synthetic “需要哪个 source 就放匹配内容”可使 centroid similarity 近似 oracle | train-only 单阈值 + 多变量 group-CV probes、shuffled/permuted/noise/no-Step0 审计 | `PENDING_RUN_EVIDENCE` |
| V15-OBS-04 | C1 | 若 version/hash/model ID 进入特征，会成为环境身份捷径 | version/hash/dimension/build path 只进 audit，不进 PM | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-05 | C0 | Strategy Step-0 若先跑完整 top-k retriever，就是改名后的 item retrieval | 只允许 family centroids + current-turn readiness；否则必须正式记 retrieval attempt | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-06 | C1 | 完全删除 Step-0 会使 MP/MS/ME/RS 选择信息不足并退化成固定动作 | 保留合法有限 observation；无 Step-0 只作为可辨识性消融 | 设计选择，待结果 |
| V15-OBS-07 | C0 | shortcut audit 一度在 candidate freeze 前读取 internal needed sources/regime | predictive oracle 仅 train 216 states；其余 split 只做无 oracle 结构审计 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-08 | C1 | 只有单特征阈值审计可能漏掉 XOR/组合 shortcut | 增加正则 logistic、浅树等低容量多变量 user-group probe | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-09 | C1 | 配置允许 precomputed embedding，但历史 state 实际为空，PM 对未见用户文本仍主要依赖 hash/OOV 表示 | 冻结 `BAAI/bge-small-en-v1.5` 精确 revision/tree hash；当前 turn 与完整可见 state 双视图；train-only PCA；development/external 同 binding | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-10 | C0 | advice request/listen boundary 的词面正则既会误杀 provider 输出，也会成为 RS target 的答案键 | 删除词面硬 gate；五类 readiness 只作为连续语义观测和独立 evaluator 因子 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-11 | C0 | 即使 Step-0 合法，若正样本 source 总有同题 item、负样本永远异题，source centroid 仍可直接读出 oracle | 非 needed source 也放入 same-topic、明确非个人/无边际价值的语义 decoy；实际 item utility gate 与 shortcut audit 共同裁决 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-OBS-12 | C1 | 冻结 encoder 若在运行时下载、漂移 revision、进入 feature identity，或开发/外部用不同 pooling，会重现 treatment mismatch | local-files-only、tree SHA、spec SHA、CLS/normalize/dimension 全绑定；身份只进 provenance/freeze，不进数值特征 | `CODE_CLOSED_RUN_UNVERIFIED` |

### 6.3 requested、retrieval、realized 与 alias

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-ACT-01 | C0 | EF 关闭时直接把 requested action 写成 effective，即使零命中 | `requested -> attempts -> realized -> prompt_equivalence` 四层合同 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ACT-02 | C1 | 把 zero-hit 一律视为非法会删除真实 retrieval failure | zero-hit 是合法后果；保留 call、latency、token、USD 和原因 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ACT-03 | C0 | 外部 evaluator 直到花完生成费用才发现 action/evidence 不一致 | generation 前重算 realized，验证 evidence/action/prompt lineage | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ACT-04 | C1 | 多个 requested actions 形成同 prompt，同一 label 被复制成多份独立证据 | shared `prompt_equivalence_id`、inverse alias weight、user/state 聚类 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ACT-05 | C1 | 物理 prompt 去重可能错误抹掉每个 requested action 的检索成本 | 质量/risk 绑定 realized prompt；attempt/cost 绑定 requested action | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ACT-06 | C1 | required-hit 若在看完 outcome 后筛样，会产生选择偏差 | 只作 response/judge 前的 positive challenge data-validity gate | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ACT-07 | C1 | 自然/external zero-hit 被 required-hit 规则删除 | natural/external 不重抽、不删除，分层报告 | 持续护栏 |

### 6.4 数据血缘、seed、Strategy Bank 与语义有效性

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-DATA-01 | C0 | EvoEmo/ES-MemEval 和 Strategy Bank 都有 ESConv 血缘；exact 清理不能证明语义独立 | source-ID、exact/Jaccard/semantic lineage 审计；外部称 ESConv-derived transfer | `PERMANENT_LIMITATION` + 部分代码护栏 |
| V15-DATA-02 | C0 | 正式 52 seed 一度直接来自与 Bank 相同的 875 source universe | 先冻结 52 source IDs，再从 Bank 删除这些完整对话实例 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-03 | C2 | “875 条 RAG bank seed”表述混淆了 seed pool 与 Bank | 875 是私有 clean seed 候选；52 被选作 development；这些 52 不在 Bank；Bank 是 11,590 cards/823 sources | 已澄清，持续防混淆 |
| V15-DATA-04 | C1 | 同一策略知识/心理支持经验的语义重合被误当成必须全部删除 | 允许领域和 family-level 经验重合；禁止同一 dialogue 实例、未来信息和测试 target 泄漏 | 设计边界 |
| V15-DATA-05 | C0 | 27 个预制审查 case 不能证明实际 468 states 语义成立 | actual-468 双 development-family、12-field 全量 pre-outcome audit | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-06 | C0 | deterministic fallback 模板包含 regime 线索，可成为答案键 | reportable corpus 最终 fallback 必须为 0；失败保留并停机 | post-repair 合同，`RUN_UNVERIFIED` |
| V15-DATA-07 | C1 | fallback 率一度只有日志没有 split-specific 硬门 | train/calibration/internal 分开；internal=0；当前生成合同进一步要求全量 0 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-08 | C1 | source age、row order、inventory、family/template 可能直接编码 regime | counterbalance、随机化和 actual shortcut probes | `PENDING_RUN_EVIDENCE` |
| V15-DATA-09 | C1 | EvoEmo chronology 曾依赖 JSON 原顺序，未验证 ID/date/reference | ISO date 稳定排序、ID/topic/reference fail-closed；不虚构 topic timestamp | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-10 | C1 | exact overlap 清零仍不等于没有母对话语义改写 | top-match/lineage 继续报告；不能称 pristine independent external | `PERMANENT_LIMITATION` |
| V15-DATA-11 | C0 | `strategy_helpful`/`strategy_harmful` 一度同时编码 RS value 与 advice readiness，模型可凭“要建议/只倾听”直接猜 RS | evaluator-only `strategy_resource_target` 与五级 `advice_readiness_target` 分开；两类 Strategy slots 在 user 内交叉配对 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-12 | C1 | 本地预设 Strategy target 可能被误当真实 outcome，形成自我实现标签 | target 仅作 pre-outcome challenge/语义审计；训练、`rs_correct` 和主结论以同 memory subset 的 blinded R0/RS utility 为准 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-13 | C0 | CLI 虽默认指向 V1.5 Bank，但可被替换成另一套 Bank 后仍形成一条内部自洽却偏离冻结方法的链 | 配置冻结 exact path/SHA/card count/source count/audit/52-seed manifest；首个付费 development 阶段在 API client 前强校验，后续由 attestation/freeze 传播 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-DATA-14 | C0 | evaluator-only `advice_readiness_target` 虽已与 Strategy value 反平衡，但真实 V8.4 的两个 Strategy surface 都没有把各自的 `light_suggestion/listen_only` 状态写进可见用户话语，PM 因而无从识别 | readiness 由本地 compiler 以每类 6 种自然句式确定性写入 current turn；52-user 内 Strategy-use/skip 双向反平衡；句式协议/hash 写入 provenance；专用环境 frozen BGE 对 12/12 句式 top-1 正确，且该门不读 outcome/Strategy target | `CODE_CLOSED_REAL_BGE_12_OF_12_FRESH_PILOT_REQUIRED` |

### 6.5 generator contract 与真实 pilot 失败

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-GEN-01 | C1 | 旧 provider 一次生成 9 cases，还同时生成 memory/evidence/oracle/rationale/IDs | provider 只看一个 case 的四个 surface 字段；证据和 evaluator rationale 本地确定性编译 | post-repair 合同，`RUN_UNVERIFIED` |
| V15-GEN-02 | C1 | whole-bundle 让一个 case 的 topic/semantic family 污染另一个 case | one case per physical call；每 case 独立 seed | post-repair 合同，`RUN_UNVERIFIED` |
| V15-GEN-03 | C2 | V8 的 `coverage_rationale<=180` 拒绝 181+ 字符真实输出 | 删除与 provider 任务无关的 rationale 字段；本地生成 | `HISTORICAL_CLOSED` |
| V15-GEN-04 | C1 | relocation/academic/workplace pilot cohort 本身自然混题 | pilot family 必须实际正交，不为测试方便制造不自然 benchmark | V8.1 改用 relocation/self-confidence/sleep |
| V15-GEN-05 | C1 | V8.1 真实输出 2/9 需要 fallback：一例 family 泄漏，一例缺少 anchor | 不放宽 9/9 门；改成逐 case + 一次受限 repair | 旧 V8.3 dry-run 已失效；post-repair pilot 待真实验证 |
| V15-GEN-06 | C1 | 无限制 retry 会把“多试几次直到好”变成选择性生成 | 每 case 初次 + 最多一次预预算 repair；attempt ledger 先写后调用 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-GEN-07 | C1 | 只修 compatibility pilot 而未同步 formal 52-user，会重建 pilot/formal mismatch | 同一 generation contract 被 base 和 V1.5 formal runner 共同使用 | 当前已同步，`RUN_UNVERIFIED` |
| V15-GEN-08 | C2 | casewise 架构把 formal 成功路径从 52 calls 提高到 468、上限 936 | 成本、token、timeout、approval 全部重新 dry-run；不能复用旧 52-call hash | 当前旧 hash 全部 stale |
| V15-GEN-09 | C1 | V8.2 用英文关键词正则把 “tips/help me figure out” 等有效建议请求判失败 | provider lint 仅保留 topic/role/chronology 结构门；意图交给连续语义表示和 actual semantic review，不再作为硬词表 | `HISTORICAL_CLOSED` + 新合同待跑 |
| V15-GEN-10 | C0 | V8.4 的结构 PASS 只证明 9 个 surface 可解析；旧 102-call 自动审核审的是另一批 27 个预制案例，没有绑定这次付费九例，却可被 formal generation 当作语义放行证据 | 自动审核 v3 同时读取 27 个确定性案例、exact paid 9-case attestation 与 24 个 hard controls；两家共 120 logical calls；report/attestation/formal generation/sweep/judging 都强制绑定同一 paid pilot SHA/contract | `CODE_CLOSED_FRESH_PILOT_AND_REVIEW_REQUIRED` |
| V15-GEN-11 | C1 | V8.4 暴露 memory-harmful current turn 缺句号形成 run-on，ME “one manageable next step” 过泛并与 MS 边际贡献接近 | compiler 使用统一句子连接器；MS 明确跨 session 模式，ME 明确一次过去事件及具体记录动作；evidence blueprint hash 随之变化 | `CODE_CLOSED_FRESH_PILOT_REQUIRED` |
| V15-GEN-12 | C1 | V8.5 的 provider schema 允许任意 role list，但返回后 lint 才要求交替并以 assistant 结尾；context_only initial+repair 都生成 `assistant,user,assistant,user`，真实消费 2 calls / `$0.0006274` 后 fail-closed | V8.6 provider 只返回 1–2 个 `{user_text, assistant_text}` exchange；本地 compiler 展开为 `user,assistant[,user,assistant]`，使交替与末尾 assistant 成为结构不变量；真实 V8.6 为 9/9 accepted、0 fallback，12 attempts 中没有该错误 | `CLOSED_V8_6_PAID_PASS` |

### 6.6 holdout、judge、训练与统计门

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-HOLD-01 | C0 | train/calibration/internal 在同一进程读取，internal 可被多次尝试 | pre-training label seal、candidate manifest、append-only one-shot ledger | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-HOLD-02 | C0 | candidate manifest 一度未预绑定 sealed internal label hash | seal 绑定 label/content/state universe/schema hash，消费时三方复核 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-JUDGE-01 | C0 | 为修 NVIDIA/Qwen 问题，开发 panel 一度换入 final GPT-4o | development 与 final 四层隔离；旧产物失效 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-JUDGE-02 | C0 | 只比较 endpoint 名会漏 alias 指向同 model/route | 比 alias、family、model、normalized base_url+model | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-JUDGE-03 | C0 | semantic runner 一度允许 CLI 换成任意两个 development endpoints | exact alias 顺序锁定配置，descriptor 和 experiment SHA 进入 attestation | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-JUDGE-04 | C0 | downstream verifier 一度不绑定当前 experiment config | 当前配置重建 endpoint descriptors 并逐项比对 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-SEM-01 | C1 | actual controls 最初只破坏 family/regime，且 `--n-controls 0` 可绕过 | 12 fields × 2 controls，数量/seed/matrix hash 冻结 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-SEM-02 | C1 | `deliberately_unrelated_control` 太明显，只测 sentinel 识别 | 使用真实 donor、标签翻转、age 矛盾、时序/grounding/strategy corruption | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-SEM-03 | C1 | order pilot 一度只检 schema，`gating_threshold=None` | 3 units × 2 schemas × 2 orders × 2 families，mean/max 数值门 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-SEM-04 | C1 | 5 条 readiness canary 太小，且 runtime 文件顶层 `PASS` 容易被误读成 readiness 全通过 | 扩为 20 条 outcome-free paraphrase challenge；runtime 与 `REPORT_ONLY_x_OF_20` 分开报告，不允许据此调 anchor | `CODE_CLOSED_LOCAL_REPORT_ONLY_14_OF_20` |
| V15-TRAIN-01 | C1 | 当前绝对 HGB 容易浪费容量预测 state 难度，而非 action 边际值 | train-only 比较 absolute、state-centered delta、rule-relative residual、rank 候选 | `PENDING_RUN_EVIDENCE` |
| V15-TRAIN-02 | C1 | 直接换大模型/RL 与 24 train users、完整 action matrix 不匹配 | 小样本可审计监督学习；RL 留给有 transition/user feedback 的后续研究 | 设计边界 |
| V15-TRAIN-03 | C1 | 缺少同观测强 rule 会把弱规则做 strawman | transparent rule 使用相同 Step-0，并作为 Gate M 主比较 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-04 | C1 | calibration 一度同时调 rule、algorithm、OOD、selector、fixed frontier | model family/rule 移到 train-group CV；calibration 只做冻结校准 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-05 | C1 | 仅选最高 CV mean 容易 winner's curse | one-standard-error rule，近最优中选更简单稳定模型 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-06 | C1 | 16 actions×states 不是独立样本，真正有效规模接近 user blocks | user-disjoint fold/bootstrap；报告 prompt-effective N 和 action stability | 持续统计护栏 |
| V15-TRAIN-07 | C1 | 仅增加冻结 embedding 仍可能因高维、小样本、环境 identity 过拟合 | 两个 384 维视图只在 train users 上 PCA 到 48 维；encoder binding 必须一致；group-CV/one-SE 决定是否保留复杂模型 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-08 | C1 | 原 no-Step0 消融仍保留 state BGE，无法区分“语言表示收益”和“Step-0 收益” | internal 一次性消费前同时冻结 full、无 Step-0、无 state-BGE、word/char-only 四格诊断；结果只解释组件贡献，不得选择或重调主 candidate，也不进入 external 主矩阵 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-09 | C1 | rule candidates 只比较聚合 action count，可能把在不同 states 上决策的两套 policy 错判为等价；诊断又到 internal 开封后才出现 | 新增 outcome-free pre-sweep grid preflight，记录每个 candidate 的 `action_by_state_sha256`、unique mapping 与 pairwise disagreement；sweep、training、candidate manifest 内容寻址绑定 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-10 | C1 | development 与 external Step-0 分布虽各自记录，却只能人工比对，且存在外部看分布后调阈值的风险 | external dry-run 读取冻结 training report，生成 calibration-vs-external quantile shift artifact；缺失来源显式标 `UNAVAILABLE`；只报告且禁止 selection/retuning | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-TRAIN-11 | C1 | residual 主算法的 no-Step0 版本会把 reference policy 从 transparent rule 改成 `M0+R0`，却可能被误写成纯 feature ablation | candidate manifest、calibration 和 internal report 显式标记 `component_removal_system_variant` 与 reference-policy change；只有非 residual 候选可称 retrained feature-set ablation | `CODE_CLOSED_RUN_UNVERIFIED` |

### 6.7 comparator、成本与 claim contract

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-COMP-01 | C0 | 外部只比 high-resource fixed，可能让固定低资源策略冒充 learned PM 的成功 | Gate M learned-vs-rule；Gate F cost-matched/ME+R0；Gate E high-resource efficiency | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-COMP-03 | C1 | 为保留“7 conditions”可能删掉关键 comparator | condition matrix 按 claim 组织，数量不是科学目标 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-COST-01 | C1 | 只统计 generator input tokens，却写“总体成本/延迟更低” | Step-0、retrieval、input/output、USD、latency 分开；主指标名称精确 | 持续 claim 边界 |
| V15-COST-02 | C1 | fixed policy 不需要 Step-0，却可能被人为收费以利于 PM | learned/rule 支付真实 Step-0；fixed 不支付未执行成本 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-COST-03 | C1 | retrieval zero-hit 仍有调用成本，若只按 realized action 会漏记 | attempt cost 绑定 requested action，prompt cost 绑定 realized evidence | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-COST-04 | C1 | 冻结 encoder 的本地推理、catalog refresh 和五类 readiness comparison 若不报告，会把“无 API token”写成“免费” | per-turn encoder input/token estimate、invocation、Step-0 latency、一次性 source/strategy catalog refresh 单列；fixed 不执行则为 0 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-CLAIM-01 | C0 | supervised single-step router 被写成 POMDP/RL/长期 distress 改善 | 固定方法名；禁止 RL、clinical、longitudinal causal claim | 持续论文护栏 |
| V15-CLAIM-02 | LIM | EvoEmo 是 ESConv-derived，且被 V1/设计过程反复检查 | 称 development-informed external transfer，不称 pristine external | `PERMANENT_LIMITATION` |
| V15-CLAIM-03 | LIM | 无人工专家/真人、final judge 为 LLM | 结果只代表冻结 rubric 下模型评估 | `PERMANENT_LIMITATION` |

### 6.8 CI、release、hash 与执行安全

| ID | 级别 | 问题 | 永久修法/护栏 | 状态 |
|---|---|---|---|---|
| V15-ENG-01 | C2 | CI 裸 `pytest` 因 `scripts.v32_contract` import 失败；本地 `python -m pytest` 把 cwd 加入 path 而掩盖 | 将模块移入 `src/metacom_pm`；clean venv 按 CI 的裸 `pytest -q` 验证 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ENG-02 | C2 | workflow path filter 未覆盖全部 V1.5 文件 | 触发范围改为 `project/**` | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ENG-03 | C2 | legacy V9 硬编码 `/home` 和旧全局 study freeze hash 阻断新 preflight | 移除机器路径；旧 freeze 标 `STALE_HISTORICAL_FREEZE/confirmatory_only`，不原地刷新 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ENG-04 | C2 | static scan 曾漏掉顶层 `scripts/v1_5_*.py` | release/freeze scan 覆盖所有 active V1.5 scripts | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ENG-05 | C2 | 本地已缓存 BGE，使纯 freeze/external wiring 单测隐式解析真实 snapshot；GitHub clean offline cache 因而 10 项失败 | wiring fixture 只构造与精确 spec/tree 绑定的类型化假 binding；生产 freeze resolver 完全不改、仍 local-only fail-closed；全仓库测试另以空 `HF_HOME` + offline 环境执行 | `CODE_CLOSED_EMPTY_CACHE_AND_CI_PASS` |
| V15-ENG-06 | C2 | external wiring fixture 以 test name 作为共享目录名，多会话并发会互删并产生瞬时 `FileNotFoundError` | 在 release root 内使用每次唯一的 `TemporaryDirectory`；既保留 freeze 路径约束又消除跨进程碰撞 | `CODE_CLOSED_FULL_TEST_PASS` |
| V15-ENG-07 | C2 | 以外部 `tmp_path` 运行 release preflight 时仍重写真实 `release_manifest.json`，并错误收录 tracked `release_preflight.json` | preflight 支持独立 `manifest_out_path`；只排除 release root 内的真实生成目标；测试所有输出均写唯一临时目录 | `CODE_CLOSED_PREFLIGHT_PASS` |
| V15-ENG-08 | C1 | `sim_eval` 虽能跑 mock tests，但 Python 3.10、Transformers 5/HF Hub 0.23 组合不符合项目合同且真实 BGE import 失败；仅锁权重也不能约束 PCA/HGB 训练数值 | 专用 Python 3.13.2 `.venv-pm-v1-5` + `PYTHONNOUSERSITE=1`；Python/package/device/dtype/user-site 与公共 3×384 canary 写入 config；development、训练入口都现场重算，candidate/freeze/external 逐段绑定 | `CODE_CLOSED_LOCAL_CANARY_PASS_FORMAL_RUN_UNVERIFIED` |
| V15-ENG-09 | C1 | 512-token `truncation=True` 没有记录，可能让 external 的历史/摘要被静默截断 | 冻结 section-aware input v2：当前话语/summary 固定预算、history 保留最近 token suffix；模型端 implicit truncation 必须为 0，逐 section 只记录计数/hash；current turn 另有完整独立 view | `CODE_CLOSED_LOCAL_TEST_PASS_FORMAL_RUN_UNVERIFIED` |
| V15-ENG-10 | C2 | transparent rule 实际用 train/train-fold 调参，但 doc/report/YAML 声称 calibration；rule decision 还冒用 learned reason | 显式记录 `train_only/train_fold_only`，独立 `TRANSPARENT_RULE_SELECTION_REASON` 并按非 fallback 处理 | `CODE_CLOSED_FULL_TEST_PASS_FORMAL_RUN_UNVERIFIED` |
| V15-ENG-11 | C1 | wrapper 虽现场验证 BGE runtime，external artifact 自身却未写入该次 runtime 与 training-distribution lineage | learned/rule 的 manifest、preflight、cost、summary、attestation 全部写 live runtime；training report 作为 attested input，另输出内容寻址的分布比较 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-ENG-12 | C0 | Memory/Strategy Step-0 曾使用旧未预算 full text，而 state BGE 使用 section-aware bounded text；长 external context 下两类 PM 特征可能基于相反的历史片段 | 单一 `prepare_visible_semantic_state` 同时生成 bounded text/current+state vectors/audit；development、EvoEmo inventory、runtime adapter 和 readiness challenge 共用；文本与向量双 hash 必须相等 | `CODE_CLOSED_REAL_BGE_STATE_SMOKE_PASS_FORMAL_RUN_UNVERIFIED` |
| V15-ENG-13 | C0 | section-aware encoder 已输出 `section_allocation`，但 `PMV2State` 严格 schema 仍只允许旧 telemetry keys；真实 BGE `case_to_state` 会直接 ValidationError，普通 fake encoder 测试未覆盖 | schema 严格接纳并逐字段校验 allocation；长 development/external 测试、篡改反例及真实 BGE strict-state no-API smoke 全部执行 | `CODE_CLOSED_REAL_BGE_STATE_SMOKE_PASS_FORMAL_RUN_UNVERIFIED` |
| V15-ENG-14 | C1 | Gemini judge 曾被当作普通 OpenAI-compatible endpoint，向 `/openai/chat/completions` 发送 OpenAI `response_format/json_schema`；真实 HTTP 400 又因 list-shaped error body 无法提取原因，容易把 transport 不兼容误判为 rubric/schema 能力不足 | endpoint identity 显式冻结 `transport`；Google judge 改用官方 native `generateContent` 的 `responseMimeType=application/json + responseJsonSchema`，schema 与本地 Pydantic 合同不放宽；错误解析递归但有界地读取 dict/list 白名单字段，只存安全摘要和 body hash；Gemini 排在 fresh full matrix 首个调用 | `CODE_CLOSED_LOCAL_MOCK_PASS_NATIVE_PAID_RUN_UNVERIFIED` |
| V15-ENG-15 | C1 | 冻结 generation-pilot 预算的回归测试未显式传 seed corpus，因本机恰有被 gitignore 的 3.3 MB development seed 文件而本地通过，GitHub clean checkout 必然 `FileNotFoundError` | 测试在 `tmp_path` 生成 52-user cohort + 1 held-out 的最小确定性合成 seed fixture，并显式传 `--seed-dialogues`；不跳过测试、不提交本地 development corpus | `CODE_CLOSED_CLEAN_CHECKOUT_FIX_CI_PENDING` |
| V15-COST-05 | C1 | 统一语义输入后仍沿用旧 4 encoder invocation / `3×full+2×current` 估算会虚报机制并掩盖真实实现 | 当前实现按两次 `[current,bounded_state]` 批量编码记录 2 invocations，token 数直接取 tokenizer telemetry；无 semantic encoder 的 legacy/fixed 路径为 0 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-REL-01 | C0 | `PAID_RUN_BLOCKED` 一度只是文档说明，各入口可直接 `--run` | 中央 release gate；每 stage 绑定 config/revision/run/cost hash | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-REL-02 | C0 | 旧 pilot、partial attempts、旧 cost hash 或旧 PASS 可能被拼接复用 | immutable fresh output dir；旧 lineage 一律 stale；不覆盖、不拼接 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-REL-03 | C1 | API 失败被误解为额度问题，或未知 attempt 被盲重试 | HTTP 前 ledger fsync；unknown 视为已花费；仅 429 支持限流判断 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-REL-04 | C0 | paid release manifest 与 review artifact index 曾对 V8.2 approval 状态相互矛盾 | 按物理 ledger 记录为 `CONSUMED_FAILED_CLOSED`，删除 active approval；V8.3 必须 fresh identity | `HISTORICAL_CLOSED` |
| V15-REL-05 | C1 | 代码/config/bank/prompt 修改后沿用旧 accepted hash | 任一输入变化使 stage approval 失效，必须 fresh dry-run + explicit approval | 持续护栏 |
| V15-REL-06 | C2 | `API_PILOT_READY` 被误读成 `CONFIRMATORY_READY` | 报告两种状态；无 checkpoint/Gate M/F/freeze 时 confirmatory=false | 持续护栏 |
| V15-REL-07 | C1 | formal generation CLI 默认指向已失效 V8.3 attestation，容易让旧 provenance 被无意继承 | 删除默认值；paid `--run` 必须显式传 fresh pilot attestation，并拒绝 V8/V8.1/V8.2/V8.3 已知历史目录 | `CODE_CLOSED_RUN_UNVERIFIED` |
| V15-REL-08 | C0 | 在 `PAID_RUN_BLOCKED` 配置上生成 cost hash、再切换 release 状态会改变全配置 SHA，使刚批准的 identity 必然失效 | 两阶段 release：先冻结 `PAID_RUN_RELEASED`，但 manifest 保持 pending/空 approvals；证明 `--run` 仍 fail-closed；只批准随后在稳定配置上生成的 post-release identity | `HISTORICAL_CLOSED` |
| V15-REL-09 | C0 | 兼容 pilot 的 transport/schema PASS 可能被误写成完整语义 PASS 并直接授权 52-user | V8.4 原始账本、费用与 PASS 完整保留但语义性淘汰；V8.5 失败账本完整保留；V8.6 独立 identity 真实 PASS 并只授权进入绑定 exact attestation 的双家族语义审核，不直接授权 52-user | `V8_4_ARCHIVED_V8_5_FAILED_V8_6_PAID_PASS_REVIEW_PENDING` |
| V15-REL-10 | C0 | V8.6 成功后虽写入 `stage_consumptions`，manifest 仍保持 `APPROVED`、旧 stage identity 和 `paid_execution_authorized=true`；中央门又不读取 consumption，换输出目录可理论性复用同一 cost identity 再次付费 | 成功后清空 approval 并关闭 paid flag；中央门从 current consumptions 与 immutable history 汇总 consumed identities，旧 identity 永久硬拒绝 | `CODE_CLOSED_FULL_TEST_PASS` |
| V15-REL-11 | C1 | V8.6 人工回填的 approval/consumption 时间分别为 `02:30/02:35Z`，晚于不可变 API ledger 与 Git commit 约 48–53 分钟，不能作为可信执行时间 | approval 改取 `dd52ff1` commit time `01:41:43Z`；consumption 改取 ledger 最后一条 terminal event `01:42:23.305920Z`，并记录 approval/code/result 三个 commit SHA | `RECONCILED_TO_IMMUTABLE_EVIDENCE` |
| V15-REL-12 | C0 | “防止同一付费调用重跑”曾实现成“stage 名一旦消费就永久禁止”，导致 transport/code 修复后即使 fresh dry-run + fresh exact approval 也无法合法重跑同一科学阶段；反过来若把旧记录移出 current slot，又可能忘记阻止历史 identity | 消费门改为全历史 identity 唯一性：同一 identity 在任何 stage/current/history 中永久拒绝；同一 stage 只有在代码/配置变化、新目录、新 dry-run identity 和独立 exact approval 全部成立时才允许再次执行 | `CODE_CLOSED_TARGETED_TEST_PASS_FULL_RUN_PENDING` |
| V15-REL-13 | C1 | V3 自动语义审核记录的 `approved_at=02:50Z` 晚于 immutable terminal ledger `02:13:57Z`，形成“不可能时序” | approval 时间按 approval commit `94e7450` 校正为 `02:10:37Z`；consumption 保持 ledger 精确时间 `02:13:57.074194Z`；结果 commit `40ce6fe` 一并绑定，旧 identity 移入 immutable history | `RECONCILED_TO_IMMUTABLE_EVIDENCE` |
| V15-REL-14 | C1 | V8.6 generation attestation 绑定整份 experiment/PM config 与共享 API 文件；只修 downstream Gemini judge transport 也会令上游 9-case 内容失效，诱发无科学意义的重复付费；若直接忽略旧 input hash，又会破坏 provenance | 不给 V8.6 事后重新盖章；V8.7 fresh contract 改为精确 `generation_config_projection`（generator endpoint、data_generation、cost planning）并继续绑定 seed、prompt/schema、每个 request payload 与共享生成代码；judge-only 配置不再进入上游 identity，generator 任何变化仍 fail-closed | `V8_7_FRESH_DRY_RUN_PASS_PAID_RUN_PENDING` |
| V15-REL-15 | C0 | V8.7 的首个候选 identity `117d…` 依赖未写入正式入口的 CLI 预算参数（`0.018/4000`）；使用默认命令会得到 `aff6…`（`2.0/12000`）。科学合同与调用计划相同但预算身份不同，审批者无法仅凭标准命令复现唯一 cost hash | 将 V1.5 pilot 的 18 attempts / `$0.018` / 4000-token 上限冻结为 `pm_v1_5.yaml` 的独立 stage contract（不污染后续共用的 token-planning mapping）；runner 从配置读取并拒绝冲突 CLI；两次独立输出目录 dry-run 必须产生相同 hash。旧 `117d…`、`aff6…`、中间结构候选 `78c…` 均禁止批准，唯一候选 `920807ec…84e8` 已真实 `CONSUMED_PASS` | `CODE_CLOSED_REPRODUCIBLE_PAID_RUN_PASS` |
| V15-REL-16 | C0 | V8.7 结果记录新造 `stage_consumptions_history` 保存 V8.6，但中央门只读取 current consumptions 与 `prior_stage_attempts_history`；在 manifest 当前关闭时测试旧 identity 会因全局状态失败，不能证明未来重新开放后仍永久拒绝 | V8.6 记录迁回既有 `prior_stage_attempts_history`；门禁兼容读取并严格验证旧平行字段；新增回归测试在 manifest 重新 `APPROVED` 后确认历史 identity 仍命中 `already been consumed`，同时 fresh identity 可合法运行 | `CODE_CLOSED_FULL_SUITE_PASS` |
| V15-REL-17 | C1 | V8.7 manifest 的 prepared/approved 时间早于其引用的 code/approval commits，形成“批准尚不存在 SHA”的不可能时序；artifact index 与核心计划仍写未执行 | prepared time 对齐 code commit `70d035b` 的 `03:31:57Z`，approval time 对齐 commit `c6e439b` 的 `03:37:55Z`，terminal time 保持 ledger `03:38:25.294639Z`，并绑定 result commit `17fb0a9`；三份当前状态文档同步为 `CONSUMED_PASS` | `RECONCILED_TO_IMMUTABLE_EVIDENCE` |

## 7. 修复本身曾引入或差点引入的新问题

这是今后最需要反复阅读的一节。每次“修一个点”至少要审查以下二阶影响。

| 原修复动机 | 新风险 | 最终原则 |
|---|---|---|
| 本地反复跑全量测试 | `python -m pytest` 隐式改变 import path，掩盖 CI 真实失败 | 验证命令必须与 clean CI 完全相同 |
| 替换不稳定 development judge | 换入 final GPT-4o，造成 evaluator leakage | 先检查角色隔离，再替换 endpoint；alias 不等于 family/model/route |
| 删除非法目录 probe | 16-way source selection变成盲猜，可能退化为 `ME+R0` | 用合法、有限、计费 Step-0；删除和保留都要有因果定义 |
| 给 PM 增加语义模型 | 只在训练 state 填 embedding、外部仍为空，或模型身份进入特征造成新环境捷径 | 同一 local snapshot/revision/tree hash/pooling 在 development 与 external 复算；身份仅审计；parity test |
| 用语义模型替代正则 | 把 semantic classifier 的 argmax 当新 oracle，仍然是“更聪明的答案键” | readiness 只作连续 observation；Strategy value 由盲评 R0/RS outcome 决定 |
| 打散 Strategy/advice 标签 | 若只交换字段而不改变 provider surface，标签会变成随机噪声 | provider 明确生成独立 readiness surface，actual-corpus judge 验证；resource target 仍非 outcome |
| 给负 source 添加同题 decoy | decoy 若实际有帮助，会污染 memory oracle；若太怪会形成模板 shortcut | 使用自然、明确非个人的同题信息；item-utility semantic gate + shortcut audit 双门，失败即重设计而非调阈值 |
| 强制 requested==realized | 将真实 zero-hit 错误当成非法数据 | 分开 requested/attempted/realized，不强求相等 |
| 要求 positive source 命中 | 若 outcome 后删除 zero-hit，会产生选择偏差 | required-hit 只能发生在 response/judge 前 |
| 为 actual semantic gate 加 controls | 仅 family/regime、可设 0、明显 sentinel，形成假安全 | 完整字段覆盖、自然 corruption、数量/seed/hash 锁定 |
| 清除 seed/Bank 共源 | “清除 875”被误解为把策略经验全部删掉 | 删除正式 52 个实例来源，不删除合法领域知识；明确 875/52/823 含义 |
| 修全仓 preflight | 原地更新旧 freeze hash 会伪造历史有效性 | 旧 freeze 只标 stale；新实验生成新 freeze |
| 收紧 provider schema | 180 字符 rationale cap 阻断与研究无关的文本 | provider 只生成它必须控制的 surface；评价字段本地编译 |
| 用 pilot 检混题 | relocation/academic/workplace 本身不正交 | pilot 也必须代表合法任务，不用人工冲突制造失败 |
| 放宽 V8.1 的 2/9 fallback | 会把不稳定生成器表面写入训练数据 | 不放宽 gate；改变请求粒度并预预算一次 repair |
| 只改 pilot 生成器 | formal 仍走 whole-bundle，重新产生 mechanism mismatch | 修改共享 contract 后逐一核对所有消费者 |
| 改成逐 case 生成 | 正式调用从 52 变 468，最大 936 | 方法改进必须同步更新成本、超时、ledger、approval、文档和 hash |
| 发布 staged approval | manifest 与审查索引可能出现不同状态 | 用户授权也是内容寻址的单一事实，不能靠文件先写成已批准 |

## 8. 不可回归宪法

以下规则高于“某个测试过了”或“某次 review 没提到”。任何一条被触碰都必须在 PR 中显式
说明；不能以“只改了一点”跳过全链检查。

1. **单一 treatment 身份。** 除研究操纵外，development、sweep、fixed comparator 和
   external 的 generator、prompt、normalization、finish reason、retriever、阈值和 EF 相同。
2. **严格 pre-action 边界。** requested action 前只有冻结的 Step-0；不得出现 item text、
   top-k、ID、oracle、outcome 或未来信息。
3. **动作四层语义。** requested、attempted、realized、prompt-equivalence 永不合并。
4. **zero-hit 不消失。** 它保留成本和失败含义，不因 realized action 改写调用历史。
5. **实例级数据隔离。** 正式 52 seed sources 不得进入 Strategy Bank；外部 lineage 只按可证
   范围陈述，不把领域知识重合误叫作弊，也不把 exact-clean 误叫完全独立。
6. **outcome 前数据门。** required-hit、semantic validity、fallback、shortcut audit 都发生在
   response/judging 前；看 outcome 后不得重生成、删 state 或改 regime。
7. **实际 corpus 优先。** 模板/27-case pilot 不能替代对真实 468 states 的检查。
8. **reportable data 零确定性 fallback。** 如果 provider/repair 仍失败，整次 run fail-closed。
9. **holdout 真正一次性。** internal bundle 先 seal，candidate 先 freeze，ledger 先 STARTED，
   outcome 只消费一次；失败后需要新的 held-out users，而不是重测同一 bundle。
10. **judge 角色与 lineage 隔离。** development/final 在 alias、family、model、route 四层分离；
    endpoint descriptor 和 experiment config hash 进入下游验证。
11. **有效样本量诚实。** user 是主要独立 block；alias labels、16 actions 和同一 state 的派生行
    不能扩大独立 N。
12. **强竞争比较器不可绕开。** rule、cost-matched fixed、`ME+R0` 和 high-resource fixed 各自
    回答不同问题；不能只展示最有利比较。
13. **成本向量不混写。** Step-0、retrieval、generator input/output、USD、latency 和实验 judge
    cost 分开；主张必须写准确 metric 名称。
14. **pilot 与 formal 同机制。** pilot 只可缩小数据量，不能更换 schema、prompt compiler、
    retry、fallback 或 provider-visible fields。
15. **失败是合法结果。** Gate M/F/E 或 pilot 未过时输出 `NOT_SUPPORTED/FAIL`；不得在同一
    holdout 上调阈值、换模型、换 comparator、删样本直到通过。
16. **版本和结果不继承。** PM-v1、legacy supplemental、V1.5_1 及其他分支的 checkpoint、
    freeze、dry-run、approval 和结果不得跨版本拼接。
17. **文档不能领先于事实。** `PASS`、`APPROVED`、`SUPPORTED`、调用数和 hash 必须与唯一
    artifact 事实一致；冲突时一律按 NO-RUN 处理。
18. **主张不超过设计。** 当前只是一轮 supervised pre-item-retrieval router 研究；不声称
    RL/POMDP、长期用户改善、临床安全或普遍外部泛化。
19. **Advice Readiness 不等于 Strategy Value。** listen-only 仍可受益于 reflection card；请求
    suggestion 也不代表当前 Bank 检索有边际价值。训练标签只能来自冻结 R0/RS outcome。
20. **语义表示也是 treatment。** encoder、revision、snapshot tree、pooling、normalization、输入
    拼接和 projection 任何一项改变，都使旧 states、shortcut report、checkpoint 和 freeze 失效。

## 9. 改动影响矩阵

任何未来 PR 必须查表。表中“失效”表示旧 artifact 不得继续作为当前链上游证据。

| 改动对象 | 至少失效的下游 | 必须重跑/复核 |
|---|---|---|
| seed pool、selected 52 或 split | generation plan、Bank exclusion、states、labels、checkpoint、internal、freeze、external | seed/Bank intersection、split/duplicate、全 development 链 |
| Strategy Bank/card/family | Step-0、RS retrieval、shortcut audit、sweep、所有 fixed/learned external | Bank lineage、family coverage、dev/external parity、全部下游 |
| generation prompt/schema/provider-visible fields | compatibility pilot、52-user corpus、semantic reports、所有后续 | V8 类 pilot → semantic pilot → formal generation |
| repair/retry/fallback 规则 | cost plan、attempt ledger、corpus provenance、fallback gate | fresh dry-run/approval 和生成链 |
| Step-0 representation/feature/quantization | shortcut audit、rule、learned model、cost、external | 468 shortcut gate、train/calibration/internal、freeze、external |
| retriever/top-k/min-score/source semantics | required-hit、realized action、prompt alias、cost、fixed comparators | pre-outcome retrieval audit、full sweep、训练和 external |
| action ID/alias/realized semantics | 16-action matrix、labels、oracle/regret、checkpoint、external | action-contract tests + 全 sweep 后链 |
| supporter model/prompt/temp/cap/finish normalization | 每个 response outcome 和外部比较 | 全部 generation/judging/训练/external |
| development judge panel/rubric | labels、algorithm selection、calibration、internal | semantic/control pilot、full judging、重新训练 |
| final judge panel/rubric/order schema | external canary/order/main outcomes | final dry-run、canary、order pilot、external judging |
| quality/risk composite、margin、utility/cost weights | selector、rule/fixed frontier、Gate M/F/E、paper claim | train/calibration/internal；若 freeze 后变更则 external 全失效 |
| model candidates/CV/one-SE 规则 | candidate manifest、internal | train-only selection、calibration、新 internal users（若旧 internal 已开） |
| external conditions/comparator | batched schema、position balance、cost、claim estimand | condition dry-run、canary/order、external main |
| pricing/token bounds | cost hash 和 stage approval | fresh dry-run、明确用户授权；通常不必重生科学数据，除非超预算停止 |
| release manifest/approval | 仅执行权限，不改变科学方法 | 对齐用户授权、config SHA、stage/run/cost hash；不得倒签 |
| 只改文档 | 通常不改变 pilot treatment；若文档改变主张、状态或冻结合同则仍需对齐配置/manifest | link/事实一致性检查；必要时重生成 review index |

## 10. 分阶段停止门

### 10.1 任何付费 pilot 前

- clean CI 命令与 GitHub Actions 相同；
- branch/commit/dirty diff 被记录；
- config、prompt、schema、Bank、seed 和 endpoint descriptors 内容寻址；
- dry-run 给出 logical calls、maximum attempts、per-call token bound、最大 USD；
- release manifest、artifact index 和用户明确批准完全一致；
- fresh output directory、无旧 ledger、无覆盖选项。

### 10.2 完整 52-user generation 前

- 当前代码创建全新 compatibility pilot 目录和 fresh dry-run identity，经独立审查和精确批准真实 PASS；任何 V8–V8.4 历史 attestation 均拒绝；
- 27 个确定性 real cases + exact paid 9-case artifact + 24 controls，经 2 个 development families 共 120 logical calls 的 semantic pilot PASS；
- pilot 与 formal 使用同一 surface-only casewise contract；
- selected 52 与 Bank source intersection 为 0；
- formal 468/936 call 预算重新 dry-run 并单独批准。

### 10.3 7,488-action sweep 前

- 468/468 states 完整且无 deterministic fallback；
- actual-468 12-field semantic gate PASS；
- required-hit 只按预注册 positive challenge 规则检查；
- train-only oracle / all-split structural shortcut audit PASS；
- outcome-free train/calibration rule-grid mapping/disagreement preflight PASS；
- retrieval/action/prompt alias contract 的 deterministic preflight PASS。

### 10.4 训练与 internal 前

- 专用 Python 3.13.2 venv 的 live runtime/canary 与 development 记录完全一致；
- 7,488 outcomes 和双-family labels 完整；
- dimension constant/duplicate/correlation/MAD gates PASS；
- model family/rule 只在 train-group CV 中选择；
- calibration 只完成冻结职责；
- sealed internal manifest 在训练前存在；
- 唯一 candidate、threshold、fixed frontier、external matrix 全部写入 candidate manifest；
- internal consumption ledger 为空且用户明确接受一次性消费。

### 10.5 external 前

- Gate M 与 Gate F 按冻结规则完成；失败即停止；
- fixed seeker 无 truncation、世界线相同、chronology report PASS；
- 新 study freeze 绑定 checkpoint、data、Bank、retrieval、conditions、judges 和 claims；
- freeze 显式绑定 full/no-Step0/no-state-BGE/lexical-only 四格 checkpoint，internal 结果不得选或重调 candidate；
- learned/rule external dry-run 自包含 live runtime lineage、section truncation gate 和只报告的 development/external Step-0 分布比较；
- 任何旧 V1/V1.5 freeze 都只作历史；
- external 每个付费 stage 再做独立 dry-run/approval。

### 10.6 论文提交前

- 只报告实际执行并 attested 的比较；
- 明确哪些是 primary、supportive、diagnostic、failed pilot 和 post-hoc；
- Gate M/F/E 各自决定可写句子，不用一个 favorable gate 覆盖另一个失败 gate；
- 报告 `NOT_SUPPORTED`、失败 pilot、fallback/repair、zero-hit、alias effective N 和所有强
  comparator；
- 不把 generator input tokens 写成 total cost，不把 LLM judge 写成人类评价；
- 不把 ESConv-derived EvoEmo 写成 pristine independent external benchmark。

## 11. 结果解释决策树

| 结果 | 允许结论 | 禁止结论 |
|---|---|---|
| Gate M、F、E 全通过 | synthetic holdout 中 learned 优于同观测 rule；对强 fixed 保持竞争力；外部对 high-resource fixed 质量非劣并减少 generator input tokens | 全部 raw quality 指标全面胜出、总成本/延迟必然更低、真实用户改善 |
| Gate M 失败，Gate E 通过 | 复杂 learned router 未证明优于透明 rule；仍可能有“某个低资源策略相对高资源 fixed 更省且质量保持”的系统结果 | learned adaptive routing 成功 |
| Gate F 对 `ME+R0` 失败 | 不能把省过高资源 fixed 包装成竞争性路由成功 | PM 不弱于强同预算 fixed |
| internal 通过，external rule/cost-matched 无优势 | 只能保留限定的 high-resource efficiency 或 synthetic mechanism finding | 外部 learned routing 泛化 |
| PM collapse 到单一动作 | 当前 observation/data/supervision 不足以支持 16-action routing | 通过放松 diversity gate 把固定动作叫 PM |
| 任一数据、lineage、judge、holdout gate 失败 | 协议失败或证据不可用；修复后需要新 run identity，必要时新 holdout | 把失败 run 中 favorable 部分抽出来作 confirmatory result |

期望的科学结果不是“PM 在所有质量指标上击败所有 fixed”。更合理的成功形态是：质量相对
高资源 fixed 非劣，risk 不增加，generator input 明显下降，并且相对同观测 rule 和强同预算
fixed 在冻结 utility 上显示可重复优势。如果数据只支持透明 rule，而不支持 learned PM，
那也是有效且应报告的研究结论。

## 12. 当前执行快照（2026-07-19）

| 项目 | 当前事实 |
|---|---|
| 分支 | `pm-v1.5_1`；V8.5 失败事实由 commit `d4fdcf1...` 归档；V8.6 role-safe exchange 修复与 fresh identity 由本快照所在 commit 标识，仍须 GitHub CI |
| tests/preflight | 专用 venv 精确 BGE runtime/canary、真实 768-d strict `PMV2State` 和 compiler-owned readiness 12/12 no-API smoke PASS；V8.6 修改后的裸全量 `pytest -q` 为 400 passed / 13 个预期 archive skip；仓库 release preflight 为 `API_PILOT_READY`，syntax/static/pytest 全 PASS，`confirmatory_ready=false`（历史 freeze 按设计不在此阶段刷新）；它们不能替代单独内容寻址的 V1.5 Bank/freeze |
| semantic runtime | 专用 `.venv-pm-v1-5`：Python 3.13.2 + exact package/device/dtype/user-site=false；冻结 3×384 public canary hash PASS；`sim_eval` 与 Conda `base` 均禁止作为正式运行环境 |
| readiness challenge | 20 个固定 outcome-free challenge：current 17/20、统一 bounded full-context 14/20；状态为 `REPORT_ONLY_14_OF_20`，只披露 BGE 粗粒度边界，不作为 outcome gate 或调参依据 |
| clean seed pool | 875 条私有候选，hash 由 artifact index 记录 |
| formal selected seeds | 52 个 source IDs |
| Strategy Bank | 11,590 cards / 823 source dialogues / 8 families；与 selected 52 交集为空 |
| V8 | 真实 FAIL：无关的 180-char rationale cap |
| V8.1 | 真实 FAIL：2/9 provider surfaces 需 fallback |
| V8.2 | 真实 FAIL：8 attempts，6 success/2 failure；旧输出与批准均 closed |
| V8.3 | 旧 dry-run `758ae052...8cf3df2` 已因当前 config/input/runtime 修复而 stale；formal CLI 也显式拒绝该历史目录 |
| V8.4 | identity `0588889c...d194b` 已真实消费并 transport/schema PASS：9/9 accepted、10 attempts、0 fallback、约 `$0.0030237`；因 readiness 与 review-lineage 缺口被语义性淘汰，禁止复用 |
| V8.5 | identity `a84c17f...05ed` 已消费并真实 FAIL：context_only initial+repair 均以 user 结束，0 accepted，2 attempts，约 `$0.0006274`；禁止复用 |
| V8.6 | role-safe exchange schema 真实 PASS：identity `64a06993...ef85`；contract `a4efb865...a756`；9/9 accepted、12 attempts、3 repairs（均为 `unique_current_user_text`）、0 fallback、约 `$0.0039084`；attestation `15135ad9...7f2c` |
| V8.7 | frozen-budget/scoped-lineage identity `920807ec...84e8` 已真实 `CONSUMED_PASS`：9/9 accepted、12 attempts、3 repairs（均为 `unique_current_user_text`）、0 fallback、18,288 input / 1,964 output tokens、约 `$0.0039216`；attestation `09f1f90b...60c5`，禁止复用 |
| paid approval | manifest 当前为 `CONSUMED_PASS`、`stage_approvals` 为空、paid flag=false；V8.6 已迁入统一历史，V8.7 保留 current consumption，二者均进入永久 consumed-identity 集；当前 `NO-RUN` |
| automated semantic review | V8.6-bound V3 identity `b4f27249...f84a` 已真实消费并因 Gemini OpenAI-compatible HTTP 400 fail-closed；绑定 V8.7 的 V4 native-Gemini fresh dry-run identity `5f3c57a0...8bf7` 已 PASS：120 logical / 360 max attempts、worst-case `$0.165096`、hard budget `$0.17`、Gemini-first；尚未批准，无当前 gate PASS |
| formal development | 未运行 |
| full sweep/judging | 未运行 |
| checkpoint/internal/freeze/external | 均未产生当前 V1.5_1 正式结果 |
| claim | `NO_CURRENT_RESULT` |

## 13. 未来每个 PR 必填模板

```text
变更目的：

触及的账本 ID：
触及的因果链层：
是否改变 treatment / estimand / comparator / claim：
是否改变 provider-visible 或 PM-visible 信息：
是否接触 train / calibration / internal / external outcome：
是否改变独立统计单位或 prompt alias：
是否改变真实部署成本：

失效的旧 artifacts/hashes/approvals：
必须重跑的 gates：
本 PR 明确不改变的机制：

clean CI 命令与结果：
dry-run identity（如适用）：
新的单一事实源：
失败时停止规则：
允许的论文句子：
禁止的论文句子：
```

审查者不得只确认“修复目标那一项测试通过”。至少还要沿第 2 节因果链向上检查信息来源、
向下检查所有消费者，并用第 9 节影响矩阵确认是否漏掉 formal、baseline、freeze、cost 或
claim evaluator。

## 14. 证据来源与维护规则

主要仓库证据：

- `PM_V1_FAILURE_LIMITATION_POSTMORTEM_ZH.md`：V1 结果、根因和范围；
- `PM_V1_5_SUPPLEMENTAL_ANALYSIS_ZH.md`：legacy V1.5 的 `ME+R0`、forced-swap 和 OOD 诊断；
- `PM_V2_1_V1_SYSTEMATIC_RESOLUTION_AUDIT_ZH.md`：V1 问题到新门禁的映射；
- `PM_V1_5_PROTOCOL_REPAIR_CONTRACT_ZH.md`：当前目标方法合同；
- `PM_V1_5_CORE_CHAIN_PLAN_ZH.md`：V1.5 执行顺序与历史 pilot 状态；
- `PM_V1_5_REVIEW_ARTIFACT_INDEX.json`：当前审查 artifact 摘要；
- `outputs/pm_v1_5_paid_run_release.json`：逐阶段付费执行 manifest；
- `release_preflight.json`：工程 preflight，不等于 confirmatory readiness。

外部复审材料对应 PR #3 `7ea2248`、PR #4 `b2333ac`、`c5c38b6`、`8910f4d` 等审查节点；
原始聊天附件只作本地审计证据，不属于可复现运行输入。

维护规则：

1. 新发现必须添加新 ID，不能静默改写旧问题的历史状态；
2. 只有同时具备代码门、测试和实际 attested run evidence 时，才可从
   `CODE_CLOSED_RUN_UNVERIFIED` 改成实证 PASS；
3. 修复导致新的方法身份、成本或调用架构时，必须同时新增“二阶风险”记录；
4. 任何 `PASS/APPROVED/SUPPORTED` 状态变化都要写明唯一 artifact 和内容哈希；
5. 探索性或失败 run 不删除，且不得把其中有利子集重新包装成确认性结果。
