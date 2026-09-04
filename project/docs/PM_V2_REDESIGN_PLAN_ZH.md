# MetaCom PM-v2.1：V1 问题的系统重设计与验证方案

更新时间：2026-07-14

> 实现分支：`codex/pm-v2-hardening`
>
> 完整执行命令见 [`PM_V2_RUNBOOK_ZH.md`](PM_V2_RUNBOOK_ZH.md)。
>
> PM 与 post-retrieval Filter 的边界见 [`PM_V2_EVIDENCE_FILTER_PROTOCOL_ZH.md`](PM_V2_EVIDENCE_FILTER_PROTOCOL_ZH.md)。
>
> PM-v1 的失败与限制复盘见 [`PM_V1_FAILURE_LIMITATION_POSTMORTEM_ZH.md`](PM_V1_FAILURE_LIMITATION_POSTMORTEM_ZH.md)。

## 1. 先给结论：哪些已经解决，哪些仍待实验证明

PM-v2.1 已经处理了 PM-v1 的已知**结构性**问题：训练/外部数据边界、`overall=support`、标签维度退化、完整 action matrix、逐维不确定性、M0/R0/RS 可学性、cost 只做 tie-break、OOD 静默外推、同预算 fixed 缺失、source-level PM 无法看到具体候选、judge 顺序与 family 污染，以及 API 无预算保护。

但“代码已经阻止旧错误”不等于“新 PM 已经在数据上学会了正确 routing”。以下三项仍是经验命题：

1. PM 是否会因情境稳定选择不同 actions，并在不需要资源时主动选择 M0/R0；
2. PM 是否真正找到 quality、risk、cost 与 uncertainty 的可泛化平衡；
3. PM 是否能在外部 EvoEmo 上达到或超过同预算固定动作，同时不过度使用 memory/strategy。

截至 2026-07-14，PM-v2.1 付费 API 调用数为 **0**，因此不能提前写“上述经验问题已完全解决”。当前可以写的是：

> PM-v2.1 has removed the known structural failure modes of PM-v1 and implements a fail-closed validation pipeline. Whether adaptive routing provides an empirical advantage remains to be established.

## 2. V1 → V2.1 问题—修复—证据矩阵

| PM-v1 问题 | PM-v2.1 处理 | 当前证据状态 |
|---|---|---|
| `overall` 与 support 完全相同，overall 失效 | judge schema 删除 `overall`；六维独立输出；训练只使用冻结 `pmv2-quality-v2` composite | 结构上已修复；仍需 human/LLM dimension audit |
| PM 近常量选动作，M0=0 或 RS 近饱和 | 九个互斥 resource-need regimes；所有 state 暴露 16 actions；learned-only diversity、entropy、M0/R0、nonfallback 与逐-regime gates | 机制已实现；是否学会仍待数据/internal/external |
| `epsilon=0` 使 cost 几乎不参与选择 | cost 直接进入每个 action 的 conservative utility；calibration 用候选无关的冻结 quality−risk−cost 标尺比较权重，禁止靠缩小自身 penalty 提高目标 | 结构上已修复；权衡质量待 calibration/internal 验证 |
| risk 维度弱、为零或与质量混在一起 | 七个独立 risk heads、分 action applicability、per-risk UCB thresholds、risk penalty 与恒定维度 gate | 结构上已修复；校准/覆盖待验证 |
| 资源开启没有相对 abstention 证明 | memory-on 必须相对**同 strategy mode** 的 M0 action 通过 benefit gate，RS 必须相对同 memory subset 的 R0 通过 strategy benefit gate；omission risk 可触发例外 | 选择规则已实现；边际收益待 holdout 验证 |
| evaluator 定义的路由需求可能在合法特征中不可见 | 付费 action pilot 前，在 exact deployable feature builder 上做 train-only user-group CV；分别检查 regime/source/memory need 与 strategy/memory helpful-vs-harmful 方向 | 必要可观测性门已实现；它不等于 learned-routing 成绩 |
| 标签低区分、judge family 偏差 | development 两个独立 family；先做平衡 compatibility pilot；完整双-family response/risk judging；逐维 MAD、constant/duplicate/correlation gates；人工盲审 | 协议已实现；真实 endpoint compatibility 尚未运行 |
| synthetic regime/family/source/item 标签可能“结构合法但语义错误” | 付费 sweep 前进行无 API、3 splits × 9 regimes 平衡人工 semantic-sanity audit；至少两名独立 annotator；逐字段 agreement/affirmative、逐 regime/逐 split gate；逐 item helpful/irrelevant/harmful 也必须人工核验；hash attestation 绑定 config/states/evaluator context | fail-closed 协议与测试已实现；正式 generated states 仍待人工标注 |
| PM 只能决定 source/policy，无法知道具体 memory/RS item 是否相关 | 正式系统改为 PM requested action → retrieval → Evidence Filter → effective action；memory filter 使用 item-level supervision/calibration/internal gate；strategy card 使用透明 contextual fail-safe；requested/effective action 分离 | 架构、训练门控与测试已实现；Filter 外部增益待公平消融 |
| 按整行 reliable 删除会改变 action/state estimand | 每个 split 保留完整聚合中位数矩阵；MAD 逐维连续降权；分维度和分 action-dimension coverage gate | 已由代码与测试约束 |
| fitted TF-IDF 外部词汇 OOV | 固定 word/char hashing；source-level centroid；retrieval-capacity-aware metadata | 已由实现/测试约束；外部泛化待验证 |
| pre-retrieval decision 泄露 retrieved item/oracle | PM state 只包含 operational metadata；actual memory、retrieved IDs、judge labels、generated response、regime/needed sources 均禁止进入 features | 已由 schema/loader/测试 fail-closed |
| synthetic catalog 比真实 EvoEmo 小，count/tokens 造成 scale shift | count 被 retrieval top-k capacity 截断；token feature 使用 bounded expected top-k tokens；age 使用 session-relative ratios；真实/开发 scale 有回归测试 | 已处理已知 metadata scale 问题 |
| train/calibration/internal 或 EvoEmo overlap | 真实 ESConv + index-aligned split/overlap manifest；user/family/current-text 三重 disjoint；固定 word/char hash 跨 split 近重复门；EvoEmo overlap 明确排除 | lineage/near-duplicate 代码已实现；正式 generated-text audit 待数据 |
| epistemic std 没有覆盖保证 | calibration 对每个 user 的全部 states×actions 先取最大残差，再拟合 head-wise split-conformal radii；YAML 只允许一个预注册 z；独立 internal users 做 coverage gate | 机制已实现；只主张 exchangeable-user 下的 head-wise marginal block coverage，不声称 13-head joint/conditional coverage |
| 状态相关，普通 bootstrap 伪独立 | ensemble 按 `user_id` group bootstrap；policy/fixed paired deltas 按 user cluster bootstrap | 已由实现/测试约束 |
| OOD 与无可行动作混为一谈 | `severe_ood` 与 `no_feasible` 两种 fallback 分开记录、分开设上限；默认 fail-closed `M0+R0` | 已实现；外部 fallback rate 待 preflight |
| 缺少强同预算固定基线 | calibration 先筛冻结 token band，再选择其中质量/utility 最强的 fixed-frontier challenger；另固定 `ME+R0`；PM 与 fixed 使用同一个 Filter；外部预计和 observed input-token 相对偏差阈值 10% | baseline pipeline 已实现；实际 token match 待生成验证 |
| Filter 后多个 action prompt 完全相同仍重复付费 | 同 state、同实际 prompt 归为 equivalence class，只做一次物理 generation，再为 requested actions 分别物化检索/过滤成本；alias 不制造独立方差 | 已实现并有回归测试 |
| 单一 judge/order 影响关键 claim | 两个与 development disjoint 的 external families；40-unit dual-order forced-swap 技术/无效性 gate；pilot units 从 full confirmatory sample 排除 | 协议已实现；结果待运行 |
| API 可能在协议错误时直接烧钱 | 所有付费脚本显式 `--dry-run/--run`、精确 call plan、预算/单次 token gate、accepted SHA、artifact attestation | 已实现；当前 paid calls=0 |

## 3. PM-v2.1 的学习目标不是“16 个 policy”

PM 是一个 state-conditional resource-routing policy。它在同一生成器前选择 16 个 resource actions：

```text
8 memory subsets × {R0, RS}
```

其中：

- `M0`：不使用长期 memory；
- `MP/MS/ME/...`：选择 profile、summary、event 的不同组合；
- `R0`：不使用 strategy retrieval；
- `RS`：使用 strategy retrieval。

所以“学会使用不同 policy”的可检验含义是：PM 在不同 state 下学会不同 resource actions，而不是训练 16 个回复模型。PM-v2.1 必须同时证明：

- context-only / memory-harmful 状态能选择 `M0`；
- strategy-harmful 或无需策略时能选择 `R0`；
- profile/summary/event/multi-source 确有边际收益时开启相应 memory；
- strategy helpful 时才开启 RS；
- 这些选择来自 learned decision，而不是 severe-OOD/no-feasible fallback；
- action distribution 有足够 entropy，最大单动作份额不过高。

## 4. 真实 seed lineage 与 development data

### 4.1 真实 ESConv train-only seed

正式入口不再接受模糊的 `<DATASET>` 占位。必须使用真实源和 strict manifest：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/tokkio/miniconda3/envs/sim_eval/bin/python \
  scripts/18_prepare_pm_v2_seed_dialogues.py \
  --inputs data/external/ESConv.json \
  --split-manifest data/strategy/esconv_split_manifest.jsonl \
  --out data/pm_v2/train_seed_dialogues.jsonl \
  --minimum-seeds 100
```

manifest 与 ESConv 按 index 一一对应，schema 固定为 `index/dialogue_id/split/excluded_for_evoemo_overlap`。只保留 train 且没有 EvoEmo overlap 的对话；任何覆盖、顺序、ID 或 hash 不一致直接失败。

### 4.2 完整规模与九个 regimes

正式 YAML 规模是：

- train：24 users × 9 states = 216；
- calibration：12 users × 9 states = 108；
- internal test：16 users × 9 states = 144；
- total：52 users、468 states、7,488 state-action outcomes。

每个 user bundle 恰好覆盖：context only、profile needed、summary needed、event needed、multi-source needed、memory harmful、strategy helpful、strategy harmful、ambiguous。

一次 bundle call 生成一个用户的九类状态，所以成功路径与冻结硬上界均为 52 calls。v4 role-slot schema + local compiler 移除了让 provider 自报结构标签的失败模式；每个 user 只授权一个请求，失败必须回到协议审查而非自动换 seed 重试。12 个 calibration users 使 90% user-block finite-sample conformal 可达。16 个独立 internal users 上的冻结 evidence threshold 是 95% Wilson 下界 `0.60`：至少 14/16 个完整 user blocks 命中才通过；13/16 失败。目标 coverage 仍是 0.90，二者不能混为一谈。正式 API generation 禁止 `--max-users`，防止用残缺数据冒充冻结设计。

### 4.3 Oracle 与 operational state 物理分离

`evaluator_contexts.jsonl` 保存 authorized full context、regime、needed sources、rationale 等，只给审计/judge。PM state 只保留调用前真实可用的信息。loader 要求两者 state IDs 精确对应，但 model feature builder 永远不读取 evaluator-only 字段。

此外，memory 的当前/future session 项被拒绝；strategy bank 的实际 count/hash 与 token/top-k contract 写入 data report。strategy catalog count 是全局常数，不进入模型 feature，避免把固定部署常数伪装成 state signal。

### 4.4 付费 action sweep 前的人类语义审计

生成 schema 能强制 `profile_needed → [MP]` 这样的结构关系，却不能证明某条自然语言案例真的需要 profile memory，也不能证明 `semantic_family` 与内容一致。因此生成完成后、compatibility pilot 规划前，额外冻结一个无 API semantic-sanity gate：

- 从 train/calibration/internal 的每个 regime 各取 YAML 冻结数量，默认覆盖完整 3×9 网格；
- packet 隐藏 state/user/split/generator 和生成器的 coverage rationale，防止审计者按来源或自我解释放行；
- 至少两名独立 annotator 分别判断 `semantic_family_match`、`regime_match`、`needed_memory_sources_match`；
- 按 YAML 检查逐字段 affirmative rate 与 pairwise agreement，以及逐 regime、逐 split affirmative rate；
- plan self-hash、保护列 hash 和 artifact attestation 绑定 config、states、memory backend、evaluator contexts、manual、packet 和全部 completed CSV；packet 展示按 source 标记的实际 synthetic memory，避免只看数值 source summary 就判断 needed sources；
- script 31 与 PM-v2 script 06 在付费调用计划之前都要求同 lineage 的 PASS，并逐行验证 runtime 是 audited PM-v2 states 的确定性转换，防止用另一份 runtime 借用 PASS。

这个 gate 只排除明显的 synthetic semantic corruption，不替代 16-action response generation/judging，也不能提前证明 PM 会学会正确 routing。

### 4.5 合法特征中的路由信号可观测性

semantic audit 通过后，script 31 使用最终 `PMV2FeatureBuilder`、固定
`M0+R0` action 和 train users 做分组交叉验证。targets 包括九类 regime、精确
needed-source、是否需要 memory，以及 strategy-helpful/harmful、
memory-helpful/harmful 两组方向性区分。输入不含 evaluator-only labels、memory
text、retrieved IDs、response 或 judge output。

这是付费前的必要条件：如果合法 feature space 连这些预注册需求都无法区分，继续
生成 180/7,488 个 actions 没有足够价值。结果由 script 31 写入 plan，但 script 34、
script 06 和 full development gate 都会从 exact states、evaluator contexts 与 YAML
重算并逐字段比较；攻击者仅重算 plan self-hash 不能绕过。即使通过，它也只说明
“存在可观测信号”，不说明最终 outcome learner 已经会路由。

## 5. Retrieval-capacity-aware pre-retrieval representation

PM-v2.1 只允许在 retrieval 前计算的特征：

- 当前文本、近期上下文、session summary 的固定 word/char hash；
- source availability；
- `min(catalog_count, retrieval_top_k)` 形式的可检索容量；
- bounded expected top-k tokens，而不是整个 catalog 尾部总 tokens；
- min/median/max age 的 session-relative ratios；
- current query 与缓存 source-level catalog centroid 的 similarity；
- 冻结 strategy top-k/token capacity。

明确禁止：

- actual memory text；
- selected/retrieved item IDs；
- item-level top-1/max/P90 similarities；
- current-state conflict/need oracle；
- regime、needed sources 或 coverage rationale；
- generated response、judge output、未来状态。

这样做解决的是可识别的 scale/leakage 根因，但不承诺固定 hashing 一定比所有 semantic encoder 更强；外部 OOD 与性能仍要实测。

## 6. `overall` 问题如何被彻底移除

### 6.1 Schema 中没有 Overall

response judge 只返回：

1. emotional support；
2. personalization；
3. memory appropriateness；
4. factual grounding；
5. temporal consistency；
6. non-intrusiveness。

因此 PM-v1 的“Overall 实际复制 Support”在数据 schema 上无法继续发生。论文也不能再把 composite 称为 LLM holistic overall。

### 6.2 `pmv2-quality-v2` 是可审计公式

```text
quality =
  0.30 × emotional_support
+ 0.20 × personalization
+ 0.15 × memory_appropriateness
+ 0.15 × factual_grounding
+ 0.10 × temporal_consistency
+ 0.10 × non_intrusiveness
```

每条 label 必须携带 composite version 和权重 hash。情感支持仍单独报告，并在 internal reportability 中有独立 delta gate；不能用 composite 掩盖 support 退化。

### 6.3 不只防 exact copy

标签批次还检查：

- response/risk 任一维度恒定；
- 两维逐行 exact duplicate rate；
- 维度绝对相关性上限；
- 每维 MAD coverage；
- 每个 action × dimension 的 MAD coverage；
- 盲化 human–LLM 一致性与人际一致性。

高相关本身不等于错误，但超过预注册上限时不能继续训练/冻结，必须先诊断 rubric 或 endpoint。

## 7. 完整 action matrix 与逐维 MAD 学习

每个 state 的所有 16 legal actions 都必须生成、双-family judging 并聚合。少一个 state-action 就是 sweep/label 失败，不能通过完整案例删除来“清洗”。

两个 judge family 的每个 response/risk dimension 分别求 median 与 MAD。训练每个 head 时使用连续权重：

```text
weight(dimension) = 1 / (1 + (MAD / scale)^2)
```

含义是：

- 中位数仍留在原始 estimand；
- 哪个维度分歧大，就只降低该 head 的影响；
- 不因某一个 risk dimension 分歧而删除同一 action 的全部质量信息；
- 但若某维或某 action-dimension 的低 MAD 覆盖不足，整个 split gate 失败。

这比 PM-v1 的单一银标签或整行可靠过滤更适合比较 16 个 actions。

## 8. 多头 outcome model、不确定性和成本模型

### 8.1 十三个直接 targets

模型直接预测六个 response heads 与七个 risk heads：selected-context misuse、unnecessary exposure、stale/conflicting use、unsupported personal claim、memory omission、strategy overuse、strategy omission。

同一 state 的所有 action rows 总权重相同。bootstrap sampling key 是 `user_id`，一个用户的状态不会被拆成伪独立 bootstrap 单位。

### 8.2 Split-conformal，不用 coverage 反向调 z

train 只拟合 outcome heads。calibration 先对每个用户的全部 states×legal actions 取最大 nonconformity，再为每个 response/risk head 和 quality composite 跨用户拟合 finite-sample split-conformal radius；YAML 的 z 列表必须恰好一个值。interval 为 ensemble mean ± epistemic term ± conformal radius。

calibration 只拟合 radius 并做结构/可达性 sanity check，不用同一拟合集的 Wilson 下界自证成功。coverage lower-bound gate 只在独立 internal users 上执行；一个 user/head 只有在其全部适用 state-action 区间都命中时才算 block hit。该保证是 head-wise、marginal 且依赖用户可交换性，不是 13-head joint 或逐 regime conditional coverage。

### 8.3 Cost 以可部署量进入选择

选择阶段的成本来自冻结的 retrieval-capacity-aware estimated resource cost，并在 calibration/internal 用 observed input tokens 检查 rank correlation 与诊断偏差。fixed baseline 的选择也使用同一 cost definition，避免一边用 estimated、一边用 observed 的口径错位。

## 9. Quality–risk–cost 的选择规则

对每个 action：

1. 预测六维 response mean/interval 与适用的 risk mean/interval；
2. 用 frozen composite 得到 quality LCB；
3. risk UCB 超过对应 per-risk threshold 时排除；
4. memory-on 相对同 strategy mode 的 M0 action 未达到 resource benefit，且该
   M0 baseline 的 omission risk 不高时排除；
5. RS 相对相同 memory subset 的 R0 未达到 strategy benefit 且 R0 strategy-omission risk 不高时排除；
6. 对可行动作计算：

```text
conservative utility
= quality LCB
- risk_weight × applicable risk UCB
- cost_weight × normalized estimated resource cost
```

7. 选择最大 utility；
8. severe OOD 与 no feasible action 分别触发、分别记录 `M0+R0` fail-closed fallback。

成本从第一层 utility 比较就参与，而不是 PM-v1 的 cost tie-breaker。resource/strategy off 也不再依赖“希望模型自己学会”，而是有相对 benefit 证据门槛。

## 10. Calibration、internal test 与“真的学会不同动作”的 gate

calibration 只选择 YAML 预注册网格中的 risk weight、cost weight、resource gain、strategy gain 和 max-risk，并选择同预算 fixed action；不能访问 internal 或 EvoEmo 分数。

internal test 只评一次，至少检查：

- quality composite 和 emotional support 相对 cost-matched fixed 的 delta；
- utility、risk 与 observed cost ratio；
- estimated/observed cost Spearman；
- 至少使用预注册数量的 learned actions；
- learned action entropy 与 maximum action share；
- M0、R0、learned nonfallback M0+R0；
- severe-OOD 与 no-feasible fallback 分别不过量；
- 平均和每一个 regime 的 alignment；
- quality/response/risk interval coverage；
- 与 fixed 的 paired delta 使用 user-cluster bootstrap CI。

fallback 选择不计作“PM 学会了 M0/R0”。因此即使总 M0 比例达标，只要都是 OOD fallback，learned-only gate 仍会失败。

internal 同时输出两个不同结论：宽松的 deployment tradeoff/non-inferiority 诊断，
以及更严格的 learned-routing advantage。只有后者才能放行 external：PM 相对
calibration-selected、在 calibration 与 internal 都满足 10% observed-token 容差的
同一 fixed action，其 quality、emotional support 和 utility 三个 user-cluster paired
bootstrap CI lower bounds 必须全部**严格大于零**。等于零或全 tie 都不能被写成
adaptive routing advantage。

## 11. Development judges 与 external judges 严格分层

`configs/pm_v2.yaml` 冻结两组不重叠 families：

- development：用于 compatibility pilot 和 7,488-action label matrix；
- external：只用于 forced-swap 与 final pointwise evaluation。

正式顺序是：

1. script 20a 先做一个 source-grounded synthetic-generator call 的结构兼容 gate；
2. scripts 20b/20c 对该九例 pilot 做双人、逐维 all-affirmative semantic gate；
3. 只有两道 pilot gate 都 PASS，script 20 才完整生成 52-user/468-state development data；
4. 全量数据双人 no-API semantic-sanity 审计通过后，script 31 重算 deployable-feature
   observability 并生成按 regime 平衡的 compatibility plan；
5. script 34 用两个 development families 各跑 response/risk，共 4-call schema
   smoke；
6. script 06 只生成 plan 中 18 states × 10 actions = 180 outcomes；
7. script 21 完成 720-call compatibility judging；除了 schema/MAD/维度门，还要求
   同一 frozen-utility oracle policy 相对最佳 fixed 在 utility、quality 和 emotional
   support 上都有预注册正 headroom；
8. 两名人工对 18 个 train-only、双盲 items 逐 family/逐维校准 development judges；
9. 只有以上 exact attestations 全 PASS，才生成完整 7,488 outcomes；
10. script 21 才允许完整 29,952-call judging。

compatibility sequence 证明 endpoint/schema/prompt 可用，并检查 train-only 标签信号、
方向分离和 optimistic oracle headroom；它仍不能证明最终 learned router 会达到 oracle，
不能作为完整训练 matrix 的替代品，也不能用于挑最有利 judge。

## 12. Human audit 和 freeze

human packet 在 YAML 中固定 72 items、sample seed、最少 annotators 与所有阈值。两个 annotators 独立填写六个 response/七个 risk dimensions；action、policy、LLM scores 对他们隐藏。

只有以下 lineage 同时一致且 gate 全 PASS，script 26 才创建 reportable freeze：

- real ESConv seed audit 与 overlap exclusion；
- 52 bundles/468 states、runtime/backend/evaluator contexts；
- full sweep manifest/attestation；
- development compatibility summary/attestation；
- full judge labels/raw/manifest/attestation；
- data/label audit；
- human audit；
- internal training report/checkpoint；
- cost-matched 与 `ME+R0` fixed checkpoints；
- EvoEmo、strategy bank、fixed seeker tracks；
- experiment/PM-v2 config hashes。

所有 reportability/selection/human thresholds 和 human sample seed 都是 YAML-only。修改 YAML 意味着新 preregistration 和新 freeze，不是原实验的“命令行补丁”。

## 13. External 设计：先 no-API preflight，再最小有价值 API

### 13.1 script 23 已禁用

旧 `scripts/23_select_pm_v2_actions.py` 接受任意 caller runtime，不能证明输入、checkpoint 与 freeze 同源，因此已 fail-closed。external action/OOD preflight 必须通过：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/tokkio/miniconda3/envs/sim_eval/bin/python \
  scripts/24_run_pm_v2_evoemo.py \
  --dry-run \
  --freeze outputs/pm_v2_study_freeze.json \
  --checkpoint outputs/pm_v2_model/pm_v2.joblib \
  --condition pm_v2 \
  --out-dir outputs/evoemo_pm_v2 \
  --max-api-calls 250 \
  --max-estimated-usd "$PMV2_EXTERNAL_GENERATION_MAX_USD" \
  --max-input-tokens-per-call 12000
```

该 dry-run 绑定 freeze、fixed tracks、checkpoint 和完整 external states，并检查
learned-only action diversity、M0/R0、两类 fallback 与 OOD。freeze 内容寻址正式
评分 turns `[3, 8]` 的 exact 204-unit universe；10-turn 全轨仍做 no-API action
diagnostic，但 generation call plan 只包含这 204 个 one-step fixed-context cases。
因此不会再为不会进入评分的 816 turns/condition 付费。失败时在 generation API 前
停止；ledger 一旦出现 `STARTED`，原始 full plan/cost evidence 只能 exact validate，
不能被后续 dry-run 重写。

### 13.2 同预算必须同时通过预计与 observed token gate

calibration-selected fixed 与 PM-v2 在相同 fixed-input unit matrix 上生成。freeze 中相对偏差阈值是 10%：

- generation 前：预计 input tokens 不匹配则不运行；
- generation 后、response judge 前：实际 `input_tokens` mean 不匹配则停止；
- unit matrix 不完全一致也直接失败。

超过阈值时可以报告“所选 fixed 未能实现外部 cost match”，但不能继续把它称为同预算证据，也不能根据 EvoEmo 分数另挑 baseline。

### 13.3 Forced-swap 是技术/无效性 gate，不是 confirmatory effect estimate

script 30 冻结抽取 40 matched units，比较 learned PM 与 cost-matched fixed：

- 两个 external judge families；
- order 0/1 强制互换；
- 160 expected calls；
- schema success、order disagreement、跨-family方向/相关性 gate；
- 若 support delta cluster-bootstrap CI upper 已低于预注册负 margin，则触发 futility stop。

它只回答“judge protocol 是否可用、关键 claim 是否值得继续花钱”，不把 40-unit pilot 当正式显著性结果。无论方向如何，这 40 units 都写入 exclusion attestation，并从 script 25 full confirmatory sample 中剔除。

### 13.4 External pointwise schema smoke

forced-swap 通过后，script 35 使用一个已经从 confirmatory sample 排除、并在 freeze
中绑定的 unit，运行与 full external 完全相同的匿名 pointwise response/risk prompt。
矩阵固定为两个 external families × 两个 schemas = 4 calls。两个 clients 必须在任何
ledger reservation/HTTP 前全部初始化；任一 credential/client 失败时 started attempts
必须为零。它只验证 structured transport、正数 usage、token bound 与 lineage，不产生
raw-family quality claim。

### 13.5 Full external pointwise

只有 forced-swap 与上述 4-call smoke 都 `PASS`，script 25 才接受
summary/attestation 并创建 full clients。正式 external：

- 单次 prompt 只看一个匿名候选；
- 两个 external families；
- 六维 response、七维 risk、冻结 composite 和 observed tokens；
- PM-v2、cost-matched fixed、`ME+R0` 及冻结的 context/structured/raw/full-history baselines；
- forced-swap units 不重用；
- paired deltas 与 user/scenario cluster bootstrap；
- negative result 原样报告。

## 14. API 价值顺序与停止决策

付费调用不是一次性全开，而是逐级获取最有决策价值的信息：

| 阶段 | 预期调用 | 只有什么通过才继续 |
|---|---:|---|
| synthetic generator schema pilot | 1 | 九 regime/三 family/schema/usage exact PASS |
| development bundles | 52 success path，52 hard max | 468 states、split/regime/schema/compiler/lineage 全 PASS |
| semantic + feature observability | 0 API | 双人语义审计与 train-only grouped-CV 必要门 PASS |
| development judge schema smoke | 4 | 两 family × response/risk exact PASS |
| compatibility action sweep | 180 | 完整 plan-bound outcomes |
| compatibility judging | 720 | 两 family schema/reliability/dimension/directional/headroom gate + attestation PASS |
| train-only human spot-check | 0 API | 两位人工、每 raw family、每维 gate PASS |
| full action sweep | 7,488 | 完整 action matrix |
| full development judging | 29,952 | 完整 labels、MAD/dimension/data gate PASS |
| human/internal/freeze | 0 API | human + internal + freeze 全 PASS |
| external generation | 204/condition | action/OOD/diversity + planned cost match PASS |
| forced-swap | 160 | observed cost match + technical/futility gate PASS |
| external pointwise schema smoke | 4 | 两 external families × response/risk exact PASS |
| full external judge | dry-run 决定 | pilot-unit exclusion与全部 attestations PASS |

每个 API 阶段都必须先精确 dry-run、核对实时价格与预算、接受 SHA。任何 gate 失败，就保留诊断产物并停止下一阶段；“已经花了前面的钱”不是继续烧后续 API 的理由。

## 15. 论文可写与不可写的边界

在只有代码、测试和 no-API preflight 时，可以写：

> PM-v2.1 removes the duplicated overall target, trains on complete action matrices with dimension-specific uncertainty weights, and uses preregistered quality–risk–cost routing with fail-closed abstention and OOD safeguards.

不能写：

- PM-v2.1 已经学会何时不用资源；
- PM-v2.1 已找到最优 quality–risk–cost balance；
- learned routing 已经优于同预算 fixed routing；
- split-conformal 保证每个用户/每种 regime 的 conditional coverage；
- forced-swap pilot 是独立 confirmatory effect evidence。

只有在完整数据、双-family 标签、human audit、internal reportability、observed cost match、forced-swap 和排除 pilot units 的 external full evaluation 都实际通过后，才能把对应经验结论升级为论文主张。若 PM-v2.1 仍不胜同预算 fixed，最重要的产出也不是隐藏负面结果，而是证明：旧的结构性混乱已经被隔离，新失败可以被准确归因到学习目标、数据或真实外部不可泛化性。
