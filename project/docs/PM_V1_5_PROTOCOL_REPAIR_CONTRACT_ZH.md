# PM-v1.5_1 方法修复合同：Step-0、路由语义与三层 Gate

更新时间：2026-07-19
合同状态：**IMPLEMENTED / V8.6 PAID COMPATIBILITY PASS / AUTOMATED SEMANTIC REVIEW PENDING**
适用对象：下一次重新生成、重新训练、重新冻结的 PM-v1.5_1 运行

任何后续局部修复还必须先检查 `PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md` 中的全链路
失效模式、不可回归宪法与改动影响矩阵，不能只验证当前报错点。

## 0. 状态与效力

本文是审查后冻结并完成代码实现的目标方法合同，不是实验结果。代码完成不等于实验
主张成立。V8.2 与 V8.5 已真实消费并 fail-closed；V8.6 已真实 PASS，但这只证明
compatibility/transport 与确定性结构门通过，不是论文结果。中央配置保持
`PAID_RUN_RELEASED`，独立 approval manifest 已在 V8.6 消费后关闭且 approval map 为空，
因此新的付费执行仍 fail-closed。在本文末尾要求的
逐阶段 clean dry-run、lineage 审查、配置哈希和新 run identity 全部完成前：

- 不允许开始新的付费 development、judging 或 external 调用；
- 旧 dry-run、旧自动语义审核、旧 cost-estimate hash 和旧 pilot 不得复用；
- 当前 checkpoint、内部报告或外部结果均不能被描述为由本合同产生；
- 任一 gate 未通过只能输出 `NOT_SUPPORTED`，不能通过调阈值改写结论。

本文取代 `PM_V1_5_CORE_CHAIN_PLAN_ZH.md` 中“完全无目录观测的 pure
pre-retrieval router”定义，作为下一次运行的优先方法合同。旧文档继续保存历史背景，但
与本文冲突时以本文为准。

中央执行门位于 `configs/pm_v1_5.yaml:execution_release`。所有含 `--run` 的 V1.5
付费入口在创建 API client 或 attempt 前都会调用同一门；只有配置状态改为
`PAID_RUN_RELEASED`，且独立 approval manifest 同时绑定当前配置哈希、release revision、
具体 stage 和新 dry-run/cost identity 时才可执行。单独传 `--run` 或复用旧 hash 均不能
绕过。

通用 `scripts/99_release_preflight.py` 已消除 V9 复现脚本中的机器专属 Python 路径，
当前静态扫描通过并返回 `API_PILOT_READY`。仓库中的旧
`outputs/study_freeze.json` 不再把整个代码审查标成 `BLOCKED`，而是明确报告为
`STALE_HISTORICAL_FREEZE / confirmatory_only`。它不会被原地改哈希；正式
PM-v1.5_1 freeze 只能在新 candidate、internal gates 与 lineage 完成后重新创建。

## 1. 方法身份与有限主张

方法正式称为：

> **基于固定来源级观测的监督式 pre-item-retrieval router**
> supervised pre-item-retrieval router with fixed source-level observations

它不是 RL、POMDP、在线学习或长期用户改善模型。训练数据来自完整 action sweep 的监督式
反事实结果；每轮只做一次前向选择，不从外部测试奖励继续更新。

允许检验的窄主张是：在冻结数据与生成合同下，learned router 是否相对强规则路由器和
竞争性固定策略获得更好的质量—风险—资源权衡，并在相对高资源固定策略保持回复质量、
不增加证据误用信号的同时减少生成器输入 tokens。它不支持临床疗效、真实用户改善、
长期强化学习或跨数据集普适性结论。

## 2. 冻结因果顺序

每个决策单元必须遵守同一顺序：

```text
current dialogue state
  -> deterministic Step-0 source observation
  -> requested_action_id
  -> retrieval_attempts
  -> realized_action_id
  -> prompt_equivalence_id
  -> supporter generation
  -> outcome judging
```

任何 item-level memory/strategy 检索结果、生成回复和 outcome label 都不得回流到
`requested_action_id`。Step-0 必须在动作之前一次性、确定性地产生并内容寻址；真正的
item-level retrieval 只能在动作之后执行。

## 3. Step-0 观测合同

### 3.0 唯一 full-state 语义输入

Memory source centroid、Strategy family centroid 与 state BGE 的 full-state 部分必须共享
同一个 section-aware bounded text 及其同一个归一化向量。当前话语和 session summary 使用
冻结预算，history 只保留最近 token suffix；该文本在送入 encoder 前已经不超过 512 tokens，
模型内部静默截断禁止作为兜底。每个 state 同时记录并强制相等：

- `step0_semantic_query_sha256 == state_embedding_query_sha256`；
- `step0_semantic_query_vector_sha256 == state_embedding_query_vector_sha256`。

Advice Readiness 有意只看 current-user 独立 view，不属于上述 full-state 相等约束。Development、
external runtime 和严格 schema 必须执行同一 helper；旧的未预算文本只能存在于不具备正式
semantic encoder 的 legacy/test compatibility path。

### 3.1 PM 可见字段

对 MP、MS、ME，每个来源仅允许固定、任务相关、有限精度的标量：

- `available`；
- bounded `item_count`；
- bounded age summary；
- expected retrieval/prompt token cost；
- `representation_valid`；
- query-to-source-centroid similarity。

Strategy 只允许：

- query 与少量预冻结 strategy-family centroids 的相似度；
- query 与五类预冻结 advice-readiness anchors 的连续语义相似度：listen-only、
  explore-first、light-suggestion、structured-plan、ambiguous；
- family availability、bounded count 与 expected cost。

禁止再用 advice/request/listen 关键词正则产生硬布尔标签。Advice Readiness 也不等于
Strategy RAG 边际价值：listen-only 可受益于 reflection/restatement；明确请求建议也可能
因 Bank 不匹配而不应开启 RS。

连续值必须冻结归一化、裁剪和量化规则。缺失表示必须显式产生 `valid=false`，不得用某个
来源特有的魔法数代替。

### 3.2 仅审计可见字段

以下字段只能进入 manifest、attestation 与日志，不能进入模型特征：

- representation/centroid algorithm version；
- encoder 或 model ID；
- embedding dimensionality；
- config/representation/catalog build SHA-256；
- 原始构建路径、环境名和 run identity。

### 3.3 明确禁止

Step-0 和 PM 输入不得包含：

- raw memory 或 Strategy Card 文本；
- item/card ID、top-k ID 或 snippet；
- item-level top-1、max、P90、top-k score；
- embedding vector、norm、mean、std；
- gold source、required source、regime label 或 judge label；
- 完整 Strategy Retriever 的先验执行结果。

若 Step-0 执行了完整 retriever，它就不再是 coarse observation，必须作为正式 retrieval
attempt 记账并缓存复用；本合同的主要机制不允许这种实现。

### 3.4 Strategy family 的冻结要求

Strategy family 必须在查看 internal-test 和 external outcome 前定义。family 构造只能使用
Strategy Bank 的开发侧信息，并记录 family 数、各 family 大小、构建算法和哈希。需要
审计单一 family 是否近似等于某个 synthetic regime，及相似度是否近似直接编码 RS
oracle；发现该 shortcut 时整次数据构造失效，而不是在结果后删除可疑 family。

Strategy Bank 与 52 个正式 development seed dialogue 还必须做到实例级不相交。当前
冻结选择先确定 52 个 seed source ID，再从主 Bank 中排除这些完整对话；结果为 11,590 张
卡、823 个来源对话，8 个策略家族仍全部覆盖。这里允许“策略类型和经验规律重合”，但
不允许训练输入与检索卡来自同一个原始 ESConv 对话实例。

`configs/pm_v1_5.yaml:strategy_bank_contract` 还固定 exact relative path、Bank SHA、
11,590/823 计数、Bank audit SHA，以及 52-source manifest 的 path/SHA/count。首个付费
development 入口必须在创建 API client 前逐项验证；CLI 不能用另一套内容替换它。后续
sweep、judging、freeze 和 external 继续验证首阶段 attestation 中的同一 hash，而不是各自
重新接受一个“看起来相似”的 Bank。

### 3.5 可见状态语义表示与数值运行时

PM 还可读取两个由同一冻结本地 encoder 产生的可部署视图：当前 user turn，以及
`current turn + recent visible dialogue + visible session summary`。完整视图必须按冻结的
section-aware 协议组装：当前 user turn 和 session summary 分别保留显式预算，剩余容量只取
保持时间顺序的 recent-dialogue suffix；不得依赖 tokenizer 的隐式右截断。当前合同固定为
`BAAI/bge-small-en-v1.5` 的精确 40-hex revision、snapshot tree SHA、CLS pooling、L2
normalization、512-token 总上限和 384 维输出；运行时禁止下载或 remote code。

报告性 development、training 和 external 还必须使用独立 `.venv-pm-v1-5` 中的精确
Python/NumPy/SciPy/scikit-learn/PyTorch/Transformers/tokenizers/Hugging Face Hub/
safetensors 版本，并设置 `PYTHONNOUSERSITE=1`。每次真实阶段都要现场重算固定 canary，
核对依赖版本、用户 site 隔离、encoder/spec/snapshot 和数值矩阵哈希；不能仅继承一次历史
`PASS` 报告。`sim_eval` 和 Conda `base` 都不是本合同的报告性运行环境。

两个视图拼接后只能在 train users 上拟合 PCA 到 48 维；calibration、internal 和 external
只能 transform。encoder/spec/hash/dimension 只进入 provenance 和 freeze，不得成为数值
identity feature。development 与 external 若 binding 不同必须在任何付费调用前 fail closed。

为防 source similarity 变成合成 oracle，普通 non-needed source 包含一条 same-topic、明确
非个人且无边际价值的目录项，再配一条异题 distractor；needed source 包含 helpful +
distractor；`memory_harmful` 的每个源则包含一条真实 unsafe contrast 和一条 same-topic
非个人 decoy，避免 harmful 分支重新形成特殊 centroid 答案键。
这只是构造假设，仍必须同时通过 actual item-utility semantic gate 和 train-only shortcut
probe，不能因为代码这样写就宣告问题已解决。

## 4. 动作、尝试、实现与 prompt alias

四个概念必须分开保存：

- `requested_action_id`：路由器在 Step-0 后请求的 4-bit 动作；
- `retrieval_attempts`：逐来源调用、命中数、失败原因、tokens、latency 与 USD；
- `realized_action_id`：根据实际进入 prompt 的 memory 来源和 Strategy 证据重算；
- `prompt_equivalence_id`：对规范化后的实际生成器输入做内容寻址。

zero-hit 是合法结果。例如 `requested=ME+RS` 可在 ME 零命中、RS 三命中时得到
`realized=M0+RS`。非法的是把 requested 写成 realized，或声称没有进入 prompt 的证据已
经生效。

完整 action sweep 仍保留 requested-action 的 intention-to-treat estimand。若多个 requested
动作产生相同实际 prompt：

- 可以只生成/判断一次以避免重复付费；
- 所有 alias 行必须显式共享同一 `prompt_equivalence_id` 和 label lineage；
- 每行仍保留独立 requested action、attempt 和 realized action；
- 训练、bootstrap 和有效样本量不得把同一生成/判断复制成独立证据。

## 5. required-hit 只是一项 outcome 前数据有效性门

required-hit 不得根据 outcome 分数筛选样本。它只用于数据构造阶段预先声明的 positive
challenge cells：若某 cell 声称某来源存在必要且可检索的证据，则在生成任何候选回复和
judge outcome 前，确定性 retrieval preflight 必须证明有命中，否则整个 cell/state
作废并按预注册规则重新构造。

自然分布和 external 中的 zero-hit 不删除、不重抽，也不视为错误。报告必须分开：

- natural/effectiveness strata；
- required-hit challenge/data-validity strata。

## 6. 候选机制与训练算法

在 internal-test 前必须同时冻结以下候选：

1. learned router + Step-0（主要机制）；
2. 强透明 rule router + 完全相同的 Step-0（主要机制 baseline）；
3. learned router without Step-0（只作内部消融，不进入外部主矩阵）；
4. cost-matched fixed action；
5. 显式 `ME+R0` 固定策略；
6. `MPMSME+RS` structured high-resource fixed。

固定策略不读取 Step-0，也不被人为添加未执行的 Step-0 成本。learned 与 rule 必须执行
同一 Step-0；learned 的总成本包含 Step-0，cost-matched fixed 按该总预算匹配。

### 6.1 训练侧候选算法

模型 family 只能在 train users 的 group-aware CV 中选择：

- 当前 absolute-outcome factorized HGB，作为保守基线；
- state-centered / paired-delta factorized HGB，作为优先候选；
- rule-relative safe residual HGB；
- LambdaMART/group ranking，作为探索候选而非默认赢家。

动作必须表示为 MP/MS/ME/RS 四个 factors，并允许预声明的 pairwise interactions。比较中
保持相同特征、用户分组、bootstrap 数、风险头和评估预算，避免同时改变目标、模型和
不确定性算法后无法归因。

rule-relative residual 必须把 `a_rule(s)` 或等价 baseline-action 编码纳入训练，并保证
同一 state 的 paired rows 与同一 user 永不跨 fold。只有下列保守条件同时满足，learned
才允许覆盖 rule action：

```text
delta_quality_LCB >= -epsilon_quality
delta_support_LCB >= -epsilon_support
delta_risk_UCB <= epsilon_risk
delta_utility_LCB > 0
```

不满足时回退到规则动作。无论采用 delta 还是 ranking，绝对 quality/risk ceiling 继续
生效，不能因相对改善而放行绝对不可接受的动作。

### 6.2 shortcut 与可辨识性审计

在候选选择前必须报告：

- Step-0 单特征和简单阈值对 source/RS oracle 的可预测性；
- 将全部 Step-0 标量联合输入正则 logistic 与浅层树的 user-group
  cross-validation probe，防止 XOR/interaction 一类组合 shortcut 绕过单特征门；
- shuffled-label、permuted-source、centroid-noise 和 no-Step-0 消融；
- 各 action factor 主效应及预声明 pairwise interaction 的覆盖；
- user/regime/environment 对 representation 的可识别性；
- alias 后独立 prompt/outcome 的真实数量。

oracle 可预测性审计只允许读取 train 的 216 个 resource/regime oracle；calibration 与
internal-test 的 252 个 state 只做不读取 resource oracle 的结构、范围、缺失和身份审计。
审计报告必须显式写入 `internal_resource_oracle_read=false`。

若简单单阈值或任一冻结的低容量多变量 probe 几乎完美恢复 synthetic label，不能把
learned PM 的高分解释为复杂状态调度。该审计不读取 action outcomes，必须在 52-user / 468-state
数据完成后立即独立运行并生成 attestation；full action sweep 不得等到训练阶段才首次运行它。

## 7. 分割、冻结与一次性 internal-test

数据角色必须硬隔离：

- `train`：选择模型 family、目标形式、interaction、结构超参数和 transparent-rule
  数值阈值；算法 family 使用 user-group CV，并以 one-standard-error rule 在近最优者中
  优先选择预注册的更简单、更稳定候选；
- `calibration`：只消费一次，用于冻结不确定性、selector margin 与 cost-matched fixed
  frontier，不再反复调整机制 baseline；
- `internal_test`：只消费一次，评估已冻结的唯一 primary candidate 和预注册消融；
- `external`：仅在 internal gates 通过并生成 study freeze 后执行。

internal-test label 文件在训练前先生成不可变 seal，记录 label 内容哈希、行数、state
universe 和 schema 哈希；candidate manifest 必须绑定这个 seal。internal-test 前写入内容
寻址 candidate manifest，至少包含代码、配置、数据、Step-0、模型、rule、阈值、
checkpoint、sealed label bundle 和输出 schema 哈希。消费动作必须写入 append-only
ledger；label 与 seal 不一致或同一 run identity 第二次读取 outcome均直接失败。查看
internal-test 后不得改变 primary candidate、阈值、baseline 或外部 condition matrix。

完整 468-state corpus 在 action response generation 前还要通过三道门：实际构造文本的
双开发家族 12-field 全量语义审核，以及按 train/calibration/internal-test 分开计算的
provider-surface fallback 上限，另加 train-only oracle / all-split structural 的 Step-0
shortcut 审计。内部测试 split 的 fallback 上限为 0；审核报告和输入
states、evaluator contexts、memory backend、Strategy Bank 和配置哈希必须由 attestation
绑定，并与下游当前实际输入逐一一致。生成协议的前置小型审查同时包含 27 个确定性
case 与 exact paid 9-case compatibility artifact，不能用未绑定真实 provider surface 的
fixture PASS 代替，也不能替代真实 468-state gate。两道自动审核各自冻结 12 字段 ×
每字段 2 个 hard controls；
零 controls、字段缺失、重复覆盖、seed/数量漂移或 control matrix hash 不一致均直接失败。
controls 使用真实候选值交换、标签翻转、age 算术矛盾、时序复制、grounding donor、
premature-strategy 注入等可读 corruption，不再使用 sentinel 字符串。

生成器接口本身冻结为 surface-only casewise 合同。每个物理请求只包含一个
semantic family/regime 的四个可见自然语言字段，不向 provider 暴露 memory、oracle label、
coverage rationale、case ID 或 evidence blueprint。每个用户的 9 个 case 分开请求；初次
schema/topic/structure lint 失败时，只允许同 case 的一次独立 seed、提前计入预算的 repair。
成功立即停止，repair 失败则整次正式生成 fail-closed；确定性 fallback 不得作为训练语料。
因此 52-user 正式生成成功路径为 468 calls，硬上限为 936 calls，而不再是旧整包 52 calls。
角色顺序不是 provider 的语义任务：provider 只填写 1–2 个
`{user_text, assistant_text}` 历史 exchange，本地 compiler 再展开为严格的
`user → assistant` turn 序列。这样 2–4 个历史 turn 的变化仍保留，但交替和“历史最后一条
必须是 assistant”成为 schema/compiler 不变量，不能再靠 prompt 或事后放松 lint。
`advice_readiness_target` 是合法的可见用户状态，不是 Strategy-value 标签：本地 compiler
用多种自然句式把它写入 current turn，并在 user 内对 Strategy-use/skip 反平衡；因此 PM
能够观察“想倾听/可接受小建议”，却不能用该句式直接猜 RS 是否产生正边际价值。

两道 semantic review 的 judge endpoint aliases 也分别在 PM 配置中按顺序锁定。runner
必须精确使用该面板，不能以任意“同样是两个独立开发家族”的 CLI override 替换。
gate report、cost plan 与 attestation 同时记录 alias、family、model、base URL；下游在接受
PASS 前必须将这些 descriptors 和 attested `experiment_config` 哈希与当前配置逐项重建、
比对。alias、model、base URL 或 experiment config 任一漂移均使旧 PASS 失效。

## 8. 三层 Gate

所有差值方向统一为 `PM - comparator`，以 user 为主要独立 bootstrap block。精确 margin、
置信水平和 utility 权重已经在新配置中冻结；任何修改都会改变配置哈希并使现有
dry-run identity 失效。

### Gate M：机制价值（learned vs strong rule）

必须同时满足：

- quality 与 emotional support 非劣；
- resource-use risk 不增加；
- total utility 的配对 CI 下界严格大于 0；
- 回退、多样性、最大 action share 和严重 OOD 预注册护栏通过。

Gate M 不通过，不能声称 learned routing 比透明规则更有价值，也不能进入外部 learned
机制主验证。

### Gate F：竞争性固定策略护栏

分别对 cost-matched fixed 和 `ME+R0` 检查：

- quality/support 非劣；
- risk 不增加；
- utility 至少非劣；
- 成本和 action usage 完整报告。

只有 utility 严格优势通过预注册 CI 时，才可写“优于”相应 fixed comparator；仅非劣时
只能写“未全面弱于/保持竞争力”。若 PM 相对任一硬护栏明显劣化，不得只用相对
`MPMSME+RS` 省 token 来包装成功。

### Gate E：外部效率（learned vs structured high-resource）

必须同时满足：

- quality 非劣；
- 分层 evidence-risk audit 不显示增加；
- observed generator input tokens 的配对 CI 上界严格小于 0；
- Gate M 与 Gate F 已通过且 lineage 完整。

Gate E 只支持“保持质量并减少生成器输入”的窄效率主张。它不自动证明总美元、总延迟或
learned routing 优于同预算 fixed。

## 9. 成本与资源报告

不得再把不同资源折成一个未经验证的“token cost”。至少分别报告：

- Step-0 encoder 本地输入 token estimate、invocation、memory/family/readiness
  comparisons 与逐 turn 耗时；
- source catalog 和 Strategy family/readiness catalog 的一次性 refresh 耗时；
- item-level memory/strategy retrieval attempts、hits 与耗时；
- observed generator input/output tokens；
- judge tokens（实验成本，不混作部署推理成本）；
- wall-clock latency；
- 按冻结价格计算的 USD。

主要效率指标暂保留 `observed generator input tokens`，因为它可直接观测且与当前主张
一致。总成本和 latency 只能描述性报告，除非另行冻结真实计价与 interleaved latency
设计。

## 10. 外部 condition matrix

外部主要矩阵按主张组织，不为保留“七条件”而添加弱 baseline：

- learned + Step-0；
- strong rule + Step-0；
- cost-matched fixed；
- `ME+R0`；
- `MPMSME+RS` structured high-resource。

`M0+R0`、raw-session top-k、full-history 可作为预注册 secondary references。no-Step-0
learned 只留在 internal ablation。任何 secondary condition 不得替代 Gate M/F/E 的指定
comparator。

若主候选是 rule-relative residual，移除 Step-0 后 residual reference 会从 transparent
rule 变为固定 `M0+R0`。因此该结果必须称为 `component-removal system variant`，不能写成
只改变一个 feature 的纯 2×2 feature ablation；非 residual 候选才允许使用后一种解释。

外部 batched schema/order pilot 使用三个预冻结 canary unit，覆盖 quality/risk、两个
order 和两个 judge family，共 24 calls。除 schema success 必须为 100% 外，mean absolute
order delta 与 maximum absolute order delta 还必须分别低于冻结阈值；只记录布尔“跑通”
不能放行主评测。

EvoEmo session 由 ISO date 升序排序，同日按原始位置稳定排序；所有 session ID、topic idx
和 `related_sessions` 引用在 freeze 前 fail-closed 校验。源数据没有 topic timestamp，因而
只主张 subsequent topics 位于完整排序历史之后，不虚构更强的 topic chronology。

## 11. Judge 角色隔离

development semantic gate、完整 action judging 与 final external judging 的端点必须在
以下四层完全不相交：

- endpoint alias；
- declared family；
- model identifier；
- normalized base URL + model route。

当前允许的 development panel 是 Gemini Flash Lite + DeepSeek Flash；final panel 是
GPT-4o + Claude。任何使用 `final_judge` 或同 family/model 的历史 development 自动审核
均不满足本合同，不能继承 PASS。隔离报告必须进入 dry-run hash、run manifest 和 artifact
attestation，并在 API key 解析与付费调用前 fail closed。

## 12. 失效与新 run identity

本合同改变方法身份、judge panel、成本计划和比较矩阵，因此以下材料全部只保留作历史：

- 旧 automated-semantic-review partial attempts 与判断；
- 旧 generation compatibility pilot；
- 所有旧 dry-run、call plan 和 accepted cost hash；
- 旧 candidate/freeze/config/attestation hash；
- 基于纯 pre-retrieval/free-probe 定义形成的 checkpoint 或结果。

新运行必须使用新的输出目录和 run identity；不得覆盖、拼接或抽取旧成功调用。

## 13. 激活清单

- [x] 本目标合同写入仓库；
- [x] CI 的 installed-package import 修复；
- [x] development/final judge 四层隔离实现与单元测试；
- [x] Step-0 schema、构建、成本和 shortcut audit 实现；
- [x] requested/attempted/realized/prompt-equivalence 全链路实现；
- [x] required-hit pre-outcome validity gate 实现；
- [x] transparent rule router 与 no-Step-0 ablation 实现；
- [x] transparent-rule 全候选映射的 outcome-free 预训练诊断、内容寻址 attestation 与
  pre-sweep 硬门实现；
- [x] train-only algorithm comparison 与 candidate manifest 实现；
- [x] full / no-Step-0 / no-state-BGE / lexical-only 的 2×2 checkpoint 在 internal-test 前
  一次冻结；internal 结果只作解释，不允许回调候选；
- [x] section-aware visible-state 输入、禁止隐式截断、精确数值运行时与 live canary lineage
  在 development/training/external/freeze 全链路实现；
- [x] one-standard-error 选择、train-only rule tuning 与 calibration 单用途实现；
- [x] pre-training internal label seal 与 append-only consumption ledger 实现；
- [x] Gate M/F/E 数值配置与测试实现；
- [x] 新 external condition matrix、freeze 与 claim wording 实现；
- [x] 52-seed/Strategy Bank 实例级隔离并保留八个策略家族；
- [x] actual 468-state semantic/fallback gate、独立 pre-sweep shortcut runner 与
  user-group 多变量 probe 实现；
- [x] EvoEmo chronology 校验与 3-unit/24-call 数值 order pilot 实现；
- [x] 所有 V1.5 `--run` 入口统一中央付费门实现；
- [x] 专用 Python 3.13.2 `.venv-pm-v1-5`、`PYTHONNOUSERSITE=1` 下与 CI 一致的裸
  `pytest -q` 全量通过；
- [ ] 新 paid-run 的 dry-run、lineage 人工审查与 approval manifest 完成。

在最后一项完成前，合同状态不得改为 `ACTIVE`。
