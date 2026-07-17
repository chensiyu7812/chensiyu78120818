# PM-v1.6 方法与确认性实验合同

状态：`FROZEN DESIGN BEFORE FORMAL EXECUTION`

## 1. 研究问题

本研究不要求 Memory 或 Strategy RAG 始终优于不使用资源的回复。它检验：

> 在长期情绪支持对话中，是否可以依据当前对话和低成本来源级观测，按状态选择 MP、MS、ME 与 Strategy RAG，在回复质量和资源误用风险不恶化的前提下，获得更好的质量—风险—成本权衡。

PM-v1.6 的确认性主张仅限冻结的 synthetic routing matrix 与 EvoEmo 模拟外部评测。

## 2. 版本纪律

PM-v1.6 在任何正式 PM-v1.5 development matrix、checkpoint、freeze 或 external result 产生前替代 PM-v1.5。旧 V1.5：

- 不删除；
- 不覆盖；
- 不复用旧 config hash、semantic-review PASS、pilot hash 或 checkpoint；
- 标记 `SUPERSEDED_BEFORE_FORMAL_EXECUTION`；
- 旧 GPT-4o development semantic-review 标记 `INVALID_FOR_PM_V1_6_PROTOCOL`。

## 3. 因果流程

```text
current dialogue state
        ↓
fixed Step-0 source observation
        ↓
strong transparent router default
        ↓
learned conservative residual override
        ↓
requested action
        ↓
item-level retrieval attempts
        ↓
realized evidence/action
        ↓
generator prompt and response
```

Step-0 固定执行于 learned PM 与 strong rule。固定动作无路由需求，不支付虚构的 Step-0 成本。

## 4. Step-0

### 4.1 PM 可见字段

每个 MP/MS/ME 来源只允许：

- available；
- bounded count；
- min/median/max age；
- estimated retrievable tokens；
- query-to-source similarity；
- representation valid。

Strategy 只允许：

- 九个冻结 strategy-family centroids 的 query similarity；
- representation valid；
- 当前话语派生的 advice requested/rejected、listening requested、clarification needed、action readiness。

### 4.2 禁止字段

PM 不能看到：

- memory/strategy 原始文本；
- item IDs；
- item-level top-k、max、P90 score；
- selected snippets；
- catalog embedding 及 norm/mean/std；
- oracle helpful/harmful；
- regime；
- stale/conflict oracle；
- representation version/hash。

版本、hash、dimension 与 catalog build 信息仅进入 audit binding。

### 4.3 相同机制

development、internal、external 必须调用同一个 `build_step0_observation` 实现、相同 encoder、维度、strategy-family 顺序和 refresh policy。

## 5. Action lineage

每条动作必须记录：

```text
requested_action_id
retrieval_attempts[source, call_count, hit_count, tokens, latency]
realized_action_id
prompt_equivalence_id
prompt_equivalence_class_size
```

zero-hit 是真实后果，不自动失败。例如：

`requested ME+RS → ME zero-hit, RS hit → realized M0+RS`

但必须保留 ME 的调用与成本。

只有日志中的 realized action 与实际进入 generator 的 evidence 不一致时 fail closed。

## 6. Alias 与共享标签

相同 prompt 可以只生成一次物理回复，但：

- quality/evidence-risk label 绑定 prompt equivalence；
- requested action 的 retrieval/cost 记录保持独立；
- 共享 quality label 使用 `1 / alias class size` 权重；
- alias rows 不能被当成独立回复扩大有效样本量；
- 所有 alias 转移矩阵与比例公开报告。

## 7. Required-hit

以下检查在 response generation 和 judging 前执行：

- profile_needed → MP hit；
- summary_needed → MS hit；
- event_needed → ME hit；
- multi_source_needed → MP/MS/ME hit；
- strategy_helpful/harmful → RS hit。

失败按冻结尝试上限重新生成整个 case/bundle，并保留失败记录。禁止根据 outcome 删除样本。

## 8. 数据防 shortcut

Natural strata 与 probe challenge strata 分开报告。必须包含：

- high-sim stale/conflicting/irrelevant；
- low lexical similarity but helpful；
- swapped source；
- multi-source redundancy；
- strategy similarity but advice rejected。

必须训练 probe-only、text-only 与 shuffled-probe 诊断。probe-only 不能成为近乎完美的 oracle key。

## 9. Strong transparent router

transparent router 使用与 learned PM 完全相同的 Step-0 观测。来源分数由 similarity、age、expected cost 和 availability 构成；RS 同时考虑 strategy-family similarity 与 advice/readiness。

参数只在 train users 上从冻结小网格选择，并使用 one-standard-error 后偏向更简单的 maximum-sources 设置。

它承担三种角色：

1. 强非神经基线；
2. learned PM 的保守默认动作；
3. 外部 learned-routing 主 comparator。

## 10. 算法竞争

internal test 前不预定某个算法必胜。train-only user-block CV 比较：

1. absolute-outcome HGB；
2. state-centered factorized HGB；
3. rule-relative residual HGB。

LambdaMART 只作排序敏感性，不承担绝对风险门。

当前 absolute HGB 已有 action bits、text×action interactions 与 paired selection，必须保留为真实竞争者。

state-centered 模型使用：

`Y(s,a) = g(s) + tau(s,a)`

rule-relative 模型必须输入 candidate action、rule action 与二者差值；rule 必须在 train users 上先冻结。

模型选择使用 policy regret、非劣违规、绝对风险、utility、系统成本、action stability 等，不以 head MSE 为主。使用 one-standard-error 原则，误差范围内选更简单模型。

## 11. Calibration

calibration 不能重选算法或 transparent rule。它只能从冻结候选网格选择 conservative override：

- candidate absolute risk UCB ≤ ceiling；
- quality delta LCB ≥ -margin；
- emotional support delta LCB ≥ -margin；
- risk delta UCB ≤ margin；
- utility delta LCB > 0。

“conservative”仅表示当前 judge/model contract 下保守，不是现实安全证明。

## 12. 成本向量

至少记录：

- Step-0 encoding latency、memory comparisons、strategy-family comparisons、amortized catalog build；
- router inference；
- item retrieval calls/hits/tokens/latency；
- generator observed input/output tokens、API cost、provider latency；
- total variable cost 与 end-to-end latency。

`generator input tokens` 必须继续单独报告，但不能称为系统总成本。固定动作不支付 Step-0，learned PM 与 strong rule 支付实际 Step-0 成本。

## 13. Internal test 封存

train/calibration 脚本不得接受 internal 路径。

internal test 前生成 candidate-family manifest，绑定：

- candidate family；
- selected primary；
- selection rule；
- train/calibration/internal hashes；
- Step-0 contract；
- strong router；
- checkpoint；
- candidate report。

评测脚本先 reserve append-only ledger，再读取 internal data。同一 internal dataset hash 只能消费一次，失败也为终态。

## 14. 三层 gate

### Gate 1：机制与数据

包括 CI、judge isolation、Step-0 forbidden fields、lineage、required-hit、alias、shortcut diagnostics、complete action matrix、cost accounting。失败则不训练。

### Gate 2：internal

Learned PM 相对 strong rule：

- quality 非劣；
- emotional support 非劣；
- absolute risk 不超过 ceiling；
- risk 不增加；
- utility CI 下界严格为正。

ME+R0 是 legacy regression guard，不是 primary algorithm baseline。PM 不得在 quality 与 utility 上显著回归至弱于 ME+R0。

失败输出 `INTERNAL_NOT_SUPPORTED`，禁止 external generation。

### Gate 3：external

七个主条件：

1. learned PM + Step-0；
2. strong rule + Step-0；
3. cost-matched fixed；
4. best high-resource fixed；
5. ME+R0 legacy anchor；
6. M0+R0；
7. session-RAG+RS。

full-history 仅描述性附录。

## 15. 外部主张

Claim A：learned PM 是否优于同观测 strong rule。  
Claim B：learned PM 是否在质量非劣、风险不增加时比 high-resource fixed 更低成本。  
Claim C：与 cost-matched fixed 的确认性 secondary comparison。  
Legacy guard：不得包装成仍全面弱于 ME+R0 的成功。

任一对应 gate 不通过，自动输出 `NOT_SUPPORTED`。

## 16. 禁止主张

不得声称：

- 临床有效；
- 真实用户情绪改善；
- 部署安全；
- RL 或 POMDP 已完成；
- 所有场景最优；
- 延迟必然下降；
- EvoEmo 完全独立于 ESConv。
