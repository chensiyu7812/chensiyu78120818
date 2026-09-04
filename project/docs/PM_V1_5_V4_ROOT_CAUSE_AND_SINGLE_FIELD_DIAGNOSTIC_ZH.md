# PM v1.5：V4 失败根因与 first-paper-scoped measurement v4.2

> 状态：`V4.1_CONSUMED_NOT_READY / V4.2_CONSUMED_8_OF_8_NOT_READY / CALIBRATION_ONLY / NOT_A_GATE`
>
> 本诊断不改变 V4 的 `CONSUMED_FAILED_CLOSED`，不授权训练、52-user generation、
> V5 或论文结论。V4 的 24 controls/36 cases 已被观察，只能作为 calibration data。

## 1. 第一篇论文的主张边界

第一篇不声称 PM 能直接识别真实用户的潜在心理需求，也不声称回复会带来真实情绪改善、
长期行为改变或临床效果。当前 PM 学习的是：

```text
冻结的合成可见状态 x
    × 16 个可执行资源动作 a
    → 冻结 generator/retrieval 下的响应
    → 双家族 LLM judge 的质量/风险代理分 + 可观测成本
    → proxy utility U(x, a)
    → 监督式 pre-item-retrieval router
```

因此可检验的窄主张是：在冻结的合成任务、资源合同和代理评分下，状态条件化 router
能否相对 fixed baselines 改善 quality-risk-cost trade-off。下列说法均禁止：

- “识别了用户真正需要什么”；
- “理解了真实用户的内心状态”；
- “改善了用户福祉/临床结局”；
- “学到了真实世界最优策略”。

没有真实用户金标签并不使这项实验自动无效，但要求结果始终称为 synthetic/LLM-judged
proxy utility，并把 judge 的测量有效性作为独立前提，而不是把 judge agreement 当成真值。

## 2. 为什么 V4 的测量工具不合格

V4 将有依赖关系的 candidate labels 和派生 rationale 一起暴露，并在一次响应中要求 12
个二元判断。candidate regime、source utility、Strategy target、readiness 与 rationale
能互相作证，输出合同又允许最省力的“全 1”。真实结果是：

- targeted controls：`0/24` 被抓住；
- DeepSeek：control 的 288 个字段全部为 1；
- Gemini：276 个 control 字段为 1、12 个非目标字段为 0，但目标 corruption 仍为 0/24；
- Gemini 与 DeepSeek 在 real cases 上还有 27 个字段分歧。

这证明 V4 不能区分正确标签与定向错误，不证明 PM 方法本身失败，也不证明真实 cases
一定错误。

## 3. v1/v2 诊断实际说明了什么

旧诊断把 6 个错误 controls 分别做成 P1/P2，共 24 logical calls。它仍有三个根本问题：

1. 全部样本都是 negative；judge 一律反驳也可能得到高分；
2. age 和 temporal order 明明可由代码计算，却交给 LLM；
3. 一个关系命题只能引用一个 evidence key，无法证明 judge 真做了跨段比较。

### v1

identity `639b8c5b…d3af` 在第 3 次物理调用因本地 Gemini usage parser 漏记
thought/tool-use token 而 fail-closed。provider 并没有账目矛盾；该 identity 永久 consumed。

### v2

identity `3153ae32…7e70` 实际 20 次物理调用：14 成功、6 失败。四个 503 和一个 read
timeout 均重试恢复；第 15 个 logical call 的首次 Gemini 响应为 HTTP 200，但 candidate
text 不是 JSON，旧 retry v2 将其归为 `other/terminal_nonretryable`，所以立即停止，绝非
“耗尽三次重试”。包含所有成功和失败付费 attempt 的精确账目为：

```text
input       5,949
output      1,780
total       7,729
estimated   $0.0013069
```

更重要的是，已完成内容只有 `7/14` 正确；一致性与传输修复不能把这个事实变成 PASS。
因此没有盲目换 identity 重跑，而是重做测量工具。

### v3

balanced identity `47481579…13fa` 真实执行两次后停止。Gemini 正确返回
`evidence_key=memory_text`；DeepSeek 的 verdict 与逐字 quote 也正确，但把该正文 quote
挂到 `evidence_key=memory_id`，而该字段只含 opaque ID。旧汇报称其“转述/改写正文”是
事实错误。原始返回还以 `We{...}` 开头；旧 API client 在同一物理调用内静默截取花括号，
绕过了 ledger-visible format repair。v3 因而永久 consumed，不能重跑。

## 4. v4 的三层真值

### 4.1 确定性事实：代码/构造真值，不调用 LLM

| 字段 | 代码判定 |
|---|---|
| `source_type_match` | proposed source 是否等于 frozen memory record 的 registered source；不从单句自然语言反推排他性 taxonomy |
| `dialogue_temporal_order_match` | history index 连续、从 user/assistant 交替、以 assistant 结束、current 未复制进 history |
| `memory_age_design_match` | `current_session >= created_session` 且 `age = current - created` |
| `surface_naturalness_match` | 只检查冻结的显式 generated-user/resource-condition/regime-label/utility-label 泄漏 marker；不冒充广义自然度金标签 |

每个字段同时包含一个原始正确项和一个定向错误项。packet 中保存代码 assessment，runner
重新计算并要求逐字段完全相同；这些项不进入 API call plan。

### 4.2 仍需语义判断的代理原语：正反成对、二元支持

| 字段 | positive | negative |
|---|---|---|
| `context_grounding_match` | 原 summary 与 history/current/provenance 一致 | 换入正交 donor summary |
| `advice_readiness_match` | 原 `explore_first` | 对同一明确先澄清的对话标 `structured_plan` |

每个字段各 1 positive + 1 negative；两家 judge 各自审查，形成 `2 × 2 × 2 = 8`
logical calls。expected verdict 只在本地 packet 中：positive=`supported`，
negative=`not_supported`，绝不进入 messages。`not_supported` 同时覆盖明确冲突和证据未能
建立全部必要条件；本研究需要的是“能否作为支持证据”，不需要强迫 judge 做额外的逻辑
哲学分类。

### 4.3 主研究中的主观边际价值：不是绝对金标签

“是否应开启 MP/MS/ME/RS”不能由当前用户一句话直接得到真实金标签。正式 PM 仍必须依靠
16-action outcome sweep：比较同一状态下不同动作产生的响应质量、风险和成本，再学习
proxy utility。语义诊断只验证测量原语，不把 source/regime oracle 伪装成真实用户意图。

## 5. v4 prompt/output 合同

每次只提供：字段定义、一个 candidate claim、该 claim 的 exhaustive evidence、判定规则。
禁止出现 polarity、expected verdict、control ID、regime、utility 或 coverage rationale。

输出为：

```json
{
  "verdict": "supported | not_supported",
  "evidence_keys": ["一个或多个 exhaustive-evidence 顶层 key"],
  "evidence_quotes": ["与 key 逐项对齐的原文"],
  "reason": "解释原子判定"
}
```

`context_grounding` 至少引用两个不同 evidence sections；`advice_readiness` 至少一个。
record-linkage metadata（例如 `memory_id`）不进入 citable evidence namespace。key 必须
唯一、quote 必须真实存在于对应 evidence。citation adherence 独立报告，用于审计 judge
是否老实指向证据；它不再把一个语义 verdict 重新定义成错误 outcome。

## 6. provider 输出格式修复边界

transport 错误（408/429/5xx/timeout）可在冻结的 10-attempt 总预算内按
`10/30/60/120/300/300/600/600/900s` 退避。`missing_field` 与
`provider_output_format` 独立计数，最多观察 2 次 malformed provider surfaces；中间的
503/timeout 不再误消耗格式修复名额：

```text
503 → malformed → 503 → valid
  transport failures: 2 / 10
  malformed surfaces: 1 / 2
  允许继续，四次 attempt 全部写入 append-only ledger
```

schema 可解析但违反 Pydantic、4xx 仍不重试。structured-schema 首先要求 exact JSON；
只允许两种不改变 JSON 值的确定性规范化：单一 Markdown JSON fence，或不超过 80 字符
前后缀包围的唯一 JSON object。原始文本、hash、规范化类型和丢弃字符数全部留痕；多
object、长 wrapper、真正 malformed JSON 仍进入 ledger-visible `provider_output_format`。
citation defect 不触发重试，但作为已完成 observation 记录。单个 provider 最终不可用时
runner 继续剩余矩阵，最终状态为 `INCONCLUSIVE_PROVIDER_AVAILABILITY`，绝不把缺失当错误
verdict 或偷偷删掉。正式 gate 的 fail-closed 行为不变。

## 7. 当前真实结果与 v4.2 状态

离线 packet：

- 12 items：4 semantic + 8 deterministic；
- observed V4 controls manifest hash 逐项验证；
- API calls：0；
- formal gate：false。

旧 v4.0 identity `acb07969…f7caf` 已真实消费：Gemini `503→503→PASS`，DeepSeek
`503→HTTP-200 non-JSON→503`；实际为 1/16 endpoint calls、0/8 paired items，花费约
`$0.0000705`。它只能判为 provider availability inconclusive，永久禁止复用。

v4.1 identity `22265947…936d` 已真实消费并完成：

```text
logical calls             16/16
physical attempts         20 (16 success + 4 recovered transient failures)
actual usage              8,785 input / 1,945 output
actual cost               about $0.0016565
deterministic accuracy    1.0
semantic verdict          0.75
citation integrity        0.875
joint                     0.6875
measurement ready         false
```

这不是 PM 方法失败；它揭示 source type/显式 leakage 不该委托给 LLM，负例不该强迫区分
`contradicted` 与 `insufficient`，citation pointer 也不该和语义 outcome 合并。中央 manifest
已清空 approval 并把该 identity 永久登记 consumed。v4.2 identity
`fdb25896e8de475ffc5241daf371bddc06fa9d1ab56825dce3c6e3628dc59d30` 也已真实消费：

```text
logical calls             8/8
physical attempts         12 (8 success + 4 recovered HTTP 5xx)
actual usage              4,864 input / 1,420 output
actual cost               about $0.0010544
deterministic accuracy    1.0
semantic verdict          0.875
citation integrity        0.875
joint                     0.75
measurement ready         false
```

Gemini 的 semantic verdict 为 4/4，但一例 context grounding 只引用一个 section，违反“两段
证据”citation 合同；DeepSeek citation 为 4/4，但把 `explore_first` 错误收窄成必须显式说
“ready for exploration”，漏掉一个正例。前者只影响 report-only citation，后者说明 readiness
必须有可执行的操作定义。V4.2 不再重跑，`pending_unapproved_dry_runs` 已清空。

## 8. 结果解释规则

报告分别给出 binary support verdict accuracy、citation integrity、joint accuracy，以及每个
endpoint/field 的 sensitivity、specificity 和 balanced accuracy。一律回答 supported 或
一律 not_supported 都不能通过成对校准。citation pointer 错误会使 citation audit 失败，
但不会改写已经正确的 binary semantic outcome。

`measurement_instrument_ready` 即便为 true，也只是 calibration diagnostic：

- 不等于 held-out accuracy；
- 不改变 V4 failure；
- 不授权训练/正式 generation；
- 不证明真实用户需求识别或真实改善；
- 不证明广义自然度、临床有效性或人类 judge 一致性。

若 v4 仍失败，先看具体 field/polarity/endpoint：失败集中于某个语义边界属于测量/定义
限制；正反都失败属于 judge 任务不适配；确定性项失败属于代码或 artifact drift。禁止通过
调松阈值、删除失败样本或反复换 identity 把失败“跑没”。
