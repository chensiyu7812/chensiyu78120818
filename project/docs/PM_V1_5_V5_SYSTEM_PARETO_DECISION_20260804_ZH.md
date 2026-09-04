# PM V1.5 V5 系统级 Pareto 决策

状态：`ONE-SHOT FRESH CONFIRMATION PASSED / RUNTIME FROZEN FOR T5`

## 2026-08-04 一次性确认结果

V5.1 已在读取任何确认回复或人评结果前冻结全部模型、特征、`.5` 阈值、retriever、typed executor、
generator、128 个 state-candidate unit、64 个 counterfactual group 和系统门。确认生成 512/512 完成，
ITT mechanical invalid 为 0；随后由不同主评分别完成 256 个盲态质量对和 512 个单臂 grounding-risk
审核。冻结聚合结果为 `PASS_SINGLE_USE_SYSTEM_GATE`，全部检查通过：

- learned PM 请求 ON 比例 `51.56%`，不是全开或全关；
- 相对 fixed-high，质量 `59` 胜、`40` 负、`157` tie/同臂，平均差 `+0.0742`，64 组聚类
  bootstrap 95% CI `[-0.0430, +0.1914]`；下界通过预设 `-0.05` 非劣门，但裕量较窄；
- 相对 always-off，质量平均差 `+0.1680`，95% CI `[+0.0352, +0.2969]`；资源没有退化成整体无用；
- learned material risk 为 `11/256=4.30%`，fixed-high 为 `13/256=5.08%`；点差 `-0.78pp`，
  95% CI `[-3.13pp,+1.17pp]`。准确结论是方向更低且非劣，不是显著降低；
- 平均 prompt tokens 为 `222.54`，fixed-high 为 `250.25`，减少 `11.07%`；
- 相对 transparent rule，质量平均差 `+0.1836`，95% CI `[+0.0586,+0.3086]`，risk 点差
  `-0.39pp`，prompt tokens 少 `19.47`；
- fabricated recall 与 internal-resource-label exposure 合计 `1` 个，fixed-high 为 `6` 个，满足上限。

系统级通过不改写原 V5 单头结论：MP/MS/ME/RS 没有任何 head 通过旧的全部 BA/Brier/leave-family
晋级门；它们仍作为有限语义能力的机制诊断。确认中的组件拆分也不得被用于结果后修模：ME 在该内部
确认集相对 always-off 的组件级质量均值为负，必须作为局限和外部检验问题报告，而不能建立 V5.2。

本结果只授权停止内部开发并进入一次 T5：sealed internal、ESConv 与 EvoEmo。所有后续结果只写入论文，
不得回流修改当前 PM、阈值、executor、RAG Bank 或记忆编译器。

## 结论

V5 四个单独 head 没有通过原先冻结的全部分类资格门，这一事实保留；但它不能继续被解释成
“PM 没学会”或“MP/MS/ME/RS 都应关闭”。这些门测的是每个组件能否逐条复现 reviewer-proxy 的
`worth_opening` hard label，而论文真正的主张是：

> 在固定资源、检索器和生成器下，学习一个低容量、可审计的预注入策略；相对高资源策略，在不
> 实质降低即时回复质量的前提下，减少 interaction-and-grounding risk 和上下文成本。

按这个主张重放冻结的 grouped-OOF 决策后，当前 learned candidate 在 FIT 上请求开启134/256个
state-candidate unit，不是全开或全关。相对 component-fixed-high：

- 质量：84次实质更好、81次实质更差、347次实质等价；
- material risk：50/512，对照为103/512，点估计减少51.5%；
- prompt tokens：均值224.4，对照250.1，减少10.3%；
- total tokens：均值274.0，对照298.9，减少8.3%。

相对 always-off，learned candidate 为148次更好、33次更差、331次tie。相对同一fixed-high参照，
transparent rule只有12次更好、70次更差，并产生95个material-risk事件。当前候选因此具有真实的
开发态Pareto信号，不能因MP/ME/RS单头BA约`.50-.60`便直接废弃。

## 为什么低单头BA仍可能形成有用PM

Balanced accuracy把每个hard ON/OFF标签视为同等重要；系统效用并不是这样：

1. 大量ON/OFF回复实质等价，路由错一个tie不会伤害质量，却仍可能节省token或避开risk；
2. false-on与false-off的代价不对称，错误开启可能引入记忆误用，错误关闭常退回可采用的基础回复；
3. 四个head的联合策略只需在关键状态做对，不需要逐条重现所有主观偏好；
4. 论文比较的是完整policy，不是要求每个head成为优秀的语义分类器。

因此，单头BA、Brier和leave-family是机制诊断与局限性证据，不再单独拥有否决与论文主张不一致的
系统候选的权力。

## Step1与Step2的责任边界

Step1只使用注入前可见状态、actual candidate descriptor和确定性后端事实，输出四个bit并投影到
合法16动作。Step1不生成回复、不选择候选ID、不解释历史，也不能提前知道Llama这次是否会漏出处。

Step2接收Step1允许的exact candidate，由typed adapter绑定owner、time、source和动作边界，再由
固定Llama-3.1-8B写回复。当前adapter只保证prompt编译确定；它没有从结构上锁住最终历史子句、出处
或原子动作数。因此fabricated recall、出处漂移、第二动作等属于冻结executor的真实outcome，而非
Step1标签无效的理由。

V5采用requested-action ITT：Step2漏用、误用、fallback、质量tie/loss都保留为“请求这个资源”的
后果。这正对应部署问题：PM要控制的是一个并不完美的资源执行器。如果learned policy仍能降低risk
和cost并保持质量，主张成立；并不要求generator在每次开启后都完美做功。

## 为什么现在不继续修Llama

提示词不能保证8B模型严格服从多条自然语言合同。V5若在看到FIT结果后继续改prompt、guard、特征或
阈值，必须重新生成所有treatment并重训；否则属于同一考卷上的结果后开发。

当前版本因此冻结executor，先回答PM主问题。若fresh confirmation失败，下一版本采用真正的
backend-locked composer，而不是第六轮prompt修补：

- MS/ME的历史出处和关键事实由typed fields确定性组成不可改写的子句；
- RS只有一个atomic move槽，结构上不存在第二动作；
- MP preference用长度、问题数、选项式/指令式等结构约束执行；
- LLM只填当前语境的自然语言连接槽；
- 失败直接使用确定性、无历史的安全回复，不进行第二次自由LLM回退。

这是executor版本变化，必须在新FIT上重训PM，不能混入V5.1确认。

## 唯一下一步

已冻结 `v5_1_system_pareto_confirmation_v1.json`。它锁定当前四个final model、scaler、`.5`阈值、
128个FRESH_CONFIRMATION unit、64个counterfactual group和两个seed，不读取任何confirmation、
sealed或external outcome。

在任何confirmation API调用前，必须先用同一冻结retriever物化actual Rank-1，并通过零outcome数据门：
与FIT没有`current state + candidate`规范化重复，counterfactual group完全隔离，pair不跨统计组，候选
owner/time正确，construction enrichment/split/topic-prefix不进入模型输入，可见cue不能完美锁死
标签或subtype。若此时发现题目构造bug，可以在零回复、零标签条件下机械修复；一旦API开始则不得再改。

一次性系统门为：

1. 相对fixed-high的质量signed preference，group-cluster bootstrap 95%下界不低于`-0.05`；
2. 相对always-off点估计不为负，且95%下界不低于`-0.05`；
3. risk相对fixed-high点估计不增加，95%上界不超过`+0.05`；
4. prompt token相对fixed-high至少减少5%；
5. 相对transparent rule质量非劣、risk与cost不更差，且至少一个维度点估计严格更好；
6. ON比例位于10%到90%，避免全开/全关；
7. fabricated recall与internal label exposure合计不超过2次，且不高于fixed-high。

通过后直接冻结16动作runtime并进入sealed internal、ESConv和EvoEmo；失败则形成bounded或negative
V1.5论文结论，不再建立第二份confirmation或回到小包盲评循环。

## 证据边界

当前数字来自FIT grouped OOF和blind-reviewer proxy，不是独立确认、临床安全或外部泛化证据。
人评不是“黄金答案”：它只为质量与interaction/grounding risk提供预先定义的测量代理；最终判断
依赖policy-level的配对差、置信区间、客观token成本和敏感性分析，而不是某一个评审者的一次标签。
