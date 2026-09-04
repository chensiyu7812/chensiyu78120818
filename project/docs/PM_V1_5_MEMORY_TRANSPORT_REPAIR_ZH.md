# PM V1.5 个性化记忆跨域迁移修订

## 结论

用户提出的问题成立，而且是强外部泛化主张的高严重度缺口。原 192 个
MP/MS/ME contrasts 只保留为小 catalog 诊断，不能再承担修复后的正式 memory-head
训练。进一步复核发现，原 64 个 RS contrasts 中有 56 个运行在旧的两条式 memory
background 上，也不能与修复后的 memory heads 拼成一个正式训练集；原混合盲评包
已经暂停，不应继续评。

Strategy RAG 与个性化记忆不能用完全相同的“同一套数据”标准：

- RS 是全局工具书，所以 development、internal、ESConv、EvoEmo 必须使用字节级相同
  的 Bank 和运行栈；
- MP/MS/ME 是用户历史的函数。不同用户、不同语料的 memory item 本来就必须不同；
  如果 EvoEmo 仍使用内部虚构用户的记忆，才是错误和数据污染。

对记忆真正必须相同的是函数合同：

`历史 → MP/MS/ME 编译 → query → retrieval/filter → candidate descriptor → PM 开关 → 注入`

不同数据经过同一合同产生不同候选，才是有意义的外部泛化。按语料临时改变 source
定义、摘要方式、chunk、Top-k、filter、descriptor 或 prompt，则与旧“两套 RAG”
属于同一种 treatment shift。

把内部和 EvoEmo 全部放进一个物理向量库并不能解决这个问题：

- 若没有强制 `(user_id, created_session < current_session)` namespace，检索到别人的
  私人记忆是严重的记忆误用和数据污染；
- 若有严格 namespace，它在逻辑上仍是每位用户各自的 catalog，只是物理上共用一个
  数据库；这可以用于部署，但不增加外部泛化证据；
- 若训练时允许读取 EvoEmo 用户内容或 outcome，外测就变成 transductive/development
  informed，不能声称从内部用户泛化到未见外部用户。

机器可读合同为
`data/pm_v1_5_contracts/memory_transport_candidate_contract_v1.json`。

## 修复前实际审计发现

下表是共享 bounded compiler 接入**之前**的原始审计，只读取 memory catalog 和
运行时状态，不读取任何 EvoEmo 回复质量、risk、judge
或 reference answer。完整数字在
`outputs/pm_v1_5_memory_transport_contract_audit_v1/memory_transport_audit.json`。

| 指标 | 内部 synthetic | EvoEmo |
|---|---:|---:|
| catalog records / users | 468 | 18 |
| MP 每 catalog 中位条数 | 2 | 7 |
| MS 每 catalog 中位条数 | 2 | 22 |
| ME 每 catalog 中位条数 | 2 | 68 |
| MP item 中位 tokens | 34 | 5 |
| MS item 中位 tokens | 41 | 41 |
| ME item 中位 tokens | 37 | 102 |
| MP 预计 Top-k tokens 中位数 | 68 | 10 |
| MS 预计 Top-k tokens 中位数 | 84 | 80.9 |
| ME 预计 Top-k tokens 中位数 | 73 | 274.3 |

最关键的不是 catalog 总量本身，而是内部每类恰好两条。修复前 MP/MS 的 Top-k=2、
ME 的 Top-k=3，因此内部 treatment 几乎不需要从同 source 的长 catalog 中作选择；
EvoEmo 的 ME 却要从中位 68 条里选 3 条。这说明当前 clean pairs 主要验证的是
“generator 是否从已给定候选获益”，不是“retriever 能否从自然 catalog 找对候选”。

MS 的单条长度和预计注入量已基本对齐。MP 和 ME 仍存在明显 representation shift；
简单对 count 做 cap 只能避免数值爆炸，不能证明分布已经匹配。

## PM 应该改成什么

V1.5 正式 PM 不应只看到 `MP/MS/ME` source 名称、全 catalog count 或 centroid。
候选检索和安全过滤先本地运行，PM 再根据候选的低容量透明描述决定是否注入：

- component/source；
- 是否存在通过 filter 的候选；
- 实际候选条数和增量 tokens；
- age；
- Top-1 relevance 和 Top-1/Top-2 margin；
- candidate 与可见当前状态的相似度；
- 与当前上下文是否冗余；
- 可由可见输入确定的 hard-exclusion flags；stale/conflict/sensitive 不得用隐藏
  evaluator 标签冒充可部署特征；
- 另外三个组件的 background bits。

完整 catalog count/tokens 只用于诊断和 OOD，不作主要 PM 特征；否则内部 8 个 prior
sessions 与 EvoEmo 更长历史之间的规模差会直接变成 domain shortcut。PM 不读取
opaque ID，不用 ID 背答案，也不负责选择具体 item。Retriever 负责选择，
PM 负责判断“这批已经找到的候选是否值得进入 generator”。本地 candidate lookup 的
成本单独记录；`off` 表示不注入，因此论文主 cost 应是 generator input tokens、注入
条目和额外延迟，而不能再声称省掉了所有 index lookup。

BAAI/BGE 在这里有一个合理而有限的角色：提供 candidate-state similarity challenger。
它不是 gold，也不能单独回答边际效用；只有在相同 grouped labels 上确实改善验证结果
才保留。

## 最终选择：内部重编译成 EvoEmo 同结构，而不是混库

若只主张“给定旧的两条候选是否值得注入”，现有 192 个 memory contrasts 可以继续。
但用户要求解决内部训练到外部 memory 的强 transport 缺口，因此选择更严格的修复：

1. 每个现有 synthetic user 有 9 个 case；按 `session_index` 排成纵向历史；
2. 只把 bundle 已有的三条 `stable_preferences` 变成 atomic `basic_info`；
   两条 52/52 完全相同的 compiler boundaries 是全局系统规则，不冒充个人 MP；
3. 每个较早 case 的可见 dialogue 变成 prior session；
4. 已有 session summary 保留；为空时使用当时的 current user text 作透明 extractive
   summary，不调用外部模型；
5. 内部与 EvoEmo 都调用同一个 `build_evo_memory`：
   - basic info → MP；
   - prior session summary → MS；
   - prior seeker turns 经同一 chunker → ME；
6. 当前 state 只读取更早 session，随后经过同一 query、retriever、filter、Top-k、
   descriptor 和 injection。

零 API 可行性结果：

| 指标 | Evo-style synthetic recompile |
|---|---:|
| 可用纵向 states | 416 |
| MP/MS/ME catalog 中位数 | 3 / 4.5 / 5 |
| MP/MS/ME catalog range | 3–3 / 1–8 / 1–14 |
| train/calibration/internal states | 192 / 96 / 128 |
| 每个 source 有非空自然检索候选 | 416 / 416 / 416 |
| 当前自然检索 | lexical score；MP/MS/ME Top-k=2/2/3 |

这不足以让内部与 EvoEmo 完全同分布，也不应该为了“长得一模一样”而复制 test。但它
已经同时做到：

- 不使用一条 EvoEmo memory 文本；
- 使用相同 compiler；
- MP 有真实 Top-2 竞争，且没有把全局规则伪装成个人事实；
- MS/ME 存在真实 catalog 竞争；
- ME Top-3 不再等于全量注入；
- 外部剩余规模偏移可以由 candidate descriptors 和 OOD gate 诚实处理。

正式 backend 已构建于
`data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/`，全部 416 个状态通过：
严格过去时、三源可用、16 个合法动作、user-group split 零重叠、零外部内容/标签读取。
候选描述另存为 `candidate_descriptors.jsonl`，其中不含 memory text、ID 或 outcome。

因此当前 contrasts 的处理改为：

- 原 192 个 memory contrasts：只保留为 legacy small-catalog candidate-effect
  diagnostic；
- 原 64 个 RS contrasts：也只保留为历史诊断，因为其中 56 个使用 legacy memory
  background；
- 在 Evo-style synthetic backend 上重新生成 192 个 MP/MS/ME formal contrasts；
- 在同一个 repaired backend 上重新生成 64 个 RS formal contrasts；
- 最终盲评只合并这 256 个 repaired-stack 独立对，不把任何旧 pair 混入正式标签。

192 个替代 contrast 的 outcome-blind blueprint 已完成：
`outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/`。每个 component
64 对、每个 one-bit background 8 对，split 为 96/48/48；候选 relevance 按冻结分位
覆盖低到高，不读取旧 `needed_memory_sources`、回复或人评 outcome。384 个新
response arms 的冻结生成计划和可恢复执行器已经完成：
`outputs/pm_v1_5_transport_repaired_memory_generation_v1/` 与
`scripts/v1_5/24bs_run_transport_repaired_memory_generation_v1_5.py`。预检确认 384/384
调用 lineage 完整。初次同构审计只通过 9/13 层，计划因此被正确暂停；完成下述修复并
重建全部 hashes 后，复审达到 13/13，正式解除执行门。随后另建并完成 64 个同栈
RS pairs 的 128/128 response arms；两批合计 512/512 次生成，最终为四组件各 64、
共 256 对。

## 2026-07-31 EvoEmo 同构审计：同一个 builder 还不等于同一个任务

零 outcome、零 API 审计见
`outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/equivalence_audit.json`。可被哈希保证的
初审 13 层机制中通过 9 层；四个阻断项是：

- 内部 RS background 使用 V4 `prompt_guidance` 编译器，当前 EvoEmo runner 仍期待
  legacy `StrategyCard` 编译器；
- candidate descriptor 只在内部 `24bp` 内联实现，外部 runner 没有调用同一个函数；
- 修订后的 PM 定义为先发现候选、后决定注入，但当前外部 runner 仍先
  `model.choose`、后 retrieval；
- 内部 prompt 暴露 `session-N`，外部 prompt 暴露原始时间字符串。

分布上也不能称“已经足够像”：

- internal MP 是三条 support preference，EvoEmo MP 是七类人口/身份 profile，
  字段名交集为 0；
- 50% internal MS 使用当前 user text 作为空 summary 的 fallback，EvoEmo 的 401 个
  session 全有 summary；
- internal 每用户 9 个重建 session，EvoEmo 为 13–33；
- 在 train+calibration 的 primary candidate feature 绝对范围内，EvoEmo 完整行覆盖率
  仅 MP=0%、MS=27.9%、ME=18.1%。

这些是初审阻断事实，不再代表当前状态。

## 2026-07-31 同构修复复审

已经实现并由内外路径共同调用：

- 全部 causal prior sessions 的同一历史编译器；内部用户自然最多 8 个过去
  session，EvoEmo 用户自然为 13–33 个，不为对齐数字而截断外测；
- 同一 deterministic MS summary/fallback 编译器；
- 同一 lexical retriever、阈值和 MP/MS/ME Top-k；
- 同一 candidate discovery 与 descriptor 函数；
- candidate discovery 后、注入与生成前的 PM 决策接口；
- 同一 V4 strategy guidance prompt compiler；
- memory timestamp 统一呈现为相对 session age，不暴露数据集特有日期格式；
- PM 使用有界 cost、相对 age、relevance buckets，raw 数值只用于审计和真实
  成本核算；正式特征不包含完整 catalog count。

随后重新编译内部 backend。恢复全历史只改变外测可用历史规模和 descriptor 的
age 表示；内部每位用户仍然只有最多 8 个合法 prior sessions，因此已生成 prompt
不变。零 API 重算确认 memory 计划 384/384、同栈 RS 计划 128/128，共 512/512 个
prompt hash 精确匹配，不需要重新调用模型。零 outcome 复审结果：

| 门 | 复审结果 |
|---|---:|
| executable mechanism | 13/13 PASS |
| MP candidate model-feature support | 100% |
| MS candidate model-feature support | 86.8% |
| ME candidate model-feature support | 89.2% |
| 内外 exact memory text overlap | 0 |
| 已生成 response arms 可复用 | 512/512 |
| 外部 outcome/reference 被用于修复 | 否 |

因此 `MIMIC_AND_CANDIDATE_SUPPORT_GATE_PASS` 已成立，付费生成可以按冻结计划执行。
这里的 support 是“204 个 outcome-blind EvoEmo 状态中，有候选行的完整透明特征
向量落在内部训练/校准特征范围内”的比例，不是内容分布相同，更不是外部效果已经
通过。冻结门为每 source 至少 80%，三源均通过；外部正式运行仍必须逐状态
in-support 或 abstain/off。

MP 仍有必须披露的语义 transport 局限：内部主要是 support preferences，EvoEmo
主要是 demographic/profile facts。二者都属于稳定 profile memory，但并非同一 subtype；
V1.5 只能检验粗粒度 candidate-conditional transport，不能主张已覆盖任意画像类型。

### 全历史是正式运行时；不能用假大目录“抹平”自然规模差

原计划要求 EvoEmo 检验长期、多来源、大 catalog 下的记忆使用，因此把外部截成最近
8 个 session 虽能让数字好看，却会削弱正式考题。最终 shared compiler 对内部与
外部都读取该用户全部合法过去历史。最新 outcome-blind 审计中的正式 catalog 是：

| source | repaired internal 每用户中位 | EvoEmo 每用户中位 | 当前差异 |
|---|---:|---:|---:|
| MP | 3 | 7 | 2.3×，且语义 subtype 不同 |
| MS | 8 | 22 | 2.75× |
| ME | 9 | 68 | 约 7.6× |

`3 / 4.5 / 5` 是把内部所有 416 个 noninitial states（包括只有 1–2 个 prior sessions
的早期状态）放在一起后的 catalog 中位数；`3 / 8 / 9` 是每个内部用户走到共同
最大历史深度时的中位数。EvoEmo `7 / 22 / 68` 正是正式全历史运行时，不再用
recent-8 把它改成较容易的考试。

正式 PM 也不读取 raw catalog count。它读取 candidate discovery 后的有界特征：
retrieved fraction、token、session-relative age、relevance/margin buckets 和饱和后的
Top-k capacity。因此 8 个 session 中的“4 个 session 以前”与 32 个 session 中的
“16 个 session 以前”具有相同相对位置，模型不会直接把外部历史长度当数据集身份。
最新 204 个 outcome-blind EvoEmo 状态中，MP 有候选 114 个且联合支持 100%；MS/ME
各 204 个有候选，联合支持分别为 86.8%/89.2%。其余候选状态按冻结 OOD 规则关闭，
不允许 learned head 外推。

这不等于“外部 retrieval 已证明正确”。当前 item-level 排名明确是
`lexical_score`，不是 BGE/cosine；外部 ME 从约 68 条中选 Top-3，显著比内部约
9 条有更强竞争。V1.5 应在模型冻结后单独报告 retrieved item 的同用户/过去时完整性、
候选存在率、词法分数/margin，以及不用于调参的外部 relevance/consistency 诊断。
不得宣称 PM 学会了挑具体记忆；PM 学的是给定已发现候选后是否值得注入。

不采用把多个无关用户/事件拼成同一人的“大目录压力样本”作为正式训练数据。那会把
目录规模差异换成 wrong-person 或人格不一致污染。若未来需要检索器压力测试，只能
使用同一用户的因果一致扩展历史，并作为 retriever robustness 诊断，不能把它冒充
PM 的 component-effect gold。V1.5 当前不需要这项额外生成即可开始既定盲评与训练。

旧 308 项混合盲评页面永久停用。唯一 replacement packet 已生成于
`outputs/pm_v1_5_transport_repaired_four_component_blind_v1/`，只包含同一 repaired
backend 上的 64 RS + 192 memory pairs，避免把 legacy treatment 误作 formal gold。
256 个主 presentation 在每个 component 内严格 A/B 位置平衡，另有 52 个反转 A/B
的重复一致性题，共 308 项。

## 外测怎样才站得住

1. 用同一个 query、retriever、filter、Top-k、descriptor、prompt 和 generator；
2. 训练和 calibration 冻结 candidate-feature support 与 OOD 门；
3. EvoEmo 不得用 outcome 改模型、阈值、chunk 或 Top-k；
4. 外部 candidate 超出训练支持时，PM 必须 abstain/off，而不是高置信外推；
5. 报告每个 component 的 in-support coverage、OOD abstention、open rate；
6. 同时报告 always-off、always-on、透明规则和固定 ME+R0 等基线；
7. 最终端到端 quality/risk/cost 仍按全量未加权结果报告，不能只挑 in-support 好看的
   子集；
8. 因为 EvoEmo 已参与过旧版本诊断和 memory chunk 修订，只能称
   `development-informed external transport/stress test`，不能称 pristine test。

若外部大部分状态 OOD，结论不是“PM 失败所以永久不用 memory”，而是：

> 内部 candidate-effect 学习成立，但当前 synthetic memory compiler 没有给外部部署
> 提供足够支持；外部泛化尚未证明。

真正的 pristine 泛化需要另一个在协议冻结后才打开的纵向语料或前瞻数据。

## 对当前执行顺序的影响

1. 暂停现有 308 项混合盲评；
2. ~~冻结 synthetic→longitudinal adapter，并编译同构 MP/MS/ME backend；~~ 已完成；
3. ~~做 outcome-blind retrieval、candidate-support 和基础 shortcut audit；~~ 已完成，
   后续仍需在有训练 label 后做 label-shortcut probe；
4. ~~按已冻结 blueprint 与 call plan 生成 192 个 repaired memory pairs；~~ 已完成
   384/384 response arms；
5. ~~在同一 repaired backend 上重新生成 64 个 RS pairs，不复用处于 legacy memory
   background 的旧 RS；~~ 已完成 128/128 response arms；
6. ~~生成唯一 replacement blind packet；~~ 已完成 256 独立对 + 52 反转重复题；
7. 解盲、做 component-on win 的最小 risk 核验；
8. 用 `state × candidate × background` 训练四个 heads；
9. internal test 过门后，先运行 EvoEmo outcome-free transport/OOD preflight；
10. 不改方法地运行 development-informed EvoEmo 最终压力测试。

这将研究主张从含混的“PM 学会使用记忆”精确化为：

> PM 能否在固定候选发现与生成栈下，学习对已发现的个性化记忆候选作有意义的
> quality-risk-cost 注入门控，并在声明支持范围内迁移到新的用户历史。
