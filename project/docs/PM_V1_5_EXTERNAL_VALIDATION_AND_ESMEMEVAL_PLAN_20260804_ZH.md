# PM V1.5 外部验证与 ES-MemEval 扩展执行计划

状态：`ACTIVE / OUTCOME-BLIND PREFLIGHT ONLY / 2026-08-04`

上位方案：`docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`  
内部冻结结论：`outputs/pm_v1_5_v5_2_confirmation_analysis_v1/confirmation_analysis.json`  
适用环境：`/home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5`

## 0. 决策摘要

下一阶段不再训练、不再修 V5.2，也不再建立第二个内部确认集。固定执行三条互补证据线：

1. `ESConv response replication`：主要检验 RS 开关、即时支持质量、边界、risk 与成本；
2. `EvoEmo response replication`：检验同一用户私有历史下 MP_PROFILE、MS、ME 的候选运输、路由和使用；
3. `ES-MemEval QA diagnostic`：用官方问题、答案和 evidence 检验记忆检索、时间推理、冲突、拒答和用户建模。

前两条回答“完整回复系统的质量—风险—成本”；第三条回答“记忆信息是否找对、读对、答对”。
三者不能互相替代，也不能用任何外部结果救活已经冻结为失败的 V5.2 内部完整系统门。

V5.2 当前最准确的研究结论保持不变：有限 PM 已显示有意义的保守路由信号，并相对透明规则改善；
相对 fixed-high 的风险和成本点估计更好，但 fixed-high 质量非劣的置信区间门和 critical grounding 门
未通过。因此外部结果只能扩大或缩小证据边界，不能改写为“内部确认通过”。

## 1. 外部考卷分别能考什么

| 证据域 | 能直接测 | 不能据此声称 |
|---|---|---|
| 内部内容独立确认 | 四组件、完整16动作、偏好反事实、wrong owner、stale/conflict、联合冲突 | 自然外部分布已泛化 |
| ESConv | RS 候选与开关、R0/RS 即时质量、边界与负担、memory absent 时正确关闭 | MP/MS/ME 长期个性化能力 |
| EvoEmo response | 每用户私有 MP_PROFILE、MS、ME 候选，纵向连续性、回复 grounding、cost/coverage | MP_PREFERENCE、所有16动作、真实用户临床效果 |
| ES-MemEval QA | 信息提取、时间推理、冲突检测、abstention、用户建模；检索与答案正确性 | 情绪支持回复质量或 PM 的 QRC 主张 |
| 内部受控补测 | 外部数据缺失的偏好、owner、边界、联合动作与 hard negative | 自然分布泛化 |

“学习—考卷同构”在本阶段的含义是：内部与外部共享 ontology、每用户隔离、memory compiler、
query builder、retriever、candidate schema、PM、Step 2、generator 和评测定义；只替换用户自己的
资源内容。ESConv 没有纵向记忆时必须自然 mask 掉 MP/MS/ME，不能伪造外部记忆。

## 2. ES-MemEval 官方版本与本地数据身份

官方论文为 [ES-MemEval](https://arxiv.org/abs/2602.01885)，代码为
[slptongji/ES-MemEval](https://github.com/slptongji/ES-MemEval)，归档为
[Zenodo 10.5281/zenodo.18338564](https://doi.org/10.5281/zenodo.18338564)。论文报告 QA
`1,209` 题：IE `271`、TR `236`、CD `226`、UM `251`、Abstention `225`。

当前公开 v1.0.0 artifact 与仓库 `main` 实际包含 `1,427` 题：

| capability | 全部18用户 | p1–p6 | p7–p12 | p13–p18 |
|---|---:|---:|---:|---:|
| Information extraction | 309 | 114 | 105 | 90 |
| Temporal reasoning | 284 | 104 | 98 | 82 |
| Conflict detection | 267 | 92 | 95 | 80 |
| User modeling | 306 | 112 | 99 | 95 |
| Abstention | 261 | 97 | 93 | 71 |
| 合计 | **1,427** | **519** | **490** | **418** |

本地 `data/external/evo_emo.json` 的 SHA256 为
`f30698e87fddaeff51270a666c654da604f487a3456ec60d2b6ae08a6fecd420`，与官方仓库
commit/tag `692624208acc077b8867698c1d6fcd998dee641a`（`v1.0.0`）当前原始文件一致。

因此论文必须写“在公开 ES-MemEval v1.0.0 artifact 的 1,427 题上评测”，不能写成精确复现论文
Table 2/3 的 1,209 题，除非以后获得作者用于质量筛选的 1,209 题 manifest。

### 2.1 暴露边界

- EvoEmo p1–p6 已是开发数据；
- p7–p18 的 response states 已在 V5.1 T5 生成中暴露，不能再称未触碰外部 response lockbox；
- p13–p18 的 `questions/answer/evidence` 是否被任何既有脚本、LLM、人工页面或分析读取，必须先做
  静态与日志审计；
- 审计若通过，p13–p18 的418题只可称 `task-disjoint QA evaluation`，不能称 user/content
  pristine lockbox；
- 审计若失败，全部1,427题均称 development-informed official benchmark，仍可报告，但不作确认性
  holdout 主张。

任何 QA 推理时不得把 gold answer、capability 标签或 evidence ID 输入 PM、检索器或 generator。

## 3. 正式比较条件

### 3.1 所有 response 域的核心主表

1. `always-off / M0+R0`；
2. `component-fixed-high`：所有有候选且通过硬门的组件全开；
3. `transparent-rule`；
4. `learned-PM-full`：冻结 V5.2 四 heads、`.5` 阈值和16动作 compiler；
5. `cost-matched-fixed`：只按当前 V5.2 input-token 表面一次性选定的固定动作。

第五项必须进入主表，否则 learned PM 相对 fixed-high 的成本优势可能只是“少用了资源”。但旧
`outputs/pm_v1_5b_internal_cost_matched_fixed_v1/cost_match_report.json` 的 `MP+RS` 来自旧 D3、旧
heads 和旧 prompt 上界，只能作历史证据。外部付费前必须在当前 V5.2 冻结表面上：

1. 穷举16个固定动作；
2. 只计算 input-token 上界；
3. 不读取 response、quality、risk、judge 或外部 outcome；
4. 按域冻结一次最接近 learned-PM 平均成本的动作；
5. 若与 always-off 或 fixed-high 完全同臂，则只记 alias，不重复生成。

### 3.2 次表与定位性 baseline

- `Raw Session Top-4 + frozen Strategy Bank`：仅 EvoEmo，检验结构化记忆是否优于原始 session RAG；
- `All Raw Sessions + frozen Strategy Bank`：仅 EvoEmo，检验压缩与选择是否值得；
- `learned-PM-conservative`：只有此前冻结定义能无歧义映射到 V5.2 时才作预声明敏感性分析；禁止按
  V5.2 结果重新挑“看起来好的组件”；
- `Legacy V1.0`：只进历史次表。只有旧策略能精确投影到当前资源与执行栈时才可另做 current-stack
  replay；不同 generator/prompt/Bank/compiler 的旧分数不能与 V5.2 作因果差值；
- `full-minus-MP/MS/ME/RS`：只在已有内部同状态回复可机械回放时作组件消融，不因外部结果新生成
  一轮调参数据。

### 3.3 不同数据域的适用矩阵

| 条件 | 内部 | ESConv | EvoEmo response | ES-MemEval QA |
|---|---|---|---|---|
| always-off / no-memory | 主 | 主 | 主 | 主 |
| fixed-high | 主 | 主（实质为 eligible RS） | 主 | typed-memory 主 |
| transparent-rule | 主 | 主 | 主 | 次要 |
| learned-PM-full | 主 | 主 | 主 | transfer stress test |
| cost-matched-fixed | 补充当前表面 | 若与既有臂别名则不重复 | 主 | 不适用 |
| Raw Session Top-4 + Strategy | 不适用 | 不适用 | 次表 | official session-RAG Top-4 主 |
| All Raw Sessions + Strategy | 不适用 | 不适用 | 次表 | full-history 主 |
| Legacy V1.0 | 历史 | 历史 | 历史 | 不适用 |

## 4. 评测合同

### 4.1 Response quality—risk—cost

全量自动统计：candidate/eligibility coverage、OOD/abstain、requested→feasible→realized action、
fallback、按组件开关率、prompt/completion tokens、API cost、latency、每用户隔离与检索命中。

最终人工只做一次预冻结 panel：

- quality：匿名 A/B，只评价可见对话和回复；
- grounding risk：显示当前对话、该臂真正获准的证据和回复，独立于质量；
- exact public surface 只评一个 representative，再传播到底层统计行；
- 预冻结20%由另一评审复核，所有 uncertain/risk 分歧集中裁决一次；
- 统计按 ESConv dialogue、EvoEmo user 聚类，不把同用户多个 state 当独立人；
- source attribution 会让 memory 臂可被猜到，因此只声称 A/B 位置与 policy identity 盲，不声称
  treatment 完全不可识别。

独立 LLM judge 在全量回复上运行，但只作稳健性与可复现性分析；不得替代人工 primary，也不得
产生 PM 训练标签。质量、risk、cost 不压成一个不透明加权总分，继续使用约束式 Pareto 报告。

### 4.2 ES-MemEval QA

主指标沿用官方：token F1、BERTScore；LLM-as-Judge 0–2 只作第三指标。按五种 capability 分层，
并单列 abstention、conflict detection。若官方 evidence 能与当前 retrieval unit 确定映射，再报告
Recall@k 与 nDCG@k；映射失败时明确写“不具备可比 retrieval metric”，不得用主题相似度伪造 gold。

QA 比较条件：

1. no-memory；
2. full history；
3. 官方 session-level RAG Top-4；
4. 当前 typed-memory fixed-high；
5. 当前 learned PM。

learned PM 在 QA 上只是跨任务压力测试：QA 通常显式要求历史，可能超出它在支持回复上学到的开关
分布。它输给 fixed-high 不等于 response PM 失败；它若节省成本且保持 QA，也只能支持“有限跨任务
运输”，不能替代 response QRC。

官方 summarization 不属于 V1.5 的输出合同，本轮不做。官方34个 dialogue-generation scenarios 可在
主实验结束后作附录稳健性实验；论文自己报告 RAG DG 的人机一致性较低，因此不能替代本项目已有的
匿名 pairwise quality/risk 评测。

## 5. 逐步执行顺序

| 阶段 | 操作 | 结束条件 | 是否调用 API/人评 |
|---|---|---|---|
| E0 | 冻结本文、总方案和失败账本 | 文档数字、版本、责任边界一致 | 否 |
| E1 | 外部 provenance / exposure / baseline 静态审计 | 数据hash、QA题数、暴露账本、等价臂、当前cost-match全部冻结 | 否 |
| E2 | 物化 ESConv/EvoEmo response 与 QA plan | 同栈、每用户隔离、无gold泄漏、调用量和预算明确 | 否 |
| E3 | retrieval-only 审计 | exact candidate lineage、coverage、R@k/nDCG可比性结论完成 | 否 |
| E4 | 单次 response 与 QA 生成 | requested=realized、fallback/invalid ITT在预设允许范围 | 是，一次 |
| E5 | 全量自动 cost/coverage/routing 与 QA objective metrics | 全部统计按冻结分母输出 | 否或judge API |
| E6 | 独立 LLM judge 稳健性 | 与人工构念分开保存 | 是 |
| E7 | 唯一最终分层人评 | primary、20% overlap、一次裁决完整 | 是，一次 |
| E8 | 聚合与论文表 | 域别、baseline、CI、局限和失败均可追溯 | 否 |

### E1 必须先回答的七个问题

1. V5.2 heads、threshold、Bank、retriever、compiler、executor、generator 和 rubric 的 hash 是否一致；
2. ESConv 122 dialogues 与 EvoEmo 204 response states 已暴露到什么层，后续准确称为什么；
3. p13–p18 的 QA answer/evidence 是否曾被读取；
4. 本地 1,427 题身份是否仍与官方 v1.0.0 一致；
5. 当前 V5.2 cost-matched fixed 是哪个动作，各域是否与其他臂别名；
6. Raw Top-4 / All Raw 的输入是否只含同一用户严格过去会话；
7. 各臂实际调用数、tokens 和预算是否在付费前冻结。

E1 不通过时停止在静态层，不调用 API。E1 通过后也不立即做人评，先完成 E2/E3，确保人评面对的是
最终冻结系统而不是又一个开发小包。

## 6. 停止规则与允许的修复

结果前允许修：错 ID、空资源、不同用户混库、gold/evidence 泄漏、动作 alias 未去重、requested 与
realized 不一致、hash 漂移。这些是 treatment 身份 bug，修后必须新 protocol 并保留旧失败记录。

结果后不允许修：按质量/risk挑动作、阈值、卡片、prompt、seed、子集、judge 或 cost baseline。
若结果暴露真实方法缺陷，只能写入局限，或升为 V1.6/新方法并使用新考卷；不能覆盖 V5.2。

外部可支持的最高结论按层级写：

- ESConv：冻结 PM 的 RS 子域在外部对话上的 repaired replication；
- EvoEmo response：冻结 PM 在合成纵向用户历史上的 repaired replication；
- ES-MemEval QA：正式公开 artifact 上的记忆检索/读取诊断；
- 内部：有限四组件 PM 的内容独立 QRC 确认失败，但存在有意义 Pareto 学习信号。

这四句话共同构成论文的诚实主张，不再追求用某一张外部表把所有能力一次性“证明通过”。

## 7. 当前进度与最近下一步

- `E0 COMPLETE`：总方案、外部执行计划和失败账本已经同步；
- `E1 COMPLETE / PASS_READY_FOR_E2_PLAN_MATERIALIZATION`：零API审计见
  `outputs/pm_v1_5_v5_2_external_e1_static_audit_v1/`。V5.2六个封存实现hash和旧外部
  response-free surface十个输入hash全部未漂移；260个外部state均未读取response/outcome，EvoEmo
  所有memory owner均与当前用户一致；
- QA暴露结论：既有loader技术上解析过完整JSON，但response路径没有访问
  `questions/question/answer/evidence/capability/summaries`，外部response panel不含这些字段，既有
  T5 prompt/outcome也没有逐字QA question命中。因此p13–p18的418题可称
  `task-disjoint QA evaluation with declared technical parse exposure`，不能称pristine lockbox；
- 当前V5.2 outcome-blind cost-match为：ESConv=`M0+R0`，与always-off完全别名；EvoEmo由p7–p12
  资格分区选为`M0+RS`，p13–p18只继承。ESConv不为别名臂重复生成；
- V5.2原子ME资格在自然外部记忆上只有p7–p12 `8/60`、p13–p18 `0/78`可执行；这是冻结方法的
  subtype/representation覆盖局限，不得用外部考卷修改compiler。外部ME不能作组件级成功主张，
  但候选缺席/abstain必须进入coverage结果；
- response核心五policy经state-action去重后为每seed 788次、两seed 1,576次；input-token上界
  1,119,634，output cap上界472,800，按既有价格代理上界约`$0.452`。QA主实验为418题×5条件
  =2,090次；精确QA token预算须在E2物化prompt/retrieval后冻结；
- `E2 COMPLETE / PASS_READY_FOR_E3_RETRIEVAL`：零API逻辑计划见
  `outputs/pm_v1_5_v5_2_external_e2_plan_v1/`。核心五policy按state-action-policy alias去重后仍为
  1,576次；EvoEmo Raw Session Top-4 / All Raw两个次表条件共276个retrieval request、双seed后
  552次；response合计计划2,128次。QA固定418题×5条件=2,090次；生成call与418条
  evaluator-only answer/evidence/capability映射物理分离；
- E2只对418条no-memory QA生成了exact messages。其余1,672条依赖retrieval或context fit，明确保留
  `PENDING_E3`，不得用占位prompt冒充exact token预算；
- `E3 COMPLETE / PASS_READY_FOR_E4_PAID_GENERATION`：本地检索与prompt seal见
  `outputs/pm_v1_5_v5_2_external_e3_retrieval_v1/`。BAAI/bge-m3固定snapshot
  `5617a9f61b028005a4858fdac845db406aefb181`、session为检索单元、Top-4；全部552条raw response和
  2,090条QA已具有exact messages/hash并在20,000 provider-neutral conservative bound内；API=0；
- 官方session evidence在341/418题可完整确定映射，evaluator-only Recall@4=`.6963`、nDCG@4=`.5997`；
  77题因abstention空evidence或官方ID/拼写不可完整映射，不伪造retrieval metric；
- QA typed fixed-high只实现MP/MS，learned PM只在392题选择MS、26题全关；ME受冻结原子表示覆盖限制，
  RS对事实QA结构性N/A。因此QA只检验当前真实可执行子集，不称四组件外部成功；
- 这里不是“ES-MemEval没有MP/ME能力”。ES-MemEval按information extraction、temporal reasoning、
  conflict detection、abstention、user modeling五种历史推理能力组织题目；PM按资源来源/功能及是否值得
  注入组织MP/MS/ME/RS。二者是交叉轴：本研究可主张资源类型、注入、grounding-risk与cost控制更细，
  但不能主张整体记忆推理严格包含或优于ES-MemEval。MP在418题中205题结构可执行但冻结response-utility
  head均未开启；MS结构可执行392题且全部开启；ME的340题typed absent、58题原子编译失败、20题候选缺失，
  属QA schema/目标与V5.2表示覆盖共同不匹配；RS为事实QA结构性N/A。详细审计见
  `outputs/pm_v1_5_es_memeval_fit_audit_v1/report.html`；
- 77题not-comparable已细分：69题abstention且official evidence为空，另1题虽标conflict detection但
  answer=`Unknown`且evidence为空；这70题没有检索正例，应只进入QA答案/拒答指标。其余7题含公开artifact
  标识问题：5题使用无公开字段对应的`timeline:n`，2题使用`p16_evnet_17`拼写。主检索结果保持n=341；
  仅作结果盲敏感性时将`timeline:n -> p16_event_n -> conv_id`及`evnet -> event`确定性修复，可得n=348、
  Recall@4=`.6930`、nDCG@4=`.5941`，与主结果`.6963/.5997`结论一致。不得静默用推定修复替代公开gold；
- `E4 STATIC FREEZE COMPLETE / AWAITING EXPLICIT PAID RELEASE`：统一付费runner为
  `scripts/v1_5/28e_run_v5_2_external_e4_generation_v1_5.py`，执行封条见
  `outputs/pm_v1_5_v5_2_external_e4_plan_v1/`。它从第一版即调用active
  `require_paid_run_release`，绑定E2/E3/V5.2、generator、seed、exact prompt、逐call token上界和实现hash；
  4,218个逻辑call均有独立physical key与最多3次仅限传输错误的持久attempt ledger预算；合法`M0+R0`
  保留为真实动作，任何资源处理失败均不得伪装成`M0+R0`回退；
- E4一次成功的费用代理上界为`$3.373812`，全部call都耗尽3次传输尝试的绝对硬上界为
  `$10.121435`；当前API=0，fresh identity为
  `2ae8f4f047f6998c8f7bd58ffeaae1e5e1cd9e2fdffff095ca56fdc109fd1579`。只有中央approval manifest
  明确绑定该stage/identity后才能运行；旧23m及其他封存runner不在此依赖链，不原地修改。
- `E5 METRICS FROZEN BEFORE E4 OUTCOMES`：自动评分合同与实现分别为
  `data/pm_v1_5_contracts/v5_2_external_e5_scoring_v1.json`和
  `scripts/v1_5/28f_score_v5_2_external_e5_v1_5.py`，封条见
  `outputs/pm_v1_5_v5_2_external_e5_scoring_v1/metric_freeze.json`。QA逐字复现公开
  ES-MemEval v1.0.0的prediction surface、set-overlap token F1和`bert-score==0.3.13`/
  `bert-base-uncased` BERTScore；模型在第一条评分前绑定唯一local snapshot并记录完整文件树hash。
  E5不把词法guard、资源出现或资源做功当response质量/risk，response自动项只覆盖cost、coverage、
  routing；LLM judge留在E6。冻结时E4 outcome读取=`false`、API=`0`。

### 6.1 E4 首次真实执行结果（2026-08-04）

用户明确批准stage=`v5_2_external_e4_generation_v1`、identity=
`2ae8f4f047f6998c8f7bd58ffeaae1e5e1cd9e2fdffff095ca56fdc109fd1579`、绝对费用上限
`$10.1214351`后，正式`.venv-pm-v1-5` runner开始执行。第一条真实调用即为
`official_session_rag_top4` QA；NVIDIA `meta/llama-3.1-8b-instruct`返回了完整provider响应，但输出从
系统task description开始复述prompt/检索内容，正好耗尽冻结`max_tokens=300`，provider
`finish_reason=length`。冻结合同将其分类为`stage_postcondition_failure / terminal_nonretryable`，不属于
HTTP/429/timeout等传输错误，因此不能使用剩余两个transport attempt盲重试；runner正确终止，4217条未调用。

该identity已按实际消费关闭：成功logical call=0、失败=1、started physical attempt=1、记录token为
input=2725/output=300，冻结代理费用=`$0.00058875`。这不是PM、检索Recall、QA答案或E5结果，而是此前
静态seal没有发现的“官方式长记忆QA prompt × 冻结Llama endpoint”真实生成兼容性失败。产物见
`outputs/pm_v1_5_v5_2_external_e4_execution_v1/ABORTED_E4.json`。不得复用该identity、提高cap后原地续跑、
跳过首条或只执行response子集；任何修复必须升新的QA-generation compatibility版本，先做有界真实endpoint
pilot，再产生全新plan/cost identity并重新取得明确批准。

这里的“新版本”严格只指 **ES-MemEval QA 调用适配层**，不指 PM、资源或研究方法重做。以下对象必须
逐哈希沿用既有冻结版本，不得重训、重标、调阈值或按外部结果修改：四个 Step1 head、特征定义、
16 动作映射、RAG Bank、MP/MS/ME compiler、检索候选与排序、每题 learned/fixed-high 开关结果、
response 生成栈、E5 指标与数据 split。允许修复的范围仅为把已冻结的同一 QA question 和同一检索
上下文，以该 NVIDIA endpoint 能正常遵循的 role/message 包装提交，并对输出做同一答案解析；不得改变
题目、证据、候选、条件或 gold 不可见性。兼容性 pilot 只是少量真实请求的 I/O 冒烟测试，不产生训练
标签、不读取答案好坏、不选择更有利的 prompt，也不重新训练 PM。通过后重新物化 4,218-call plan 的
原因仅是旧 identity 已被一次真实调用消耗且 provider-visible QA message hash 发生变化；此前所有训练、
人评、检索审计和冻结 PM 结果全部继承，不从头开始。

### 6.2 QA endpoint adapter V2：零费用准备状态

已将修复严格限制为独立的 QA I/O adapter：移除会诱发 Llama 复述的完整示例与
`Task Description / Input Format`元模板，保留同一 question、同一已冻结检索 payload、同一五种 QA
条件、同一模型、temperature、seed 和 300-token cap。pilot 固定使用首次失败问题在五种条件下各一次，
只检查`finish_reason=complete`、单行`Answer:`输出、非空与无 prompt echo；**不读取 gold、不判断答案
正确性、不产生 PM 标签**。

零费用 dry-run 已独立复现两次：5 个 logical call，call-plan hash=
`2d546de619ef2721bbf07e5167c234a45dd65b00b00221ca5f5637de0747f6a1`，identity=
`1984d476258e7d522fb087010be2c2f17342f06f2fe4a4199bd0732c8114ce91`。一次尝试各自的费用代理上界
为`$0.0039519`；包含每条最多三次纯传输尝试的绝对上界为`$0.0118557`。用户随后以精确
identity/上限授权真实 pilot；结果为`CONSUMED_PASS`：五种条件5/5均在第一次物理请求完成，0次重试，
均输出单行`Answer:`且无 prompt echo。最长的`full_history`请求实际使用11,975 input tokens，说明通过
不只来自短上下文；总实际使用14,996 input / 45 output tokens，费用代理`$0.0022764`。该pilot没有读取
或评分答案正确性，不能作为 QA 性能结果。

### 6.3 E4 V2 冻结状态

pilot通过后已生成 E4 V2 零费用冻结计划。response-core 1,576条和response-raw 552条继续逐字节引用
V1的相同源计划；QA仍为418题×5条件=2,090条，question、condition、session/typed候选、检索分数、
learned开关、seed与output cap均逐项相同，唯一 provider-visible 变化是已经资格验证的QA message
adapter。总计仍为4,218 logical calls；PM、路由、资源、检索与E5指标变化均为false。

重复冻结调用得到相同 physical-plan hash=
`d6190bf1aeec0062bbcd058cd33c64d10d5f084a522cec3eebbb5da6260343e7`和identity=
`0db16b65af2e8f37804a7a1676f20cb727d98b81d40f5ef420fdb20e59edbb30`。一次尝试各自的费用代理上界
为`$3.05562315`，包含每条最多三次纯传输尝试的绝对上界为`$9.16686945`。当前完整E4 V2尚未获得
绑定该identity/上限的明确授权，真实调用为0。

用户随后精确授权E4 V2。真实运行先成功38条，随后同一QA调用连续三次收到HTTP 429；10秒、30秒
退避仍处于provider分钟级限流窗，runner按每调用三次上限终止。38条成功使用103,655 input / 609
output tokens，费用代理`$0.01591365`；内容postcondition与QA prompt echo失败均为0。因此该前缀不形成
外部研究结果，但其exact outcome可在相同message/model/seed/参数下确定性carry-forward。

已零费用冻结paced continuation：carry 38条，只发送剩余4,180条（QA 2,052、response-core 1,576、
response-raw 552）；最少2秒/调用，429退避60/60秒，每调用物理上限仍为3。provider-visible输入与
科学方法均不变。重复冻结identity=
`dbbf2f8faa8487cc46fc880b7a784aaf069e847b750c668393c0c40aaf0769d7`，一次尝试各自费用代理上界
`$3.0274368`，绝对上界`$9.0823104`。该continuation尚未获得绑定新identity/上限的明确授权，API=0。
