# PM V1.5 历史方法冻结：可审计的预注入资源路由

状态：`SUPERSEDED FOR FUTURE EXECUTION / 2026-08-02`

> **规范入口已迁移：** 后续唯一执行方案为
> `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`，机器合同为
> `data/pm_v1_5_contracts/final_execution_plan_v3.json`。本文保留 2026-08-02 当日方法演进、
> 物化 bug 与 Step 2 失败证据，不再单独授权训练、人评或外部测试。

本文优先于此前把 PM 写成“检索前 component-effect 预测器”或“直接预测单次回复
胜负”的段落。历史文件与结果保留作失败证据，不删除、不冒充当前方法。

### 2026-08-02 晚间执行物化纠正（优先于本文旧执行状态）

`M0+R0` 是完整16动作中的合法默认动作，也是 PM 学会“不调用资源”的必要输出。所谓
nonempty 硬门只适用于 Step1 已开启的组件：开启某组件时，其选定资源必须非空并真实进入
prompt；Step1 关闭全部组件时，active resource set 为空正是正确行为，绝不为通过检查而
伪造资源。

旧V2正式计划存在纯物化bug：RS排序wrapper没有解析成完整card，导致已请求RS的payload为空；
memory又混淆Top-k候选描述与实际注入。该run在498/1121时停止并仅作失败证据。纠正合同为：
RS按card_id解析完整三字段；MP/MS/ME的Step1仍可看Top-k candidate descriptor，但Step2只
注入原排序rank-1。candidate lineage与execution lineage、requested action与realized action
分别记录。

纠正后的44-call内部/开发门不读取p7–p18外部结果：单组件32/32按请求执行，多组件11/12，
schema错误0；唯一失败是guard正确拦截虚构过去效果并回退为`M0+R0`。机器门之后只做一次
固定16条的功能/误用语义核验，不重做quality盲评。EvoEmo的p1–p6已定为development，
p7–p12为untouched qualification，p13–p18为untouched lockbox；方法冻结后按此顺序消费。

### 2026-08-02 深夜语义门纠正（当前最高优先级）

固定16条人工语义门已经失败：结构化主评的全组件同时做功为`7/16`，MP=`2/8`、
MS=`6/8`、ME=`3/10`、RS=`10/10`，primary与delivered material misuse各1。另一份更严格、
但字段不完全兼容的独立审核也判定MP/ME及多组件路径未资格化。两份结果均保留，不平均、
不追加第三人投票。

该结果不能解释为“MS新晋级、ME被取代”。旧ME的`PASS_PROMOTED`只发生在
`STEP1_INTERNAL_CONSTRUCT`；本轮测的是`STEP2_FUNCTIONAL_EXECUTION`。进一步审计发现旧ME
确认集存在candidate-gold错绑：8个正例实际检索Top-3均为重复的普通同主题背景句，而非gold
设想的可复用事件。EvoEmo的ME又是任意过去seeker episode，并不保证存在动作、结果或可复用
机制。因此历史ME晋级现降级为无效的candidate-conditional资格证据；p7–p12外部资格测试继续
封存，直至exact rank-1候选标签与ME subtype合同重建完成。

以后禁止无层级写“某组件通过/失败”。所有状态必须写明：
`版本 × domain/split × STEP1_INTERNAL_CONSTRUCT / REPRESENTATION_TRANSPORT /
RETRIEVAL_CANDIDATE_FIT / STEP2_FUNCTIONAL_EXECUTION / END_TO_END_QRC_EFFECT`。

## 1. 主张有没有改变

没有改变研究目标：PM 仍然要在即时回复质量、明确的 interaction-and-grounding
risk 与资源成本之间找到一个可用平衡。改变的是可验证的操作定义：

> 一个低容量、可审计的预注入资源路由器，能在固定资源与生成栈下学习保守的资源
> 机会决策；在测得的支持范围内，相比固定高资源策略减少进入 generator 的上下文
> 成本，同时保持即时支持质量，并不增加明确的 interaction-and-grounding risk。

这不是缩小成“只做一个分类器”。它把原主张拆成四个能分别验证的环节：候选是否
存在且合格、PM 是否允许注入、generator 是否正确使用、最终系统是否达到
quality-risk-cost 约束。

## 2. 当时冻结的系统（历史）

执行顺序固定为：

1. 同一 compiler、catalog、query builder、retriever 发现 MP/MS/ME/RS 候选；
2. owner、严格过去、冲突、重复、明确边界、高风险和 token budget 硬门；
3. 四个独立的 L2 logistic opportunity heads；
4. 未晋级、OOD、无候选或低置信组件关闭；
5. 四个 bit 机械编译为既有 16 个合法动作之一；
6. 只将通过的资源注入 source-matched prompt；
7. 固定 Llama generator 写回复。

PM 的决策时点是 `post-candidate-discovery / pre-injection / pre-generation`。因此它能
减少注入条目和 generator input tokens，不能声称省掉已经发生的候选检索调用。

## 3. Step 1 与 Step 2 分工

Step 1 学的是 outcome-blind resource opportunity：当前真实候选是否合格、相关、非
重复、不过时、不过界且值得允许进入 prompt。它不读取候选 ID、数据集身份、未来 turn、
response、judge、quality/risk 结果或外部答案。

Step 2 不是第二个 PM 分类器，而是 source-matched execution：

- MP preference 改变表达方式或互动负担；MP profile 只提供当前有关的背景；
- MS 用于承接严格过去的会话主线；
- ME 用于当前相关的具体过去事件；
- RS 提供 topic-agnostic support move，不提供领域事实结论；
- 无关、冗余或冲突证据必须忽略。

最终 blind quality/risk/cost 实验验证整条 pipeline；它不反过来改 Step 1 标签。

## 4. 学习方法

每个组件单独使用标准化 L2 logistic regression：

\[
\mathcal L_h=-\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]+\lambda\lVert w_h\rVert_2^2
\]

固定 `C=0.3`、threshold `0.5`、五个 seed。按 user/dialogue grouped OOF，同时报告
leave-condition-family-out、Brier gain、balanced accuracy、recall、specificity、
on/off 比例和 seed agreement。不使用 focal loss、PPO、直接 16 分类或外测调参。

训练集被有意构造成 1:1，因此输出是 routing decision score，不解释为自然部署概率。

## 5. 2026-08-02 唯一表示修复结果

旧 raw term-frequency cosine 在内部短历史与 EvoEmo 长历史上不是同一尺度。按失败账本
只允许一次修复：去掉固定英语功能词和 compiler 包装词后，将 query/candidate 的不同
内容词交集映射为 `0 / 0.5 / 1`（零个、一个、至少两个）。检索排序仍使用同一 lexical
retriever；只替换 PM head 的尺度敏感输入。

- 旧 48 个 fit 用户原样重编译，没有重新生成回复或重做人评；
- 另造 16 个全新 V3 confirmation 用户，零 API、零外部内容、零 outcome；
- ME：所有内部门通过，正式晋级；
- MP：V3 BA `.75`，但 specificity `.50`，不晋级；
- MS：V3 BA `.75`、specificity `.50`，且条件族留出 BA `.604`，不晋级；
- outcome-blind EvoEmo head-eligible support coverage：MP/MS/ME 均 `1.00`；
- 未晋级 MP/MS 在运行时强制关闭，不再改阈值、词表或造 V4 confirmation；
- ME 在 EvoEmo 的 198 个非硬关闭状态全部预测开启。这只说明当前 opportunity 构念在
  外部目录中普遍成立，不证明每次注入会提高回复，必须由最终系统实验约束。

因此，这次修复解决了“表示不可运输”，但没有解决 MP/MS 的细粒度低相关区分。这里的
`fail-closed` 只定义“当前有资格作为保守运行时的组件”，不等于永久删除 MP/MS，也不等于
证明这两类资源无用。最终实验同时冻结两个 learned arms：

- `learned-PM-full`：四个已经训练好的 head 全部运行，不再调参；它直接检验原始四资源
  PM 主张，但在 MP/MS 未过确认门前属于 exploratory/full-claim arm，不能冒充已资格化部署；
- `learned-PM-conservative`：只允许 RS 与 ME，MP/MS fail-closed；它是目前证据支持边界内的
  保守敏感性/部署候选。

这两个条件必须同时保留。否则只跑保守版会通过“手动关掉失败组件”弱化原始主张；只跑
完整版又会把未过门的 MP/MS 写成已可靠。

### 5.1 V1.5b 根因修复（同日后续，优先于上段旧 head 状态）

旧 scale-stable MP/MS 的未晋级结论原样保留，不能改写。另建的 V1.5b 没有调旧阈值，
而是修复了标签—背景捷径、同特征异标签和 source 构念混淆：MP 显式区分 preference/
profile、scope、entity 与 incremental information；MS 显式区分 same issue、current goal、
prior distinction/outcome 与 resolved marker。新的 MP/MS head 在64个 fit 用户和32个 sealed
confirmation 用户上通过冻结内部门，并在 EvoEmo outcome-blind transport 中达到 MP `1.00`、
MS `.951` 的 head-eligible representation coverage。它证明新构念可学习和可运输，不证明
外部效用。

最终 `learned-PM-full` 使用这组 V1.5b MP/MS head，加已冻结 ME/RS head，直接检验四资源
主张；`learned-PM-conservative` 仍只允许 ME/RS，作为当前最保守的资格边界。两者同表，
因此既没有手动掩盖 MP/MS，也没有把内部 construct pass 冒充部署资格。

## 6. 已有结果如何使用

不重跑：Strategy Bank 建设与人评、RS clean pairs、D2/D3 回复生成与质量/风险人评、
旧 memory compiler/transport、BGE 资格赛、EvoEmo fixed tracks、32 条 generator execution。

继续有效但限界使用：

- 80-card V4 Bank 和同一检索栈继续作为固定 RS 资源；
- RS opportunity head 是 development-informed basic pass，不是外部确认；
- natural Top-1 在 H2 仅 `19/31` 合适，故 RS 必须保留 hard filter/abstain，不能写成
  已证明检索正确；
- BGE-small 没有在任务内 fit-only 资格赛带来预设增益，保持拒绝，不再更换 embedding；
- 自动 generator judges 的 exact agreement 仅 `20/64`，不得作为 gold；
- 原 10 条人工 execution check 的两位评审按预冻结门均未通过；共同结论是 Step 2
  存在陈旧证据升级、表面使用、错误忽略和不可裁决样本。两份结果不平均，也不反写
  Step 1 标签。一次且仅一次 source-matched prompt 修复已经执行，并生成 10 条全新开发态
  核验包；该包全部声明 use，只能验证正向执行与误用，不能单独证明 ignore 路径。
- generator 自报的 `concise_decision_reason` 仅为审计 telemetry，禁止作为 PM gold、自动标签
  或论文效果证据。
- 修复后新10条已由两位评审完成。可导入正式标注为：可裁决`10/10`、声明一致`9/10`、
  决定合理`10/10`、真正做功`7/10`、material misuse=`1/10`；独立敏感性结果为
  `10/10、8/10、10/10、5/10、0/10`。两者都只在功能执行门失败，因此结论不依赖
  misuse松紧：资源机会被认可，但generator只在`50%–70%`样本中稳定落实资源。
- 一次prompt修复额度已经消费，不做第三次prompt调整或更多execution人评。最终系统实验
  必须把functional treatment adherence作为过程指标并披露，不能把“正确决定use”写成
  “资源已经被有效利用”，也不能按事后看起来最好的ME三条单独追认组件资格。
- V1.5b 随后的32条四组件功能核验给出主评做功`26/32`、misuse=`7/32`，summary-only
  敏感性评审做功`24/32`、misuse=`6/32`。两套口径均未过预冻结misuse门，因此 Step 2
  generator 独立资格失败；该失败不回写Step 1标签。新发现的`fabricated_recall`（虚构
  “上次你发现某方法有效”）与陈旧事实升级、panic冲突和危险社交回避已经进入冻结的
  fail-closed guard。开发回放捕获7/7、误拦2条只能作为开发诊断，最终未见系统实验不得
  再调规则。
- 单资源execution核验不能直接覆盖16动作中的多资源组合。现已冻结 bundle execution：
  一次生成一个共享回复，但MP/MS/ME/RS分别返回有界application status；机器门看到全部
  授权证据，避免跨资源误判。任一已声明applied的组件触发明确grounding guard时，共享回复
  整体作废并只允许一次resource-free fallback；仅有界cannot_apply时只将对应bit实现为OFF。
  requested action、realized action、fallback calls和全部tokens分开记录。

## 7. 最终比较与消融

### 7.1 主系统表

以下六种策略共享完全相同的测试状态、候选发现、当前资源库、检索器、source-matched
prompt 与 generator。区别只允许发生在路由策略：

1. `always-off / Context Only`：M0+R0，回答“不用额外资源能做到什么”；
2. `fixed-high-resource`：所有通过硬门且有候选的组件均注入，给出资源与成本上界；
3. `transparent-rule`：固定人工规则，不学习，回答 learned PM 是否胜过透明启发式；
4. `learned-PM-full`：冻结的 MP/MS/ME/RS 四 head 全运行，直接检验原始完整主张；
5. `learned-PM-conservative`：RS basic + ME promoted，MP/MS fail-closed，给出当前资格边界内
   的保守系统结果；
6. `cost-matched-fixed`：在内部 calibration 仅按 token 成本预先选定并冻结的固定动作，
   回答 learned PM 的收益是否只是“用了不同资源量”。禁止看 EvoEmo/ESConv 的 quality 或
   risk 后再从16动作中挑“best fixed”。

成本匹配已在32个独立内部D3状态上完成：只比较同一bundle prompt的保守input-token上界，
16个固定动作全部参加，未读取response、quality、risk、judge或外测。冻结结果为`MP+RS`，
平均token上界与`learned-PM-full`相差`4.4%`。旧V1中很强的`ME+R0`仍可作为额外固定护栏，
但不再冒充本轮cost-matched comparator。

### 7.2 从 V1.0 继承的次要基线

V1.0 的合理基线定义应保留，但旧分数不能直接和 V1.5 拼表：

- `Raw Session Top-4 + Strategy`：检验结构化记忆是否优于原始会话 RAG；
- `All Raw Sessions + Strategy`：检验压缩与选择是否值得其复杂度；
- `Cost-Matched Event Memory`：并入主表的同预算固定动作；
- `Legacy V1.0 Learned PM`：若旧模型与旧资源栈可完整复现，作为“历史完整系统”次表；
  因 generator、prompt、Bank、memory compiler 均不同，它不能承担当前同栈因果归因。只有在
  预先定义并验证旧16动作到当前16动作的精确投影后，才可另报 current-stack replay。

### 7.3 定位性消融

- 每组件单 bit contrast（同状态只改变 MP/MS/ME/RS）；
- full logistic、去 candidate match、去 grounding/nonredundancy、transparent rule、已拒绝
  BGE challenger；
- generic/raw resource prompt 对 source-matched prompt 的 Step 2 使用消融；
- 16 个 action mapping 做机械审计，不为每个 action 穷举一轮大模型生成。

回复盲评只回答端到端采用偏好，不能判定是谁的锅。每个样本必须另存
`state -> retrieved candidates -> hard gates -> head score/decision -> requested/effective action ->
injected item -> functional use/misuse -> quality/risk/cost` trace，分别计算 PM、retrieval、
generator 和 end-to-end 四层指标。

## 8. 指标与通过条件

不合成总分。顺序是：

1. Step 1：至少一个组件的 grouped OOF Brier gain 为正、能同时 on/off；
2. quality：learned PM 对 always-off 与 fixed-high 的 blind NetWin 非劣界；
3. risk：explicit boundary、unsupported personal claim、stale/conflicting use、
   excessive directiveness 分项不明显恶化；
4. cost：对 fixed-high 的平均 generator input tokens 至少降低 10%；
5. retrieval calls 只作候选发现遥测，预期 learned 与 fixed-high 相同，不作为 PM 节省。

如果 quality 或 risk 不过，成本下降不能挽救结论。若只有 ME/RS 获支持，论文必须写
“partial component support”，不得声称学会所有 16 动作。

## 9. 从现在到最终实验的最短路径

1. 冻结修复后source-matched prompt为“功能执行未资格化的当前实现”，停止prompt修复和
   execution人评；不改PM数据和标签，ignore路径明确列为未独立资格化；
2. ~~将两个learned arms接入同一个16-action compiler~~：已完成。16/16动作可达，full
   保留四bit，conservative只mask MP/MS，所有policy均不能绕过候选/硬门；
3. ~~冻结cost-matched fixed~~：已完成，内部token-only结果为`MP+RS`；
4. ~~冻结外测panel~~：V1先冻结ESConv 169个正式test dialogue各一个outcome-blind
   hash状态，EvoEmo为102条固定open-loop track × turns 3/8=`204`状态。首次执行的前62个
   call暴露纯格式兼容问题：Llama会额外或重复报告未请求component status；这47个已触及
   ESConv dialogue全部降级为format pilot且永不进正式结果。V2正式panel保留其余122个
   未触及ESConv dialogue和全部204个未触及EvoEmo状态；前者按dialogue、后者按18个user
   聚类。唯一格式修复是：一致的requested status可用；extra status只记录不参与routing；
   相同duplicate折叠；冲突duplicate仍fail-closed。PM、candidate、prompt、semantic guard
   和回复均未因pilot改变；
5. ~~生成六策略outcome-blind routing trace与调用计划~~：V2共326 states、1,956 policy
   assignments；相同state/action去重后为1,121次primary calls，节省835次重复生成；最坏
   795次resource-free fallback，保守费用上界约`$0.80`。正式执行已启动，运行结果不得再
   修改route、resource、prompt或guard；
6. 冻结方法后一次性运行 ESConv/EvoEmo 的六个主表条件；V1.0 raw-session/all-history
   条件作为次表，可按预冻结子样本控制成本；
7. 外测只出结果，禁止再修改 PM；失败按 component、retrieval、execution、quality、risk、
   cost 分层解释。

当前不再需要新的 RAG 大人评、support-need 标注、BAAI 替换或回复质量训练标签。

### 9.1 物化纠正后的实际剩余路径

上面第9节所称“V2正式执行已启动”已经被晚间物化审计取代，不能再续跑旧V2。当前唯一
路径为：

1. 完成已固定的16条Step2功能/误用核验；它不评PM开关、不评检索、不做A/B质量；
2. 若功能执行与material misuse门通过，冻结V4 messages、完整RS与memory rank-1表面；
3. 用新表面重新计算内部token-only cost-matched fixed，禁止沿用旧Top-k注入成本结论；
4. 先运行p7–p12 qualification，只按预冻结系统门决定是否允许消费p13–p18 lockbox；
5. ESConv用新protocol重跑纠正后的122个dialogue并明确披露其因纯物化bug重跑；
6. 最终只比较同一纠正栈下六个主条件，旧V2任何回复均不得并入正式表。

若16条语义门失败，停止外部正式执行并只修Step2实现；不得改PM标签、路由head、Bank、
外部panel或测试门槛。若通过，之后外部结果只能用于报告和分层归因，不能再回流修改系统。
