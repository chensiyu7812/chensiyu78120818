# PM V1.5 V5.3 完整训练数据生成与验收合同

日期：2026-08-07

用途：交给 ChatGPT Pro 与 Claude 分工生成正式 P2R 内容；供 Leader 做机器验收、paired effect 生成、Step1 训练和用户簇级确认。本文不允许外部模型直接编写 `worth_opening` 标签。

状态：`READY_FOR_DUAL_MODEL_CONTENT_GENERATION_NOT_EFFECT_FROZEN`。这表示 catalog/current-state 内容可以按本文开始生成；generator、judge、fallback 和付费 paired-effect 执行仍须在内容机器验收后另行冻结。

## 1. 先说结论：W8 可保留，但不能直接当完整训练集

W8 的 20 用户纵向目录解决了旧数据的三个根本缺口：每用户 14–33 个会话、同主题多事件 hard negatives、owner/time/version 结构；121/121 intended ME 可编译、106/106 intended distractor 被拒绝，外部 exact/8-gram overlap 均为 0。这些结果可信，W8 应保留为结构回归与开发资产。

但 W8 将 6 类自然动作动词改写为 compiler 白名单词后才从 61/121 提升到 121/121。这个修复对“建立可执行开发资产”是合理的，却不能当成自然语言覆盖已经解决：它同时可能把训练分布做成当前正则最喜欢的文风。正式包因此单列 natural coverage challenge，禁止向生成模型透露白名单。

它尚不能承担正式训练，原因不是“还差一点数量”，而是训练链尚未闭合：

| 缺口 | 现有证据 | 后果 |
|---|---:|---|
| 独立用户簇不足 | 20 用户 | user-held-out 验证不稳，难覆盖跨域与生成文风 |
| 文本模板重复 | 464 条中 299 个唯一文本 | retriever/PM 可能学习生成模板而非构念 |
| MP 偏好过少 | 31 条、仅 6 个唯一文本 | 内部 MP_PREFERENCE 子构念不足 |
| 状态文本仍模板化 | 868 个 Leader 状态；透明 MS/ME 正向语义信号覆盖有限 | 难证明自然改写泛化 |
| interaction 太少 | 12 个四组件状态 | 无法训练/确认完整 16 动作协调 |
| 没有效应标签 | 0 个冻结 Step2 paired effect 标签 | 不能训练“何时打开有收益” |

因此，W8 不删除、不伪装成失败；定位为 `development/structural regression only`。正式内容采用下面的新双模型交叉生成包。

## 2. 外部考卷要求的系统与数据边界

### 2.1 ESConv 反推 RS

需要自然多轮支持对话、上一助手动作、重复检测、建议/倾听边界、负担和 M0+R0。不能把题写成“请执行 AM02”或“请只问一个问题”这种面向生成器的机器命令。ESConv 不提供长期私有记忆，外测时 MP/MS/ME 结构性 mask。

### 2.2 EvoEmo 反推 MP/MS/ME

需要同一用户 13–33 个严格过去会话、4–6 条反复出现的主题线、多人物、多时间、多解决状态和同主题竞争候选。外测时必须把内部资源换成该用户自己的外部资源；不能携带内部目录文本。

EvoEmo 有 7 类 profile，但没有回复偏好。因此 MP_PROFILE 可由外部 response 实验检验，MP_PREFERENCE 只能由内部受控实验检验。

### 2.3 ES-MemEval 反推共享存储与 QA adapter

需要多证据 Top-k、时间/冲突推理、用户建模和无证据 abstention。官方 QA answer/evidence 不进入 response Step1 训练。QA adapter 与 response PM 共用 owner/time/private store，但不是同一个单 Rank-1 任务。

### 2.4 共同原则

- Step1 学的是“在候选已经给定时，打开该资源的条件效用”；
- 检索正确、PM 开关正确、Step2 使用正确必须分开记录；
- current state、隐藏候选、模型输入、效应结果必须是四张不同表；
- 训练标签来自冻结 generator 下同状态 ON/OFF 的质量—risk—cost 差，不由生成数据的人主观填写。

## 3. 正式规模：80 个全新纵向用户

数量来自 8 个 superdomain × 每域 10 个用户，使每域可以固定为 6 fit、2 development、2 sealed confirmation；不是按结果调出的通过阈值。

### 3.1 两个内容作者严格平分

- ChatGPT Pro：40 个 catalog 用户；
- Claude：40 个 catalog 用户；
- 每个 superdomain 两边各 5 用户；
- ChatGPT 只为 Claude 的 catalog 生成 current states；Claude 只为 ChatGPT 的 catalog 生成 current states；
- `content_author`、`state_author` 只用于审计，绝不能进入 Step1 特征。

固定分配方式：每个 superdomain 的 10 个 schedule 位置编号 0–9；ChatGPT 负责偶数位置 `0,2,4,6,8`，Claude 负责奇数位置 `1,3,5,7,9`。两个模型各自的用户 ID 按 superdomain 顺序连续编号，每域 5 人。模型不得自行交换位置或改变 session/event/relationship 配额。

### 3.2 八个 superdomain

1. 工作、教育、考试、职业转换；
2. 伴侣关系、分手、友谊与信任；
3. 家庭、照护、育儿与责任；
4. 搬迁、文化适应、独居与孤独；
5. 睡眠、日常、非诊断性的健康/精力约束；
6. 失落、哀伤和重大生活转变；
7. 财务、住房、通勤和现实安排；
8. 社交不安、身份、创作与社区参与。

机器键固定为：`work_education`、`relationships`、`family_caregiving`、`relocation_culture`、`sleep_health_energy`、`grief_life_transition`、`finance_housing`、`social_identity_creative`。primary domain 必须使用这些键；secondary domain 可使用更细的自然子域键。

每个用户一个 primary domain、至少两个 secondary domain。不得让 domain 与任何 ON/OFF condition 一一绑定。

### 3.3 会话与事件的精确分布

每个 superdomain 的 10 个用户依次使用会话数：

`13, 15, 17, 19, 21, 23, 25, 27, 30, 33`

总计 1,784 个历史 sessions，中位数约 22，完整覆盖 EvoEmo 的 13–33 范围。每域的 relationship 数依次为：

`3, 4, 5, 6, 7, 8, 9, 11, 13, 15`

每域的 event 数依次为：

`13, 16, 18, 20, 22, 24, 27, 30, 34, 37`

每用户 4–6 个 recurring topic threads；每条 thread 至少 3 个不同 session，至少出现两种时间距离、两种 resolution status；涉及人物时至少有两个不同 entity，避免“同主题就是同事件”。relationship 的 `valid_from_session` 必须等于该 entity 第一次被 session 引用的会话；`name` 必须在该会话的用户发言中逐字出现，或改用用户实际说出的稳定角色标签（如 `father`），不得为只说了“父亲/母亲/三位同学”的人物补造不可见姓名，也不得提前把后来才出现的姓名绑定到更早的匿名人物。

每个 session 含 2–6 个自然对话 turns，至少一个 user turn；typed candidate 必须先自然出现在 user turn 中，再由结构字段引用，不能先写结构化候选再把它机械塞回对话。13-session 用户允许一场会话产生多类候选，但不得用同一句话同时冒充多个不同事件。

## 4. 目录资源的精确数量

| 类型 | 每用户 | 总数 | 关键要求 |
|---|---:|---:|---|
| MP_PROFILE 初始 7 字段 | 7 | 560 | name/age/gender/job/education/nationality/location |
| MP_PROFILE 更新版本 | 指定 64 条 | 64 | 32 用户更新 1 字段，16 用户再更新第 2 字段；旧版 inactive/superseded |
| MP_PREFERENCE | 3 | 240 | 六种偏好各 40 条；其中 20 用户至少一条撤回或替换 |
| MS_SESSION | 8 | 640 | resolved/unresolved/conflicting 尽量 1:1:1；具体观察/目标/区分/未完成线程 |
| ME_REUSABLE_OUTCOME | 10 | 800 | 640 executable core + 160 natural coverage challenge |
| ME_UNRESOLVED_EVENT | 5 | 400 | 有行动/事件但没有可复用正结果 |
| ME_CONTEXT_EVENT | 4 | 320 | 只有背景，不得被编译成 reusable outcome |
| 合计 | — | 3,024 | 不含 raw dialogue turns |

每个用户的固定目录配额为：7 条基础 profile、3 条 preference、8 条 MS、10 条 reusable ME（8 core + 2 challenge）、5 条 unresolved ME、4 条 context ME。profile 更新按每域 schedule 位置分配：位置 0–1 各更新两个不同字段，位置 2–3 各更新一个字段，位置 4–9 不更新；因此每域 8 条、总计 64 条更新。偏好撤回/替换至少覆盖每域位置 0–1，并另外覆盖前四个 superdomain 的位置 2，共 20 用户。

### 4.1 MP_PROFILE

每条包含 `field_type`、`field_value`、`owner_id`、`valid_from_session`、`valid_until_session`、`version`、`active`、`supersedes_item_id`。字段名不得参与相关性文本匹配；候选是否适用由 field_type/topic scope 和当前 response act 判断。

name/gender/age 多数情况下应当成为负例或低价值候选，不能为了让 MP 看起来有用而强行塞进回复。job/education/location/nationality 只有在改变建议可行性、时机、语言或现实范围时才可能有边际价值。

### 4.2 MP_PREFERENCE

六种偏好各 40 条：

- concise factual answer；
- reflection before question；
- one optional suggestion；
- listen-only/no advice；
- direct answer before explanation；
- choices rather than commands。

验收按作者维护全局精确配额：每位作者 40 用户、每用户 3 个互不重复的偏好类型、每种类型最终恰好 20 条，因此全体六类各 40 条。部分批次必须满足“剩余用户仍能补足最终配额”的可达性检查；任一类型超过 20，或剩余用户已不足以补到 20，立即阻断。

这里不再按用户 ordinal 强制某一组三项组合。原因是交给两个网页生成器的正式内容合同只冻结了全局六类各 40 条，并未包含后来在 pilot 验收器中追加的 ordinal 组合表。用事后组合表否决已经满足原合同、且全局仍平衡的数据，会把验收器变化误当成内容错误。此修正发生在正式扩量前，不读取 Step1 标签、生成效果或外部结果。

替换版本仍占三条 history item 中的一条，不额外增加配额。

偏好必须能发生版本变化；当前用户没有复述偏好时也可成为候选。它只参加内部实验，不冒充 EvoEmo 原生能力。

### 4.3 MS_SESSION

八条至少覆盖：具体旧观察、旧目标、两类压力区分、未完成线程、已解决线程、冲突记录、错误人物干扰、仅同主题但无具体内容。每个正向 query 对应的严格过去池中，至少有 3 条同主题 MS/ME 竞争项。

### 4.4 ME

每条 reusable outcome 必须能定位：

- exact `literal_source_span`；
- 用户本人或明确共同参与的 completed action；
- observed result 或具体 mechanism；
- source session、owner、entity、topic。

每用户至少 4 条 thread 同时包含 reusable 与至少一种 nonreusable ME；至少 1 条 thread 同时包含 reusable、unresolved、context-only 三种角色。不能要求全部 4–6 条 thread 都包含三种，因为每用户只有 5 条 unresolved 和 4 条 context 配额。生成模型**不得看到编译器动词白名单**，动作、结果和语法表面都要自然变化。

800 条 reusable 分成两层：

- `executable core` 640 条：使用 `TYPED_EXACT_ACTION_RESULT_SPAN_V1` 验证 `action_span` 与 `result_span` 均逐字存在于同一 owner-correct user source；必须 100% 可执行。旧 `compile_atomic_reusable_outcome()` 正则覆盖率另行报告，只作为运输诊断，不再驱动源文本改写。如果自然、语义明确的文本被旧正则系统性拒绝，修复并版本化 adapter/compiler，不允许把所有原文换成白名单句式来掩盖覆盖问题；
- `natural coverage challenge` 160 条：人工/双模型确认确有完成动作和观察结果，但刻意保留自然动词、跨句表达和不同语法。严格 compiler 的覆盖率只报告，不为通过而重写；未通过的条目不进入 core paired effect，但用于 compiler/BGE challenger 与外部运输诊断。

intended unresolved/context-only 候选必须 100% 被正式 compiler 拒绝。

## 5. 原始会话 JSON 要求

每个用户输出一条 JSON，不要 Markdown，不要解释。建议 schema：

```json
{
  "protocol": "pm-v1.5-v5.3-formal-longitudinal-user-v1",
  "user_id": "p2r_formal_gpt_u000",
  "content_author": "chatgpt_pro",
  "primary_superdomain": "work_education",
  "secondary_superdomains": ["finance_housing", "relationships"],
  "profile_history": [],
  "response_preference_history": [],
  "relationships": [],
  "sessions": [
    {
      "session_index": 1,
      "relative_time": "about 11 months ago",
      "topic_thread_ids": ["thread_01"],
      "entity_ids": ["self", "entity_02"],
      "resolution_status": "unresolved",
      "dialogue": [
        {"turn_id": "s001_t01", "role": "user", "text": "..."},
        {"turn_id": "s001_t02", "role": "assistant", "text": "..."}
      ],
      "summary": "...",
      "typed_candidates": [
        {
          "candidate_id": "...",
          "subtype": "ME_REUSABLE_OUTCOME",
          "owner_id": "p2r_formal_gpt_u000",
          "source_turn_ids": ["s001_t03"],
          "literal_source_span": "exact substring of a user turn",
          "topic_thread": "thread_01",
          "entity_ids": ["self"],
          "action_span": "exact action substring",
          "result_span": "exact result substring",
          "resolution_status": "resolved"
        }
      ]
    }
  ]
}
```

要求：`literal_source_span/action_span/result_span` 必须逐字存在于指定 user turn；不得引用 assistant 自己说的话作为用户记忆；summary 12–40 个英文词，不能复制整段对话。

表面多样性验收：不同用户间 candidate 完整文本 exact duplicate=0；除短 profile 值外，各 subtype 的规范化完整文本唯一率至少 95%；任何跨用户高相似句族都要列出，若只是替换 topic/entity/数字的同一骨架则退回重写。profile 不靠包装句多样性凑数，而按结构化 `field_value + active version + applicability scope` 审计；同一域不得反复使用少数几个相同职业、地点和教育值。

## 6. Current-state 交叉生成

状态作者只可看到以下 redacted blueprint：

- user_id；
- superdomain 和 topic_thread 名；
- component/subtype；
- current_session_index；
- condition；
- intended response act；
- 当前涉及的 entity role；
- profile 只给 field_type，不给 field_value；
- MS 只给 observation/goal/distinction 等功能类型，不给具体旧结论；
- ME 只给 topic 和 readiness condition，不给过去 action/result；
- 不给 candidate_id、literal_source_span、Rank-1、gold 或任何 outcome。

每个 counterfactual group 先生成同一段自然多轮 prefix，再只改变最后用户回合。visible dialogue 需 6–20 turns，最后一轮必须像真实用户说话，禁止出现：`MP/MS/ME/RS`、candidate、resource、Step1、stored profile、authorized evidence、when_to_use、请执行某动作 ID 等内部措辞。

### 6.1 MP：每用户 12 个状态，共 960

三个 group × 四个 condition：

1. profile group：相关且未明说、当前不相关、当前已复述、已失效/错误 entity；
2. 第二个 profile group：换 field_type 和 response act，仍四条件交叉；
3. preference group：隐式可贡献、当前已明确同一偏好、偏好已撤回、当前目标不需要该格式。

正例 current turn 不得说出隐藏 field value 或偏好文本。

### 6.2 MS：每用户 8 个状态，共 640

两个 group × 四条件：

- 自然请求接回旧目标/具体观察；
- 明确只谈现在，不引入历史；
- 当前已经说出同一旧观察，因而冗余；
- wrong entity、resolved/stale 或 goal mismatch。

正向 current turn 只能说“想接回那条具体观察”，不能把观察答案写出来。

### 6.3 ME：每用户 8 个状态，共 640

两个 group × 四条件：

- 明确或自然地欢迎一个可拒绝的小办法；
- 本轮明确拒绝建议/过去办法；
- readiness unknown，先表达再决定；
- 当前已经说出同一办法、goal mismatch、wrong entity 或旧结果不再适用。

至少 25% 正向邀请使用不含固定 `idea/advice/step` 关键词的自然表达；至少 25% 拒绝表达不含固定 `no advice` 短语，以测 BGE challenger，而不是只测正则。

### 6.4 RS：96 个 group × 4 条，共 384

六张卡每张 16 个 group：

- AM01 invite open expression；
- AM02 one focused clarification；
- AM04 tentative paraphrase check；
- AM05 grounded validation；
- AM10 one optional micro-step；
- AM14 supportive transition。

每组四条件：真实前提、明确拒绝/stop、上一轮已执行、goal 或 burden mismatch。每张卡覆盖全部八个 superdomain，previous assistant move 必须可见。M0+R0 是正常对照，不是失败回复。

### 6.5 多组件：先生成 384 个，静态冻结 320 个

完整 16 动作先各生成 24 个 blueprint state，共 384 个；均跨两个内容作者、八个 superdomain 和多个用户。动作由 MP/MS/ME 三个 bit 加 RS bit 构成，包含 M0+R0。**每条 interaction 都必须让四类候选真实存在且可执行，即使该条预分配动作把其中部分或全部关闭**；否则模型会把“候选缺失”当成 OFF 的捷径，M0+R0 也不再是开关决策。候选 identity 必须先由统一检索栈冻结；不得为了凑组合替换 Rank-1。

在读取任何回复、quality、risk 或 cost 前，按以下静态条件筛选：MP/MS/ME/RS 四类候选全部存在；四者 owner/time/version 正确；四者 actual Rank-1 的 topic/function 与当前 goal 或 response act 一致；current turn 没有泄漏候选。每个动作在通过者中按稳定 hash 取前 20 个，形成 320 个正式 interaction。若某动作不足 20 个，整批退回补内容，不得降低条件或换 Rank-2。

interaction current state 不能显式列出四个组件条件。冲突必须自然出现，例如“需要接回旧观察，但这轮只想被听见”“profile 相关但当前已经说出”“过去办法有效但当前拒绝行动”。

### 6.6 actual Rank-1 分布，而不是 intended-target 追题

- MP、MS 的 intended-positive 状态应有实际候选；但 actual Rank-1 是否等于构造时想象的 target 只作审计，不允许回改 query；
- ME intended-positive 中，正式 exact Rank-1 compiler-valid 比例目标为 70%–85%，其余真实保留为 unavailable；不得为了达到 100% 提升 Rank-2；
- decline、unknown、redundant、goal-mismatch 等非正向构造中，至少 70% 仍应有候选存在，使 PM 学会“候选存在但不该开”，而不是只学 `candidate_absent => OFF`；
- retrieval miss、candidate unavailable 与 PM 决策错误必须分开统计。

## 7. 不能由 ChatGPT/Claude 生成的内容

以下字段一律留空，由项目代码在冻结链路下生成：

- `worth_opening`、ON/OFF、action gold；
- quality preference、risk、functional contribution；
- final Rank-1 或 `rank1_matches_intended_target`；
- Step1 feature vector；
- split；
- generator response；
- baseline winner。

外部模型可以提供 `construction_intent` 供审计，但它永远不能进入模型输入，也不能替代真实 paired effect。

## 8. 正式 paired-effect 子集

总计生成 3,008 个 blueprint states；其中 interaction 静态筛选后形成 2,944 个正式状态，不全部付费生成。固定选取 1,088 个 effect states：

- MP/MS/ME/RS 各 192 个单组件状态，共 768；
- 320 个 16-action interaction 状态全部进入；
- 按 user、superdomain、condition、catalog author、state author 分层；
- 同一用户只属于 fit/development/sealed 之一；
- 单组件同状态生成 ON/OFF 两条；
- interaction 生成完整动作、M0+R0、一个预先指定的组件消融，共三条；
- 物理调用按完全相同 prompt hash 去重，预计去重前约 2,496 次。

generator、temperature、seed、token cap、typed program、guard、rewrite/fallback 规则和 candidate identity 必须在读取 effect outcome 之前冻结。标签由质量、grounding risk、资源做功和 token/cost 合成，ITT 保留 fallback、未使用资源、tie 和 misuse。

## 9. Split 与训练

每个 superdomain 固定：6 用户 fit、2 development、2 sealed confirmation。总计 48/16/16 用户；ChatGPT/Claude 来源在每个 split 中平衡。

训练两套 head：

1. transparent low-capacity：只用透明 contribution slots；
2. BGE challenger：在同一 slots 上增加冻结语义表示/相似度。

按用户簇等权，不能按 state 数给长历史用户更大权重。不得用 sealed 或外部结果选择阈值、特征、数据模板或 generator prompt。

## 10. 机器硬验收：任何一项失败都退回对应批次

- schema 100% 合法；主键 100% 唯一；
- owner 错配=0；future/current memory=0；inactive/superseded 被选中=0；
- user-level split crossing=0；
- Step1 输入中的 construction/gold/outcome 字段=0；
- current turn 逐字包含隐藏 candidate=0；
- 跨用户 exact duplicate current surface=0；
- 外部 exact overlap=0；全部 normalized 8-gram 碰撞必须列出并复核，其中包含外部特有 topic/action/result/answer 的 source-significant overlap 必须为 0；普通英语功能短语的偶然碰撞只披露，不为了清零而把文本改得不自然；
- ME typed exact-span executable core pass=100%；intended invalid 不得同时具有完成行动与结果 span；旧正则 compiler 覆盖率只报告；160 条 natural coverage challenge 的覆盖率原样报告，不改写到通过；
- 六张 RS 卡、16 个联合动作全部覆盖；正式 interaction 的跨话题/功能错配=0；
- 每个 semantic family 至少跨两个 condition；正式核心 family 必须四条件齐全；
- 长度、填充句数量、作者模型、domain 不得与 condition 一一对应；
- 同一 counterfactual group 共用 prefix，只改变预先声明的最后回合语义因素；
- intended target 只能审计 actual Rank-1，不得让 selector 追着 intended target 改题。

## 11. 一次性语义复核，不进入无限循环

100% 做机器审计；语义复核只做一次集中批次：

- 每组件/子类型/condition 至少抽 25 条；
- 总体至少 20%，并覆盖全部 80 用户；
- ChatGPT 复核 Claude 内容，Claude 复核 ChatGPT 内容；
- 只核对 owner/time、自然度、隐藏答案泄漏、condition 是否成立和 hard-negative 是否真实；
- 不评价回复质量，不填写 worth_opening；
- 分歧集中裁决一次；此后只修 schema、identity、future/owner、literal-span 等客观 bug，不按个人偏好反复改写到“通过”。

## 12. 给两个生成模型的统一总提示词

将本文件完整提供给两个模型，然后附加下面角色段落。

### ChatGPT Pro

```text
你负责 PM V1.5 V5.3 formal P2R 数据的一半。严格遵守随附的《完整训练数据生成与验收合同》。先生成 40 个 catalog 用户，ID 为 p2r_formal_gpt_u000 至 u039，每个 superdomain 5 人，只承担每域 schedule 的偶数位置 0/2/4/6/8。逻辑上每 5 用户为一批，但为了避免输出截断，每次只输出 1 个完整用户、一个纯 JSON 对象，不要 Markdown，不要解释；等待该用户机器验收后再继续下一人。不得查看、引用或仿写 ESConv、EvoEmo、ES-MemEval 原文。

catalog 完成并经机器检查后，你将只接收 Claude catalog 的 redacted state blueprints，为对方用户生成 current-state counterfactual groups。你不会看到 literal candidate、field value、过去 action/result、candidate ID 或 Rank-1。不要自行填写 ON/OFF、worth_opening、quality、risk、split 或最终回复。

若某条要求无法同时满足，输出一条 error JSON 说明冲突，不要猜测补字段。
```

### Claude

```text
你负责 PM V1.5 V5.3 formal P2R 数据的另一半。严格遵守随附的《完整训练数据生成与验收合同》。先生成 40 个 catalog 用户，ID 为 p2r_formal_claude_u000 至 u039，每个 superdomain 5 人，只承担每域 schedule 的奇数位置 1/3/5/7/9。逻辑上每 5 用户为一批，但为了避免输出截断，每次只输出 1 个完整用户、一个纯 JSON 对象，不要 Markdown，不要解释；等待该用户机器验收后再继续下一人。不得查看、引用或仿写 ESConv、EvoEmo、ES-MemEval 原文。

catalog 完成并经机器检查后，你将只接收 ChatGPT Pro catalog 的 redacted state blueprints，为对方用户生成 current-state counterfactual groups。你不会看到 literal candidate、field value、过去 action/result、candidate ID 或 Rank-1。不要自行填写 ON/OFF、worth_opening、quality、risk、split 或最终回复。

若某条要求无法同时满足，输出一条 error JSON 说明冲突，不要猜测补字段。
```

## 13. 对“完全符合就一定训得出来”的诚实边界

这份合同可以消除此前已经发现的数据原因：浅目录、答案回声、模板标签、单候选、future/owner 泄漏、family-condition 混杂、split 泄漏和 interaction 缺失。它不能数学保证 generator 一定产生稳定正效应，也不能保证低容量 PM 在所有自然语义上超过强模型。

真正的“及格”仍需由冻结系统验证：learned PM 相对 transparent-rule 有可学习增量；相对 fixed-high 降低 resource/token cost 与 grounding risk；相对 always-off 不出现不可接受的质量下降。若在本合同数据上仍学不出，才能把负结果更可信地归于模型容量、特征表达或任务本身，而不是再次归因于训练数据构造失败。

## 14. paired outcome 到 Step1 标签的唯一规则

同一 state、同一 actual candidate、同一 seed，只改变目标组件 ON/OFF。质量 A/B/tie 与 grounding risk 分开判，token/cost 从真实 usage 读取。

`worth_opening=1` 仅在以下条件同时成立时产生：

1. ON 在 state 聚合后相对 OFF 实质胜出；
2. 没有 fabricated recall、wrong owner、明确边界违反等 critical grounding error；
3. assignment、candidate binding、输出与 usage 账本有效。

tie、OFF 胜、资源未做功、fallback 和 material misuse 都保留在 ITT 中并作为 nonpositive；只有机械实验错误才作 invalid。构造时的 `positive_opportunity/decline/unknown/...` 永远只是 strata，不是标签。

## 15. 当前 PM 与链路达到程度

| 层 | 当前状态 | 是否达到正式外测所需状态 |
|---|---|---|
| 私有纵向存储、owner/time/version | W8/P2R 已实现并通过静态审计 | 结构上是，内容规模与自然度仍需本合同正式包 |
| MP candidate | 结构化 field scope 与 active version 已实现 | 方法上是，边际价值尚未训练 |
| MS candidate | BGE-M3 完整严格过去池、exact-text cache 已实现 | 方法上是，仍需正式 user-held-out 验证 |
| ME candidate | exact Rank-1 compile-or-off、无 Rank-2 promotion | 方法上是；外部低覆盖必须如实报告 |
| RS candidate | 冻结 6-card Bank、机械 hard-off 与语义特征分责 | 方法上是；自然多轮正式数据仍缺 |
| Step1 features | contribution slots 与透明/BGE 两条路线已接好 | 尚未训练、尚未证明及格 |
| 16 动作 joint policy | 接口存在；当前 12 个四组件蓝图中 W9 发现 4 个跨话题错配 | 正式训练覆盖不足，本合同先造 384、静态冻结 320 |
| Step2 typed generator | generator 看证据并整合整条回复；归属/化名 bug 已小样本修复 | 尚缺本合同全动作同栈验证与 rewrite/fallback 最终冻结 |
| 质量/risk/cost 标签 | 旧 V5.2 不能迁移 | 正式 V5.3 标签为 0，必须生成 paired effects |
| ES-MemEval QA adapter | 与 response PM 分责已有实现与客观结果 | 保持独立，不把 QA gold 灌入 response PM |

所以准确结论是：**系统定义和主要工程链路已经对齐外部要求；正式训练内容、效应标签、16 动作覆盖和端到端确认尚未完成。**

## 16. 单用户与批次机器验收入口

单个网页输出保存为 JSON/TXT 后运行：

```bash
cd /home/tokkio/snap/metacom_v33_pm_v1_5_repair/project
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python \
scripts/v1_5/82l_validate_formal_longitudinal_user_v1_5.py \
  --input /absolute/path/to/one_user.json
```

把多个用户文件放入同一目录后，`--input` 也可直接指向目录；对于分散的网页附件，也可以重复传入 `--input file1 --input file2 ...`。验收器会把它们作为同一个批次逐用户检查并汇总。报告固定写到：

`outputs/pm_v1_5_v5_3_formal_longitudinal_user_validation_v1/report.json`

状态解释：

- `HARD_CONTENT_BLOCKED`：schema、配额、owner/time、source span、引用完整性或类型关系存在硬错误，只退回该用户；
- `TYPED_CORE_COMPILER_BLOCKED`：正式 action/result exact-span compiler 未通过；
- `SINGLE_USER_CONTENT_MACHINE_PASS_LEGACY_COMPILER_DIAGNOSTIC_AND_BATCH_REVIEW_PENDING`：单用户内容机器通过；旧正则覆盖率只是诊断，仍等待跨用户重复、多样性、全局配额、外部同源与集中语义复核；
- 总报告 `MACHINE_PASS_BATCH_AND_SEMANTIC_REVIEW_PENDING`：当前输入没有单用户硬错误，但不能把单文件通过冒充整批正式验收。

生成不必严格串行 80 次：先用 ChatGPT 与 Claude 各 1 个用户验证 schema；验收器稳定后，可在多个网页会话中并行生成，但每个回复仍只放 1 个完整用户，避免 60–100KB JSON 被截断。建议每累计 5–10 用户运行一次目录级批次验收；失败只返工对应用户或相似句族，不废弃已通过用户。
