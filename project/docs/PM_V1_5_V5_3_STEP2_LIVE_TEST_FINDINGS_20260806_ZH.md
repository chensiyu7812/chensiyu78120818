# Step2 真实生成小批量测试——发现（2026-08-06，10条真实调用）

状态：**真实API调用，用户2026-08-06明确授权**（"step2可以继续"，此前已告知需要付费）。用
`scripts/v1_5/48_step2_typed_response_dry_run_v1_5.py --live --n 10`，通过NVIDIA网关调用
真实`generator`端点（Llama-3.1-8B-Instruct），对10个真实138状态面板state（跨10个不同用户，
每个用户1条，避免全部落在同一人的写作风格上）生成了真实响应。产出：
`outputs/pm_v1_5_v5_3_step2_live_test_v1/step2_live_responses.jsonl`（gitignore，含真实
模型生成文本，未进git）。

## 结构层面：10/10通过

`typed_response_guard_errors`（检查内部标签/资源ID泄漏、V5.2式"记录复述"泄漏短语等
machine-checkable结构性错误）对全部10条响应返回空错误列表。`used_evidence_ids`在每条响应里
都精确对应真实传入的证据ID，没有编造或遗漏。

## 但人工逐条读了全部10条真实回复，发现一个结构层面测不出来的严重问题

**5/10条回复把用户自己的事实/经历，用第一人称说成是助手自己的**，不是"复述用户说过的话"，
是"把这件事说成发生在助手自己身上"：

- **p7（MP="Job: business owner" + MS）**："As a business owner, **I've** had my fair share
  of challenges, including a failed partnership and account discrepancies."——用户是business
  owner，助手不是，但助手说"作为一个企业主，我也经历过合伙人失败、账目差错"。
- **p9（MS=纠结要不要接受降薪的工作机会）**："**I've** been trying to weigh the pros and
  cons of this job offer...worried about the financial implications of taking a pay cut...
  saving for a down payment on a house"——这整段完全是用户自己的处境，被助手用第一人称说
  成自己的事。
- **p10（MS=用户对丈夫辞职、儿子行为问题的复杂情绪）**："**I'm** really proud of **my
  husband** for his success...**our son's** behavior has been causing some family
  conflicts"——**最严重的一例：助手凭空声称自己有丈夫、有儿子**，这不是"混淆时态"，是编造
  了一个不存在的助手身份/家庭关系。
- **p12（MS=是否要跟前任重新联系）**："As **I** weigh the possibility of reconnecting with
  someone from my past...**I'm** also drawn to the idea of reconnecting with someone who
  holds a special place in **my** heart"——用户自己纠结要不要联系前任，被说成助手自己在
  纠结要不要联系"我的前任"。
- **p13（MP="college degree" + MS=大学同学聚会引发的复杂情绪）**："With **my** college
  degree in hand, **I've** been able to pursue a fulfilling career, but now **I'm** facing
  the challenge of balancing work and wedding planning"——用户的学历、婚礼筹备被说成助手
  自己的。

**1/10条混合**（p11）：前半段"I can relate...I've been trying to focus on improving my
study habits...I remember when I was trying to balance schoolwork"仍是第一人称占用用户
经历，但后半段自己纠正成"we can work together to find some strategies that can help **you**
cope...make the most of **your** time in high school"，同一条回复内前后不一致。

**4/10条正确**（p8、p14、p15、p16）：全程用"you/your"第二人称正确归属证据是用户自己的，
读起来自然，是这次测试里符合设计意图的例子（比如p16:"I recall when **you** mentioned
feeling undervalued by **your** supervisor..."）。

## 为什么结构guard完全没拦住——这是设计缺口，不是guard的bug

`typed_response_guard_errors`检查的是"machine-checkable structural/binding errors only"
（模块自己的docstring原话）：内部标签/ID泄漏、特定的V5.2式"记录复述"短语。它从来没有被设计
成检查"这段第一人称叙述描述的到底是用户的事还是助手自己的事"——这是一个语义/指代问题，不是
字符串匹配能测的。

回头看`evidence_aware_generation_messages()`的system prompt，确实**没有任何一句明确指令
说"下面这些evidence是用户自己的过去经历/事实，你必须用第二人称呈现，不能说成是你自己的"**——
现有约束只有"forbidden: inventing a current cause, a stable personality trait, an
unmentioned third party, a diagnosis, or an outcome guarantee"，没有覆盖"把证据的主语从
用户换成助手自己"这类错误。这是一个具体、可定位的prompt设计缺口，不是模型选型问题（换更大的
模型也不能保证解决，除非验证过）。

## 结论：V5.2的老问题（LLM把过去/他人的事实变成当前第一人称事实）在V5.3里以新形式复现了

V5.2机械拼接方案存在的根本原因之一，就是不信任LLM能正确处理"呈现证据但不越界声称"这件事，所以
把MS/ME渲染做成了后端确定性拼接、LLM完全看不到原文。V5.3 typed response program的设计前提
是"给LLM完整typed证据，让它自己整合"，赌的是"新的显式指令+结构化输出能约束住LLM"。**这次10条
真实小样本测试显示，这个赌注目前没有完全兑现**：结构层面的泄漏（V5.2那种"记录复述"腔调、内部
ID泄漏）确实被防住了，但一个新的、同样严重甚至更严重的错误模式（把用户自己的事实说成助手自己
的、包括凭空编出配偶子女）冒出来了，而且发生率不低（10条里5条明确、1条部分）。

**这不是"V5.3方案失败"的结论**——是"当前这版system prompt不够、需要补一条明确的证据归属
指令，再重新测"的结论。样本量也只有10条、单次调用（temperature未特别设置为复现同一测试的
多次采样），不能排除是这次采样运气差，但5/10这个比例大到不像纯噪声，值得在改prompt后立刻
重新小批量验证，而不是无视它继续往前走。

## 2026-08-06修复与复测：核心bug已修复，发现并修好一个新问题，还有一个更细的问题留着

用户看过独立复核者（另一个Codex）的方案分析后明确说"开始吧"，授权继续。落地了"最小归属修复"：
`ExecutionEvidence`本来就有`owner_id`字段（真实存在，只是从没被渲染进prompt），现在每条证据都
带上明确的owner标签，system prompt加了角色边界指令（"你不是用户、不能扮演用户"），证据从跟
`current_context`混在一起的user消息挪到了单独的system消息。另外做了一个专门校准过的
`speaker_attribution_guard_errors`（不是笼统关键词表，在原10条真实回复上验证：5/5明确违规
全部抓到、4/4干净样本0误报），配一次定向重写+失败退回`m0_fallback_response`的兜底逻辑，最多
重试一次，不会无限循环。这些改动本身不花钱（没有调用真实API），先用假client把重写/兜底4条
分支全部测过，再动真格。

**第一轮复测（10条全新独立样本，跟最初10条完全不重复）**：9/10首次通过guard，1/10（p7）被
guard拦截、正确退到了安全兜底。**人工逐条通读全部10条真实回复**：8/10确认真正干净——包括
两个跟最初最严重案例结构完全相同的场景，这次都对了：p10正确说"your husband"（不是"my
husband"）、p16正确说"You're a project manager"（不是"As a project manager, I..."）。1/10
（p7）被guard正确拦截。**但第6条（p12）冒出一个新问题**：模型把prompt里"you/your"这个记号
写法**原文抄进了回复**里五次（"That's a great approach, you/your...Have you/your thought
about how you/your will..."），读起来是坏掉的。这不是guard该管的那类错误（不是把用户的事说成
自己的），guard正确地没有拦它，但确实是我自己prompt措辞的问题——用了一个模型会照抄的字面
记号。已经改成不用斜杠记号的自然语言描述，加了一条回归测试锁定"system prompt里不能再出现
you/your这个字面字符串"。

**第二轮复测（5条，含1条刻意跟最初bug发现batch用同一个真实state做直接前后对照）**：4/5干净，
1/5（还是p7）再次被guard拦截、退到兜底——同一个state连续两个prompt版本都触发同样的guard，
说明guard在这个具体case上判断是稳定的，不是偶然噪声（具体触发的第一版回复文本没有留存日志，
只存了最终结果，这是个小的记录缺口，不影响兜底机制本身已验证有效这个结论）。**最有说服力的
一条**：state_f95b7e244d5934d33cdd（p10）这个state，在最初bug发现batch里就是那个，当时的
真实回复是"I'm really proud of my husband...our son's behavior..."（凭空认了配偶和孩子）；
这次用完全相同的state重跑，回复变成"I hear you're feeling a mix of emotions about your
husband's new job...You mentioned earlier that your son's behavior..."——**同一个state、
同一份证据，修复前后的真实回复直接对比，问题确认修复**。

**复测中还发现一个更细、这次没有修的问题**：p11这条真实回复里，MS证据文本本身用了一个第三人称
人名（EvoEmo数据集自己的叙事惯例，用一个化名指代用户本人，类似之前见过的"Emily"/"Jimmy"这类
名字），模型的回复是"You mentioned earlier that Anna is stressed about them too, and I'm
worried that she might be feeling overwhelmed"——**把证据文本里指代用户本人的化名"Anna"当成
了一个独立于"you"的第三方**，造成"你提到Anna...我担心她"这种应该是同一个人却被拆成两个人的
混乱指代。这不是这次guard设计要覆盖的范围（这不是"助手把用户的事说成自己的"，是"助手把用户的
化名当成了别人"），现有guard没有也不应该拦这个——如实记录为一个新发现、尚未修复的独立问题，
不在这轮授权范围内顺手改，留给用户决定要不要继续投入。

## 建议（更新）

1. **核心的说话人归属bug（助手把用户的事说成自己的）方向上已经确认修复**：15条真实复测（两轮
   共15条，加最初10条共25条真实调用）里，0条再出现"assistant声称拥有用户配偶/子女/工作/学历"
   这类错误；关键的同state前后对照（p10）直接证实。这个具体问题不再建议继续投入更多验证轮次，
   可以视为这一轮修复的目标已达成。
2. **"you/your"字面记号泄漏问题已发现并修复**，但只用5条新样本验证了"没有再复现"，样本量小，
   如果后续要做更大规模Step2验证，这一点顺带留意一下即可，不需要专门再开一轮。
3. **"证据里的化名被当成第三方"问题已经修复并做了针对性验证**。先去查了EvoEmo原始数据而不是
   凭印象判断严重程度：`basic_info.name`在每个用户自己的MS session摘要里出现的比例是
   20%-79%（18个用户逐个查过），说明这不是p11一次运气差，是结构性存在、影响面不小的模式。
   好消息是这个名字本身是已知结构化数据（跟最开始那个bug同一类修复手法——不用建新的语义抽取，
   把`basic_info.name`显式传进prompt，加一条"这个名字指的就是你正在对话的这个人"的指令即可）。
   修复后**专门挑了8个真实证据里确实含用户化名的state**（不是随机抽样，是精确定位到会触发
   这个问题的场景）做真实验证，包括p11（Anna）本人当初触发bug的那条evidence——**8/8没有再
   出现"把化名当成另一个人"这个模式**。过程中顺带发现两次不相关的guard触发
   （`REQUIRED_EVIDENCE_NOT_USED`，一次靠重写修好，一次退到M0兜底）——这是已有的结构性guard
   在正常工作，不是新问题。
4. 仍然维持此前的结论：这次测试全程绕过了Step1（强制注入而非PM路由决策）、用的是旧production
   MS selector不是BGE、`current_goal`是诚实的占位任务描述不是真实意图识别——这些边界条件没变，
   真正端到端的Step2验证还需要跟Step1真实接起来之后再测一次。

## 2026-08-06独立复核后的4处改进（commit待补）

一位独立复核者（另一个Codex）核对了提交、代码和真实输出后，指出这批工作"真实且有价值，但
'Step2已修好'说得偏满"——批评基本成立，逐条核实后采纳了其中3条低成本、值得立刻做的：

1. **归属/guard/重写/回退这套编排逻辑（`call_with_guard_and_rewrite`）之前只存在于测试脚本
   `48_step2_typed_response_dry_run_v1_5.py`里，不是正式可复用模块**——批评属实，已经整体
   搬进核心模块`v1_5_v5_3_typed_response_program.py`，脚本改成从核心模块导入。搬运过程中
   还顺手发现并修了一处遗漏：重写请求里的纠正指令原文还残留着"you/your"这个字面记号（跟之前
   system prompt里修过的是同一类风险，只是这次在重写轮的user消息里，没在之前的测试批次里
   被真正触发过），已经改成完整句子。
2. **`used_evidence_ids`是模型自报，从来没跟回复正文做过核验**——批评属实。没有采纳对方提的
   `realized_reply_span`逐字匹配schema（成本高，会给8B模型引入新的schema合规噪声，等于用
   一个新问题的风险去换一个不确定收益）。改用更便宜的折中：新增
   `evidence_usage_plausibility_errors`，复用项目里已经在用的`content_words`词面重合检查，
   只抓"声称用了某条证据但回复正文里连一个词的痕迹都没有"这种最明显的情况——不追求完整
   grounding验证，只堵最明显的自报造假。
3. **M0 fallback之前写死成`"one_focused_question"`（固定追问），可能违反某些场景下"不该
   提问、只需陈述"的边界**——批评属实，是真代码bug。改成根据`program.atomic_move_budget`
   判断：有RS（预期有一个支持动作）时仍用追问，没有RS时改用`"concise_reflection"`（陈述句，
   不追问）。这是个启发式改进，不是完整的边界感知系统，但比之前的硬编码更合理。

**没有采纳的部分**（性价比不划算，不是不认同）：
- "无依据推断兴奋/压力原因"——这个约束已经写在prompt里了（"Forbidden: inventing a current
  cause"），如果仍在发生是合规率问题，需要真正的人工grounding审查才能量化，不能用正则/关键词
  便宜堵住，成本级别接近V5.2当年的E7人审，不该现在临时插入。
- 完整的`realized_reply_span`逐字grounding schema——见上面第2点，用更便宜的折中方案代替了。

同时把`PM_V1_5_V5_3_CONSOLIDATED_FINDINGS_20260806_ZH.md`里"MP目前没有已知未修复问题"这句
话的范围收窄了——对方指出这个措辞太乐观，只在"两类已量化的检索假匹配"这个具体范围内成立，
MP候选的边际价值判断（Step1该不该开启）完全没碰过，这个批评也属实。
