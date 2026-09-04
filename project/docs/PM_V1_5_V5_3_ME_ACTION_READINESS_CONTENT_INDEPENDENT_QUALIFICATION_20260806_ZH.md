# ME行动准备度三态观察器——真正的内容独立资格验证（2026-08-06）

## 为什么需要这次验证

账本`V15-MEAS-53`已经指出：新三态观察器（`observe_action_readiness()`）之前只在已经
消费过的V3/V5.2开发构造行上回放通过（128/128），这批数据的自然语言表述本身就是设计/
调这个正则时参照的对象，不能算泛化证据。这次是真正独立的检验：从EvoEmo全部18个用户、
4452条真实seeker发言里按固定种子随机抽70条（`seed=42`），逐条只看发言本身（跟观察器
拿到的信息完全一致，不看supporter前情），人工独立判定真实标签，再跟观察器预测比较。
零API调用，人工标签在脚本里固定写死，不是脚本自己生成再自己打分。

脚本：`63_me_action_readiness_content_independent_qualification_v1_5.py`

## 结果

| 指标 | 结果 |
|---|---:|
| 总体一致率（70条） | 98.6%（69/70） |
| 人工标注INVITES_ACTION数 | 2 |
| INVITES_ACTION recall | 50%（1/2） |
| 人工标注DECLINES_ACTION数 | 0（本抽样中没有真实样本） |
| UNKNOWN precision（不误报） | 100%（68/68，零假阳性） |

**两个真实INVITES_ACTION例子**：
- "How would you suggest meeting people in person?" → 观察器正确识别（命中
  `(?:can|could|would)\s+you\s+(?:suggest|recommend)`分支）
- "Where can I go for support?" → **观察器漏判**，判成UNKNOWN。"where can I go/find for X"
  这种"请求具体资源/去处"的自然问法，目前不在`_INVITE_RE`任何一条分支里。

**DECLINES_ACTION专项检索**：在全部4452条真实seeker发言里用宽松关键词（"just listen"、
"don't want advice"等）额外检索，只命中4条，且逐条读完发现**全部是回顾性感谢/一般偏好描述**
（"they were... just listened"、"you have been helpful by just listening"），不是当前
回合明确拒绝建议的发言。**在这个真实语料里，找不到一条能作为DECLINES_ACTION正例的自然
发言**——不是观察器漏判，是这个具体、当面拒绝建议的言语行为在这份真实支持性对话语料里
本身极其罕见（可能不存在）。

## 结论

1. **精度是真实、可信的**：70条真实发言里零假阳性，观察器没有把日常倾诉、感谢、反思类
   发言误判成邀请或拒绝行动——符合模块自己"缺信号默认UNKNOWN，不强行归类"的设计目标。
2. **INVITES_ACTION召回率50%是真实的、不够高的**，但样本太小（真实正例只有2个）不能just
   据此重写正则——这正是这次session反复确认过的教训（`不继续扩正则到通过`，见
   `V15-MEAS-53`的修复要求）。已发现的具体缺口（"where can I go/find for X"这种请求
   具体去处/资源的问法）值得记录，不必现在就扩正则。
3. **DECLINES_ACTION在这份真实语料里没有可测的正例**——这不是这次验证的失败，是一个
   诚实、值得披露的真实边界：这个三态里的"拒绝"分支，主要防线不是"这个语料里常见"，
   而是"万一出现要接住"；真实频率看起来接近0。

## 对P2的建议

- 不必现在扩`_INVITE_RE`去覆盖"where can I go for X"这一类——先记录，等正式FIT数据
  的资格验证阶段积累更多真实未见样本后再判断是否值得改；
- 正式FIT的P2资格验证阶段应该用类似方法（真实语料+人工独立标注+零API），而不是只用
  已消费的开发构造行回放；
- `UNKNOWN`占比这么高（68/70）是符合预期的，不代表这个槽位没用——它的价值在于"少数
  真正明确的邀请/拒绝出现时不能漏、不能误报"，不是"每条发言都要归类"。
