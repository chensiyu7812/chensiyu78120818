# ME行动准备度——平衡、内容独立资格集的真实结果（2026-08-06）

## 为什么需要这一批（区别于脚本63）

脚本63的70条真实EvoEmo样本只有2条真实INVITES_ACTION、0条真实DECLINES_ACTION——只够做
真实频率/假阳性审计，不够估计这两个稀有类别的召回率。这次新建**平衡**集：3类×6种自然
表达模板×10个话题=180条，全部新写，不复制/改写EvoEmo或ES-MemEval原句。DECLINES用内部
构造的自然拒绝表达（因为EvoEmo结构上没有这类正例，见脚本63的语料级检索结果）。

脚本：`67_me_action_readiness_balanced_qualification_v1_5.py`。拆分卫生：每条同时标注
`semantic_family`（话题）、`template_family`（类别+表达形式）、`user_id`（每个话题一个
合成用户），脚本内程序化断言`(template_family, semantic_family)`组合不重复，不是口头
保证。70个真实样本继续只用于真实频率/假阳性审计，不参与这次的召回估计。

## 结果：精度依然干净，但召回率明显不够高——比脚本63那次乐观的50%/无法测更真实

| 类别 | n | recall |
|---|---:|---:|
| INVITES_ACTION | 60 | **50.0%**（30/60命中） |
| DECLINES_ACTION | 60 | **33.3%**（20/60命中） |
| UNKNOWN | 60 | 100%（零假阳性，跟脚本63一致） |

**INVITES_ACTION命中的3种表达**："What should I do about X?"、"Do you have any advice
for handling X?"、"Could you suggest a way to approach X?"——都是`_INVITE_RE`已经覆盖的
直接问法。**漏判的3种**："Is there something I could try for X?"、"Where can I go for
help with X?"（脚本63已发现的缺口）、"I'd be open to hearing an idea for dealing with
X."——都是间接、非疑问句式的邀请。

**DECLINES_ACTION命中的2种**："I don't want advice about X right now, I just need to
vent."、"Please don't tell me what to do about X, I just need you to listen."——命中
`_DECLINE_RE`里"do not want advice"和"just want to listen"的直接分支。**漏判的4种**：
"I'm not looking for suggestions..."、"I already know what to do..., I just want to
talk it through"、"Let's not jump to solutions...yet"、"I just need to vent..., not fix
it right now"——都是更委婉、不直接说"不要建议"的自然拒绝方式。

## 诚实结论

这个观察器的真实形状是：**精度高、召回率中等偏低**——不是脚本63那次(70条真实样本仅2/0个
正例)看起来的"98.6%一致率"，那个数字主要是UNKNOWN占绝大多数带来的。真实的邀请/拒绝识别
能力，面对自然、多样的表达时，大约只能捕捉到一半（INVITES）到三分之一（DECLINES）。

**这不构成阻塞**，因为经过这一轮的候选层分责修复（见
`PM_V1_5_V5_3_CANDIDATE_LAYER_RESPONSIBILITY_SPLIT_20260806_ZH.md`），这个信号现在只是
Step1的一个输入特征，不再替任何组件做"要不要考虑候选"的硬性裁决——召回率不足只意味着这个
特征在真实召回不到的场景下提供的信息量是"不知道"（正确地落回UNKNOWN，不是错误地判成
DECLINES），不会导致候选被错误地拒之门外。

## 交给leader的问题

是否需要现在就扩大正则覆盖这些间接表达，还是先如实记录、留到正式FIT阶段积累更大规模
真实/合成数据后再做——这是一个成本/收益判断，不在本轮单方面决定。
