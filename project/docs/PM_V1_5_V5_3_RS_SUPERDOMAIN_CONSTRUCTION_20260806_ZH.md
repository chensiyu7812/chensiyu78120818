# 新RS natural-language superdomain构造（2026-08-06）

## 方法

脚本`69_build_rs_superdomain_v1_5.py`。直接回应RS更正发现的根因：V3构造的RS
"user turn"是指令式口吻（"Please ask exactly one focused question...do not add
another support move"），不是自然求助者语言。这次为6卡系统的5个语义机会家族（AM01/
AM02/AM04/AM05/AM10；AM14在真实runtime里已不作为机会，见`v1_5_strategy_rag_runtime.py`
注释）各写一条全新、自然的用户发言，加2个边界负例。

全部通过这一轮新增的`rs_mechanical_candidate_pool()`（候选池只在机械条件下清空）+
`rs_transparent_rule_top1()`（明确标注的transparent-rule基线，不是Step1本身）真实验证，
不是断言标签。

## 结果：5个语义家族4/5命中，边界负例2/2正确

| 家族 | 结果 |
|---|---|
| AM01 open_expression | 命中 |
| AM02 focused_clarification | 命中 |
| AM04 paraphrase_check | 命中 |
| AM05 grounded_validation | **未命中** |
| AM10 offer_one_optional_micro_step | 命中 |

**AM05未命中的真实原因**："I feel so overwhelmed by all of this"里"feel"和
"overwhelmed"之间多了一个"so"，`_FEELING_RE`只认`i\s+feel\s+overwhelmed`这种紧邻写法，
不允许中间插入副词——跟这次session早些时候在ME正则里修过的"也/just/really"副词间隔缺口
是同一类问题，只是这次出现在RS。**没有为了让它命中而改写句子**，如实记录为一个真实、
小的正则缺口，是否要修交给leader决定。

**边界负例**：`explicit_stop`正确机械hard-off（候选池清空，reason=`explicit_stop`）；
"已上一轮执行AM02"的重复负例正确地只排除AM02（候选池剩5个，不是清空），且因为这句话本身
只对应AM02的语义条件、其余4个家族都不匹配，transparent-rule正确返回"无信号"（不会为了
凑结果误选到别的家族）。

## 跟候选层分责的关系

这次全部5个家族case，就算语义条件没命中（如AM05这次），候选池依然是完整的6个move
（或排除已执行项后的5个）——不会像V3旧构造那样直接清空。语义命中与否只影响
`rs_transparent_rule_top1()`这个**明确标注的baseline**要不要选出一个偏好，不影响候选
本身是否存在，符合这一轮"语义观察只做特征/baseline，不再替Step1做最终裁决"的目标。
