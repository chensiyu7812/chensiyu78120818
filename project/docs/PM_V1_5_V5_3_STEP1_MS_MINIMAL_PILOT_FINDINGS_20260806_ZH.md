# Step1最小试点：MS单组件，真实生成+真实judge打分（2026-08-06）

延续MP最小试点（`PM_V1_5_V5_3_STEP1_MP_MINIMAL_PILOT_FINDINGS_20260806_ZH.md`）同样的模式，
用户授权"扩展到ME/MS/RS"后执行。脚本`53_step1_ms_minimal_pilot_v1_5.py`，逻辑跟MP试点完全
一致（复用同一个`v1_5_mvp_judge.py`真实judge instrument），唯一区别：测的是"带MS（用已经
定下来的生产方法BGE-M3）vs 不带MS（M0+R0）"，不是"带不带MP"。MS在138/138真实state上都会
命中候选（跟ME形成鲜明对比，见下面ME的部分），所以样本没有稀缺性问题，n=12覆盖全部12个用户。

## 意外发现一个真实、影响面比MS本身更大的bug：M0+R0基线本身会编造证据引用

第一次真实批跑（12个state）就发现：**"不带MS"这一侧（M0+R0，也就是"这一轮不需要任何记忆
组件"这个最基础、最常见的场景）里，6/12（50%）触发`TRACE_REFERENCES_UNAUTHORIZED_EVIDENCE_
ID`guard，其中4个直接兜底成8个字的通用句子。** 这条guard是在检查`used_evidence_ids`里
有没有不存在的id——但M0+R0根本没给任何证据，所以任何非空的`used_evidence_ids`都会触发。

去代码里直接复现（同一个真实state，同样的构造方式，跳过整个pilot管线单独调用）：模型返回
`used_evidence_ids: ["user_message_1"]`——一个自己编出来的id，不是记忆事实的幻觉，是把
"用户当前这句话"当成了一个可引用的证据条目。根源：`evidence_aware_generation_messages()`
从来没告诉过模型"`used_evidence_ids`只能是下面'Background facts'列表里的id，用户当前发言
不是可引用的证据"——M0+R0分支的系统提示词只说了"没有记忆证据"，但schema本身仍然要求每次
都填这个字段，8B模型倾向于"我确实用到了用户这句话的信息，所以应该填点什么"，而不是理解成
"这个字段只对应那份专门列出来的证据清单"。

**这条bug影响的不只是MS这次测试**——它是M0+R0这个核心路径本身的问题，任何"不带X"的对照组
（MP试点的、RS试点的、以及V5.3真实生产中任何一次PM判断"这轮不需要注入记忆"的真实调用）都会
撞上同一个问题。`v1_5_v5_3_typed_response_program.py`的docstring里其实早就写明"这个真空
路径从来没被真实测试过"（2026-08-06当天早些时候修M0+R0悬空引用问题时留的注释）——这次是
第一次真实跑到这条路径，也是第一次发现这个新问题。

**修复**：在`evidence_aware_generation_messages()`里加了一条明确指令："used_evidence_ids
must only contain evidence_id values copied exactly from the 'Background facts' list
below...never invent an id like 'user_message_1'...if there is no Background facts list
...used_evidence_ids must be an empty list."——同一个真实state、同样的seed，修复前后对照：
修复前`used_evidence_ids: ["user_message_1"]`，修复后`used_evidence_ids: []`，直接复现
确认。

**修复后重新跑了一遍全部12个state**：M0+R0侧的触发率从6/12降到4/12——**明显改善，但没有
完全消除**。残余的4例（p8/p14/p16/p17）单独复现时用不同种子有时能拿到干净结果，说明这是
模型层面残留的、概率性的倾向，不是这次修复没生效的迹象；进一步降到0可能需要更强的手段（比如
prompt里加一条"错误示例"的few-shot，或换模型），这次没有再往下投入，按"明显改善、非完美"
如实记录。

## MS本身的结果（干净、修复后的批次）

12/12都拿到了有效的judge判定（第一次批次因为一次真实503+这个bug连带，丢了1个）：

| 结果 | 数量 |
|---|---|
| A（带MS更好） | 6 |
| B（不带MS更好） | 2 |
| tie | 4 |

**带MS这一侧修复后全部12个state都是clean（0次兜底）**，跟不带MS这一侧残余的4/12兜底形成
对比——这本身也是个真实信号：**给了具体、贴合上下文的MS证据后，模型的输出反而更稳定**，不是
只有"更详细"这一个优点。

6/12真实偏向"带MS更好"，2/12偏向"不带"，4/12 tie——方向上支持MS组件本身有正面效果，跟
qualifying trial（BGE vs 词面，138/138，bge胜率70.7%）的方向一致，但这次测的是不同的问题
（"有没有MS"而不是"哪种MS候选更好"），是一个独立的、新的正面证据，不是重复验证同一个结论。

## 风险judge结果：没有发现MS特有的新风险模式

复查了risk judge在两侧标出的violation，跟MP试点一样，多数`unsupported_personal_claim`/
`stale_or_conflicting_evidence_use`要么是judge没拿到别名映射导致的已知假阳性（跟p11案例
同一类问题，MP试点文档里已经记录过），要么是评论宽泛没有指向MS证据本身的具体内容。没有找到
新的、MS特有的、需要单独处理的风险模式。

## 结论

1. **发现并修复了一个真实的、影响全组件的核心bug**（M0+R0基线编造证据引用），这个发现的
   价值超出了MS试点本身——直接影响"5.3能不能确保正确学出来"这个问题，因为M0+R0是最基础、
   出现频率最高的场景之一。已修复，效果明显但不完美，如实记录。
2. **MS组件本身**：干净重测后6A/2B/4tie，方向上支持"有MS比没有好"，且带MS的一侧输出更
   稳定（0次兜底 vs 不带MS一侧4次）。
