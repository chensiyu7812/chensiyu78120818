# MP/MS候选级特征域对比（2026-08-06，路线图"MP/MS/RS域对比审计"第一部分）

之前只对ME做过训练域vs EvoEmo域的候选级对比（脚本60/61）。这次用同一方法论补上MP和MS
（脚本`64_mp_ms_contribution_slot_domain_comparison_v1_5.py`），训练域这次直接用修正后
正确的当前V3构造（`pm_v1_5_v3_effect_blueprint_v1`，2026-08-03），不再犯之前用错旧
468-card backend的错误。

**已知限制（如实披露）**：V3导出只有exact Rank-1候选，没有完整Top-k候选池，所以训练域
一侧的`topk_*`聚合字段退化成n=1；本报告只依赖`rank1_*`和语义槽位字段，这些字段直接用
唯一候选计算，不受此限制影响。EvoEmo一侧走的是完整真实检索流程，`topk_*`是真正意义上
的聚合。

## MS：没有发现ME那种严重的构念合格率域间差距——好消息

| 特征 | 训练域(n=160) | EvoEmo域(n=138) |
|---|---:|---:|
| `has_specific_prior_observation`（候选通过`compile_atomic_session_observation`） | **100%** | **100%** |
| `continuity_request` | 0% | 0% |
| `current_redundant` | 69.4% | 30.4% |
| `rank1_injected_tokens`（均值） | 26 | 46 |

MS最核心的合格率指标——候选是否通过编译器——两个域都是100%，跟ME`past_action_result`
训练域0%对比EvoEmo域78.3%的严重域间差距完全不是一回事。**MS目前没有发现ME那种"训练语料
自己都造不出合格候选"的问题**。

`continuity_request`（这次新加的、还未独立验证过的正则，专门识别"上次说过"、"还是同一个
问题"这类连续性请求）在两个域都是0%——这是一个真实、诚实的负结果：这个信号目前设计得太窄，
两个域都测不出来，不是域间差距，是这个具体槽位现在对MS几乎没有贡献，需要重新设计或接受
它暂时无效。

token成本差距（26 vs 46，约1.8倍）方向跟ME的token差距（约2.5倍）一致：EvoEmo真实session
摘要普遍比合成语料写得更长，这是内容风格的真实差异，不是bug。

## MP：发现一个结构性、比"信号太窄"更根本的问题——EvoEmo里几乎没有MP_PREFERENCE这个构念

| 特征 | 训练域(n=160) | EvoEmo域(n=27，138个state里只有27个MP被检索为Rank-1) |
|---|---:|---:|
| `profile_goal_needs_advice_or_arrangement` | 0%（80/80） | 0%（27/27） |
| `preference_applies_to_response_act` | 100%（80/80） | **n=0，无法计算** |
| `current_redundant` | 33.1% | 0% |
| `rank1_injected_tokens`（均值） | 28 | 6 |

**`profile_goal_needs_advice_or_arrangement`两个域都是0%**：这是MP复用的同一个
`explicit_advice_welcome`信号（跟ME的`current_action_invitation`底层是同一族正则）。
两个域都测不出来——**这不是ME独有的问题，是这整族"是否明确邀请建议/行动"的信号，在MP和
ME两个不同组件上都表现出同样的过窄**，是一个跨组件的、系统性的信号校准问题，不是ME单独
的构念模糊。

**`preference_applies_to_response_act`在EvoEmo一侧样本量是0，不是脚本bug**：直接读取
`build_evo_memory()`产出的全部126条真实MP条目（覆盖全部18用户），逐条核对，**全部是
"Age: 35"、"Education: Bachelor's degree"这类field-value事实，没有一条是"prefers X
风格/格式"这类偏好陈述**。换句话说：**EvoEmo这份真实外部数据的MP来源里，`MP_PREFERENCE`
这个子构念在结构上就不存在，不是没被检索到，是源头数据里根本没有**。而训练语料里
`MP_PREFERENCE`占了一半（80/160）。

这跟账本已有的`V15-MEM-54`是同一个根因的进一步量化：不只是"EvoEmo的Job/Education大多
应该判off"，是**EvoEmo这份数据本身没有偏好类MP内容可供检索**，所以训练语料里为
`MP_PREFERENCE`投入的一半数据量，在这份外部集上永远没有真实内容可以验证——不是"泛化
效果差"，是"这个子构念在这份外部数据里没有可比对象"。

## 结论与建议

1. MS的合格率没有域间差距，比ME乐观，暂不需要额外修复；
2. `explicit_advice_welcome`族信号（ME的`current_action_invitation`、MP的
   `profile_goal_needs_advice_or_arrangement`）在两个组件、两个域全部测不出来，应该当
   一个跨组件的共享问题处理（重新校准或换方法），不要在ME和MP各自重复修一遍；
3. MP_PREFERENCE这个子构念在EvoEmo没有真实可比数据，需要如实披露为"该子构念的外部验证
   范围受EvoEmo数据本身限制"，不能等着看它在EvoEmo上"泛化"，因为源头就没有内容；
4. RS的域对比审计还没做，RS结构跟MP/MS/ME不同（没有记忆目录），需要单独设计方法。
