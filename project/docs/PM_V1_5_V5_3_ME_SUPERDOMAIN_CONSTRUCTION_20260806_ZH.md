# 新ME superdomain构造：真实密度、真实compiler/Rank-1双重验证（2026-08-06）

## 方法

脚本`68_build_me_superdomain_v1_5.py`。8个全新语义族（job_transition/chronic_pain/
parenting_conflict/breakup_recovery/exam_anxiety/career_change/housing_move/
friendship_rift），每个族4种current-turn变体（positive/current_redundant/
explicit_decline/goal_mismatch）= 32个构造state。每个state的ME目录有7条真实候选
（不是旧468-card语料固定的2条）：intended-target（用本session已验证通过编译器的模板
"When dealing with X, I <action>, and it helped me <benefit>."）、同主题不同事件的干扰项、
同主题无结果项、纯背景CONTEXT_EVENT项、以及3条完全不相关话题的干扰项——密度和真实EvoEmo
（用户级37-109条）方向一致，不是旧语料的稀疏目录。

**intended-positive一律不手工塞入**：每个state都真的跑一遍`me_subtype_hints()`+
`discover_final_typed_memory_candidates()`拿到真正的Rank-1，再用
`compile_atomic_reusable_outcome()`核实是否通过编译器、Rank-1是否等于设计意图的target——
两项都满足才算verified，任何一项不满足直接丢弃，不做二次加工。

## 结果：32个构造state，14个双重验证通过（43.75%），比预想的更能说明问题

| negative_kind | 构造数 | 验证通过 |
|---|---:|---:|
| positive（应该开） | 8 | 4 |
| current_redundant | 8 | 4 |
| explicit_decline | 8 | 4 |
| goal_mismatch | 8 | 2 |

**第一版（当前turn文本泛泛地说"could use an idea"）：positive类0/8全部失败**——排查后
发现同主题干扰项（"I also tried a different approach first..."）系统性地压过intended
target，赢得Rank-1。**改进（把当前turn文本改成跟target自己陈述的具体收益词面对齐，
比如"chronic pain"族用"...want to finally figure out my actual triggers"直接呼应target
的"...it helped me figure out my actual triggers"）后，positive从0/8提升到4/8**——
有真实提升，但依然接近一半的state，即使query已经刻意贴合target的具体措辞，同主题干扰项
仍然赢得Rank-1。

## 这不是构造失败，是这次session反复发现的同一个根因在受控条件下的再次复现

抽查一个失败案例（chronic_pain）：query含"...figure out my actual triggers"，跟target
"...it helped me figure out my actual triggers"逐字重合；但排序仍然选中了干扰项
"I also tried a different approach first, and it helped me realize what didn't work"。
说明当前`match_level`/`lexical_score`的打分機制，在**两个候选都合法、都同主题**时，
不能稳定地按"跟当前具体需求最贴合的那一条"来排序——这正是`V15-DATA-79`/`V15-MEAS-55`已经
指出、但当时只在真实EvoEmo/训练数据里观察到的现象，这次用**全新构造、query已经刻意对齐**
的干净数据把它复现了一遍，说明这不是训练语料脏或话术不自然的问题，是排序机制本身在候选
密度提高、同主题竞争出现时的真实局限。

## 诚实的规模披露

这次只构造了32个state（8家族×4变体），远低于运行手册4.2节"每组件至少128个独立group"的
正式下限——这一轮零API、时间有限，没有勉强凑到128个只为了看起来"达标"。14个双重验证通过
的state已经是真实、可用的ME superdomain种子（`outputs/pm_v1_5_v5_3_me_superdomain_v1/
states_verified.json`），可以在扩大规模时复用同一套构造+双重验证方法论；18个未通过的
state不是"脏数据"，是排序鲁棒性问题的真实证据，原样保留在`states_all.json`里供进一步
分析，不删除。

## 交给leader的问题

扩大到128+规模前，需要先决定：是继续用"查询措辞对齐target"这种构造技巧去人工提高通过率
（治标），还是先解决排序在同主题竞争下的鲁棒性问题本身（治本，`V15-MEAS-55`已经提过这个
需要算法设计决策）——这次的43.75%通过率说明光靠构造技巧，天花板可能不高。
