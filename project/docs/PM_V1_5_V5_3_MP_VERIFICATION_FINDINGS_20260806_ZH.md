# MP 同栈核实——发现（单标注员/单次代码核查，探索性）

状态：**探索性，同栈可信（用的是真实`discover_final_typed_memory_candidates`选择器和真实
`build_evo_memory`候选池），但只跑了一遍，没有第二独立核查者**。目的是回答"MP的真实问题是什么"，
不是给MP做资格赛（MP不存在"要不要换BGE"这个问题，见下）。

**2026-08-06更新：第4节里提到的字段名前缀修正已经实现并验证，不再只是"未来方向"**。见
`src/metacom_pm/v1_5_candidate_discovery.py`的`_mp_match_document()`（commit `8812492`）：
在MP的内容匹配阶段把"Job: office worker"这类字段值前缀（"Job: "）先剥离，只用值本身
（"office worker"）参与匹配，`item.text`本身（渲染用）不变，MS/ME完全不受影响。用真实138状态
面板重跑`scripts/v1_5/46_mp_verification_same_stack_v1_5.py`确认修复生效：选中≥1条候选的
状态数从55降到33/138；p12("office worker")、p14("freelancer")、p18("software engineer")
三个此前100%靠"job"这个词撞词的用户，现在全部正确选出0条候选。新增2个回归测试
（`test_mp_label_prefix_does_not_create_false_content_match`、
`test_mp_value_content_still_matches_after_label_stripped`）锁定这个行为。

**2026-08-06二次更新：(b)类"字段值内通用填充词"问题也已经修复**（commit `5edcf0d`）。
`_mp_match_document()`现在还会剥离字段值里的括号状态标注（如"(in progress)"），先查过
范围再动手：全部18个用户126个`basic_info`字段里，只有p11这1条带括号标注，所以这是一个
针对已观察到模式的完整修复，不是扩大停用词表（那会有误伤MS/ME真实信号的风险）。真实重跑：
p11的非零选中从9降到3（精确对应之前定位的6次"progress"撞词全部消失），整体MP非零选中状态
从33降到27/138。**第4节里"p14张冠李戴"这个案例也确认是(a)类字段名前缀问题的另一种表现，
不是独立的owner/entity机制**——(a)类修复后p14已经自然降到0次选中，不需要额外的
`field_type`/`field_value`/`owner`结构化改动。

复现：`scripts/v1_5/46_mp_verification_same_stack_v1_5.py`，产出：
`outputs/pm_v1_5_v5_3_mp_verification_v1/mp_selection_all_states.jsonl`（gitignore，未进git，
不含私有对话原文之外的额外私有信息，字段本身就是`current_user_text`等已在其他资格赛里处理过的
同类数据）。

## 0. 结论先说：MP不是排序/embedding模型问题，是"粗粒度内容匹配对通用词免疫力不足"的问题

跟MS不同，MP的候选池极小（每用户约6-8条`basic_info`静态短事实，如"Job: X"/"Education: Y"/
"Location: Z"），`build_evo_memory()`确认EvoEmo从不设置`mp_subtype`元数据，所以选择器里专门给
偏好类MP做的`preference_scope_match_level`分支从不触发——MP排序退化成跟MS结构相同的"内容匹配
层级(0/0.5/1) + typed tier + lexical tie-break"，只是候选池小得多。**换嵌入模型（BGE）对这个
问题没有直接意义**：下面会证明真正的失败模式不是"两个候选打分打得差不多、需要更好的语义排序"，
而是"唯一能超过匹配下限的候选，是靠通用词/字段名本身撞上的，跟这条静态事实是否真的相关无关"。

## 1. 总体分布：138个状态里，83个（60%）MP什么都选不出来

| | 状态数 |
|---|---:|
| MP候选池里没有任何一条达到内容匹配下限（选中0条） | 83/138 |
| MP选中≥1条候选 | 55/138 |
| 硬约束（`describe_memory_candidate`因果断言）触发次数 | 0/138 |

这本身就说明MP大部分时候是"结构性沉默"——6-8条极短静态事实很难跟一段情绪化的诉说文本产生哪怕
一个内容词的字面重合，这不是bug，是符合"MP候选池小"这个已知形状的预期行为。真正值得深挖的是
剩下55个状态、68次具体选中（部分状态选中不止1条）里发生了什么。

## 2. 对68次具体选中，逐条用真实`final_typed_content_match_level`核实匹配到底靠哪个词

方法：对每次选中，把MP静态事实拆成"字段名前缀"（如"Job"）和"字段值"（如"office worker"），
用真实生产分词函数`final_typed_content_words`（跟`discover_final_typed_memory_candidates`用的
完全一样的函数，不是我自己重新实现的近似版）分别跟查询做交集，看重合词到底来自字段值（真实主题
信号）还是字段名前缀/字段值里的通用填充词（跟事实本身是否相关无关的偶然撞词）。

| 匹配来源 | 次数 | 占比 |
|---|---:|---:|
| 字段值里的词有真实重合（可信的主题信号） | 44 | 64.7% |
| 仅靠字段名前缀本身（如"job"）或字段值里的通用填充词重合，字段值实质内容无重合 | 24 | 35.3% |

**这24次里，5个用户占了全部**：

- **p12（"Job: office worker"，6/6次全部如此）**：查询里只要出现泛泛的"job"这个词（不管是不是
  在说这个用户自己的工作），就会跟"Job: office worker"里的字段名前缀"Job"重合，匹配层级直接
  记为0.5——跟"office worker"这个具体职业值本身有没有出现在对话里完全无关。
- **p14（"Job: freelancer"，6/6次全部如此）**：同样的字段名前缀撞词；抽查发现查询原文其实是在
  说"我伴侣的工作"（"my partner's job"），不是用户自己的自由职业身份，字段名前缀撞词导致"张冠
  李戴"——检索选中了正确类别、错误归属对象的事实。
- **p18（"Job: software engineer"，6/6次）**、**p15（"Job: graduate student"，6/7次，另1次是
  字段值真实重合）**：同一模式。
- **p11（"Education: high school (in progress)"，9次选中里的6次）**：这次不是字段名前缀，而是
  字段值本身里的通用填充词"progress"——EvoEmo里大量情绪支持类回合会说"healing is not a linear
  process"、"I'm proud of myself"、"process my feelings"等，这些短语里带出的"progress"跟
  静态事实文本"(in progress)"里的"progress"发生了字面重合，即使对话内容明明是在谈"跟John的
  偶遇"、"参加聚会"这类完全跟在读高中与否无关的话题。

## 3. 这跟MS的问题是同一类，但换个位置发作

MS资格赛已经确认：生产环境的粗粒度content-match会被"stress/anxious/work"这类高频治疗话术词
骗到选中完全不相关的旧记录（40%完全不搭边 vs BGE的4%）。这里的MP核查证明**同一个粗粒度0/0.5/1
匹配层级机制，在候选池换成"字段名:字段值"这种结构化短事实时，会以两种新形式暴露同一个弱点**：
(a) 字段名前缀本身（Job/Education/Location）被当成普通内容词参与匹配，制造"提到职业类通用词
就命中这条职业事实"的假阳性；(b) 字段值里偶然带的通用词（"progress"）被泛用语境词撞上。

## 4. 结论：MP需要的是Step1候选表示/资格判断修正，不是切换embedding模型

- 换BGE不能解决(a)类问题：字段名前缀撞词是字面拼写层面的问题，BGE语义相似度大概率同样会因为
  "job"这个词本身在文本里出现而给出较高相似度（语义上"job"确实和"job"相关，语义模型不会区分
  "字段名"和"内容"）。
  这需要在检索之前的候选渲染阶段把字段名前缀从匹配用文本里剔除——**(a)类已经这样修了，见文首
  2026-08-06更新**。
- (b)类问题（"progress"撞词）**已修复**（见文首二次更新）：在MP专属的匹配文本处理里剔除候选
  值本身携带的括号状态标注，没有动全局`CONTENT_MATCH_IGNORED_WORDS`，不影响MS/ME的真实信号。
- 这印证（而不是推翻）Codex此前"MP的问题在Step1资格判断/渲染方式，不在检索排序"的判断，现在有
  了具体、可复现、代码核实过的量化证据：**68次选中里35%是通用词而非事实内容驱动的假阳性**，
  且集中在5个可点名的用户/事实上，不是随机噪声。
