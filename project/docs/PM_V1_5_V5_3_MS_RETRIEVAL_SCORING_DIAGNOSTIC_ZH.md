# V5.3 检索打分诊断（MS/MP/ME）——探索性发现，未经正式资格赛确认

**2026-08-06更新：本文档两处结论已被后续核查修正/取代，读者请先看这里：**
1. **"生产环境=裸lexical_score"这个前提是错的。** 真实P2/runtime selector是
   `v1_5_candidate_discovery.discover_final_typed_memory_candidates`（内容词过滤→粗粒度
   content-match tier→typed tier→lexical仅tie-break），不是本文档下面测的裸lexical。用正确
   baseline重新做的同栈资格赛见`PM_V1_5_V5_3_MS_QUALIFYING_TRIAL_FINDINGS_20260806_ZH.md`——
   结论方向没变（BGE配对显著更优，p≈0.0019），但具体数字不同：真正的信号在"完全不相关"这个
   最坏情况上（生产40% vs BGE 4%），不是"exact fit到70%"。
2. **"EvoEmo只有4条ME候选"是错的。** 这个数字来自我自己用regex直接扫原始对话、绕开了真正的
   `evoemo.build_evo_memory()`编译器。实测单个用户通过`build_evo_memory()`产出**109条ME
   候选**（不是全库4条）。ME真正的问题（覆盖率而非候选量）需要重新核实，本文档第3节的ME结论
   暂不可信，待后续更新。

状态：**探索性诊断，不是已审计结果，不是正式资格赛**。原始产出目录
`outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1/`按项目artifact政策gitignore，
未进git；本文档是sanitized summary，逐文件SHA256见同目录
`PM_V1_5_V5_3_MS_RETRIEVAL_DIAGNOSTIC_ARTIFACT_MANIFEST_20260805_ZH.json`。
生成脚本：`scripts/v1_5/42_diagnose_ms_retrieval_scoring_v1_5.py`、
`scripts/v1_5/43_diagnose_ms_rank1_on_real_states_v1_5.py`、
`scripts/v1_5/44_diagnose_mp_me_rank1_on_real_states_v1_5.py`。

## 0. 四处需要先纠正的表述（在早期讨论中出现过，此处按纠正后的口径重写）

1. **不称当前正式检索器为"裸lexical"**：V5.2正式使用的是typed candidate selector
   （候选发现→owner/version/eligibility检查→typed candidate构造→打分排序→exact Rank-1执行的完整
   链路，见`src/metacom_pm/v1_5_typed_resource_adapter.py`、`retrieval.py`）。本诊断只观察到其中
   **排序打分这一步**（`retrieval.MemoryRetriever.retrieve()`）使用`text.lexical_score`（无IDF、
   无停用词的词频余弦）。这是对打分函数的诊断，不是对整个typed selector的诊断。
2. **候选池构造未显式验证因果边界，不代表已发生结果泄漏**：脚本43/44构造MS/MP/ME候选池时，没有
   显式按`created_session < current_session_index`过滤（正式协议的因果边界定义见
   `scripts/v1_5/26l_materialize_v5_1_t5_surfaces_v1_5.py`）。这是候选池合同不精确、存在潜在泄漏
   风险，但对138个状态的最终Rank-1逐条核查后：**lexical选中未来/当前会话0次，BGE选中未来/当前会话
   0次**——没有已发生的结果泄漏。正式资格赛必须先修正候选池构造，本诊断的方向性结论不能因这个已知
   缺口被推翻，但也不能被拿来当作"已经过因果边界审计"。
3. **不把MS检索称为"已确认的唯一/主导瓶颈"**：本诊断只能说"候选排序质量是一个已观察到的、可能重要
   的瓶颈来源"；根据其他讨论，最终质量差异还可能来自Step1价值判断、Step2执行、开场白质量、评审噪声
   等多个环节共同作用，这些需要分层责任账本（V5.3计划§2.6）才能拆开，本诊断没有做那个拆分。
4. **数字口径**：下面的百分比来自本次会话作者对samples的单人单遍人工判读，**没有保存成可审计的逐条
   标签表，没有第二标注者，没有裁决**——这正是被指出的方法学缺口。数字只作探索性方向参考，不能写成
   "已审计事实"，正式结论需要走V5.3计划§7.4定义的三轮正式语义评测流程（一次primary、20%独立重叠、
   一次集中裁决）。

## 1. ES-MemEval代理任务上的打分函数对比（零新增API/模型成本）

复用ES-MemEval已冻结的BGE-M3 embedding与gold标签（341道题、同一批候选session池），对当前生产用的
`lexical_score`、已有但未用于排序的`content_word_match_level`、以及BGE-M3做了头对头对比：

| 打分方式 | recall@4 | nDCG@4 |
|---|---:|---:|
| lexical_score（当前用于排序） | 0.4034 | 0.3121 |
| content_word_match_level（现有代码，仅作Step1特征，未用于排序） | 0.4911 | 0.3691 |
| BGE-M3（ES-MemEval官方已冻结数字） | 0.6963 | 0.5997 |

这部分数字是机器计算、有gold标签支持的，不是人工判读，相对可信；但候选文档是ES-MemEval的整段原始
对话，跟MS实际候选（session摘要）表示形式不同，只能作为同一用户池、同一类检索任务的旁证。

## 2. 真实138个production state上的观察（单人单遍人工判读，未经二次核验）

用生产查询构造函数（`seeker_only_context_query`）和更接近真实候选表示的文本
（`data/external/evo_emo.json`每session的`summary`字段）在全部138个state上跑了lexical/BGE的
Rank-1对比：两者选择不一致的比例为86.2%（119/138）。

对50个随机state做了单人判读（口径：精确对题/同域但细节有误/完全不搭边），**未保存逐条标签表**：

| | 精确对题 | 同域细节有误 | 完全不搭边 |
|---|---:|---:|---:|
| lexical | ~36% | ~32% | ~32% |
| BGE-M3 | ~70% | ~28% | ~2% |

方向性观察（不是确认结论）：候选池明显噪声较高时，换用BGE-M3打分似乎能大幅降低"完全不搭边"的比例。
是否成立、成立到什么程度，需要走正式资格赛重新测量。

## 3. MP / ME：不是同一个问题，数据也不支持同样的观察

- **MP**（basic_info，每用户6条静态短事实）：抽查实际分数确认lexical对典型查询给出全部候选0.0分
  （无字面重合，之前样本里"nationality: American"反复出现是脚本tie-break的字母序伪影，不是真实
  lexical信号）；BGE则表现出另一种退化（约23/30样本Rank-1选中"age: NN"，与话题无关）。这不支持
  "换BGE解决MP问题"——候选池小、内容短，两个打分函数在这个任务形状上都不可靠，需要另外的方案（比如
  Step1规则触发），不是打分函数选型问题。
- **ME**：用生产环境同一个`compile_atomic_reusable_outcome`函数扫全部18用户对话，只抽出4条候选、
  覆盖3个用户，138个state里只有6个凑够≥2候选可比较。样本量不足以得出任何lexical/BGE结论。

## 4. 下一步

正式结论只能来自V5.3计划里定义的、与生产链路（typed candidate selector、`created_session <
current_session_index`因果边界、typed query构造、tie-break、`discover_final_typed_memory_candidates`）
完全一致的"精确同栈MS资格赛"，且需要预注册标签体系（exact_fit/related_but_wrong_detail/unrelated/
wrong_owner_entity/stale_conflicting）、独立重叠评审与集中裁决、逐状态可审计JSONL——不是本诊断这种
单人单遍抽样。版本治理完成前不启动该资格赛。
