# W6：ME Rank-1 reranker零API资格比较（2026-08-06）

## 范围声明

本文件只回答一个问题：在同一批已存在的32个ME superdomain state（脚本68，8家族×4
current-turn变体，每state同一7-candidate目录、同一owner、同一因果边界）上，三种**预先
写定、读结果前不再改**的排序方法谁的Rank-1更准。**不物化正式P2蓝图，不生成任何paired
回复，不训练PM，不读取quality/risk/outcome。**

## 三种方法（代码：`src/metacom_pm/v1_5_v5_3_semantic_me_retrieval.py`，opt-in，未接入
production；对比脚本`scripts/v1_5/70w_qualify_me_reranker_v1_5.py`）

1. **current_production**：`discover_final_typed_memory_candidates()` + `me_subtype_hints()`
   （词面内容匹配层+typed_tier二级排序），全程使用的真实候选发现路径；
2. **bge_m3**：BGE-M3 cosine，对全部7条候选直接排序，不做任何预过滤；
3. **hybrid_filter_then_bge**：先只保留strictly-prior（`created_session < session_index`）
   且通过`compile_atomic_reusable_outcome`的候选（owner由单用户候选池天然保证，
   `MemoryItem`本身不带owner字段），再对幸存候选做BGE-M3排序；tie按
   （`created_session`降序、`memory_id`降序）确定性处理。

三者在读取任何结果前已经写死在代码里，运行期间没有再加第四种方法或调权重。

## 结果：没有一种方法明显更好，两阶段hybrid反而在主指标上最差

| 方法 | intended target exact Rank-1 | Rank-1 compiler-valid率 | 完全无关率 | owner/future违规 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| current_production | 46.9%（15/32） | 96.9% | 0.0% | 0/0 | 0.001s |
| bge_m3（纯，无过滤） | 46.9%（15/32） | **28.1%** | 0.0% | 0/0 | 0.307s |
| hybrid_filter_then_bge | **31.3%（10/32）** | 100.0% | 0.0% | 0/0 | 0.159s |

**两两对比（是否命中intended target的win/loss/tie）**：

- current_production vs bge_m3：6胜6负20平——完全打平，没有谁更优；
- current_production vs hybrid：**5胜0负27平**——production全面不劣于hybrid，hybrid
  没有任何一个production答错而它答对的case；
- bge_m3 vs hybrid：7胜2负23平——纯BGE反而比"更聪明"的两阶段方案更准。

## 为什么两阶段方案反而更差：一个真实、值得记录的反直觉发现

`hybrid_filter_then_bge`的compiler-valid率是100%（设计上必然），但这不等于排序更准——
过滤掉context-only/no-result候选后，**剩下的候选往往是target和同主题干扰项两个都合法、
都compiler-valid的候选**，这时候BGE-M3语义相似度一样分不清"哪一条是当前具体需要的那个
动作"，跟production的词面排序面对同一个问题时表现差不多（甚至更差）。**纯BGE-M3
（不过滤）compiler-valid率只有28.1%，是因为它经常把简短的CONTEXT_EVENT背景句排到
Rank-1**——语义上"是同一个话题"，但没有能力区分"这条有没有具体动作和结果"，这正是当初
需要compiler过滤的原因；但一旦真的过滤，剩下的合法候选之间BGE-M3又没有优势。

## 按family拆分（8族×4个state，不是32个独立用户，只做描述性展示）

| family | current_production | bge_m3 | hybrid |
|---|---:|---:|---:|
| breakup_recovery | 0.25 | 0.50 | 0.00 |
| career_change | 1.00 | 0.50 | 0.75 |
| chronic_pain | 0.00 | 0.25 | 0.00 |
| exam_anxiety | 1.00 | 0.50 | 0.75 |
| friendship_rift | 0.75 | 0.50 | 0.50 |
| housing_move | 0.00 | 0.50 | 0.00 |
| job_transition | 0.75 | 0.50 | 0.50 |
| parenting_conflict | 0.00 | 0.50 | 0.00 |

production呈现明显的两极分化（要么0%要么75-100%，取决于该family的query措辞是否巧合
贴合target词面），bge_m3在几乎所有family上稳定在50%左右（不精但更均匀），hybrid在
production表现好的family上接近但不超过production，在production表现差的family上同样差。

## 延迟/内存：MS已加载同一BGE时，ME的增量成本只是encode调用本身

BGE-M3模型加载耗时4.77秒（一次性，CPU）。这次运行里`bge_m3`和`hybrid_filter_then_bge`
共用同一个`BgeM3Encoder`实例——如果MS已经在同一进程里加载了同一份BGE-M3权重
（`v1_5_v5_3_semantic_ms_retrieval.BgeM3Encoder`，同一snapshot），ME复用它的**增量成本
不是再加载一次~2GB模型**，只是每次调用的encode时间：bge_m3平均0.307秒/state，hybrid
0.159秒/state（候选更少，编码更快）。current_production（纯词面）平均0.001秒/state，
快两个数量级——如果最终决定不采用BGE，纯速度也是一个真实的理由。

## 一个顺带发现的、诚实记录的小问题（不在本次任务范围内修）

复核时发现脚本68原始"verified"计数（14/32）和这次`current_production_matches_intended_
target`计数（15/32）差1个state（`me_sd_me_sd_u013`，breakup_recovery/positive）：
这个state的**target本身**（"...I deleted the old photos in one sitting instead of
slowly, and it helped me stop reopening the wound every day."）虽然仍然正确地成为
Rank-1，但`compile_atomic_reusable_outcome`判它不通过——具体是哪个从句触发了拒绝还没
排查，只如实记录这个1-state的差异，不在本次W6任务范围内修复编译器。

## 结论与建议（不擅自切换production default）

**没有一种方法"明显改善"**，按运行手册第7.3节和本次任务第7条要求：不推荐把
`bge_m3`或`hybrid_filter_then_bge`设为production默认，两个模块保留为opt-in代码
（已实现、已测试、未接线），供leader复核后决定是否需要进一步调查（比如换一种候选
构造方式让BGE-M3的语义优势能体现出来），而不是现在就采用。当前最值得记录的结论是：
**ME"同主题候选竞争"这个排序鲁棒性问题，不是靠换一个更"聪明"的排序器就能解决的**，
三种方法在这个具体维度上表现相近或更差，指向问题可能出在候选表示或候选构造本身，
不只是排序算法的选择。
