# 按真实 v1 实现补全方法与主张

本文件根据当前工作树的冻结实现、已有实验记录和 2026-09-14 离线重放整理。不更改 PM checkpoint 或选择阈值。新增 judge 检查的结果另见 `outputs/reviewer_supplement_20260914/results/`。

## 1. 规则基线的精确定义

`StrongRulePolicy` 位于 `src/metacom_pm/policies.py`。查询文本由当前会话摘要、可见会话消息和当前用户消息拼接、规范化得到。外部 EvoEmo 适配器的会话摘要为空，保留最近 8 条先前消息。

对可用的 MP/MS/ME 来源，Rule 用 64 维 `HashingVectorizer` 将查询向量化：word unigram/bigram、lowercase、非负哈希、L2 归一化。将查询与该来源的 catalog fingerprint 做 cosine similarity。按相似度降序，保留达到各来源阈值的前 `max_sources` 个；相似度相同时按实现中的来源标识排序。冻结配置是：

| 参数 | 值 |
|---|---:|
| MP threshold | 0.0 |
| MS threshold | 0.0 |
| ME threshold | 0.0 |
| max_sources | 2 |
| strategy threshold | 0.08 |

RS 的置信度是查询与全部 Strategy Bank cards 的最大词频 cosine similarity，达到 0.08 时开启，否则 R0。Rule 会读取来源 catalog fingerprint；最终稳定 PM 使用 `text_metadata_stable` 特征，不读取这些 fingerprint。这一点应如实描述，不能把两者可见信息说成完全一样。

这套规则在本次 204 个状态上全部选 `MSE+RS`，即 MS+ME+RS；离线重放逐项验证了这一点。它在这批样本上表现为常量配置，不代表其算法对任何状态都恒定。

## 2. 九种开发 inventory variants

构造代码：`src/metacom_pm/preparation.py`，构造标签在 `data/synthetic/audit_only.jsonl`，标签本身不输入 PM 或生成器。

每个源状态构造以下 9 种库存，每种在完整开发数据中出现 192 次：

| Variant | 可用来源 / 内容 |
|---|---|
| none | 无长期记忆 |
| mp | MP |
| ms | MS |
| me_high | 一个与当前上下文词法相关度最高的 ME item |
| me_low | 一个相关度最低的 ME item |
| mp_ms | MP + MS |
| mp_me | MP + high ME |
| ms_me | MS + low ME |
| mp_ms_me | MP + MS + high ME |

high/low 根据相同查询的词法相似度排序，并以不透明 memory ID 破平；ME high/low 的可用性和条目数量一致，但不能据此声称它们所有元数据（例如文本长度）都严格相同。实际库存签名重复时去重。

共 192 源状态、16 合成用户、12 语义家族、36 种当前话语、1,728 runtime cards。动作数量为：2 个动作的 192 cards，4 个的 768，8 个的 576，16 个的 192，总计 11,136 state-action responses。

## 3. 开发评分量纲与训练

开发 `response-preference score` 不等于外部的 1–5 分 Emotional Support。`labels.action_response_scores` 在每张卡内，把成对偏好转为 +1/−1/0，求带 ridge=0.10 的动作潜分差最小二乘解，居中后，对一个动作与其余合法动作的潜分差做 sigmoid 并取均值，得到 [0,1] 的开发诊断分数。它不是人工支持度评分或临床收益。

可部署 response ranker 是基于状态–动作特征差的 LogisticRegression。训练排除 tie，镜像每一对的方向，每张卡总权重归一；运行时对各状态–动作的 decision function 做 sigmoid。该单动作预测分数并非经过独立概率校准的“回答成功概率”。

其余 5 个头用 Ridge 学习，再将预测裁剪到 [0,1]：

- misuse：各已选 memory source 的不必要暴露、陈旧/冲突使用、无依据个人断言的最大严重度；M0 对应无依据个人断言。原 0–2 标注归一到 [0,1]。
- omission：M0 的遗漏与 omission appropriateness，以及 M2b 的 selected-set omission；按 `labels.py` 取适用风险的最大值。
- memory decision quality：来源集合 appropriateness，并受 M2b selected-set sufficiency 的较低值约束。
- strategy risk：RS 的过度结构化/过早建议，或 R0 的遗漏机会、不恰当建议、低 omission appropriateness。
- strategy decision quality：RS 的相关性与利用情况均值，或 R0 的 omission appropriateness。

这些目标都是评价标注的预测，不是临床安全概率。应把稿中的 “calibrated risk constraints” 改为 “thresholds on predicted annotation-based risk scores”，除非另有独立校准证据。

## 4. legal actions、可行集合与空集合

legal actions 只由资源是否可用决定：缺少一个来源时，不能选择包含该来源的动作。feasible set 是合法动作中再通过预测风险阈值的子集。两个概念不能机械统一成同一个名称。

部署冻结值为 epsilon=0、tau_misuse=0.35、tau_omission=1.0、tau_strategy=1.0。预测被裁剪到 [0,1]，因此后两个约束在部署中不会排除动作。可行集合内先取预测回复分最高者；完全并列时才按预计成本、memory quality、strategy quality、response score、action ID 依次破平。

预计成本为各来源估算的 top-k token 数之和，加 RS 开启时的 260 token 常量，再加每个来源调用 24 的惩罚。这是检索前 proxy，不是测得的完整 generator input tokens，也不是实测毫秒数。不能把本次重放的 proxy 均值与旧开发表的执行后总 input tokens 直接比较。

没有动作通过约束时，确认性运行直接抛出错误停止；不会静默改用 M0。只有开发网格搜索可以显式启用最小归一化违约的 fallback，最后选择流程排除使用 fallback 的配置。

## 5. 本次实际决策统计

数据：`outputs/reviewer_supplement_20260914/offline/*selector*`。外部 204 状态逐一重构并核验完整状态 SHA256；PM 与 Rule 的输出逐项匹配原重跑动作。

| 行为 | 外部 204 状态 | 开发 1,728 cards 的 OOF 预测重放 |
|---|---:|---:|
| 最高回复分完全并列 | 0 | 0 |
| 成本项改变最终动作 | 0 | 0 |
| 至少删除一个风险不合格动作 | 200 | 170 |
| 移除风险筛选后最终动作改变 | 0 | 97 |
| omission / strategy 阈值删除动作 | 0 / 0 | 0 / 0 |
| 使用 constraint fallback | 0 | 0 |
| RS 开启 | 192/204（94.12%） | 1,287/1,728（74.48%） |

开发重放采用各 user fold 的留出模型，但统一应用最终冻结的 consensus 阈值；它不是重新执行每折原来的阈值选择，也不是新的独立测试集结果。因此动作分布可以不同于旧 selection 文件按各折各自设置汇总的分布。

在外部 204 状态上，所有状态有完整 16 个合法动作；最终结果恰好等于在全部合法动作上直接取 response ranker 的最高分。由此可以明确说：本样本中成本破平与风险筛选没有改变实际选择。不能将观察到的资源节省说成已经实证归因于这两个环节，也不能由此推断风险头在任何分布中都无用。

## 6. RS 开启率的直接证据

开发已有同记忆配置、仅交换 RS/R0 的 5,568 对双顺序 response judgments：RS 胜 2,673，R0 胜 1,313，tie 1,582；其中 1,452 对双顺序不一致并被映射为 tie。该标签分布属于开发数据，不是外部结果。

使用留出模型预测时，同记忆配置的 RS 分更高的有 4,147/5,568 对；在外部则为 1,536/1,632 对（94.12%）。在外部模型为 RS 赋予较高回复偏好分，比仅查看后置 strategy-quality 头更直接地解释了开启行为。ESConv 的 2,275 个 M0+RS/M0+R0 对比由 GPT-4o 评分，RS 胜 696、tie 457、R0 胜 1,122；该差异提示迁移与评价环境差异，不能单独归因为 Gemini 或 GPT-4o。

## 7. 外部 memory 和固定输入的对应

`build_evo_memory`：MP 为 basic_info 的独立字段；MS 为历史 session summary；ME 为每个历史 session 内所有 seeker 消息按时间拼接成一个条目，而非 evaluator 的 gold event timeline。观测 annotations、future topics、gold related-session 标签和参考答案不输入 PM 或生成器。

204 状态 = 34 scenarios × 3 fixed seeker seeds × turns 3/8；共 18 用户。每个状态的六个回复共享当前输入；旧 GPT-4o 表的六个策略与新 Table III 的 PM/Rule/Fixed × Filter ON/OFF 是两套候选集合，不能混分。固定 seeker seed 也不等于 PM 训练 seed。

生成器实际使用最近 8 条先前消息和当前 seeker 消息，外部 current-session summary 为空。旧主 judge 还可以看到更完整的固定上下文与 evaluator-only authorized context。新增历史需求审查使用生成器实际可见窗口，不能因为 judge 看到某段更早内容，就说 Context Only 也看到了。

## 8. PM 对 Context Only：已有配对区间须呈现

来源是旧 `response_statistical_summary.json`，不是本轮新 judge 结果。以下为 204 状态均值差和 scenario 聚类区间；旧 user CI 使用等用户权重，点估计另列在 CSV 中。未做多重比较联合校正。

| 指标 | PM−Context Only | 95% scenario CI |
|---|---:|---:|
| Emotional Support | −0.0882 | [−0.1667, −0.0147] |
| Personalization | −0.0980 | [−0.1569, −0.0392] |
| Memory Appropriateness | −0.0539 | [−0.0931, −0.0147] |
| Factual Grounding | −0.0098 | [−0.0343, +0.0147] |
| Temporal Consistency | −0.0098 | [−0.0294, +0.0098] |
| Non-Intrusiveness | −0.0245 | [−0.0539, +0.0049] |

前三个逐项区间为负。因此对 Context Only 不只是均值未获优势，旧逐项区间也对 PM 不利。原始 per-state V4 score 文件未找到，本轮仅提取并核对现存统计与均值，不能声称独立从原始分数重新计算了这些 CI。

## 9. ME+R0：已存在的负面诊断与换序结果

已从本地 `origin/main` commit `fbe7781ef5ceda2b1bc8e3f892b24f20e7e09cea` 恢复原报告及找到的执行代码到 `offline/existing_cost_matched_evidence/`，附校验和。部分被报告引用的脚本及原始 score/response 文件没有包含在已检查的 checkout 中；这些数字是有来源的既有报告，不冒充本轮原始数据重算。

ME+R0 是在开发 validation 的 mean cost proxy 不高于 PM 的 fixed candidates 中，按既定质量/风险/成本规则选出的动作，不是看到 EvoEmo 成绩后挑选。所谓匹配是平均预算近似匹配，不是逐状态 token 完全相等。旧报告实测输入：PM 1,294.6，ME+R0 1,283.8 tokens。

报告包含 204 状态的统一两候选 GPT-4o 评分：PM−ME+R0 的 Support −0.172，所报 CI [−0.250, −0.093]；Personalization −0.201，[−0.275, −0.123]。不要将两候选实验的绝对分插入原六候选主表。

另外已有 40 状态 × 双顺序 = 80 次 forced-swap 评分：Support 差 −0.150，[−0.250, −0.050]；Personalization −0.138，[−0.238, −0.050]；resolved preference 为 PM 2、ME+R0 11、tie 27，顺序不一致 1。其代码对双顺序合并后的单位值作 5,000 次单位 bootstrap，不能改称用户聚类 CI。应该把这些结果放到实验部分的独立诊断段落，不能仅用“提示词不同”搁置它们。

原报告与可恢复代码还有一处口径差异：文字称双顺序不一致即记 tie；代码实际上在一边 tie、另一边有胜者时保留胜者，只有两个顺序选出相反胜者时才记为 `order_disagreement` 并归 tie。因此“顺序不一致 1”不能解读为只有 1 个状态的两次标签不同。原始 score/judgment 文件缺失，不能重新确认 2/11/27 的合并口径；正文优先引用文档报告的双顺序平均分差，并披露原始材料未在本次 checkout 中恢复。

新增盲审显示，引用核验本身也有失败，最终分析保留所有 204 状态的首次原始标签及核验标志；不把部分状态的重试结果选择性替换进首次比例。完整说明见补充总报告。

## 10. 论文措辞与术语

- 摘要明确 synthetic development 和 EvoEmo virtual users，并限定省资源的比较对象为 Rule 与 Fixed Structured。
- 外部质量指标统一叫 Emotional Support；旧输出 `overall` 是其别名，不增加第七维。新重跑的 `overall` 是独立的总体分，表注另行解释，不把两者混为同一列。
- CI 跨零不证明等效；保持 observed mean profiles closely matched，不声称严格非劣性。
- 本批证据未证明学习型 PM 比 Context Only 或 ME+R0 更值得采用；贡献应限定为可部署的来源路由实现、受控开发信号，以及相对指定高资源结构化基线的节省。
- 如果图中的 s 表示完整决策输入，可写 s_t=(x_t,I_t)。不要把图 1(c) 的 MP+ME 示例改成 MP+MS；它会改变动作含义。
- 当前代码已有公开 GitHub 分支，可以给真实仓库地址与冻结 commit；不作尚不存在的全数据公开承诺。
- 当前证据支持输入资源减少；端到端延迟保持辅助诊断。如果要升级为明确的时延优势，需要另行控制部署条件测量，不能把离线 PM 计时与不同批次的生成时间直接相加。

可插入论文的英文 LaTeX 片段见本目录 `PAPER_INSERTS_EN.tex`。它们是针对可核对稿件的增补，没有直接修改未知最新状态的 Overleaf。
