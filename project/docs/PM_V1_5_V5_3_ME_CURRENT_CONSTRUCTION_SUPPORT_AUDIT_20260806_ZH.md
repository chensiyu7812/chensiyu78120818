# V5.3 ME 当前构造支持审计（2026-08-06）

## 技术结论

旧报告中的 ME `past_action_result=0%` 不适用于当前正式构造：它读取的是已经淘汰的
468-card PMV2 backend。当前 V3 exact Rank-1 物化面中，128/128 个 effect state 的 ME 候选
都通过同一个 atomic reusable-outcome compiler，并与私有构造目标精确绑定。

这修正了“泛化已经解决但 ME 训不出来”的错误叙述。更准确的现状是：**当前候选构造有能力提供
ME 正/非正机制；真正尚未验证的是全新 V5.3 数据上的行动准备度观察和真实 ON/OFF paired effect。**

## 数据范围与粒度

- 构造意图：`data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl`；
- exact Rank-1 物化：`outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl`；
- 审计单位：一个 state 的一个 ME exact Rank-1 candidate；
- 总计160行，其中effect 128行、eligibility audit 32行；
- 不读取质量、risk、功能、A/B偏好或外部 outcome；API调用为0。

## 关键检查结果

| 检查 | 结果 | 能证明什么 | 不能证明什么 |
|---|---:|---|---|
| effect exact Rank-1通过ME compiler | 128/128 | 当前构造能提供“过去动作+结果”候选 | 资源开启后一定提高回复 |
| effect exact Rank-1与构造target精确绑定 | 128/128 | materialization没有把别的候选错绑到Rank-1 | 检索在自然外部语料上一定正确 |
| 私有blueprint标记action-ready | 128/128 | 这批开发构造原本就在测行动型目标 | 该意图可直接作worth-opening gold |
| 旧`explicit_advice_welcome`识别 | 0/128 | 旧观察器不适配当前自然表述 | 用户真实不欢迎行动 |
| 新三态观察器开发回放识别 | 128/128 | 新观察器覆盖了已消费的开发表述 | 对新用户/新表达已经泛化 |

Eligibility audit 的32行不是全部正例：其中24行 compiler-valid，另外8行按设计为context-only或
unresolved等负构念。该分布符合资格审计目的，不能拿32作effect分母。

## 为什么旧0%会误导

脚本60的“训练域”来自旧 PMV2 backend；脚本61读取的是当前 V3 exact Rank-1 构造。两者不是同一
数据世代，也不是同一用途。旧0%是有用的淘汰证据：它说明旧backend不得复用；但把它外推为
“当前构造也缺ME支持”会错误触发旧模板修补、外部比例追平或subtype压过topic relevance。

正确的数据质量边界是：

1. 不原地修补、重标旧backend；
2. 不以EvoEmo的78.3% subtype prevalence作为内部目标；
3. 不让形式合格但话题不相关的outcome压过相关候选；
4. V5.3新数据只要求覆盖正/非正机制和部署特征支持，并与外部文本内容不重叠。

## 尚未完成与不确定性

- 新三态行动准备度只在已经消费的V3/V5.2构造上开发回放通过，尚无内容独立资格证据；
- 128/128是候选构造/绑定检查，不是Step1准确率，也不是Step2质量或端到端QRC结果；
- 当前V3行已经被旧实验消费，不能重新命名为V5.3 fresh confirmation；
- ME真实效应仍须在同state、同candidate、同seed的ME-ON/OFF回复对上测量。

## 冻结的下一步

1. 在P2生成内容独立的V5.3 ME superdomain：至少40个 intended-positive 与40个 nonpositive
   independent group，按用户、语义族、反事实组隔离；
2. intended-positive在任何API前强制通过compiler和exact Rank-1绑定；
3. 行动准备度保留三态，`UNKNOWN`不作OFF gold；先做一次内容独立资格，不反复扩规则；
4. 随后与MP/MS/RS一起进入一次FIT paired generation，效应标签由真实回复QRC产生，而不是由
   构造意图或eligibility人工代替。

图表省略原因：本审计只有少量离散计数和合同判断，精确审计表比图形更不易混淆分母。

