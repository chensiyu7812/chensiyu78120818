# PM-v1.6 审查入口

状态：`PREREGISTERED IMPLEMENTATION / NO FORMAL RESULT`

PM-v1.6 在任何正式 468×16 development sweep、checkpoint、study freeze 或外部结果产生前，替代 PM-v1.5。旧 V1.5 保留为不可变审计历史，状态为：

`SUPERSEDED_BEFORE_FORMAL_EXECUTION`

## 为什么升级协议

PM-v1.5 虽然已经统一 Strategy Bank、supporter prompt、retrieval 参数与 Evidence Filter 状态，但其 PM 在选择 item-level retrieval 前使用了内容派生的目录相似度与 embedding 统计量，却仍称为 pure pre-retrieval router。PM-v1.6 将这一隐性机制改为公开、固定、低成本、可计费的 Step-0 来源级观测，并删除无任务语义、容易形成环境身份捷径的目录 embedding norm/mean/std。

## PM-v1.6 的方法定位

PM-v1.6 是：

> 基于固定来源级粗观测的监督式 pre-item-retrieval 资源路由器。

它不是强化学习、POMDP、临床系统或真实用户改善研究。

运行流程：

1. 当前用户话语、最近对话和摘要形成 query；
2. Step-0 只计算 MP/MS/ME 来源级相似度、有限目录元数据、策略族相似度与当前 advice/readiness；
3. strong transparent router 给出默认动作；
4. train-only 选出的 learned outcome model 仅在冻结的保守 override 条件满足时覆盖默认动作；
5. 执行 item-level retrieval；
6. 从实际进入 generator 的证据推导 realized action；
7. 记录完整成本向量并生成回复。

## 关键防护

- PM-visible Step-0 不包含原始 memory、item ID、top-k item score、catalog embedding、oracle label、stale/conflict oracle、representation hash/version。
- representation hash/version 仅进入 audit binding。
- requested、retrieval attempts、realized、prompt alias 分层记录。
- zero-hit 合法，但保留检索调用与成本。
- 相同 prompt 可物理复用回复，但共享 quality label 采用 inverse alias-class weight，不能制造伪样本量。
- required-hit 是 response/judge 前的数据有效性门，不允许看 outcome 后删样本。
- development judge 只允许 Gemini + DeepSeek；任何 GPT-4o/Claude family/model/base URL 进入开发阶段都 fail closed。
- current absolute、state-centered、rule-relative 三种 HGB 模型只在 train users 内通过 user-block CV 竞争。
- calibration 只能选择 override 阈值，不能重新选择算法或 transparent rule。
- internal test 通过 append-only ledger 一次性消费。
- 外部必须同时包含 learned PM、同观测 strong rule、cost-matched fixed、high-resource fixed、ME+R0 legacy anchor、M0+R0 与 session-RAG。
- 任一主 gate 失败，输出 `NOT_SUPPORTED`。

## 审查顺序

1. `project/docs/PM_V1_6_METHOD_CONTRACT_ZH.md`
2. `project/configs/pm_v1_6.yaml`
3. `project/src/metacom_pm/pm_v1_6_contracts.py`
4. `project/src/metacom_pm/pm_v1_6_step0.py`
5. `project/src/metacom_pm/pm_v1_6_action_lineage.py`
6. `project/src/metacom_pm/pm_v1_6_router.py`
7. `project/src/metacom_pm/pm_v1_6_models.py`
8. `project/src/metacom_pm/pm_v1_6_training.py`
9. `project/src/metacom_pm/pm_v1_6_internal.py`
10. `project/src/metacom_pm/pm_v1_6_claims.py`
11. `project/scripts/v1_6/`
12. `project/tests/test_pm_v1_6_*.py`

## 当前执行状态

- 新方法合同：已写入代码与配置；
- CI 路径和 clean pytest import：已修订；
- final/development judge 隔离：已代码化；
- Step-0、action lineage、alias、成本向量：已实现；
- transparent router 与三算法 train-only competition：已实现；
- internal one-time ledger 与 hierarchical claim assessment：已实现；
- 正式 development generation、sweep、judge、training、internal test、freeze、external generation：尚未运行；
- 当前 claim：`NO_CURRENT_RESULT`。
