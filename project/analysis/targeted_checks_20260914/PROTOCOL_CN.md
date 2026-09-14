# PM v1 两项定向补充：执行前协议

日期 2026-09-14。本轮是已有结果后的探索性补充，不修改 v1 checkpoint、阈值、原主表或此前补充文件。新增 API 预算为 **0 美元**，仅本地计算。无根据 PM 输赢选择样本、重试、调整参数或停止的规则。

## A. 动作置换诊断

- 使用已有 1,728 synthetic cards 的 user-held-out 模型预测重放，统一使用最终冻结 consensus 阈值。这不是新的独立测试集，也不重新调参。
- 在相同 fold、相同合法动作集合的 cards 内置换 PM 动作，精确保留每层各动作数量和来源/RS 调用分布。动作始终合法；不额外施加新状态的预测风险约束，风险变化作为结果报告。
- 使用已保存逐动作 judge 标签与实际 sweep input-token 记录评价，绝不用 PM 自身预测分评价效果。报告 response-preference score、misuse、omission、strategy risk、input tokens 和来源调用数。
- 直接计算各层动作频率下的条件期望作为无状态匹配对照，并用 10,000 次置换（seed=20260915）展示置换分布。它不是精确相同 token 预算的对照，实际 token 必须单列。
- PM−置换期望：按 16 用户聚类 bootstrap 10,000 次（seed=20260916），保留 card 均值估计量；bootstrap 中固定本数据估得的对照动作频率。区间是条件性的开发诊断，不宣称新的无偏泛化性能。另列各 fold、各 inventory 可用集合结果。
- Context Only (M0+R0)、ME+R0、Full Structured 使用原 FixedPolicy 的不可用来源删除规则。它们是解释性固定参照，不在新数据上挑选最优动作。

## B. 本地时延试跑

- 固定使用此前换序诊断的全部 48 个状态（18 用户、34 scenarios，seed×turn 六层各 8 个）；复用选样，不根据时延或质量再筛选。
- 五组：PM OFF、Rule OFF、Fixed OFF、Context Only、ME+R0。后三者固定动作来自同一 v1 设定；五组同一 selective system prompt、retriever、tokenizer、模型与解码。新加两组只测时延，不获得可与旧协议混合的新质量分。
- NVIDIA RTX A6000 上，同一已冻结 Llama 3.1 8B BF16、SDPA、greedy、自然 EOS。无实验输出 token 上限；到模型上下文上限仍无 EOS 属于失败，不当作完整回复。
- 模型/静态索引常驻，初始化和磁盘模型加载不计在线时延。已有 runtime catalog 为预构建状态；计时从该状态反序列化开始，包含策略决策、查询构造、实际检索、prompt 构造、tokenization、生成和最终文本解码。这里不包含远程网络、数据库或 UI，因此称本地处理链路时延，不直接外推线上端到端体验。
- 各方法共享相同的 PreparedStrategyRetriever 文档特征预计算。每次请求前清空 query cache，禁止跨方法/请求复用查询结果；同一请求内 Rule confidence 与 RS retrieval 可以复用。所有实际输出必须与原三组冻结 prompt/action 完全一致，否则停止。
- 先用 1 个固定状态逐方法做不计入统计的 warmup；再做 3 轮，每轮 48×5 次，合计 **720 次测量**。顺序用 seed=20260917 预先随机化各轮状态顺序及状态内方法顺序；不并发，batch size=1，无跨请求 KV/prefix cache。
- 首要指标：本地链路到首个可见非空文本的时间、到完整回复的时间；同时保留到首 token 时间、策略/检索/prompt/tokenization/generation 分项、输入/输出 tokens、完整输出与停止原因。
- 先在每个状态内平均 3 次测量，再比较 PM−Rule / PM−Fixed；Context Only / ME+R0 为参照。用户聚类 10,000 次为主，scenario 聚类为敏感性，seed=20260918。P95 为全请求描述性数值，不拿重复次数当新增独立用户样本。
- 保存每轮完整生成结果，并检查原三组与旧回复的一致性。不同则报告，不能把旧质量分自动转移到新回复。试跑不替换 204 状态主表，也不证明质量等效。
- 技术错误导致暂停并保留日志。恢复时仅运行没有成功记录的请求；不删除慢请求或针对 PM 输赢追加重复。独立文件锁防止重复运行。

## 产物与决策

执行前固定协议、代码、样本和顺序的哈希。保存逐 card 对照、置换分布、逐请求时延、分项统计、CI、验证记录及中文报告。只有观察到的证据进入建议措辞：时延不降则继续主张输入/调用资源减少；置换无收益则不声称已证明状态适配价值。
