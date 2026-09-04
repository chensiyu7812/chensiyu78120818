# PM V1.5 训练信号、算法与16动作完整诊断

状态：`TECHNICAL DIAGNOSIS / EXTERNAL QUALIFICATION BLOCKED`

## 技术结论

当前问题不是“每次只评十几条，所以模型拿最新十几条训练”。绝大多数10/16/32条数据根本不进入
Step1训练，它们分别是retrieval、generator执行或回复质量门禁。真正训练规模是RS 69条、MP 64条、
MS 64条、历史ME 48条；但有效语义条件远少于行数，而且ME数据已因candidate-gold错绑失效。

现用算法——四个独立、标准化、L2正则logistic head，`C=.3`、threshold `.5`、五seed、grouped
OOF——作为低容量可审计组合器是合适的。失败根因主要不在“没有换更复杂分类器”，而在它收到的
标签和特征没有独立、没有在exact candidate粒度对齐，也没有覆盖外部候选的功能语义。直接改成
大模型、BAAI或16分类只会把这些问题藏起来。

## 每批小数据到底做什么

| 批次 | n | 独立单位 | 实际用途 | 是否训练Step1 |
|---|---:|---|---|---|
| RS opportunity fit | 69 | dialogue state | 11个透明factor的logistic训练 | 是 |
| RS H2 | 48 | dialogue | 人工机会transfer验证；已是development-informed | 否 |
| RS paired effect | 32 | same-state pair | 证明固定RS有时赢、有时不赢 | 当前opportunity head不使用 |
| MP V1.5b fit / confirm | 64 / 32 | synthetic user | 8种condition模板的内部构念学习/确认 | 64训练，32确认 |
| MS V1.5b fit / confirm | 64 / 32 | synthetic user | 8种condition模板的内部构念学习/确认 | 64训练，32确认 |
| ME scale-stable fit / confirm | 48 / 16 | synthetic user | 旧内部构念训练/确认 | 曾训练，现因gold错绑失效 |
| D3 quality blind | 160 | response pair | component direct-effect质量测量 | 否 |
| V1.5b Step2 gate | 32 | response | generator功能与misuse开发门 | 否 |
| corrected Step2 gate | 16 | state/action response | 多组件功能与misuse最终内部门 | 否 |

因此不存在“永远使用最新十几条”。正确逻辑应是所有artifact按角色累计：训练集只训练，确认集只
做一次确认，开发门失败用于定位，external qualification与lockbox只在冻结后消费。新证据只有在
发现上游构念无效时才会撤销旧结论，例如本轮ME候选—gold错绑，而不是因为它日期更新。

## 算法为什么会出现漂亮内部结果却外部失灵

### 有效样本量远小于表面行数

MP/MS的64条fit各由8种condition×8个topic组成，32条confirmation由同样8种condition×4个新topic
组成。换`community choir`为`pottery market`只产生新词面，不产生新判断逻辑。模型确认的是同一
模板能否换名词复现，而不是是否能处理新型profile、summary或event。

MP有11个输入特征、MS有13个、RS有11个；每个memory head fit只有32个正例。L2可抑制系数爆炸，
不能补出缺失的语义变化。尤其MP/MS确认BA=1.0更应作为“任务可能被造得过直”警报，而不是稳健性
证明。

### 标签与特征并未真正独立

V1.5b已经修掉background bit捷径，但candidate/current文本、target，以及
`profile_relevance/current_goal_fit/prior_outcome`等factor仍由同一condition builder共同产生。
这相当于出题人先决定答案，再同时写答案和提示。logistic真正学到的是人工factor组合，复杂的
自然语言语义仍由regex和模板负责。

### 四个独立head不是错误，直接16分类才是错误方向

四个bit分别学习可隔离单组件问题，也保留了`M0+R0`。直接训练16分类需要远多于当前数据，还会
让稀有动作没有样本，并重新混淆哪个组件产生效果。需要补的不是16分类器，而是：

1. 每个head使用正确的exact candidate监督；
2. 四bit合成后增加联合可执行性projection；
3. requested、feasible、realized action分开记录；
4. 最终用同栈端到端实验验证quality-risk-cost。

## 当前精准问题清单

| 层 | 问题 | 影响组件 | 类型 | 是否已有可行解 |
|---|---|---|---|---|
| 数据单位 | gold没有绑定实际rank-1 | ME，其他组件需防回归 | Critical data | 有：candidate-first labeling与ID/text不变量 |
| 构念 | 同名source内部外部功能不同 | MP、ME | Critical data/definition | 有：typed subtype与功能合同 |
| 独立性 | label与factor同源于condition builder | MP、MS、ME | Critical measurement | 有：独立labeler与feature builder |
| 覆盖 | 8种模板不代表外部候选空间 | MP、MS、ME | High data | 有：按语义轴造新逻辑而非换topic |
| 样本/容量 | 11–13特征配32个正例 | RS、MP、MS | High model/data | 有：降到5–7 factor或扩大fit/confirm |
| RS检索 | H2 Top-1仅19/31合适 | RS | High retrieval | 有：hard abstain、exact-card fit训练与fresh确认 |
| RS路由 | H2 BA=.680但recall=.419且Brier差于prior | RS | High model/data | 有：使用真正人标candidate opportunity，不再rule replay |
| MP外部 | 114候选开109，多数只是Job/Education | MP | Critical distribution | 有：profile增量/目标/实体门，普通profile作负例 |
| MS外部 | 204状态只开4，但给定资源做功6/8 | MS | High calibration/data | 有：外部形态summary的on/off覆盖与fresh确认 |
| ME外部 | 任意episode被当作event/outcome | ME | Critical ontology | 有：仅REUSABLE_OUTCOME为V1.5 ME-on |
| 联合动作 | 合法bit组合可能合同互斥 | 全部 | Critical action | 有：outcome-blind joint feasibility projection |
| Step2绑定 | 过去时短语不能证明用了指定证据 | MP/MS/ME | Critical generator | 有：component→evidence→function绑定与fallback |
| 端到端 | 质量偏好不能自动定位PM/检索/generator责任 | 全部 | High evaluation | 有：逐层trace与原子指标 |

## 解决到什么程度，才能说四组件学会且16动作成立

没有任何数据量能预先保证四组件一定通过；可以冻结一组足以支持结论的必要且联合充分的实验门。
只有以下全部通过，才能使用“完整四资源PM”的主张：

1. **数据门**：100% label绑定实际rank-1；user/topic/template跨split零重叠；无target/condition泄漏；
   每个支持subtype在fit与fresh confirmation均有正负例。
2. **单head门**：每个组件在fresh grouped confirmation同时达到BA、recall、specificity与正Brier
   gain门；五seed稳定且真实产生on/off；报告置信区间，不以raw accuracy替代。
3. **retrieval门**：candidate opportunity、Top-1 fit、hard exclusion、abstain分别达标；不能以
   Top-3里存在好候选替代实际rank-1。
4. **联合动作门**：16个动作机械可达；所有组合冲突都由预注册projection处理；新状态中
   requested→feasible→realized lineage完整。
5. **Step2门**：每组件证据绑定和真正做功分别至少达预冻结比例，material misuse不越界；无解
   合同样本从能力分母剔除但单独计冲突率。
6. **外部门**：先p7–p12 qualification，再按冻结规则决定是否消费p13–p18；外部结果不回流。
7. **系统门**：learned-full相对always-off、fixed-high、transparent-rule与cost-matched fixed在
   quality上非劣、risk不恶化、generator input成本下降；否则只能主张partial component support。

### 最小数据与容量建议

V1.5快速方案不是无限人评，而是把零API候选监督做对：

- 每head保留5–7个预注册、可解释factor；
- 至少96个独立exact-candidate fit状态（48/48）和48个fresh confirmation状态（24/24）；
- fit至少12个语义/condition family，confirmation至少6个真正未见逻辑族；
- 若坚持MP/MS当前11–13个feature，则至少提升到约160 fit/80 confirmation，并仍需正则与区间；
- 只对候选标签做约20%的双人抽审与所有边界/冲突项复核，不再全量盲评每次回复。

这些是V1.5可操作的工程下限，不是数学保证。真正判定仍由fresh confirmation和外部冻结门完成。

## 外部研究能补什么，不能补什么

- CASE与ESCA可作为RS/Step2的后续对照：前者加强认知—情感表达，后者显式分离strategy planner与
  strategy-aligned generator；它们不能修复memory candidate gold。
- ESC-Eval适合最后的多轮响应质量与持续支持外测，不适合直接生成PM开关标签。
- LongMemEval的`indexing→retrieval→reading`分层、knowledge update与abstention正好支持本项目把
  检索、路由、证据读取分开；LoCoMo的persona+temporal event graph可用于构造内容不重叠的类
  EvoEmo内部纵向环境。
- 这些工作应作为数据结构、分层评测和消融参考，不应把V1.5扩成训练大型planner或用户模拟器。

## 下一步最短实施顺序

1. 建立唯一evidence registry，停止“最新packet覆盖旧结论”；
2. 重建MP/MS/ME candidate-first数据合同，先修ME与MP subtype；
3. 将每head feature压到5–7个，RS也改用exact-card人标机会而非rule-replay gold；
4. 零API完成模板泄漏、exact-candidate、split与有效样本量预检；
5. 训练四个L2 logistic head并在fresh confirmation一次性判定；
6. 实现joint feasibility与selected-evidence binding；
7. 只做一轮有界Step2语义门，然后依次消费p7–p12与p13–p18。

这条路线保留原主张和16动作，不依赖把失败组件手动关掉，也不需要把项目改造成大型深度模型研究。
