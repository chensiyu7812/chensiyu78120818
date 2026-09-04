# PM V1.5 H1 v2 标注与数据质量审计

状态：`HOLD_DO_NOT_TRAIN`

## 结论

本轮主审的 256 条导出**格式正确、ID 完整、双门派生无错误**，因此人评工作没有作废；它可作为
开发证据保存。但当前 H1 v2 不能冻结为 confirmation/sealed gold，也不能开始正式训练。

阻断原因不是“PM 必须全选对”，而是训练和考试尚未独立、数据仍有可直接走捷径的结构：

1. 去掉不改变路由判断的装饰性历史前缀后，发现 40 组、80 个状态在
   `current_user_text + 四个 exact candidate` 上完全相同且跨 split；
2. MS 的人评真标签中，恒定预测 OFF 已达 `174/236 = .7373`，FIT、fresh、sealed 三区均超过
   预冻结的 `<.70` 门槛；
3. ME 仅凭当前消息中的固定句
   `A tentative option based on what helped before is welcome.` 即得到约 `.8900` balanced accuracy，
   说明模型可以不读 exact candidate；
4. 第二评审目前只有汇总，没有 64 条逐行 JSONL，无法计算 raw agreement、Cohen κ、逐组件一致率
   或裁决具体分歧。
5. MP_PREFERENCE 72 个候选全部被判 OFF，RS 的 strategy family 也与标签强绑定；当前数据无法证明
   PM 会在偏好真正有增量时打开 MP，也无法证明 RS 不是只认 family 名。

所以正确状态是：`主标注可用作 development evidence；现有 split 不可用作独立资格证据`。

## 主标注本身是否可信

| 检查 | 结果 |
|---|---:|
| 导出行数 / 唯一 blind state | 256 / 256 |
| 与冻结 packet ID 对齐 | 256 / 256 |
| component decision 数 | 1,024 |
| 双门到 ON/OFF 的机械派生错误 | 0 |
| subtype / absent 合同错误 | 0 |
| evidence code 缺失 | 0 |
| 主审 ABSTAIN | 0 |

这说明文件能用、主审遵循了页面 schema，但不等于每个语义裁决都已最终确认。第二评审指出的
`d2c5f711...` RS 项确有冲突：原子动作表面上与“保持简短、先倾听”相符，但 exact card 的
execution profile 和 `when_not_to_use` 明确排除低负担/listen-only 状态。按“审核整个 exact card”
而不是只看 support_move 的合同，该项应进入逐行裁决，倾向门1 `no` / OFF，而不是直接采用主审 ON。

## 真实标签分布

以下只统计 candidate-present；candidate-absent 自动 OFF 不参与恒定基线门。

| 组件 | ON | OFF | 多数类恒定准确率 | 结论 |
|---|---:|---:|---:|---|
| MP | 75 | 134 | .6411 | 分布门通过 |
| MS | 62 | 174 | .7373 | 分布门失败 |
| ME | 121 | 115 | .5127 | 分布门通过，但 cue shortcut 失败 |
| RS | 84 | 140 | .6250 | 分布门通过 |

MS 不是因为 ON 少就“资源无用”。这里说明的是：当前题目不足以证明学习，因为一个完全不看输入的
always-OFF baseline 已经很高。修复目标不是强行把 MS 开多，而是补足真正具备非冗余用途、同时
能由可见候选因素辨识的 MS 状态。

subtype/family 审计还显示：

- MP_PREFERENCE：`0 ON / 72 OFF`；MP 的 75 个 ON 全来自 MP_PROFILE；
- ME_REUSABLE_OUTCOME：`121 ON / 14 OFF`，而 CONTEXT_EVENT 与 UNRESOLVED_EVENT 全 OFF；这是
  typed ontology 的强先验，但14个反例仍要求模型检查目标、实体和边界，不能只看 subtype；
- RS Question：`24 ON / 106 OFF`；Restatement：`35/1`；Providing Suggestions：`1/33`；
  Reflection：`24/0`。只看 family 的回顾性 BA 约 `.848`。

因此 repair set 必须包含 MP_PREFERENCE 的真实 ON、建议卡的真实 ON、reflection/restatement 的
OFF，以及各 family 内的最小反事实；否则“学会四组件”会退化成“记住 subtype/family 默认值”。

## 为什么静态门之前会误判 PASS

原静态门把完整 `visible_dialogue` 放进 duplicate signature。生成器给不同 split 添加了不同的
“Earlier in this conversation set...”前缀，因此 JSON 不同；但当前请求和四个 exact candidates
完全相同，路由决策并没有增加新信息。

审计后的决策面定义为：

`current_user_text + MP/MS/ME/RS 四个 exact Rank-1 candidate texts`

在这个定义下：

- 40 个重复组、80 个受影响状态；
- 11 组连接 FIT–FRESH；
- 29 组连接 FRESH–SEALED；
- 同一重复组的主审标签全部一致，反而会虚高 confirmation/sealed 的表面稳定性。

静态门代码已增加这一决策面检查；旧冻结 artifact 保留为失败证据，不篡改成“当时已经通过”。

## 现有透明特征为何还不能训练

把完全相同的透明 feature vector 视为模型不可区分的状态，得到如下诊断上限：

| 组件 | present rows | distinct vectors | mixed-label rows | oracle BA ceiling |
|---|---:|---:|---:|---:|
| MP | 209 | 6 | 114 | .8545 |
| MS | 236 | 12 | 133 | .7712 |
| ME | 236 | 14 | 133 | .9272 |
| RS | 224 | 8 | 198 | .8131 |

`oracle ceiling` 是把每个 feature vector 事后赋予最有利标签的诊断，不是可报告模型成绩。MS 即使
事后看完全部标签，现有向量的 balanced-accuracy 上限也只有约 `.77`；真实 FIT→fresh/sealed
泛化只会更低。

源码层根因也明确：

- MP 的 `candidate_incremental_information` 仅检查候选全文是否作为字符串出现在上下文，209 个
  present 项全部为 1，不能表达人评的 89 个 `CURRENTLY_REDUNDANT`；
- RS 的 `candidate_nonredundancy` 用完整 support_move 字符串做 exact containment，224 个 present
  项全部为 1，不能识别“上一轮已经问过同类问题”或“当前明确不要重复”；
- MS 的 goal / distinction 特征过粗，无法区分事实回忆目标、当前支持目标、已解决旧目标和只是同题；
- ME 的 action/result 特征有意义，但当前固定 invitation cue 与候选 subtype 强绑定，导致不读候选
  也能猜标签。

这不是换 logistic loss、换 BAAI 或增加同模板行数能解决的问题。先修可观察 factor 和反事实覆盖，
再比较透明 logistic 与 BAAI challenger 才有意义。

## RS 的 ON 到底表示什么

RS 没有改回“卡片看起来相关就开”。H1 v2 的两门是：

1. exact Rank-1 card 对当前 owner、目标、边界和负担是允许且安全的；
2. 它相对 R0 增加一个当前尚未规定、尚未执行的具体动作或约束。

双 yes 才是 **ex-ante incremental opportunity ON**。它不是单次生成后实际赢，也不是“所有支持场景
都开”。Step 2 再检验 generator 是否真正使用，H3 再检验实际 quality–risk–cost 净收益。于是早先
“没有收益不如不开”的原则没有消失，而是被拆成：回复前学习可观察机会，回复后独立验证真实收益。

## 两位评审现在能得出什么

在固定 64-state overlap 上，主审 present-candidate 汇总为：

| 组件 | 主审 ON/OFF | 第二评审汇报 ON/OFF/ABSTAIN |
|---|---:|---:|
| MP | 16 / 34 | 17 / 33 / 0 |
| MS | 16 / 42 | 13 / 45 / 0 |
| ME | 32 / 26 | 32 / 26 / 0 |
| RS | 18 / 37 | 21 / 33 / 1 |

总量很接近，但相同边际分布不等于逐项一致。没有第二评审 64 行原始 JSONL，不能从这些数字反推
kappa；也不能把两份按多数票合并。该文件是当前唯一需要用户补交的已有标注资产。

## 最小返工方案

不重做人评 256 条，也不把所有旧数据删掉。按以下顺序一次修复：

1. 保存全部主标注为 development evidence；收到第二评审逐行 JSONL 后只裁决 overlap 分歧；
2. 修复静态门，使 decision-surface duplicate 在出页面前硬失败；
3. 用 H1 evidence code 定义 5–7 个真正可部署的透明 factors：
   - MP：candidate subtype、当前是否已明说同一偏好/事实、profile 是否对当前任务产生功能增量、实体/边界；
   - MS：事实回忆 vs 当前支持目标、prior distinction 是否回答当前问题、resolved/stale、当前已重复；
   - ME：typed action+result/mechanism、实体与目标匹配、当前已重复；
   - RS：prior same move、explicit non-repeat/listen boundary、完整 execution-profile hard exclusion、
     family/goal/burden fit；
4. 构造一个而非连续小包的 repair set，同时完成三件事：替换40个跨split副本、补足MS可辨识正例、
   给ME加入 invitation×candidate-validity 的四格反事实，并在MP subtype与RS family内部同时覆盖
   ON/OFF；
5. 新 confirmation 与 sealed 在任何模型拟合前冻结。旧 fresh/sealed 标签只能参与开发诊断，不参与
   最终资格或论文数字；
6. 通过静态门和 H1 IAA 后，才训练四个 L2 logistic heads；BAAI仅作预注册 challenger。H2/H3
   继续沿用既有定义，不重做过去的 RS/记忆质量人评。

repair set 的确切数量应由反事实四格、每组件独立语义组和事件数共同决定，不能为了少标任意压缩。
当前估计为一个约 64–96 state 的最后整包，而不是再做多轮 10/16/32 条零散人评；最终数量在第二
评审逐行文件导入并完成去重后冻结。

## 产物

- 机器审计：`outputs/pm_v1_5_final_h1_v2_annotation_audit_v1/audit_report.json`
- 主标注规范化副本：`outputs/pm_v1_5_final_h1_v2_annotation_audit_v1/primary_annotations.jsonl`
- 审计脚本：`scripts/v1_5/25m_audit_final_h1_v2_annotations_v1_5.py`
- 回归测试：`tests/test_v1_5_final_h1_v2_annotation_audit.py`

本报告没有关闭 MP/MS/ME/RS 中任何组件，也没有缩小16动作接口。它只阻止一份仍会泄漏与走捷径
的数据被误称为独立 gold。
