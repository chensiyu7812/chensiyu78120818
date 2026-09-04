# PM V1.5b：MP/MS 与 Generator 根因修复报告

日期：2026-08-02  
状态：`MP/MS internal construct PASS；external representation transport PASS；generator human gate robust FAIL；fail-closed guard frozen for final system validation`

## 结论

这次没有把 MP/MS 永久关闭，也没有通过降低阈值把它们“调到通过”。旧 V1.5
作为冻结对照保留；另建 V1.5b 分支，分别修复了两个真正根因：

1. MP/MS 数据中的标签—背景捷径和同特征异标签；
2. generator 在 PM 已允许资源后重新自由决定 use/ignore、并以复制审计 span
   代替“资源真正做功”的错误接口。

修复后，MP/MS 在零 API、全新主题的受控确认集上通过预先冻结的内部学习门；在
EvoEmo 上用同一 compiler、query、retriever、候选去重和 feature builder 做
outcome-blind transport，进入 learned head 的表示覆盖为 MP `1.000`、MS
`.951`。这只能证明表示可运输，不能称作 EvoEmo 路由准确率。

新的 generator 合同已经在原来同一批 32 个状态和资源上真实运行。32 条都返回
`applied`；旧机器门通过30条。但随后两份最小功能核验稳健地判定 generator 未过门：
完整主评做功26/32、misuse7/32，summary-only敏感性做功24/32、misuse6/32。
两份评审的细节阈值不同，却都远超预冻结的misuse≤1/32，因此不能晋级为已资格化
generator。

## 一、旧 MP/MS 为什么失败

### 1. 数据不是单纯“小”，而是机制性泄漏

旧补充数据的 condition 和 background 都由同一个 `local_index` 生成。V3 中：

- MP target 可由 `background_ME_on` 16/16 恢复；
- MS target 可由 `background_ME_on == background_MP_on` 16/16 恢复。

旧 48-row fit 还出现：

- MP：3 个相同 feature pattern 同时对应正负标签，影响 7 行；
- MS：2 个相同 feature pattern 同时对应正负标签，影响 9 行。

因此继续增加同机制数据、调 C、调 threshold 或换一个更大的通用 embedding，均不
能从根本上解决。

### 2. BAAI 没解决的是构念，不是参数量

BGE-small 以前作为一个 generic cosine/PCA 特征加入时，MP 无增益、MS 增益未到
冻结门、ME 反而退化。一个通用相似度无法区分：

- MP：用户当前欢迎某种交互偏好，还是明确拒绝它；profile 说的是用户本人，还是
  同主题的第三方；
- MS：是否是同一个未解决问题、同一个当前目标，还是同主题但不同任务；历史是否
  真提供了当前缺少的 distinction/outcome。

所以 V1.5b 没有把 BAAI 换成另一个排行榜模型，而是先把 source-specific 构念显式化。

## 二、V1.5b 的 MP/MS 修复

### 1. 数据合同

每个 head：

- fit：64 个独立用户，32 on / 32 off；
- sealed confirmation：32 个独立用户，16 on / 16 off；
- 8 个均衡 condition families；
- 每个 target 内三项 background bits 的分布完全相同；
- 同时包含低词面真阳性、高词面错 scope/goal 负例、冗余、错实体与现实无关项；
- fit/confirmation 用户、topic families 与表面实现分开；
- 零 API、零回复结果、零外部 outcome。

自动预检：

- nuisance-only logistic BA：四个 split/head 均 `.50`；
- 同 feature pattern 异标签：四个 split/head 均 `0`；
- 64/32 行数、类别、用户、条件、背景交叉均通过。

### 2. 表示

MP 拆成：

- preference scope fit；
- profile topic relevance；
- profile entity scope；
- incremental/nonredundant information；
- current scope conflict。

MS 拆成：

- same-issue level；
- current conversational-goal fit；
- current continuity invitation；
- prior outcome/distinction；
- resolved marker；
- incremental/nonredundant information。

Top-k 内同 source、规范化文本完全相同的候选先去重，避免两条重复 summary 被当成
两份独立证据。16 个动作和 MP/MS/ME/RS 四个 bit 均未改变。

### 3. 内部和外部结果的正确解释

内部 controlled construct 的 grouped OOF 与 sealed confirmation 均为 BA/recall/
specificity `1.00`，nuisance-only BA `.50`。这个 1.00 说明冻结的透明构念在新主题
中可表达，不等于真实回复质量 100%，也不等于外部准确率。

EvoEmo outcome-blind transport：

| 组件 | states | hard-off | eligible in-support | predicted on |
|---|---:|---:|---:|---:|
| MP | 204 | 90 | 114/114 | 109 |
| MS | 204 | 0 | 194/204 | 4 |

MP 候选可用时偏向开启，MS 极保守。由于没有读外部 routing gold 或回复结果，这只是
被冻结的行为分布，不能拿它继续调模型。最终由同栈系统 quality–risk–cost 比较判断
这种行为是否有意义。

## 三、Generator 根因与修复

旧接口的问题：

- PM 已允许注入，generator 又被要求自由 use/ignore；
- generator 被要求复制 resource/response exact spans 和写 reason；
- validator 主要验证记账与字面复制，不能验证功能贡献；
- 因此会出现“正确提到历史，但后续建议与历史无关”的表面执行。

V1.5b 新接口：

- 运行时先生成 `ResourceExecutionPlan`；
- plan 固定 source function、temporal stance、required contribution 和 forbidden
  inferences；
- generator 只可返回 `applied` 或有界的 `cannot_apply`；
- 不再要求复制私有 ID、resource span、response span 或自由 reason；
- `cannot_apply` 或机器验证失败时，realized component bit=OFF，最多一次无资源回退，
  两次调用成本全部计入。

32 条实际运行：

- structured output：32/32；
- machine valid：30/32；
- fallback：2；
- realized-on：MP 8/8、MS 7/8、ME 7/8、RS 8/8。

两条被拦截的具体模式都是过去来源丢失：一条 ME 直接给“写未发送草稿”，一条 MS
直接把休息内疚与照护工作量连接起来，都没有说明这是过去信息或核对当前是否仍成立。
这说明新的机器门确实拦住了旧评审曾指出的 temporal promotion 问题。

人工核验进一步发现旧机器门漏掉了更严重的问题：两条回复虚构“上次你发现某方法
有帮助”，但资源根本没有任何历史结果；另有过去原因直接升级为当前、假定旧任务仍
存在、建议减少社交接触，以及用户明确说写作会触发panic后仍派写作任务。故新增
`fabricated_recall` misuse 类别，并把这四类可观察错误加入 fail-closed validator。
在同一开发32条回放中，新guard捕获7/7已知misuse，另拦2条保守false-positive，
development specificity `.92`。因为guard由本批失败形成，这个回放不是独立资格证据；
它现在冻结，只能由最终未见系统risk实验验证。

## 四、现在还不能声称什么

目前不能声称：

- MP/MS 已在 EvoEmo 上达到某个 routing accuracy；
- 30/32 machine-valid 等于 30/32 functional use；
- V1.5b 一定优于旧 V1.5 或 fixed-high；
- 新表示证明 BAAI/其他语义模型永远无用；
- 任意新增 memory/RAG 内容都无需重训。

## 五、最终系统准备进度

generator 人工门已经失败，责任归于 Step2 execution，不回写 PM 标签。现在不再追加
generator 人评或在这32条上继续调提示词。冻结的新 fail-closed guard 进入一次性主系统
比较：被拦截的资源回复实际记为component OFF，并最多生成一次无资源fallback，所有
额外tokens/calls计入成本。最终未见 quality 与risk 结果决定这个“PM+执行器+guard”
系统是否仍能达到质量不劣、risk不增加和成本下降；若不能，V1.5如实报告完整系统
未达标，同时保留MP/MS opportunity head本身的受控学习证据。

主系统比较保留：always-off、fixed-high、transparent-rule、learned-PM-full、
learned-PM-conservative、internal-only cost-matched fixed，以及同当前栈的 V1 raw-session
次级 baselines。最终报告 quality noninferiority、interaction-and-grounding risk 与
generator input token/call fallback cost。

截至本报告同日后续，以下前置工作已经零API完成：

- 共用四bit→16动作compiler机械审计通过；full保留四head，conservative只mask MP/MS；
- 多资源bundle Step2接口及fail-closed realized-action投影完成；
- 32个内部D3用户、16个固定动作的token-only cost match冻结为`MP+RS`，相对learned-full
  平均input-token上界偏差`4.4%`；未读任何response/outcome；
- 初始ESConv 169-dialogue panel中的47个dialogue被一次execution-format pilot触及，因
  generator额外/重复报告component statuses而整体排除；这62个调用不进入正式证据。
  V2正式panel为剩余122个未触及ESConv dialogue + 全部204个未触及EvoEmo states；
- V2六策略共1,956个assignments；同state/effective-action复用后冻结为1,121次primary
  calls，节省835次重复生成，最坏含全部fallback约`$0.80`；正式可恢复执行已经启动。

## 六、可复现入口

- 修复合同：`data/pm_v1_5_contracts/v1_5b_mp_ms_generator_root_repair_v1.json`
- 反捷径数据：`outputs/pm_v1_5b_mp_ms_antishortcut_v1/preflight_report.json`
- MP/MS fit：`outputs/pm_v1_5b_mp_ms_opportunity_heads_v1/fit_report.json`
- EvoEmo transport：`outputs/pm_v1_5b_evoemo_mp_ms_transport_v1/transport_report.json`
- generator 执行：`outputs/pm_v1_5b_resource_application_v1/execution_summary.json`
- 人工核验聚合：`outputs/pm_v1_5b_resource_application_human_check_v1/qualification_report.json`
- fail-closed guard开发审计：`outputs/pm_v1_5b_development_safety_guard_audit_v1/guard_audit_report.json`
- 16动作compiler：`outputs/pm_v1_5b_final_policy_compiler_audit_v1/compiler_audit_report.json`
- 内部cost match：`outputs/pm_v1_5b_internal_cost_matched_fixed_v1/cost_match_report.json`
- 外测panel冻结：`outputs/pm_v1_5b_final_external_panel_v1/panel_freeze_report.json`
- execution-format pilot关账：`outputs/pm_v1_5b_final_system_generation_v1_execution/execution_format_pilot_closeout.json`
- V2正式panel：`outputs/pm_v1_5b_final_external_panel_v2/panel_freeze_report.json`
- V2正式调用计划：`outputs/pm_v1_5b_final_system_generation_v2_candidate/generation_preflight.json`
- 全局失败账本：`docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`
