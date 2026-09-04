# MetaCom V3.2 研究计划：长期情感支持数字人中的检索前证据资源分配 Policy Manager

版本：2026-06-20；补充更新：2026-06-21  
状态：V3.2 方案设计、代码恢复与 API-ready 工程骨架合并完成；正式 API judge pilot、full judging、PM 训练和最终测试尚未完成。  
项目目录：`/home/tokkio/esconv_experiment_bundle/policy_manager_35`

## 0. 一句话定位

本研究不是训练一个新的情感支持大语言模型，而是研究一个独立的 **Policy Manager（PM）决策层**：

> 在固定回复生成器不变的情况下，PM 能否仅基于检索前可见状态，决定应该激活哪类记忆和策略资源，使长期情感支持回复在质量、记忆适当性、风险和成本之间取得更好平衡？

V3.2 的核心修正是：PM 必须是 **pre-evidence / pre-retrieval resource allocation policy**，而不是先把所有候选证据都检索出来再排序的 evidence reranker。

## 1. 为什么需要 V3.2

早期 V1/V2/V3.1 已经证明了一个重要方向：更多 RAG、更多记忆、更多 Strategy RAG 并不总是更好，规则门控也容易误用记忆或过度注入上下文。

但旧方案存在多类会颠覆论文主张的问题，因此当前全部旧 headline 结果降级为 development / legacy artifacts：

- PM 使用了检索后特征，因果顺序不成立；
- Strategy RAG 被 evidence judge 当作长期记忆奖励；
- M0 正确不使用记忆没有得到 omission reward；
- response judge 绝对评分严重饱和；
- test alpha 和 claim gates 被多次查看，final test 不再干净；
- 旧结果把 internal utility、external overall、cost 口径混在一起；
- 部分数据模板、hidden proxy、外部验证口径存在审计风险。

V3.2 的目标不是在旧结果上修表，而是重建一个逻辑干净的第一篇论文版本：

1. PM 输入只使用部署时检索前可见信息；
2. 旧 response trajectories 可以作为 development source，但旧 labels / utility / oracle / checkpoints 作废；
3. 使用新的 pairwise response judge 和分离式 memory / strategy judge 重新打标签；
4. 训练一个检索前资源分配 PM；
5. 在冻结 validation 超参数后，在未触碰 final test 上评价。

## 2. 研究问题

### RQ1：检索前 PM 是否能优于规则资源门控？

规则门控通常根据显式关键词、会话深度或固定启发式决定是否调用记忆。它不能直接学习“这个资源在当前状态下是否真的值得调用”。V3.2 检验 PM 是否能在只看检索前特征的条件下，学会比 rule policy 更好的资源分配。

### RQ2：选择性结构化证据是否优于全量或固定 RAG？

全量 raw-session RAG 代价高，且容易引入无关、过时或敏感历史。V3.2 检验 PM 是否可以用更低的检索和上下文成本，达到接近或超过强固定策略和 raw-session RAG 的效果。

### RQ3：Strategy RAG 是否应该作为可选资源而非 always-on？

旧实验显示 Strategy RAG 的边际收益依赖情境：在 M0 或某些情绪承接场景下可能有益，在已有记忆时可能过度结构化或降低自然性。V3.2 将 Strategy RAG 作为策略轴动作，由 PM 决定是否激活。

### RQ4：PM 学到的是资源条件推理，还是模板记忆？

V3.2 数据仍有受控模板成分，因此必须通过 availability counterfactual、seen/unseen text-profile 分析、metadata-only / text-only probe 等诊断，验证 PM 是否真的利用 memory availability metadata，而不是记住固定句式。

## 3. 研究对象与边界

### 3.1 研究对象

研究对象是 **对话策略决策层**，不是 LLM 本身。PM 输出结构化资源动作：

```json
{
  "memory_action": "M0 | MP | MS | ME | MPMS | MPE | MSE | MPMSME",
  "strategy_action": "R0 | RS"
}
```

PM 决定 **激活什么资源**，固定 support generator 决定 **如何表达回复**。

### 3.2 非研究对象

本篇论文不主张：

- 端到端训练 supporter LLM；
- 真实临床疗效或真实用户情绪改善；
- 完整 offline RL / deep RL；
- 多模态情绪识别；
- Tokkio 真实部署实验；
- ESC-Eval benchmark 官方完整复现。

这些可以作为后续论文或 future work。

注意：ES-MemEval / EvoEmo 18-user longitudinal evaluation **不是** future work，而是本篇长期记忆外部验证的必要组成部分。若未完全复现官方评价脚本和官方表格，则论文中应称为 official-aligned 或 ES-MemEval-style full 18-user evaluation；但不能只做 n=30 抽样来替代完整 18 用户外部验证。

## 4. V3.2 系统定义

### 4.1 两阶段因果流程

V3.2 必须严格遵守以下顺序：

```text
Step 1: Pre-evidence PM
  input: current text, recent dialogue, session depth, memory inventory metadata, action id
  output: resource action a = memory_action + strategy_action

Step 2: Evidence execution and response
  retrieve/filter only selected memory source(s)
  retrieve strategy only if RS
  build prompt
  fixed generator produces response
```

关键约束：

- Step 1 不能读取 `memory_context`；
- Step 1 不能读取 `selected_memory_ids`；
- Step 1 不能读取检索分数、实际 evidence chars、response 或 judge score；
- Step 1 可以读取 memory inventory metadata，例如某类记忆是否存在、数量、年龄范围、目录标签统计；
- Step 2 才执行真实检索和回复生成。

因此，V3.2 PM 是 resource allocator，不是 post-retrieval reranker。

### 4.2 动作空间

V3.2 使用 8 类 memory action × 2 类 strategy action。

Memory action:

- `M0`: 不调用长期记忆；
- `MP`: Preference Memory，偏好、习惯、支持风格；
- `MS`: Summary Memory，跨会话摘要和长期模式；
- `ME`: Event Memory，具体事件或片段；
- `MPMS`: MP + MS；
- `MPE`: MP + ME；
- `MSE`: MS + ME；
- `MPMSME`: MP + MS + ME。

Strategy action:

- `R0`: 不调用 Strategy RAG；
- `RS`: 调用 Strategy RAG。

完整动作形如 `ME+RS`、`M0+R0`、`MPMSME+RS`。

### 4.3 固定资源与 PM 可控资源

固定资源：

- 当前用户发言；
- 近期对话上下文；
- 安全约束和基础系统提示；
- 必要的非 PM 控制信息。

PM 可控资源：

- 长期记忆源：MP / MS / ME；
- Strategy RAG：RS。

Raw-session RAG 不作为 PM 主动作，而作为强基线或 diagnostic baseline。

## 5. 数据设计

### 5.1 当前可用数据

当前项目中已恢复和保留的数据：

- `data/evidence_allocation_v1/support_gradient_v31/cards_runtime.jsonl`
  - 192 cards；
  - `split_user`: train=120, val=36, test=36；
  - `split_semantic`: train=96, val=48, test=48；
  - 用作 V3.2 development card source。
- `outputs/v31_api_full/action_sweep_results.jsonl`
  - V31 API action trajectories；
  - 仅复用 response / selected evidence / trajectories；
  - 旧 judgment / utility / oracle / checkpoint 全部作废。
- `outputs/v32_source/action_outcomes_sanitized.jsonl`
  - V32 sanitized source；
  - 移除旧 judge、utility、oracle、private seeker fields、semantic IDs、retrieval scores 等。
- `outputs/v32_pair_graph_n20/pair_graph.jsonl`
  - n=20 judge pilot pair graph。

已恢复的外部资源：

- `data/evoemo_eval_v2/`
  - EvoEmo / ES-MemEval-derived longitudinal raw/eval cards；
  - `evo_emo.json`: 18 users；
  - `cards_evoemo_eval_v2.jsonl`: 401 longitudinal cards；
  - 本篇长期记忆外部验证必须使用完整 18 用户，而不是只用 n=30 抽样；
  - 旧结果只作 legacy reference，正式结果必须用 V3.2 冻结 PM 重新评估。
- `evidence_allocation_v1/esceval_adapter/`
  - ESC-Eval adapter；
  - 当前只作为后续 simulator work 的接口资源。
- `data/strategy/strategy_cards_v13.jsonl`
  - Strategy cards。

### 5.2 当前数据限制

当前 `cards_runtime.jsonl` 中：

- `semantic_family` 未填充；
- top-level `support_need` 未填充；
- 因此不能直接声称 semantic-family generalization 或 support-need-stratified evaluation；
- 若论文需要这些 claim，必须重建字段并重新验证。

V3.2 第一篇可以采用更保守表述：

> controlled counterfactual evaluation with user-held-out splits and resource-availability variation.

不能声称：

> open-domain natural-language generalization.

### 5.3 Final test 原则

当前 192 cards 和 V31 trajectories 已经过多轮开发查看，因此只能作为 development set。

正式论文主结果需要：

- 在 development set 上完成 judge prompt、utility 形式、model class、alpha/epsilon/tau 等所有选择；
- 冻结所有超参数；
- 重新生成或保留一份从未查看的新 final test；
- final test 只跑一次 confirmatory evaluation。

## 6. V3.2 Judge 设计

V3.2 不再使用单一绝对 1-5 overall 分数作为主要训练目标。原因是旧 response judge 饱和严重，绝对分数无法稳定区分动作。

### 6.1 Response Pairwise Judge

输入：

- 可见用户上下文；
- response A；
- response B。

不可见：

- action id；
- memory evidence；
- strategy evidence；
- selected memory ids；
- old utility。

输出：

```json
{
  "preference": "A | B | tie",
  "empathy": "A | B | tie",
  "contextual_fit": "A | B | tie",
  "guidance_fit": "A | B | tie",
  "non_intrusiveness": "A | B | tie",
  "coherence": "A | B | tie",
  "reason": "..."
}
```

用途：

- 训练或估计 response quality / response ranking；
- 检查 evidence 对回复本身是否有可观测影响；
- 通过 A/B reversal consistency 检查 judge 稳定性。

Go/no-go diagnostics：

- judge failure rate = 0；
- reversal consistency 不能过低；
- tie rate 如果 >80%，说明 response signal 仍然太弱；
- tie rate 不是唯一硬门槛，但必须解释。

### 6.2 M1 Inventory Audit

输入：

- 可见上下文；
- available memory inventory。

输出每个 memory item 的：

- current relevance；
- potential helpfulness；
- likely stale/conflicting；
- sensitive/intrusive risk。

作用：

- 审计 memory inventory 的机会和风险；
- 不直接泄露给 M2；
- 不作为 seeker prompt；
- 不能让 PM 在部署时看到 judge 结果。

### 6.3 M2 Memory Action Audit

输入：

- 可见上下文；
- available memory inventory；
- selected memory items；
- generated response。

若 memory action = `M0`：

```json
{
  "memory_action_type": "M0",
  "omission_appropriateness": 0|1|2,
  "missed_useful_sources": ["MP|MS|ME"],
  "unsupported_personal_claim": 0|1|2,
  "reason": "..."
}
```

若 memory action != `M0`：

```json
{
  "memory_action_type": "memory_used",
  "source_set_appropriateness": 0|1|2,
  "selected_sources": [
    {
      "source": "MP|MS|ME",
      "relevance": 0|1|2,
      "utilization": 0|1|2,
      "unused_retrieval": 0|1|2,
      "stale_or_conflicting": 0|1|2,
      "unnecessary_exposure": 0|1|2
    }
  ],
  "missed_useful_sources": ["MP|MS|ME"],
  "unsupported_personal_claim": 0|1|2,
  "reason": "..."
}
```

关键修复：

- M0 正确不使用记忆会获得 omission appropriateness；
- M0+RS 不得获得 memory-use credit；
- Strategy RAG 不再被当作长期记忆奖励。

### 6.4 Strategy Audit

仅用于 RS 动作：

```json
{
  "strategy_action_type": "RS",
  "strategy_relevance": 0|1|2,
  "strategy_utilization": 0|1|2,
  "over_structuring": 0|1|2,
  "premature_advice": 0|1|2,
  "reason": "..."
}
```

Strategy 的主效果仍然应优先通过同 memory 条件下的 pairwise comparison 估计：

```text
M+R0 vs M+RS
```

Strategy audit 主要用于解释和 misuse diagnostics。

## 7. Pair Graph 设计

V3.2 使用 sparse pair graph，而不是所有动作全排列。

主要 pair 类型：

- `strategy_same_memory`: 同 memory 下 R0 vs RS；
- `memory_vs_m0`: 单一 memory source vs M0；
- `combo_vs_subset`: 组合动作 vs 子集动作；
- `combo_vs_single`: 组合动作 vs 单一 source。

约束：

- 只从 sanitized source 构建；
- 不允许 silently add fallback edges；
- pair graph 必须 connected，否则 fail closed；
- richer action 在 A/B 位置上尽量平衡；
- 部分 pair 做 A/B reversal，用于 position bias 检查；
- pair loss 训练时应按 card 平衡，而不是把所有 pair 当独立样本。

## 8. PM 学习目标

V3.2 不建议继续使用旧式单一加权 utility：

```text
U = w_response Q_response + w_memory Q_memory + w_strategy Q_strategy - alpha cost - risk
```

因为 memory / strategy 的好处可能已经反映在 response pairwise quality 中，再加一次 memory score 会重复奖励。

更干净的决策框架是约束式或分头式：

```text
A_epsilon = {
  a : Q_response(x,a) >= max_a Q_response(x,a) - epsilon
      and Q_misuse(x,a) <= tau
}

a* = argmax_{a in A_epsilon} [
  beta_m Q_memory_decision(x,a)
  + beta_s Q_strategy_decision(x,a)
  - alpha Cost(a)
]
```

其中：

- `Q_response`: 由 blind pairwise response judge 学到；
- `Q_memory_decision`: memory use / omission appropriateness；
- `Q_strategy_decision`: strategy relevance/utilization minus over-structuring/premature-advice；
- `Q_misuse`: stale/conflicting/unnecessary exposure/unsupported personal claim；
- `Cost(a)`: expected retrieval/context cost；
- `epsilon, tau, alpha, beta_m, beta_s` 必须在 validation 上冻结。

若 pairwise response pilot 仍然高度饱和，则 response head 不能支撑此框架，必须回到实验设计层面重新思考 generator / task difficulty / evidence impact。

## 9. 基线设计

正式评估至少需要以下基线：

1. `M0+R0`: no memory, no strategy；
2. `M0+RS`: strategy-only fixed；
3. validation-selected best fixed action；
4. budget-matched fixed action；
5. rule policy；
6. metadata-only router；
7. text-only router；
8. raw-session RAG diagnostic；
9. all-evidence / full structured diagnostic。

关键要求：

- best fixed 必须在 validation 上选择，不能在 test 上选择；
- final test 不能再扫 alpha；
- PM 必须不仅超过 weak rule，还要对强固定策略有解释充分的优势；
- 如果 PM 只比 M0+RS 略高且成本更高，不能声称强 adaptivity。

## 10. 评价指标

### 10.1 Judge / label health

- response pair tie rate；
- pairwise A/B reversal consistency；
- judge failure count；
- memory schema violation count；
- M0+RS memory violation count；
- strategy schema violation count；
- calibration control pass rate。

### 10.2 Policy quality

- pairwise response win/tie/loss；
- response ranking regret；
- memory decision appropriateness；
- omission appropriateness for M0；
- unnecessary exposure；
- stale/conflict risk；
- unsupported personal claim；
- strategy over-structuring / premature advice；
- retrieval/context cost；
- selected action distribution；
- PM vs rule / fixed / raw RAG paired differences。

### 10.3 Generalization diagnostics

- user-held-out performance；
- semantic-held-out performance if semantic fields are rebuilt；
- seen vs unseen text-profile combinations；
- availability counterfactual:
  - same text, different memory availability；
  - PM action should change when resource availability changes.

## 11. 实验阶段

### Stage A：恢复与审计资源

已完成：

- V31/V32 current resources restored；
- EvoEmo and ESC-Eval adapter resources restored；
- legacy outputs/models isolated under legacy directories；
- V32 sanitized source validator passes。

### Stage B：V3.2 judge pilot

当前目标是小规模 API pilot。它只检验 generator / judge / schema 是否可用，**不是**论文最终实验，也不需要先接入官方 ESConv 或重建正式 Strategy Bank。

当前使用：

- `data/synthetic/runtime_states.jsonl`
- `data/synthetic/memory_backend.jsonl`
- `data/strategy/strategy_cards.jsonl`（pilot-only Strategy Bank）

运行：

```bash
PYTHONPATH=src python scripts/05_generator_variance.py --n-cards 8
PYTHONPATH=src python scripts/06_run_action_sweep.py --max-cards 20
PYTHONPATH=src python scripts/07_run_judge_pilot.py --max-cards 20
PYTHONPATH=src python scripts/08_analyze_judge_pilot.py
```

Go/no-go：

- no judge failures；
- no schema violation；
- M0+RS memory violation = 0；
- M0 omission score rate = 1；
- non-M0 sourcewise assessment rate = 1；
- response tie rate and reversal consistency are diagnostically acceptable。
- gate 输出 `PILOT_GO_FULL_JUDGING`。

### Stage C：全量 development re-judge

如果 pilot 通过：

- 对 full development source 进行 pairwise response judge；
- 进行 memory action audit；
- 进行 strategy audit；
- 保存完整 V3.2 labels。

旧 labels / utility / oracle 不复用。

### Stage D：训练 V3.2 PM

训练目标：

- response pairwise ranker；
- memory decision head；
- strategy decision head；
- misuse/risk head；
- cost-aware selector。

训练原则：

- train split 训练；
- validation 冻结超参数；
- test 只用于一次性报告；
- pair loss 按 card 平衡；
- 低方差/常数特征删除或冻结，不能让 OOD 激活随机权重。

### Stage E：内部 development evaluation

报告：

- PM vs validation-selected fixed；
- PM vs rule；
- PM vs raw-session RAG diagnostic；
- PM action distribution；
- cost / risk / response quality trade-off；
- availability counterfactual。

这阶段仍是 development，不是 final claim。

### Stage F：外部验证

可用外部资源：

- EvoEmo / ES-MemEval-derived restored data；
- ESC-Eval adapter；
- ESConv old data/cache。

当前第一篇建议：

- ESConv 用于 Strategy routing / response-quality / false-memory-control；
- EvoEmo / ES-MemEval-derived 18-user data 是长期记忆外部主验证，必须全 18 用户运行；
- ESC-Eval 暂时只作为 simulator adapter / future work，除非正式构建闭环多轮 evaluation；
- 若未完全复现 ES-MemEval 官方脚本，称为 official-aligned / ES-MemEval-style，而不是官方表格复现。

外部验证必须注意：

- PM 与 baselines 使用相同 generator；
- baselines 是否也允许 RS 必须公平；
- judge prompt 不得偏向记忆系统；
- response-only 与 evidence-use 指标分开；
- 外部样本不用于 PM 训练或调参。

### Stage G：Final test

正式论文主 claim 需要一份 untouched final test：

- 在 development 上冻结所有设计；
- final test 不扫 alpha；
- final test 不选 best fixed；
- final test 不修改 judge prompt；
- final test 失败就诚实报告，不再回头调。

## 12. 当前文件与职责

### V3.2 API-ready current package

当前已合并 `MetaCom_PM_API_Ready_Frozen_20260621.zip` 的主工程骨架，并在本项目目录下通过语法编译和 `--skip-tests` preflight。它应作为后续 V3.2 的主代码路径。

- `src/metacom_pm/contracts.py`
  - 严格 action schema、runtime state、memory backend、judgment schema、action outcome contract。
- `src/metacom_pm/features.py`
  - pre-evidence feature builder；只使用当前对话、inventory metadata、catalog fingerprint、action id。
- `src/metacom_pm/sweep.py`
  - clean action sweep；先由 PM/baseline 选择 action，再只检索被选资源生成 response。
- `src/metacom_pm/judging.py`
  - response pairwise judge、M1 inventory audit、M0 omission audit、M2 memory-use audit、strategy audit；保存 raw judge calls 和 run manifest。
- `src/metacom_pm/pilot_analysis.py`
  - judge pilot gate；检查 held-out controls、A/B reversal、一致性、completion counts。
- `src/metacom_pm/training.py`
  - V3.2 PM trainer：pairwise response ranker + misuse regressor + memory decision regressor。
- `src/metacom_pm/selection.py`
  - validation-only selection：best fixed、budget-matched fixed、PM epsilon/tau、strong rule grid。
- `src/metacom_pm/freeze.py`
  - study-freeze 创建与校验；正式外部评测必须 fail-closed。
- `src/metacom_pm/evoemo.py` 与 `src/metacom_pm/evo_metrics.py`
  - EvoEmo / ES-MemEval-style 18-user external generation 与 official-aligned / selective-memory metrics。
- `src/metacom_pm/esconv.py`
  - ESConv Strategy routing / response-quality / false-memory-control evaluation。
- `scripts/00_prepare_synthetic.py` 到 `scripts/20_freeze_study.py`
  - 编号化端到端运行脚本。
- `scripts/99_release_preflight.py`
  - 静态、数据、Strategy Bank、EvoEmo、config、freeze preflight。
- `configs/experiment.yaml`
  - API endpoint/model 配置模板；当前仍为 placeholder，正式 API 运行前必须填写。
- `README_CN.md`、`docs/RUNBOOK_CN.md`、`docs/STUDY_PROTOCOL_CN.md`
  - 运行手册、边界声明和冻结协议。

### Current development data

- `data/synthetic/runtime_states.jsonl`
  - 1728 runtime cards = 192 states × 9 counterfactual inventory variants；
  - 16 synthetic users、12 semantic families、36 unique current texts；
  - 用于 development / training，不作为最终外部 claim。
- `data/synthetic/memory_backend.jsonl`
  - 与 runtime cards 物理分离的 memory backend。
- `data/synthetic/pair_graph.jsonl`
  - sparse connected pair graph；约 19,776 training pairs，另含 reversal / same-order repeats。
- `data/external/evo_emo.json`
  - EvoEmo / ES-MemEval-derived 18 users、401 sessions、34 scenarios。
- `data/strategy/strategy_cards.jsonl`
  - pilot-only Strategy Bank；正式 ESConv/EvoEmo confirmatory 前必须用官方 ESConv 重建并做 overlap audit。

### Legacy / development-only files

以下资源保留用于审计与历史复核，但不能直接作为 V3.2 论文主结论：

- `data/evidence_allocation_v1/support_gradient_v31/cards_runtime.jsonl`
- `outputs/v31_api_full/action_sweep_results.jsonl`
- `outputs/v32_source/action_outcomes_sanitized.jsonl`
- `outputs/v32_pair_graph_n20/pair_graph.jsonl`
- `src/metacom_pm/v32_contract.py`（唯一实现；`scripts/v32_contract.py` 仅为兼容 wrapper）
- `scripts/build_v32_rejudge_source.py`
- `scripts/run_v32_judge_pilot.py`
- `scripts/analyze_v32_judge_pilot.py`

### Restored external resources

- `data/evoemo_eval_v2/`
- `data/strategy/`
- `evidence_allocation_v1/esceval_adapter/`
- `scripts/eval_evoemo_generation_judge_v1.py`
- `scripts/run_esceval_pm_adapter_v1.py`

### Legacy resources

- `outputs/legacy_paper_artifacts_20260620_1358/`
- `checkpoints/legacy_paper_artifacts_20260620_1358/`
- `legacy_paper_artifacts_20260620_1358/`

Legacy resources can be used for audit/history, not for current V3.2 claims.

## 13. 论文主张的安全版本

如果 V3.2 pilot、full re-judge、training 和 final test 都通过，论文主张可以写为：

> We formulate evidence use in longitudinal emotional-support dialogue as a pre-retrieval resource allocation problem. Our Policy Manager selects which memory and strategy resources to activate using only deployment-observable state and inventory metadata. With blind pairwise response evaluation and separate memory/strategy audits, we show that learned pre-evidence allocation can improve over heuristic resource gates and avoid unnecessary retrieval, while preserving response quality and reducing memory misuse risk.

中文：

> 本研究将长期情感支持对话中的证据使用建模为检索前资源分配问题。PM 只基于部署时可见状态和记忆目录元数据，选择激活哪类记忆与策略资源。通过盲式 pairwise 回复评价和分离式记忆/策略审计，我们检验学习式检索前分配是否能优于规则门控，并减少不必要检索与记忆误用。

不能写：

- “PM 已经证明真实改善用户情绪”；
- “PM 完整实现 RL”；
- “PM 在开放域自然语言上泛化”；
- “旧 n=88 / EvoEmo 表格证明最终结论”。

## 14. 主要风险与防护

### 风险 1：response pairwise 仍然饱和

防护：

- pilot 阶段看 tie rate；
- 若 tie rate 太高，不能继续训练 PM；
- 需要修改任务难度、response generation setup 或 judge prompt。

### 风险 2：PM 只学固定模板

防护：

- availability counterfactual；
- text-only / metadata-only probe；
- seen vs unseen text-profile analysis；
- 谨慎声明 controlled setting。

### 风险 3：Strategy 再次被当成 memory reward

防护：

- M0+RS memory violation hard gate；
- memory audit 与 strategy audit 分离；
- M0 omission 单独评分。

### 风险 4：test 被反复查看

防护：

- 当前 development set 不作为 final test；
- 新 final test 只跑一次；
- validation 冻结所有参数。

### 风险 5：baseline 太弱

防护：

- 加 validation-selected fixed；
- 加 budget-matched fixed；
- 加 text-only / metadata-only router；
- 加 raw-session diagnostic。

## 15. 与后续 RL 版本的关系

V3.2 不是最终博士课题的全部，只是第一篇论文的稳健版本。

合理路线：

1. 第一篇：pre-retrieval contextual resource allocation PM；
2. 第二篇：接入真实用户模拟器，构建 RL environment；
3. 第三篇：deep RL / future-oriented reward；
4. 第四步：Tokkio 多模态输入，音频/视频/文本状态估计；
5. 最后：真实用户或专家人评。

V3.2 的价值是给 RL 版本打地基：

- 明确动作空间；
- 明确资源成本；
- 明确 judge 与风险指标；
- 明确不能让 seeker / judge / PM 看到不该看的东西；
- 明确 causal boundary。

## 16. 下一步执行清单

短期分成三道门，不要混在一起。

### 16.1 当前可做：API pilot

目的：确认 generator、judge、pair graph、memory/strategy schema 和断点恢复能真实跑通。

这一步 **不需要** 先接入官方 ESConv，也 **不需要** 重建正式 Strategy Bank；可以使用当前 pilot-only Strategy Bank。

1. 填写 `configs/experiment.yaml` 中 generator / seeker / training_judge / final_judge 的 endpoint 和 model；
2. 重新运行 `scripts/99_release_preflight.py --skip-tests`，确认状态至少为 `API_PILOT_READY`；
3. 跑 generator variance：`scripts/05_generator_variance.py --n-cards 8`；
4. 跑 20-card action sweep：`scripts/06_run_action_sweep.py --max-cards 20`；
5. 跑 V3.2 judge pilot：`scripts/07_run_judge_pilot.py --max-cards 20`；
6. 分析 pilot gate：`scripts/08_analyze_judge_pilot.py`。

只有 gate 输出 `PILOT_GO_FULL_JUDGING`，才进入下一步。

### 16.2 Pilot 通过后：development full judging / training

目的：在 synthetic development set 上重打 V3.2 labels、训练 PM、只用 validation 冻结选择规则。

1. 跑 full synthetic action sweep；
2. 跑 full judging；
3. 训练多 seed / 多 fold PM 和 learned baselines；
4. 在 validation 上冻结 best fixed、budget-matched fixed、epsilon、tau、strong rule grid；
5. 用冻结结构训练 final PM checkpoint。

这一步仍然是 development，不是论文最终证据。

### 16.3 正式外部实验前：confirmatory hygiene

目的：锁住可报告实验，避免 Strategy Bank 或外部测试泄漏。

1. 接入官方 ESConv：`scripts/02_download_esconv.py`；
2. 只用 ESConv train 重建 Strategy Bank：`scripts/03_build_strategy_bank.py`；
3. 构建 ESConv test runtime：`scripts/12_build_esconv_test.py`；
4. 重新运行 preflight，确认官方 ESConv、overlap audit、ESConv test runtime 和配置都有效；
5. 创建 `outputs/study_freeze.json`；
6. 只有状态达到 `CONFIRMATORY_READY`，才运行 ESConv 和 EvoEmo / ES-MemEval-style full 18-user external evaluation。

当前不建议：

- 直接用旧 V31 checkpoint；
- 直接引用旧 n=88 表；
- 直接用旧 EvoEmo 表当主结果；
- 用 n=30 EvoEmo 抽样替代完整 18-user long-term external evaluation；
- 在未解决 pairwise 饱和前跑全量训练。

## 17. 冻结版协议补充：论文题目、定位与相关工作边界

建议英文题目方向：

> When Should an Emotional Support Agent Remember?  
> Pre-Evidence Resource Allocation for Longitudinal Emotional Support Conversations

中文题目方向：

> 情感支持对话中的选择性记忆与策略检索：一种检索前 Policy Manager

第一篇论文的准确方法名称应为：

> supervised pre-evidence resource allocation policy

不要称为：

- RL；
- POMDP；
- RLHF / DPO / GRPO；
- 端到端 ESC generation model；
- clinical safety system。

与先行工作的叙事边界：

- 许多 ESC 工作关注单会话语境理解、支持策略选择、共情生成或生成器训练；
- D2RCU 类工作可以作为“增强生成器利用 demonstration / persona / cognitive understanding”的对比背景，但本研究不直接优化生成器；
- future-oriented reward / RL 类工作关注多轮未来收益和端到端回复策略优化，本研究只做固定生成器外部的资源调度；
- ES-MemEval / EvoEmo 类长期 ESC benchmark 说明长期记忆需要信息提取、时序推理、冲突检测、拒答和用户建模。本研究第一篇必须在完整 18-user longitudinal setting 上验证长期记忆资源分配；若不能完全复现官方脚本，则明确称为 official-aligned / ES-MemEval-style，而不是声称复现官方 leaderboard。

因此论文应从一开始明确：

> PM is a decision layer outside the generator. It selects which resource to activate before evidence retrieval, under quality, misuse-risk, and cost constraints.

## 18. 冻结版研究问题与假设

当前文档已有 RQ1-RQ4。冻结版可进一步写成以下可检验假设。

### H1：Strategy RAG 路由

在无长期记忆或普通 ESC 场景中，PM 是否能合理决定是否调用 Strategy RAG？

成功标准不是必须“质量大幅提升”，也可以是：

- 相对 always-RS 质量非劣但 RS 调用更少；
- 相对 always-R0 在需要承接或策略支持时质量更好；
- 相对 strong rule strategy router 有更好质量-成本权衡。

### H2：长期记忆选择

在跨 session 情境中，PM 是否能选择合适的 MP / MS / ME 组合，并避免无关、过时、冲突或突兀记忆？

成功标准：

- 记忆决策适当性更高；
- stale/conflict/unnecessary exposure 更低；
- response quality 不低于最强固定或强规则基线。

### H3：条件自适应

PM 是否真正根据当前语境和 memory inventory 改变动作，而不是退化为固定策略？

必须比较：

- validation-selected best fixed action；
- budget-matched fixed action；
- strong rule router；
- text-only router；
- metadata-only router。

还要做反事实：

- same text / different inventory；
- same inventory / different context；
- relevant memory available vs unavailable；
- stale/conflicting memory present vs absent。

### H4：资源效率

PM 是否在质量和记忆风险不劣的前提下减少资源消耗？

至少报告：

- memory retrieval calls；
- strategy retrieval calls；
- selected evidence tokens；
- total input tokens；
- completion tokens；
- p50 / p95 latency；
- API 或 GPU cost。

必须区分：

- offline action sweep cost；
- deployment policy cost。

### H5：生成器依赖

PM 的效果不能只存在于一个精挑细选的生成模型上。

理想设置：

- 一个主生成器；
- 一个跨模型家族 robustness generator；
- 生成器选择在 development pilot 前冻结；
- 不能根据 test 上哪个 generator 让 PM 赢得最多来选择。

若时间不足，H5 可作为 secondary / robustness，而不是第一篇主 claim。

## 19. PM 输入：允许项与禁止项

V3.2 的 PM 输入必须是 deployment-observable、pre-evidence 的。

### 允许输入

- 当前用户文本；
- 当前 session 最近对话；
- 当前 session 专用摘要；
- turn index / session depth；
- MP / MS / ME 是否存在；
- 每类资源条目数；
- 每类资源年龄范围、最小/最大 age；
- source-level catalog descriptor；
- 由历史 memory 在写入时预计算的 source-level catalog embedding；
- 当前文本 embedding 与 catalog embedding 的相似度；
- action one-hot；
- action 预计成本。

重要细节：

- catalog embedding 必须由当时已经存在的历史 memory 构建；
- catalog similarity 是轻量目录探测，不是读取实际 evidence text；
- 如果使用 query embedding / catalog similarity，它的计算成本和延迟必须计入 cost。

因此比 “pre-retrieval” 更精确的表述是：

> pre-evidence resource allocation

即 PM 可以读取轻量目录信息，但不能读取实际 evidence snippets。

### 禁止输入

- actual memory snippets；
- Strategy RAG snippets；
- selected memory IDs；
- retrieval scores；
- top-k 检索结果；
- 实际检索后的上下文长度；
- 生成回复；
- seeker reaction；
- judge 分数；
- gold action；
- expected source；
- role card；
- scenario label；
- problem type gold；
- emotion type gold；
- relevant-session annotations；
- stale/conflict 审计金标；
- future session 或 future event。

若任一禁止项进入 PM scorer，实验应视为 invalid。

## 20. 资源构建规则

### MP：Persona / Preference Memory

只存储稳定或慢变信息：

- 用户偏好；
- 沟通偏好；
- 稳定身份；
- 长期关系；
- 长期限制或习惯。

不得把大量具体事件直接塞入 MP。

### MS：Memory Summary

跨 session 压缩信息：

- 长期模式；
- 反复出现的问题；
- 用户状态演变；
- 多次会话形成的概括。

MS 只能由当前时间点之前的会话构建。

### ME：Episodic Memory

带时间信息的具体事件：

- 某次分手；
- 工作变化；
- 家庭冲突；
- 医疗或生活事件；
- 后续更新事件。

ME 必须保留时间和更新顺序。

### RS：Strategy RAG

只包含通用情感支持资源：

- ESC strategy；
- 支持方式；
- 承接与追问方式；
- 适当建议方式；
- delexicalized examples。

不得包含当前用户的长期个人事实。

## 21. Strategy RAG 构建与泄漏控制

Strategy RAG cards 只能从 ESConv train split 构建。

禁止：

- 使用 ESConv dev/test；
- 使用 ESConv gold strategy labels 作为检索 query；
- 使用 problem type gold；
- 使用 emotion type gold；
- 使用 hidden support-need label；
- 将完整训练回复作为唯一 guidance；
- 保留可识别测试样本的具体个人事实。

建议 card 格式：

```json
{
  "strategy_type": "...",
  "preceding_context": "...",
  "generalized_guidance": "...",
  "delexicalized_example": "..."
}
```

处理要求：

- 删除人名、日期、具体身份；
- 对 cards 去重和近重复过滤；
- 若 EvoEmo / external evaluation 与 ESConv 存在重叠，应通过 normalized text hash 或 near-duplicate detection 删除可能泄漏的 cards。

Strategy RAG query 只能使用：

- 当前用户文本；
- 当前 session context；
- 当前 session summary。

## 22. 数据使用边界

### A. Synthetic / support-gradient 数据

用途：

- PM 训练；
- judge prompt development；
- model selection；
- counterfactual analysis；
- internal development diagnostics。

不能作为最终外部效果证明。

合成数据正确生成顺序：

1. 独立生成完整用户背景和多 session 历史；
2. 生成当前 session 和当前用户话语；
3. 从过去历史构建 MP / MS / ME；
4. 构造反事实 inventory；
5. 枚举合法动作并生成候选回复；
6. 对候选回复和记忆使用重新评价；
7. 训练 PM。

禁止：

- 先指定目标动作，再反向生成一句明显应该使用某动作的用户文本；
- 把 action cue phrase 写进 current user text；
- 把 gold source 或 private need 作为 PM 可见输入。

反事实 inventory 应覆盖：

- 无可用长期记忆；
- 有相关 MP；
- 有相关 MS；
- 有相关 ME；
- 有无关 ME；
- 有过时 ME；
- 有旧事实和更新事实；
- 三类 memory 都存在但本轮不需要；
- 多源中只有一源相关；
- 用户偏好不主动提旧事。

### B. ESConv

ESConv 是单 session 数据，因此不能用来证明长期记忆能力。

正式 ESConv evaluation 中，MP / MS / ME 不可用，合法动作只有：

- `M0+R0`
- `M0+RS`

ESConv 主要评价：

- Strategy RAG 路由；
- 普通 ESC 回复质量；
- 无长期历史时是否避免编造用户个人经历。

可称为：

- Strategy routing；
- memory abstention；
- false-memory control。

不能称为：

- long-term memory improvement。

### C. EvoEmo / ES-MemEval-derived 18-user longitudinal evaluation

当前项目已恢复 EvoEmo / ES-MemEval-derived 数据：

- `data/evoemo_eval_v2/evo_emo.json`: 18 users；
- `data/evoemo_eval_v2/cards_evoemo_eval_v2.jsonl`: 401 longitudinal cards；
- 每个 user 有多 session history、questions、summaries、event experience 等长期记忆相关字段。

这条线是第一篇的长期记忆外部主验证，不能降级为 future work，也不能用 n=30 抽样替代。

建议设计两个轨道：

1. official-comparable track：
   - No Memory；
   - Full History；
   - official-style RAG；
   - PM；
   - 如果官方指标可完整实现，则报告 observation recall、weighted score、long-term memory、personalization、emotional support；
   - 如果只是对齐官方思想但没有完全复现官方脚本，则命名为 official-aligned / ES-MemEval-style track。
2. selective-memory extension：
   - 中性 supporter prompt；
   - 不强制主动提记忆；
   - 评价 omission、misuse、stale/conflict、unnecessary exposure。

EvoEmo / ES-MemEval-derived long-term evaluation 必须遵守时间约束：

- 只能使用 `timestamp < current scenario timestamp` 的内容；
- 禁止 future session；
- 禁止 benchmark reference observations；
- 禁止 relevant prior sessions gold；
- 禁止 event timeline gold；
- 禁止 reference response。

## 23. 候选回复生成协议

所有 PM 和 baseline 条件必须共享：

- 相同生成模型；
- 相同基础 system prompt；
- 相同当前 session context；
- 相同检索器；
- 相同 top-k；
- 相同 token cap；
- 相同 decoding 参数。

唯一变化是：

> 谁选择了哪些资源。

生成器选择不能根据 test 胜负决定。

建议在 development pilot 前比较：

- 一个中型 Qwen；
- 一个 Llama 系模型；
- 可选更强生成器作为 ceiling。

选择标准：

- 基础 ESC 质量合格；
- 能实际利用 memory 和 strategy；
- 不出现所有动作完全相同的 ceiling；
- 不出现生成质量过低的 floor；
- 同一 action 随机波动低于不同 action 差异；
- 成本可承担。

确认性测试建议：

- `temperature=0`；
- 或每个条件使用 matched seeds。

不能让不同动作的随机采样差异被误认为资源选择效果。

## 24. 训练目标细化

推荐模型结构：

```text
Frozen context encoder
  + inventory metadata
  + catalog similarity
  + action embedding
  -> 2-layer MLP action scorer
```

主要输出：

- `Q_response(x,a)`：该动作产生更优回复的相对分数；
- `Q_misuse(x,a)`：记忆误用风险；
- `Q_memory_decision(x,a)`：记忆资源决策适切性；
- `Q_strategy_decision(x,a)`：策略资源决策适切性；
- `C(x,a)`：预计成本，可由确定性公式或独立简单模型得到。

主要损失：

```text
P(a_i > a_j | x) = sigmoid(f_theta(x,a_i) - f_theta(x,a_j))
```

使用 pairwise ranking loss，而不是回归旧的 1-5 overall。

每个 state/card 内所有 pair loss 总权重归一为 1，避免动作较多的卡片贡献更大权重。

训练流程：

- scaler 只在 train fit；
- 删除零方差特征；
- 做 OOD/range 检查；
- 至少 5 个随机 seed；
- validation 选择结构、阈值和规则；
- final test 前冻结全部参数。

## 25. 部署决策规则

不采用简单加权总分：

```text
U = response + memory + strategy - cost
```

避免重复计算 memory / strategy 对 response 的影响。

推荐约束式选择：

```text
q_max = max_a Q_response(x,a)

A_epsilon = {
  a:
    Q_response(x,a) >= q_max - epsilon
    and Q_misuse(x,a) <= tau
}

a* = argmin_{a in A_epsilon} C(x,a)
```

如果多个动作成本相近，再用 memory-decision / strategy-decision score 作为 tie-break。

必须在 validation 上冻结：

- `epsilon`；
- `tau`；
- cost budget；
- tie-break threshold。

## 26. 实验与统计协议补充

### Experiment 0：泄漏与数据审计

正式实验前必须输出：

- exact duplicate；
- near duplicate；
- semantic similarity；
- Strategy card 与 ESConv test 重叠；
- Strategy card 与 EvoEmo session 重叠；
- future memory 检查；
- gold/private 字段扫描；
- action cue phrase audit；
- text-only shortcut probe；
- metadata-only shortcut probe。

任何 gold、future 或 test contamination 均停止实验。

### Experiment 1：Synthetic development

目的：

- 验证 judge；
- 训练 PM；
- 做模型消融；
- 选择阈值；
- 分析 counterfactual sensitivity。

不作为最终外部效果证明。

### Experiment 2：ESConv Strategy / abstention evaluation

比较：

- PM；
- always-R0；
- always-RS；
- validation-selected best fixed；
- strong rule strategy router。

指标：

- direct pairwise response preference；
- W/T/L；
- cluster bootstrap 95% CI；
- unsupported personal claims；
- RS call rate；
- token/cost。

### Experiment 3：Longitudinal memory evaluation

EvoEmo / ES-MemEval-derived 18-user evaluation：

- 必须覆盖全部 18 users；
- 优先使用全部 401 cards；若因预算使用子集，必须预注册 sampling rule，并且不能作为完整外部主结论；
- user-cluster bootstrap；
- 不在看到结果后重采样；
- 时间截断；
- 中性 prompt 和 selective-memory 指标。

若补齐或确认 ES-MemEval official-compatible protocol：

- official-comparable track 与 selective-memory extension 必须分开报告。

### Experiment 4：ESC-Eval 多轮辅助实验

仅在 PM 冻结后执行。

用途：

- 多轮互动质量；
- PM 是否过度调用 memory；
- RS 是否导致过度结构化；
- 普通 ESC 情境中是否保持 M0。

限制：

- role card 只给 seeker simulator；
- PM 只能看自然对话；
- simulator 不看 PM action 或 evidence；
- 结果为辅助，不作为唯一主结论。

### 统计分析

ESConv：

- 以 dialogue 为 cluster；
- paired bootstrap；
- human majority vote；
- 多 seed 时使用 card × seed 配对。

EvoEmo / long-term：

- 以 user 为 cluster；
- hierarchical bootstrap: user -> scenario -> turn；
- leave-one-user-out；
- 检查是否单个用户主导结论。

多重比较：

- 预注册一个主要 ESConv baseline；
- 预注册一个主要 long-term baseline；
- 两个 co-primary endpoints；
- 其余作为 secondary/exploratory。

## 27. 成功标准与失败解释

不要用一个混合 `claim_ready` 决定论文是否成立。

### 成功模式 A：质量和记忆优势

相对 strongest baseline：

- response preference 显著更好；
- memory primary metric 更好；
- misuse 不更差；
- 成本不超过预注册上限。

### 成功模式 B：质量非劣、资源更优

相对 strongest baseline：

- response quality 满足预注册非劣界；
- memory quality 满足非劣；
- retrieval calls 或 input tokens 显著下降；
- misuse 不更差。

### 若 PM 不优于 strong rule

可报告：

> 在当前部署可见信息下，强规则已足以完成资源调度，学习型 PM 未表现出稳定优势。

不能强行写 PM 胜利。

### 若 PM 只降低成本

可报告：

> PM 在保持回复质量和记忆风险非劣的同时，减少不必要资源调用。

### 若 PM 只提高 memory relevance

不能写：

> PM 改善了 ESC 回复质量。

只能写：

> PM 改善了资源选择或记忆适切性。

### 若强生成器下所有动作都接近 tie

不能换弱生成器只为制造优势。

应报告：

> 强生成器对资源变化不敏感，PM 的主要价值体现在资源效率和 memory-use reliability。

## 28. 人工评估与第二 Judge

自动 judge 不能成为唯一证据。

建议：

- 10%–20% response pairs 由第二 judge 复评；
- 抽取一个人工盲评子集；
- 3 名标注者；
- A/B 顺序随机；
- 不显示 action 或 evidence；
- 报告多数票；
- 报告 Fleiss' kappa 或 Krippendorff's alpha；
- 报告自动 judge 与人工方向一致性。

Judge 模型和回复生成模型尽量使用不同模型家族。

## 29. 端到端实验流程冻结版

1. 冻结论文 scope 与研究假设；
2. 冻结 ESConv / synthetic / EvoEmo 数据边界；
3. 用 ESConv train 构建 Strategy RAG；
4. 删除与 ESConv test / EvoEmo 重叠的 cards；
5. 进行 generator sensitivity pilot；
6. 生成或确认 synthetic longitudinal users；
7. 构建 MP / MS / ME 与反事实 inventory；
8. 运行 clean action sweep；
9. 运行 judge calibration controls；
10. 小规模 judge pilot；
11. 冻结 judge prompt 和 pair graph；
12. 全量标注 development action outcomes；
13. 训练多 seed PM 和 learned baselines；
14. 在 validation 冻结 epsilon / tau / budget；
15. 锁定 study protocol；
16. 一次性运行 final test；
17. 一次性运行 ESConv / EvoEmo-ES-MemEval 18-user external evaluation；
18. 运行人工盲评；
19. 可选运行 ESC-Eval 和第二生成器 robustness；
20. 生成统计报告和论文表格。

## 30. 最终论文可以主张什么

最强且合理的主张：

> 本文提出一个固定生成器之外的、检索前的资源分配 PM。该 PM 仅利用当前对话和轻量 memory inventory 信息，动态决定是否调用 Persona、长期摘要、事件记忆与 ESC Strategy RAG。我们分别评价其回复质量、长期记忆使用和资源成本，并检验其相对固定策略和强规则策略的质量—记忆可靠性—成本权衡。

不能主张：

- 降低真实用户 distress；
- 提升临床安全；
- 完成多模态数字人；
- 实现 RL / POMDP / RLHF；
- 直接优于 D2RCU 或 future-oriented RL 工作；
- 证明所有 LLM 上普遍有效；
- 只凭 ESConv 证明长期记忆能力；
- 只凭 memory score 证明回复质量更好。

核心逻辑：

> Synthetic / support-gradient 数据用于学习如何路由；ESConv 检验一般回复质量和 Strategy 路由；EvoEmo / ES-MemEval-derived full 18-user evaluation 检验跨 session 记忆；多者共同验证 PM 是否能在固定生成器下实现可靠、节制且有条件适应性的资源分配。
