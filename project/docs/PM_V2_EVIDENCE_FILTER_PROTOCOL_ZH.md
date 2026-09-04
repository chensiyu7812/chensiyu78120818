# PM V2 Evidence Filter 协议

更新时间：2026-07-14

## 1. 为什么 PM 之后仍需要 Filter

PM V2 是 pre-retrieval policy manager。它在看不到具体 memory snippet 和 strategy card 的条件下，根据当前对话、可部署 inventory metadata、质量下界、动作适用风险上界和成本选择一个 requested action。

因此 PM 能学习“这个状态大概是否值得调用 MP/MS/ME/RS”，但不能直接证明检索返回的某个具体 item：

- 与当前问题相关；
- 没有过时或冲突；
- 放进 prompt 后真的有帮助；
- 不会造成不必要暴露或生硬的 strategy overuse。

V2 的正式系统定义为：

```text
deployable state
  -> PM requested action (source/policy level)
  -> retrieval candidates
  -> Evidence Filter (item level)
  -> effective action + generator prompt
```

PM 与 Filter 不是互相替代：PM 控制是否付出资源调用和候选暴露成本，Filter 处理调用后才可观察的 item-level 错误。

## 2. requested 与 effective action

每条 outcome 同时保存：

- `action_id` / `requested_action_id`：PM 或 fixed policy 在检索前请求的动作；
- `effective_action_id`：过滤后真正进入 generator prompt 的资源组合；
- candidate/kept/dropped IDs、tokens 和逐 item 决策；
- filter mode、config SHA、checkpoint SHA；
- requested retrieval calls 与过滤后 generation input cost。

如果 PM 请求 `MPMSME+RS`，Filter 最后全部丢弃，effective action 可以是 `M0+R0`；但这条记录仍然是 PM 请求高资源后被 Filter 纠正，不能算作 PM 学会 abstention。

所有 PM action diversity、M0/R0 使用率和 learned abstention 主张必须基于 requested action，并排除 fallback；effective action 只用于描述实际生成暴露与成本。

## 3. Memory item 监督信号

旧的“needed source = 该来源所有 item 都有用”标签不够细。V2 数据契约现在要求每个 synthetic memory item 带 evaluator-only：

```text
item_utility = helpful | irrelevant | harmful
```

过滤器正例必须同时满足：

```text
item_utility == helpful
source ∈ needed_memory_sources
stale == false
conflicts_with_current_state == false
```

约束还要求：

- helpful sources 与 `needed_memory_sources` 精确一致；
- 每个 needed source 至少有一个 helpful item；
- 每个 needed source 同时至少有一个 irrelevant/harmful distractor；
- helpful item 不得 stale/conflicting；
- `memory_harmful` case 至少有一个 harmful item。

这些 label 只存在于隔离的 evaluator context，不进入 PM state、retrieval backend或 generator inference prompt。人工 semantic-sanity packet 显示 item 文本和候选 utility label，并要求双人填写 `memory_item_utility_match`。

## 4. Filter 模型与门控

`scripts/36_train_pm_v2_evidence_filter.py` 只使用：

- train users：拟合 word/char hashing + metadata 的 logistic model；
- calibration users：从冻结 threshold grid 选择满足 recall/FPR 的阈值；
- internal-test users：检查 recall、precision、specificity。

推理特征仅含当前文本、可见对话上下文、候选 item 文本、source、age 和 token estimate，不读取 regime、needed source、stale/conflict 或 item utility oracle。

正式 action sweep、study freeze 和 EvoEmo generation 会验证：

- checkpoint/report/attestation 内容 hash；
- PM V2 config hash；
- training status `COMPLETE`；
- calibration/internal status `PASS`；
- model contract hash。

任一缺失或漂移都在 API 之前 fail closed。

训练命令（0 API）：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/tokkio/miniconda3/envs/sim_eval/bin/python \
  scripts/36_train_pm_v2_evidence_filter.py
```

当前尚无生成后的 V2 development data，因此不存在可报告为成功的 filter checkpoint。

## 5. Strategy card 的边界

RS 候选当前使用 `lexical_contextual_fail_safe`：Filter 读取实际 retrieved card 的 retrieval text，并与当前 turn 和可见 context 做相关性门控、去重与 item cap。

这比 PM 只按 catalog metadata 请求 RS 更细，但它不是“已学习 card-level causal benefit”。论文应准确写成 deterministic contextual filter。若后续要把 strategy filtering 写成 learned contribution，需要使用 train-only、action-paired R0/RS outcomes 训练独立模型，并重新冻结 full development sweep；不能从 external result 反向调阈值。

## 6. 公平基线与消融

主系统比较必须遵守：

```text
PM + Filter       vs. cost-matched fixed + 同一个 Filter
PM + Filter       vs. ME+R0 fixed + 同一个 Filter
```

不能只给 PM 使用 Filter，否则差异混合了 routing 与 filtering。

Filter 的因果增益应作为独立消融：

```text
PM + Filter       vs. PM only
fixed + Filter    vs. fixed only
```

至少报告 support、六维 quality、动作适用 risk、candidate/kept/dropped evidence、observed tokens 和 context misuse。只有当相同 requested policy 下 Filter 带来稳定收益，才可主张 Filter 是必要系统层。

## 7. Prompt-equivalence 去重

不同 requested actions 经过 Filter 后可能产生同一个 effective evidence prompt。V2 将同一 state 下的相同 prompt 放进一个 equivalence class：

- generator 只做一次物理调用；
- 回复、provider usage 和 request hash由该 class 共享；
- 每个 requested action 仍分别保存候选检索、Filter 决策和 requested cost；
- alias 不能制造独立样本量或独立响应方差。

这既避免浪费 API，也防止相同 prompt 因重复采样出现虚假的 action quality 差异。

## 8. 当前可主张与不可主张

可以写：

> PM V2 implements a two-stage source-routing and item-filtering architecture. The PM chooses resources before retrieval, while an attested post-retrieval filter can remove irrelevant or unsafe memory items before generation.

当前不可以写：

- PM V2 + Filter 已经优于 `ME+R0`；
- Filter 已经证明提高外部支持质量；
- effective `M0/R0` 等同于 PM learned abstention；
- strategy card filtering 已学习到因果 usefulness；
- synthetic internal filter gate 等同于真实用户效度。

这些结论必须等待真实 V2 数据、checkpoint、同过滤器公平基线、PM-only 消融和独立 external evaluation。
