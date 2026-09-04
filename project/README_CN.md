# MetaCom-PM：检索前情绪支持资源分配

> 当前 PM-v1.5 唯一执行路线见
> `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`，机器可读阶段与门槛见
> `data/pm_v1_5_contracts/final_execution_plan_v2.json`。旧的 minimum/core-chain/reverse-
> designed/canonical-freeze 文档只保留为历史证据，不再分别下达任务。当前旧 ESConv states
> 含 corpus-level `situation` privileged input，不得进入新训练或最终评测。

本包实现第一篇论文的冻结范围：**纯文本、固定生成器之外、监督式 pre-evidence Policy Manager**。PM 只在实际取回证据前选择 MP、MS、ME 与 ESConv Strategy RAG，不实现 POMDP、RL、distress、安全门控、多模态或记忆写回。

## 数据边界

- `data/synthetic/runtime_states.jsonl` 与 `memory_backend.jsonl`：已清洗、物理隔离的合成训练输入。含旧 private/audit 字段的历史源卡不随 clean release 分发。
- `data/external/evo_emo.json`：EvoEmo 全部 18 用户、401 历史会话、34 个对话生成场景；只用于冻结后的外部评测。
- `data/strategy/strategy_cards.jsonl`：随包附带的**预构建 pilot 策略库**。它只允许 generator/judge pilot，不能用于论文确认性结果。
- 正式 ESConv 结果前，必须将本地官方 `ESConv.json` 导入 `data/external/ESConv.json`，重建 Strategy Bank，并通过 ESConv↔EvoEmo 重叠审计。
- ESConv test 从不进入 Strategy RAG；EvoEmo 从不进入 PM 训练、阈值选择、prompt 修改或 generator 选择。

## 本地 ESConv 接入

项目已经支持直接复制你本地现有的官方文件：

```bash
python scripts/02_download_esconv.py \
  --source /你的本地项目/ESConv.json \
  --overwrite
python scripts/03_build_strategy_bank.py
python scripts/12_build_esconv_test.py
```

若不提供 `--source`，脚本会从固定官方 commit 下载。两种方式都会记录 SHA256；Strategy Bank 只使用 dialogue-level train split，并排除与 EvoEmo 重叠的对话。

## 两级门禁

1. `release_preflight.json.status == API_PILOT_READY`：只说明静态、数据与 mock 测试通过，可以运行小规模真实 API pilot。
2. `release_preflight.json.confirmatory_ready == true` 且 `outputs/study_freeze.json` 有效：才允许正式 ESConv/EvoEmo 脚本运行。

正式脚本 `13`–`17` 会强制校验：

- config；
- checkpoint；
- validation-only selection；
- M2b selected-set omission labels（正式 V3.3 训练/调参必需）；
- official ESConv / frozen EvoEmo；
- rebuilt Strategy Bank；
- ESConv test runtime；
- EvoEmo fixed seeker tracks 及其 attestation；
- prompt 与核心代码哈希。

任何文件变化、换 checkpoint、换 selection、缩小 ESConv/EvoEmo 子集，都会硬失败。

## 必须遵守的运行原则

1. 先填写 `configs/experiment.yaml` 的真实 endpoint 与 model 名称。
2. 先运行 generator variance 和 20-card judge pilot。
3. 只有 judge gate 通过才允许全量生成标签和训练 PM。
4. 用官方 ESConv 重建 Strategy Bank 后重新运行 preflight；只有 `confirmatory_ready=true` 才能冻结研究。
5. 冻结后不得根据 ESConv test 或 EvoEmo 结果更换 generator、修改阈值、规则、prompt 或样本子集。

完整命令见 `docs/RUNBOOK_CN.md`，研究边界见 `docs/STUDY_PROTOCOL_CN.md`。


## 本地 ESConv 与正式门禁

本地已有官方 `ESConv.json` 时运行：

```bash
python scripts/02_download_esconv.py --source /path/to/ESConv.json --overwrite
python scripts/03_build_strategy_bank.py
python scripts/12_build_esconv_test.py
```

正式外部评测必须先运行 `scripts/20_freeze_study.py`。脚本 13–17 默认 fail-closed；只有显式 `--allow-unfrozen-debug` 才能做不可报告的调试。
