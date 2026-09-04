# API 运行手册

## 0. 环境

```bash
cd MetaCom_PM_API_Ready_Frozen_20260621
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp configs/experiment.example.yaml configs/experiment.yaml
# 编辑 configs/experiment.yaml：填写真实 endpoint / model；API key 只放环境变量
PYTHONPATH=src python scripts/99_release_preflight.py --root . --out release_preflight.json
```

初始包的 `status` 应为 `API_PILOT_READY`。由于尚未接入你的本地官方 ESConv，`confirmatory_ready` 初始为 `false`，这是预期行为。

### 0.1 PM-v1.5_1 冻结语义运行环境

PM-v1.5_1 的 reportable development/external 路径不能使用历史 `sim_eval`
环境。它虽可运行 mock tests，但其 Python/Transformers/Hugging Face Hub 组合不符合
当前合同，且无法真实加载冻结 BGE。也不要直接使用可变的 Conda `base`；`base` 中的
Python 3.13.2 只可作为创建器，所有正式命令都在仓库专用 venv 中执行：

```bash
cd <仓库根目录>
python3.13 -m venv .venv-pm-v1-5
source .venv-pm-v1-5/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install -c project/constraints/pm_v1_5_runtime.txt -e 'project[dev]'
python -m pip check
cd project
python scripts/v1_5/19_preflight_semantic_runtime_v1_5.py
```

正式 shell 保持 `PYTHONNOUSERSITE=1`，不要在中途 `conda activate sim_eval` 或
`conda activate base`。只有顶层 `semantic_runtime_status=PASS` 才能继续。该门同时绑定
Python、关键 package、CPU/float32、user-site 关闭状态、固定公开文本和 3×384 数值矩阵；
development data generation 与训练入口都会现场重算，candidate manifest、study freeze
和 external artifact 再逐段绑定。readiness challenge 独立显示为
`REPORT_ONLY_x_OF_20`；它是能力边界诊断，不是 runtime PASS，也不得用其结果在
internal/external 后反向改模型。

52-user states 生成后、任何 7,488-response action sweep 前，必须先执行不读 outcome 的
rule-grid 预检；该产物同时被 sweep 和训练内容寻址验证：

```bash
python scripts/v1_5/20b_preflight_rule_grid_v1_5.py
```

正式 52-user `--run` 还必须显式传入本次新 pilot 的
`--generation-pilot-attestation`。脚本没有默认值，并会拒绝 V8–V8.5
这些已消费或已失效的历史目录。V8.4 虽在 transport/schema 层真实 PASS，也因
readiness surface 和 review-lineage 缺口被归档；V8.5 因 provider role list 末尾为 user
真实 fail-closed。二者均不得复用；V8.6 role-safe exchange 合同已真实 PASS，identity
在消费后关闭，只允许其 exact attestation 进入后续自动语义审核。该审核必须同时覆盖 27 个确定性案例、exact paid 9-case
artifact 和 24 个 hard controls（双开发家族共 120 logical calls）。

## 1. 接入本地官方 ESConv，并重建 Strategy Bank

已有本地官方文件时：

```bash
PYTHONPATH=src python scripts/02_download_esconv.py \
  --source /绝对路径/ESConv.json \
  --overwrite
```

没有本地文件时，可从固定官方 commit 下载：

```bash
PYTHONPATH=src python scripts/02_download_esconv.py --overwrite
```

然后执行：

```bash
PYTHONPATH=src python scripts/03_build_strategy_bank.py
PYTHONPATH=src python scripts/12_build_esconv_test.py
PYTHONPATH=src python scripts/01_audit_and_make_folds.py
PYTHONPATH=src python scripts/04_build_pair_graph.py
PYTHONPATH=src python scripts/99_release_preflight.py --root . --out release_preflight.json
```

`03_build_strategy_bank.py`：

- 只使用固定 dialogue-level ESConv train split；
- ESConv validation/test 不进入 Strategy RAG；
- 自动排除与 EvoEmo exact/near-duplicate 的 ESConv dialogue；
- 写入 ESConv、EvoEmo、Strategy Bank 的 SHA256 和 overlap audit。

此时若 endpoint 配置已填写、官方 ESConv 接入正确，preflight 中除
`study_freeze_valid` 之外的 confirmatory data checks 应通过。完整的
`release_preflight.json.confirmatory_ready=true` 只能在第 4 步创建
`outputs/study_freeze.json` 之后出现。

## 2. 付费前小实验

```bash
PYTHONPATH=src python scripts/05_generator_variance.py --n-cards 8
PYTHONPATH=src python scripts/06_run_action_sweep.py --max-cards 20
PYTHONPATH=src python scripts/07_run_judge_pilot.py --max-cards 20
PYTHONPATH=src python scripts/08_analyze_judge_pilot.py
```

只有 pilot gate 明确允许 full judging 才继续。当前 response measurement 使用
`docs/MEASUREMENT_PROTOCOL_V3.md`：training-eligible pairs 采用 dual-order
debiasing；hard controls 与 boundary diagnostics 分层；raw dual-order agreement、
raw reversal / raw position bias 作为 diagnostic；response quality 在后续作为非劣约束而不是默认优越性结论。Dry-run 或 mock 通过不等于真实 judge 已校准。

## 3. Synthetic 全量标签与 PM 训练

```bash
PYTHONPATH=src python scripts/06_run_action_sweep.py
PYTHONPATH=src python scripts/09_run_full_judging.py
PYTHONPATH=src python scripts/09b_run_m2b_audit.py

# 推荐完整 CV grid 与多 seed
PYTHONPATH=src python scripts/10b_train_cv_grid.py \
  --feature-modes text_only metadata_only text_metadata \
  --m2b-path outputs/m2b_selected_set_omission_gemini_flash_lite_v3/memory_selected_set_omission_judgments.jsonl

# validation-only 选择 policy、epsilon、tau、strong rule、best fixed
PYTHONPATH=src python scripts/11_tune_validation.py \
  --checkpoint outputs/models_cv_m2b/text_metadata_user_fold0_seed17.joblib \
  --m2b-path outputs/m2b_selected_set_omission_gemini_flash_lite_v3/memory_selected_set_omission_judgments.jsonl

# 用冻结结构在全部 synthetic development data 上重训 final PM
PYTHONPATH=src python scripts/11b_train_final_pm.py \
  --feature-mode text_metadata \
  --m2b-path outputs/m2b_selected_set_omission_gemini_flash_lite_v3/memory_selected_set_omission_judgments.jsonl
```

不得读取 ESConv test 或 EvoEmo 结果来选择模型、阈值或规则。正式 V3.3 训练和 validation selection 默认要求 M2b selected-set omission 标签；`--allow-no-m2b` 只能用于非报告性 debug。

## 4. 创建 study freeze

先再次运行 preflight，并确认 `confirmatory_ready=true`：

```bash
PYTHONPATH=src python scripts/99_release_preflight.py --root . --out release_preflight.json
```

然后冻结：

```bash
PYTHONPATH=src python scripts/15a_build_evoemo_fixed_tracks.py
PYTHONPATH=src python scripts/20_freeze_study.py \
  --checkpoint outputs/models/FINAL_MODEL.joblib \
  --selection outputs/selection.json
```

Freeze 会锁定：

- config 与 endpoint/model 名称；
- final checkpoint；
- validation-only selection；
- synthetic provenance；
- official ESConv；
- frozen EvoEmo；
- rebuilt Strategy Bank 与 overlap audit；
- ESConv test runtime；
- EvoEmo fixed seeker tracks 及其 attestation；
- prompts、retrieval、policy、metrics 和 confirmatory scripts。

## 5. ESConv 确认性回复质量评测

```bash
PYTHONPATH=src python scripts/13_run_esconv_sweep.py
PYTHONPATH=src python scripts/14_eval_esconv.py \
  --checkpoint outputs/models/FINAL_MODEL.joblib \
  --selection outputs/selection.json
```

正式脚本强制要求有效 freeze，并强制使用完整 ESConv test。ESConv 没有跨日长期历史；这里主评 R0/RS 路由与回复质量，只辅助报告 false-memory abstention。

## 6. EvoEmo / ES-MemEval 全 18 用户评测

官方轨道：

```bash
PYTHONPATH=src python scripts/15_run_evoemo.py \
  --protocol official \
  --interaction-mode fixed \
  --simulator-id seeker_main \
  --checkpoint outputs/models/FINAL_MODEL.joblib \
  --selection outputs/selection.json

PYTHONPATH=src python scripts/16_eval_evoemo_official.py \
  --generation-attestation outputs/evoemo_official/artifact_attestation.json
```

选择性记忆轨道：

```bash
PYTHONPATH=src python scripts/15_run_evoemo.py \
  --protocol selective \
  --interaction-mode fixed \
  --simulator-id seeker_main \
  --checkpoint outputs/models/FINAL_MODEL.joblib \
  --selection outputs/selection.json

PYTHONPATH=src python scripts/17_eval_evoemo_selective.py \
  --generation-attestation outputs/evoemo_selective/artifact_attestation.json
```

确认性脚本：

- 强制运行全部 18 用户、34 scenarios；
- 禁止 `--max-scenarios`；
- seeds 必须与冻结 config 中的 `robustness_seeds` 完全一致；
- 不允许换 checkpoint、selection、Strategy Bank 或 config。

## 7. 人工盲评

```bash
PYTHONPATH=src python scripts/18_export_human_eval.py --comparisons YOUR_COMPARISON.jsonl
# 每位标注者填写 CSV：pair_code,preference,notes
PYTHONPATH=src python scripts/19_analyze_human_eval.py --annotations ann1.csv ann2.csv ann3.csv
```

## 8. 断点恢复与审计

所有 API runner 使用 JSONL 完成键断点恢复，并保存 raw response、usage、请求 hash 与 run manifest。模型、prompt、数据或配置变化后，旧输出目录不得继续复用。正式冻结后，任何冻结文件变化都会在脚本入口硬失败。


## 8. 两级门禁

- `API_PILOT_READY`：只能运行 generator variance、20-card action sweep 和 judge pilot。
- `CONFIRMATORY_READY`：官方 ESConv、无重叠 Strategy Bank、ESConv test runtime、完整配置和 study freeze 全部有效，才允许正式 ESConv/EvoEmo。

无论静态 preflight 是否通过，真实 judge pilot 仍必须输出 `PILOT_GO_FULL_JUDGING`，否则不得进行全量 judging 或训练。
