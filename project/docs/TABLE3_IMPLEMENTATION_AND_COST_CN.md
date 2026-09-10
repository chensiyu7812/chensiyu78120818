**表 III 落地与 API 预算，2026-09-10**

**预算更新：用户现已要求总 API 费用低于 $10，并确认 NVIDIA 生成为免费。当前建议改用 [mini Batch＋GPT-4o 小样本检查方案](TABLE3_UNDER10_CN.md)，预计 $6.53、含余量约 $8.17。下文的全量 GPT-4o 报价及生成费用待核验说明保留为旧方案记录，不再是当前执行建议。**

建议采用原 204 个 fixed-context states，六组 `PM / Rule / Structured Fixed × Filter OFF / ON`。本轮已完成真实输入上的离线落地：六组配置、memory-only Filter、候选检索、生成 prompts、质量／风险／遗漏评分 prompt builders、成本估计和输入 hash 清单。没有生成新回复或评分，尚未实现／启动付费 API 调度与最终统计输出。

执行工作区为 `/home/tokkio/chensiyu78120818-table3`，分支 `experiment/pm-v1-table3`，基于冻结提交 `c16608343fe60e92e57c622fdc24738efe57d08c`；原 checkout 保持不变。

**已经具体实现的部分。**

- [配置](../configs/table3_rerun.json)：完整 34 scenarios × seeds 101/202/303 × turns 3/8；三种 v1 policy；统一 Filter。
- [Filter、候选规划与评分 prompts](../src/metacom_pm/table3.py)：使用原 PM、state、memory retriever 和 generation prompt。为加速离线规划，Strategy 卡的词频和范数缓存一次；打分公式、排序与 tie-break 保持 v1 一致。
- [离线入口](../scripts/21_plan_table3.py)：不读取 API keys、不调用推理 API，生成完整 unit／arm 清单与费用估计。
- [输出](../outputs/table3_rerun_v1/plan/manifest.json)：204 units、1,224 条生成任务，含明确的 code／config／artifact hashes；旧轨迹按原内容 hash 导入，不改写历史 attestation。
- [成本明细](../outputs/table3_rerun_v1/plan/cost_estimate.json)、[过滤行为](../outputs/table3_rerun_v1/plan/retrieval_diagnostic.json)、[依赖锁](../configs/table3_requirements.txt)。

23 项相关测试通过。另对真实数据检查了全部 612 个 policy-unit 配对：OFF／ON 候选一致、Strategy 一致、requested action 一致、ON 为候选子集；全部 OFF prompt 与原 v1 函数一致。三种 policy 在这 204 个 units 上的动作分布均与历史 V4 记录一致。还以完整 Strategy Bank 的两个真实查询对照了缓存规划器与原检索器的候选和 confidence，完全一致。

**Filter 的确定版本。**

直接采用后续 `f8508da:project/configs/pm_v2.yaml` 中已有的 memory 参数，移植为 memory-only 模式。这是既有参数的诊断复用，不是重新在当前结果上搜索最优阈值；不接后续 learned helpfulness model。

| Source | Current relevance 最低分 | Context relevance 最低分 | 最多保留 |
|---|---:|---:|---:|
| MP | 0.08 | 0.08 | 2 |
| MS | 0.18 | 0.22 | 2 |
| ME | 0.50 | 1.00 | 1 |

相关性条件是 current 或 context 达标。ME 若超过 120 个估算 tokens，还要求 current score ≥0.65。全 memory 最多保留 3 条、去除规范化文本重复、允许为空，最后保留原检索顺序。Strategy 不过滤、不去重、不改顺序。这里的 lexical relevance 是词频 cosine，不宣称它等于语义相关性。

真实离线输入上的过滤行为如下；这些是资源记录，不是新的回复质量／风险结果：

| Policy | Filter 前平均 memory tokens | Filter 后平均 memory tokens | 平均逻辑源调用数（含 RS） | 过滤后空 memory |
|---|---:|---:|---:|---:|
| PM | 683.3 | 71.5 | 2.716 | 42/204 |
| Rule | 1024.6 | 91.5 | 3.000 | 0/204 |
| Structured Fixed | 1037.8 | 93.0 | 4.000 | 0/204 |

memory tokens 沿用 v1 的 chars/4 估计；调用数按 Filter 前请求的来源统计，OFF／ON 相同。PM 有 42 个 state 的 memory 被删空，必须保留这些结果并通过独立 omission 评分检查信息损失，不据此剔除样本。逻辑源数不是外部 HTTP 请求数；离线规划中的计时也不是部署延迟 benchmark。

**预计费用。**

按 OpenAI 官方 GPT-4o 文本价格计算：每百万输入 tokens $2.50、输出 $10.00。本次建议将 judge 明确锁为 `gpt-4o-2024-08-06`，而不依赖未来可能漂移的 alias；这不是对历史 judge 后台 revision 的确认。[官方 GPT-4o 模型与价格](https://developers.openai.com/api/docs/models/gpt-4o)。

| 工作 | 推理次数 | 预计输入 tokens | 预计输出 tokens | 普通 API 估价 |
|---|---:|---:|---:|---:|
| Llama 生成六组回复 | 1,224 | 约 1.318M | 上限约 0.122M（Llama 口径） | NVIDIA 账户路线待核验 |
| 六候选匿名质量评分 | 204 | 约 1.542M | 0.184M | $5.69 |
| 独立 selected-evidence 风险审计 | 1,224 | 约 3.946M | 0.428M | $14.15 |
| 独立 authorized-context 遗漏检查 | 1,224 | 约 7.899M | 0.220M | $21.95 |
| **全量 OpenAI 评分** | **2,652** | **约 13.386M** | **约 0.832M** | **$41.79** |

费用公式为 `(输入 tokens×2.50 + 输出 tokens×10.00)/1,000,000`。输入来自完整真实上下文、实际候选／保留证据和完整评分 prompt，使用 o200k_base BPE，并为未来每条生成回复预留 100 个 GPT tokens。Llama 与 GPT tokenizer 不同，模型未生成前回复长度只能估计；Chat Completions 的 schema 包装也只能估计。judge 预期输出按每次 quality=900、risk=350、omission=180 tokens 预算，并非已发生用量。

普通调用另计 12-unit pilot（质量双顺序、完整风险／遗漏检查）约 $2.79；这是按全量平均长度外推的 pilot 估计，尚未按选定 12 个单位生成正式 pilot jobs。加 25% 余量，OpenAI 预算约 **$55.73，建议预留 $60**。生成端 pilot 另有 72 条生成；全量复跑和 pilot 均发生时总基础生成为 1,296 条。此费用包括预期余量，不是数学上保证不超的账单上限；实际调度器仍需预算计数和停止条件。

按设定的 judge 输出上限和 10% 输入包装余量，全量评分的较保守规划值约 $51.09，尚不含 pilot／重试。上述余量不应被理解为可以无限重试；同一失败请求的尝试次数和已返回 usage 均须计入账本。

若全量评分走 OpenAI Batch、pilot 走普通调用：全量约 $20.89，连同 pilot 和 25% 余量约 **$29.61，建议预留 $30–35**。Batch 价格比同步调用低 50%，适合这种离线评分；按账户 queued-token quota 分批，并按 `custom_id` 合并结果，不能依赖返回行顺序。Batch 单批采用 24 小时 completion window，总体工期还受批次数与账户额度影响。[官方 Batch 文档](https://developers.openai.com/api/docs/guides/batch)。

不建议为省费用默认更换成低价 judge。$60 普通预算已经包含完整风险和遗漏检查；仅做质量＋风险约 $19.84，但它省去了验证删除记忆是否损伤有效信息的一项关键证据。

**NVIDIA 生成端有一个实际待解决点。**

原模型页面目前标明 Free Endpoint = Deprecated，Partner Endpoint 与 Download Available。[NVIDIA Llama 3.1 8B 官方页面](https://build.nvidia.com/meta/llama-3_1-8b-instruct)。本次还直接读取公开的 `https://integrate.api.nvidia.com/v1/models`，80 个模型里没有 `meta/llama-3.1-8b-instruct`。

这说明不能先承诺“原 NVIDIA 免费 API 一定可用”。本会话没有注入 `NVIDIA_API_KEY`／`OPENAI_API_KEY`，因此没有做带账户的推理测试。下一步先用现有 NVIDIA 账户确认旧 endpoint 是否仍能调用；如果账户仍可免费调用，生成账单可按 $0，以上 OpenAI 预算就是主要 API 支出。若旧 endpoint 确实失效，应先恢复相同 Llama 3.1 8B 权重的可用服务并确定价格，再冻结执行 endpoint；不静默换成 70B 或 Nemotron。自部署同模型可以没有按 token 的生成 API 账单，但 GPU／部署成本需另算。

因此当前可给出的准确报价是：**OpenAI 普通完整预算约 $60，或 Batch 约 $30–35；生成端费用和整次运行总账单待 NVIDIA 路线核验，尚不能保证总额。** 已保存的 cost JSON 将 generator cost 与 total invoice 留为 null，没有虚填为零。

**接下来的执行顺序。**

1. 用已有账户确认生成 endpoint，保持 Llama 3.1 8B 权重和 v1 解码设定（temperature=0、max_tokens=100、seed=原 seed+turn）。明确服务端 model/revision；若 endpoint 变化，更新 config 后重新生成 plan hash。
2. 补齐 API 执行器：消费已冻结的 1,224 条生成任务，逐次保存 request、response、provider usage、返回 model／fingerprint、错误／重试和账单；按 unit 内轮换顺序调用，唯一键去重和断点续跑。不能把无 usage 的失败请求默认当作已知零成本。
3. 固定选取 12 units 做技术 pilot，检查生成可用、解析、候选换序稳定性。继续使用完整 204-unit 主样本，pilot 用途和与主样本重叠明确记录；不要求“全新未使用”。先固定技术门槛和参数，不依赖 PM 排名是否有利来决定是否继续。
4. 用实际新回复生成评分 requests：204 条 quality、1,224 条 risk、1,224 条 omission；质量保留六候选位置平衡，两个审计都为匿名 pointwise。普通或 Batch 只改变执行渠道，不改变模型／prompt／指标。
5. 完整收齐配对结果后导出表 III 与 paired CIs；以 user-cluster bootstrap 为主、scenario-cluster 为补充。均值、误用比例／严重度、major verdict、遗漏及资源分别记录，缺失 arm 不能默默当作零分或删除。

上述第 2–5 步的联网调度／结果导入／统计导出尚未写成完整 runner，本轮不把离线 plan 标为“已全量可运行”。现有产物足以审查具体实验、验证输入一致性并给出基于真实 prompt 的费用预算。

离线重建命令（须使用新的空输出目录）：

```bash
cd /home/tokkio/chensiyu78120818-table3/project
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/tokkio/audits/pm-v1-table3-20260910/.venv/bin/python \
  scripts/21_plan_table3.py --out outputs/table3_rerun_v1/plan_new
```

本轮付费推理调用为 0；已有表 III 的质量／风险数字尚未被替换。
