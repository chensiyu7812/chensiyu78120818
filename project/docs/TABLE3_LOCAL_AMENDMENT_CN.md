**表 III 本地生成修订，用户已授权，2026-09-10**

父冻结 `668378d317aa06248960e367fe521621a0c32e75b889da603c71c80e573ce290` 的第一条 NVIDIA 请求返回 HTTP 410：原 Llama 3.1 8B 服务于 2026-08-26 退役。父运行只有这一条免费生成失败记录，OpenAI 调用和费用均为 0。用户确认“可以的，推进吧”，授权在本机 A6000 运行同一 8B 模型。

本次建立 `frozen_local_v2`，保留父冻结、失败响应和原执行器代码。204 个状态、六组、生成 messages／seed／temperature=0／max_tokens=100、PM checkpoint、Filter、全部 judge 模型和评分规则、pilot 门槛、$9 限额及统计定义均沿用父版本。

本地生成采用 `NousResearch/Meta-Llama-3.1-8B-Instruct` 固定 revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`。四个 safetensors 分片下载后逐个计算 SHA256，与 Meta 官方 `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659` 的公开文件元数据核对一致。权重共 16,060,556,376 bytes；无量化，以 BF16 在 A6000 上运行。

tokenizer.json、config.json、generation_config.json 的公开源文件哈希也一致。官方当前 tokenizer_config.json 与镜像不同，直接下载官方文件返回 401，本次固定公开镜像自带的 tokenizer 配置。其纯文本模板为：一个 BOS、每条消息的角色头和 EOT、最后的 assistant 头，去除每条消息首尾空白。对全部 1,224 个实际生成输入，独立构造同一格式并逐条核对文本和 token IDs；没有插入日期、工具描述或额外 system 指令。消息边界格式另与 [Meta 公开 llama3 实现](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/chat_format.py)核对。

本地服务只监听 `127.0.0.1:18081`，仅接受预先登记的实验 prompts、seed 与解码参数。贪心解码，最多生成 100 tokens，停止 IDs 为 `[128001,128008,128009]`，不对生成文本做 tokenizer 空格清理。使用固定环境的 Transformers/PyTorch SDPA 实现；启动时核验权重、tokenizer、代码和依赖版本。每条响应记录模型资产 hash、输入 token hash、输出 token IDs 和用量。云 API 密钥不会发送到本地服务。

不能声称本地数值计算或 chat wrapper 与已退役 NVIDIA 服务逐位一致；这是明确记录的生成服务修订。生成本身没有外部 API 账单，评分仍按原先约 $8.17（含计划余量）方案执行，并由账本限制在 $9 预留线内。

运行目录为 `outputs/table3_under10/run_local_v2`。先完成 12 个状态的生成及 mini／GPT-4o 检查；通过冻结门槛后继续完整生成和 mini Batch 评分。完整生成保存后停止本实验的本地模型服务，释放 A6000，再等待 Batch 完成。失败时保存所有结果和费用，不根据方法排名改变门槛。

```bash
cd /home/tokkio/chensiyu78120818-table3/project
set -a
source /home/tokkio/metacom5_3.env
set +a
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
/home/tokkio/audits/pm-v1-table3-20260910/.venv/bin/python scripts/25_run_table3_local.py verify
/home/tokkio/audits/pm-v1-table3-20260910/.venv/bin/python scripts/25_run_table3_local.py run
```

模型文件保存在工作区外 `/home/tokkio/models/table3-llama31-8b-d10aef7`，Git 只保存资产清单和哈希。两个用户级服务分别负责本地模型与实验：`pm-table3-llama-local-v2.service`、`pm-table3-local-v2.service`。原冻结验证仍应通过。
