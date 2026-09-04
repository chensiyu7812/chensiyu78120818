# Step2 生成器兼容门——本环境可行性核实（2026-08-06）

**2026-08-06二次修正：本文档最初的核心结论错了，用户和另一位Codex都指出了这一点，这里先纠正
再往下看。** 最初版本断言"本环境没有任何可用的LLM API凭据，真实生成调用在这里物理上无法执行"
——这是错的。我的凭据检查只查了当前shell已导出的环境变量和项目目录内的`.env`文件，没有搜索
用户家目录。用户家目录下实际存在`~/.metacom_v1_5_secrets.env`（以及`~/.pm_v1_5_judge.env`、
`~/.pm_v1_5_openai.env`），`source`一下就能激活，里面确认包含`OPENAI_API_KEY`、
`NVIDIA_API_KEY`、`GEMINI_API_KEY`、`ANTHROPIC_API_KEY`等真实凭据的变量名（只核对了变量名
存在，没有打印、没有查看任何密钥值）。**真实LLM调用在这个环境里是可以执行的，此前"物理上无法
执行"的结论是我的检查范围不够全面导致的错误结论，不是环境的真实限制**。下面第一部分的"环境
凭据检查"保留作为记录（说明了错误检查过程本身），但其结论已被推翻，请以本节和文末更新为准。

## 背景

V5.2机械拼接方案（`compose_locked_response`：LLM生成的`primary_response`与后端模板化
`locked_clauses`做确定性字符串拼接，LLM从不看到MS/ME原文）已确认是导致早期实验里MS/ME内容
经常读起来生硬、割裂的架构原因。V5.3的替代方案（typed response program，
`v1_5_v5_3_typed_response_program.py`）把这个改成"生成器必须看到完整typed证据，产出一次性
整合响应"，代码已经写完，但从未跑过完整的真实兼容门——即真实调用LLM API，检查生成器在这个新
契约下是否真的产出结构合法、内容忠实的响应，而不只是通过离线单元测试。

## 核实内容

1. **环境凭据检查（原始版本，结论已被文首2026-08-06修正推翻，保留记录检查过程本身）**：
   - `env | grep -iE "api_key|openai|anthropic|openrouter"` → 空（只有`CLAUDE_CODE_EXECPATH`
     这个跟本次任务无关的变量）。
   - 项目根目录及`project/`目录下无`.env`/`.env.*`文件。
   - `src/metacom_pm/api.py`的`Endpoint.api_key`属性：读取`os.environ.get(self.api_key_env,
     "")`，为空时直接`raise RuntimeError(f"Environment variable {self.api_key_env} is not
     set")`——代码本身的fail-closed设计确认了"没配置就跑不起来"，这部分观察本身没错。
   - ~~结论：本环境没有任何可用的LLM API凭据~~——**错误**。检查范围只覆盖了当前shell环境变量
     和项目目录内的`.env`文件，没有搜索用户家目录下按约定命名的密钥文件
     （`~/.metacom_v1_5_secrets.env`等）。`RuntimeError`确实会在密钥未`source`时触发，但这
     只说明"当前shell没有加载密钥"，不能跳到"密钥不存在"这个更强的结论。

2. **能做的部分：结构/契约层面的离线校验，全部通过**：
   - `test_v1_5_v5_3_typed_response_program.py`：11个单元测试全部通过（`pytest`，0.06秒）。
   - 项目里带"v5_3"标记的全部测试文件：合计跑通，无失败。
   - 这些测试覆盖的是typed response program的输入/输出schema、typed证据的结构化传递、以及不
     依赖真实模型输出内容的纯逻辑路径——**验证的是"新契约的管道结构是对的"，不是"真实LLM在这个
     新契约下会写出什么样的响应、会不会跑题或幻觉"**。

## 结论：Step2真实生成器兼容性——已验证部分与未验证部分要分开说

| 已验证（本次可以做，也做了） | 未验证（本环境做不了，需要额外授权/凭据才能做） |
|---|---|
| typed response program的输入/输出结构、schema合法性（11个单元测试通过） | 真实LLM在"必须整合typed MS/ME/MP证据、一次性生成响应"这个新约束下，输出是否忠实、是否真的不再出现V5.2那种生硬拼接感 |
| 与V5.2 locked composer并存的代码路径没有相互破坏（同一批测试跑通） | 真实响应是否会因为要同时满足typed证据整合+对话连贯性两个目标而产生新的失败模式（如遗漏某条证据、误改证据措辞） |
| 无API凭据环境下的fail-closed行为符合预期（`RuntimeError`而非静默跳过或使用假数据） | 端到端的人工/自动质量评审（这需要真实响应文本才能进行，本次完全没有生成任何真实响应） |

**如实结论（2026-08-06修正后）：Step2生成器兼容门在本次会话里没有被"通过"，也没有被"跳过"，
但此前"本环境无法执行"的判断本身是错的**。真实缺口不是凭据，是：(1) 没有任何脚本把
`v1_5_v5_3_typed_response_program.py`接到真实端点上（见下方补充）；(2) 真实调用产生费用，
此前指令明确要求"不要立即付费运行"，需要用户明确授权才能实际发起调用。这两步都不在本次"接手
自主推进"的授权范围内自动执行，留给用户决定是否、何时授权执行——但请注意，"是否执行"跟"能不能
执行"是两件事，本环境**能**执行，只是还没有被授权/没有被接线。

## 2026-08-06补充：具体缺什么、怎么补——纯研究，没有调用任何API、没有花钱

按用户要求"研究一下Step2怎么弄"，查了项目已有的端点配置和调用基础设施，不是猜测：

### 凭据不是从零开始配——项目已经有一份写死了真实模型的配置

`project/configs/experiment.yaml`（不是`.example`模板，是已经填好真实`base_url`/`model`的
正式配置）里，跟Step2生成直接相关的端点是`generator`角色：

```yaml
generator:
  base_url: "https://integrate.api.nvidia.com"
  model: "meta/llama-3.1-8b-instruct"
  api_key_env: "NVIDIA_API_KEY"
  family: "llama"
```

配置里的注释解释了选型理由（"Supporter generator: Llama-8B is mid-strength, more sensitive
to evidence than 70B models, and realistic for digital-human deployment"）。**这个凭据缺口
其实已经不存在了**：`~/.metacom_v1_5_secrets.env`已经包含真实的`NVIDIA_API_KEY`，`source`
一下就能激活，不需要重新决定用哪家、哪个模型——这个决定已经在配置里做过了，密钥也已经有了。
配置注释里另外提到NVIDIA网关有个"free/prototyping tier"（40 RPM硬顶，见
`training_judge_deepseek_flash`条目的注释），对一次小样本Step2检查而言，实际花费大概率很低，
但具体额度、是否已经产生历史费用由用户自己在NVIDIA账号里确认，这里不替用户估算或承诺费用。

### 光有密钥还不够——目前没有任何脚本把typed_response_program接到真实端点上

搜了`project/scripts/`全目录，**没有任何脚本调用`v1_5_v5_3_typed_response_program.py`**——
这跟该模块自己文档字符串里写的状态一致："implemented, unit-tested, NOT yet wired into any
generation pipeline"。也就是说即使现在就拿到`NVIDIA_API_KEY`，也还差一个新脚本：加载几个
真实typed证据状态（MS/MP/ME核查这几天已经产出了现成的候选数据可以复用）、按
typed_response_program的输入契约组装、通过`OpenAICompatibleClient`（`api.py`里已有的调用
封装）调用`generator`端点、把响应存下来供人工核对是否忠实整合了证据、没有编造。项目里已有
脚本（如`project/scripts/v1_5/27i_materialize_v5_2_confirmation_plan_v1_5.py`等）演示了
"读`experiment.yaml`配置→`endpoint_from_config`→调用generator端点"这一套现成模式，新脚本
照这个模式写，不是从零发明调用方式。

### 结论：脚本已经写好了，缺的只剩"一句明确授权"

**2026-08-06补充：新脚本已经写完并验证**，`scripts/v1_5/48_step2_typed_response_dry_run_v1_5.py`
——用真实`discover_final_typed_memory_candidates`/`build_evo_memory`（跟脚本45-47同一套）
从真实138状态面板里取一个MP+MS都有候选的state，组装成真实`TypedResponseProgram`，构造真实
生成消息，默认（不带`--live`）**不发起任何网络请求**，只用一个手写的、明确标注为虚构的假
响应验证`parse_generator_response_dict`+`typed_response_guard_errors`这条解析/校验链路是
对的——已经实测跑通：正常响应校验通过（无错误），刻意构造一条V5.2式泄漏短语（"An earlier
session recorded..."）的假响应被正确拦截（`RECORD_LOG_PHRASING_LEAK`）。`--live`模式需要
`NVIDIA_API_KEY`，会发起真实付费调用，其代码路径（`OpenAICompatibleClient`/`endpoint_from_
config`的import）延迟到`if args.live`分支内部，dry-run和直接import这个文件都不可能碰到
网络——本次会话没有传过`--live`。

如果用户想推进：(1) `source ~/.metacom_v1_5_secrets.env`（凭据已经就绪，不需要新申请）；
(2) 明确授权可以花一点钱做一次小样本（比如10-20个真实state）验证跑；(3) 跑
`python scripts/v1_5/48_step2_typed_response_dry_run_v1_5.py --live`（或者我扩展这个脚本
支持批量跑多个state）。在此之前，这一项保持"脚本已就绪、未执行"的状态，不会因为"脚本写好了"
就擅自发起真实调用——脚本就绪不等于已经获得执行授权，这是两件独立的事。

## 2026-08-06最终更新：用户已授权，真实小批量已跑，发现一个严重问题

用户明确说"step2可以继续"后，跑了10条真实调用（跨10个不同用户），脚本扩展支持批量
（`--live --n 10`）。**结构层面10/10通过guard**，但人工逐条读全部真实回复后发现：
**5/10条回复把用户自己的事实/经历用第一人称说成是助手自己的**，包括一例助手凭空声称自己
有"丈夫"和"儿子"（实际是用户自己的家人）。这不是guard的bug——guard检查的是内部标签/ID泄漏
这类结构性错误，从来没设计成检查"这段第一人称叙述的主语到底该是用户还是助手"。完整发现、
逐条例子、根因分析（system prompt缺一条明确的证据归属指令）见
`PM_V1_5_V5_3_STEP2_LIVE_TEST_FINDINGS_20260806_ZH.md`。

**这推翻了"typed response program已经解决V5.2那种越界声称问题"这个假设**——同类问题以新
形式复现了。不是方案失败，是当前prompt需要补一条明确指令后重新验证，本次会话没有做这个
修正，留给用户决定下一步。
