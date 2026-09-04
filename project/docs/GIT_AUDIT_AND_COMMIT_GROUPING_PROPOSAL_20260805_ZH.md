# Git 只读审计 + 提交分组建议（2026-08-05）

状态：**只读审计，未执行任何 git 写操作**（未`add`、未`commit`、未`checkout`、未`reset`）。

## 1. 基本事实

- 当前分支：`recovered/pm-v1.5-current-20260728`（本地恢复分支，尚未push，基于
  `origin/pm-v1.5-hybrid-retrieval`）。
- 远程还有：`origin/main`、`pm-v1-frozen`、`pm-v1.5-supplemental`、`pm-v1.5_1`、
  `pm-v2-redesign`、`route-b-v4-redesign`、`agent/pm-v1-5-conference-review`。
- 最近一次真实commit：`ce0dfc5`（2026-07-28）"Recover current PM v1.5 source snapshot after
  workspace cleanup"——**自那以后到现在（跨越7/28到8/5，超过一周）的所有工作都没有再提交过**。
- `git status --porcelain`：**723条untracked（??），30条已修改未暂存（M）**。
- `git diff --stat`（30条modified文件）：1754行新增、90行删除。

## 2. 关键发现：不是"一批V5.3新文件"，是"一周多的整个项目都没提交"

723条untracked覆盖了几乎整个`src/metacom_pm/`（100+模块）、`scripts/v1_5/`（脚本编号从12到44，
横跨V3 observation pipeline、V5 FIT、V5.1、V5.2 E1-E7、strategy bank v3、support need factorized
bakeoff、ESConv auxiliary generation等至少6-8个历史阶段）、几乎全部`tests/`（130+测试文件）、
以及大量`docs/`和`data/pm_v1_5_contracts/`。这不是"V5.3做完了忘记提交"，是**从7/28恢复之后就再
没做过一次干净提交**。

**直接后果：像`docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`（已跟踪、+319/-x行）这种文件，它的
diff是"7/28到现在全部累积修改"，不可能干净拆出"只有今天V5.3那几条"——git在7/28和现在之间没有任何
中间检查点，没法做这种切割。** `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`更极端：它整个文件都是
untracked（从来没提交过），800+行内容代表的是过去一周多的完整累积产出，不是今天的孤立新增。

这两个文件我建议**先不纳入本轮提交**，单列一类，等你确认要不要"一次性提交多天累积内容并如实标注
提交信息"，还是希望换个方式处理（见下方分组4）。

## 3. 重要安全发现：.gitignore 对当前输出命名规则已经过时

`/project/outputs/pm_v1_5*/`、`/project/data/pm_v1_5/`这类规则**确实生效**——用`git check-ignore`
验证过，像`pm_v1_5_dual_domain_weak_fit_only_candidate_v4/`（273M）这类大目录已经被正确排除，
不会被`git status`看到，也不会被`git add`意外收进来，这点是安全的。

但两个问题：

1. **不匹配`pm_v1_5_*`前缀的批量生成目录完全没被忽略**，目前untracked里就有约300-400MB这类内容
   （`data/esconv_test_v1_5*`共108M、`data/pm_v1_5_formal_v8_*`系列共约100M、
   `outputs/esconv_auxiliary_*`系列共约55M等）。这些看命名就是批量生成产物，符合`.gitignore`
   注释里"应该留在本地或content-addressed artifact vault"的描述，只是因为改了命名习惯没被现有
   规则覆盖。**建议先扩充`.gitignore`再做任何`git add`，避免以后有人一次`git add -A`就把几百MB
   生成数据带进仓库。**
2. **我自己这次新建的`outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1/`（228K，纯诊断
   JSON/JSONL/MD，不是批量生成物）意外命中了`pm_v1_5*/`这条排除规则，默认不会被`git add`收录。**
   这条目录体积很小、内容是分析结论不是原始生成数据，建议加一条`.gitignore`例外
   （`!/project/outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1/`）让它能被正常提交，
   而不是搬到别的路径（会破坏这几轮对话里已经引用的路径）。

## 4. 建议的提交分组

**只提出分组，不执行提交，等你确认。**

### 分组0（前置，建议最先做）：修一下 `.gitignore`
- 追加规则排除`esconv_test_v1_5*`、`esconv_auxiliary_v1_5*`、`pm_v1_5_formal_v8_*`这类
  非`pm_v1_5`前缀但明显是批量生成产物的目录（具体清单可以再列，先确认要不要这么做）。
- 加一条例外放行`outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1/`。
- 这一步本身也是一次独立、干净的commit。

### 分组1：V5.3治理文档（部分可以干净提交，部分不能）
可以干净提交（untracked、内容自成一体、无历史纠缠）：
- `docs/PM_V1_5_V5_3_INTEGRATED_EVIDENCE_EXECUTION_PLAN_20260805_ZH.md`
- `data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json`（已用`json.load`校验过
  可解析，见我们上一轮的验证记录）

**不能干净提交，需要你决定怎么处理**（原因见第2节）：
- `docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`（tracked, +319/-x，混合多天历史）
- `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`（untracked，整份从未提交，800+行累积内容）
- `docs/PM_V1_5_CORE_CHAIN_PLAN_ZH.md`（tracked, +84行，同样的问题）

### 分组2：typed response executor（不是我写的，Codex那边的产出，我只验证过测试通过）
- `src/metacom_pm/v1_5_v5_3_typed_response_program.py`
- `tests/test_v1_5_v5_3_typed_response_program.py`

### 分组3：semantic MS retrieval（我这次写的实现）
- `src/metacom_pm/v1_5_v5_3_semantic_ms_retrieval.py`
- `tests/test_v1_5_v5_3_semantic_ms_retrieval.py`

### 分组4：检索诊断脚本与诊断产物（我这次写的）
- `scripts/v1_5/42_diagnose_ms_retrieval_scoring_v1_5.py`
- `scripts/v1_5/43_diagnose_ms_rank1_on_real_states_v1_5.py`
- `scripts/v1_5/44_diagnose_mp_me_rank1_on_real_states_v1_5.py`
- `outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1/`（需要先做分组0的gitignore例外）

### 分组5：其余历史修改——本轮明确不动，只做粗分类供你后续单独审计

723条untracked里去掉分组1-4后剩下的部分，按前缀粗分（不是精确清单，只是让你知道大致构成）：

| 前缀/目录特征 | 大致内容 | 数量级 |
|---|---|---|
| `scripts/v1_5/22r-29a_*.py` | support need factorized bakeoff、strategy bank v3、V5 FIT、
  V5.1、V5.2 E1-E7 pipeline脚本 | 约170个脚本 |
| `src/metacom_pm/v1_5_*.py`（非v5_3前缀） | V3 observation/eligibility、strategy rag、
  support need、typed candidate adapter等历史模块 | 约50个模块 |
| `tests/test_v1_5_*.py`（非v5_3前缀） | 上述模块对应测试 | 约130个 |
| `docs/PM_V1_5_*_2026073*/080*_ZH.md` | 7/30-8/4期间的阶段性报告文档 | 约35份 |
| `data/pm_v1_5_contracts/*.json`（非v5_3前缀） | 历史阶段机器合同 | 约50份 |
| `outputs/`下各类生成/评审产物目录 | 见第3节，部分应gitignore、部分是有意义的冻结产物 | 数百个目录 |
| `configs/`、`README_CN.md`等已跟踪文件的修改 | 零散配置调整 | 数个文件 |

这一层建议你（或者带着这份清单跟Codex核对）单独做一轮"哪些是真正要保留的研究产物、哪些是可以
gitignore掉的中间产物"的分类，不建议这次顺手混着提交。

## 5. 下一步（等你确认后再做，本轮不执行）

1. 确认分组0的.gitignore改动内容。
2. 确认分组1里两份"不能干净拆分"的文档怎么处理（整份提交并如实注明"累积多日内容"，还是暂缓）。
3. 按分组1(部分)→2→3→4的顺序，用**显式路径**（不用`git add .`/`-A`）分批提交，每次提交前重跑
   相关测试；每次commit记录SHA。
4. 分组5留给专门的一轮审计，不在这次流程里处理。
5. 全部确认后再考虑要不要建`v5.3-dev`分支和`pre-implementation-freeze` tag——这两步本轮也不做。
