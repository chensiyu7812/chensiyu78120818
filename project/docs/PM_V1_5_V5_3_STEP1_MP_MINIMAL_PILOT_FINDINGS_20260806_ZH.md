# Step1最小试点：MP单组件，真实生成+真实judge打分（2026-08-06，用户授权后执行）

## 目的

用户原话："ME，MP，MS，step1，step2全都进行到能开始跑5.3……包括MS怎么还没定下来"，之后就"要不要
先跑一个完整Step1"讨论过，我提出的方案是：先不做全量Step1训练（成本是几十到几百次真实调用/组件，
风险太高、没验证过管线本身能不能跑通就投入不划算），先做一个**最小试点**——只测MP一个组件，
10-20个真实state，"带MP vs 不带MP"配对生成，加一次真实judge打分，只回答一个问题：**"生成→judge
打分→提取出能用的信号"这条完整链路，跑不跑得通？** 用户批准："可以，那先尝试。推进吧。"

## 复用了什么，新写了什么

**复用、未修改**：
- state发现+配对生成逻辑，跟`50_mp_paired_generation_effect_v1_5.py`完全一致（同样的
  `call_with_guard_and_rewrite`兜底处理）；
- judge本体，跟`v1_5_mvp_judge.py`完全一致——这是项目里已经建好、验证过的正式judge instrument
  （不是这次新写的）：匿名A/B配对偏好judge（正反顺序各跑一次，`resolve_ab_ba_quality()`合并，
  不一致就判"tie"，不做多数投票/平均）+ 4项pointwise风险判定（`unsupported_personal_claim`等，
  要求引用原文/证据的逐字substring，不能编）。

**新写的**：脚本`52_step1_mp_minimal_pilot_v1_5.py`，把上面两块接起来，每个真实state跑：
2次生成（带MP/不带MP）+ 2次质量judge（正反序）+ 2次风险judge（每个arm一次）= 6次真实调用。

**用的judge endpoint**：`training_judge`（Gemini 2.5 Flash Lite）——不是`24au`那条脚本用的
"已通过资质认证的judge"（那个需要先跑qualification report，是给最终、要写进结论的正式研究用的）。
这里明确只是管线验证+方向性信号，不是发表级结论。

## 真实跑的时候发现并修了2个脚本bug（跟ME那次是同一类问题：先小样本冒烟测试，再上量）

1. **judge遇到真实503瞬时错误，直接把整批跑崩了**：`client.chat()`自带的内部重试（retries=2）没
   兜住Gemini的一次真实503，异常直接往外抛，导致整个pilot在第1个state的第2次judge调用就崩溃。
   修复：给judge调用加了外层try/except，遇到provider错误就把这一条判定标记为不可用、跳过，不让
   一次瞬时故障拖垮整批。
2. **格式修复重试形同虚设**：原来的写法是"先拿到schema校验通过的parse结果就直接返回"，"逐字
   引用必须真实存在"这个更严格的校验是在外面单独做的——意味着"格式修复"提示词其实永远不会针对
   "引用不是逐字"这种失败被触发（只有整体parse失败才会重试）。修复：把校验回调塞进重试循环内部，
   跟`24au_run_rs_paired_effect_judge_v1_5.py`的写法对齐——现在真的会在第一次引用校验失败后，
   带着"必须逐字复制"的提示词重试一次。修复前后对照（n=2冒烟测试）：修复前风险judge 2/4次因为
   引用校验失败放弃；修复后同样2个state只剩1/4次失败（且日志里能看到重试确实被触发了）。

两个bug都已修复，脚本内有详细注释记录发现过程和修复方式。

## 真实结果（全部6个MP会触发的state，唯一存在的全集，不是抽样）

MP只在6个state上会真的选中候选（这是`50`号脚本已经确认过的真实上限，不是这次n=15参数限制的），
覆盖6个不同用户。全部6个都真实跑完，没有中途崩溃。

| 用户 | MP事实 | with_mp状态 | without_mp状态 | 质量judge结果 |
|---|---|---|---|---|
| p7 | Job: business owner | fixed_by_rewrite（触发`POSSIBLE_ASSISTANT_SELF_ATTRIBUTION_OF_USER_FACT`，重写后过） | clean | tie（正反序不一致，判tie） |
| p11 | Education: high school (in progress) | clean | clean | tie（正反序不一致，判tie） |
| p13 | Education: college degree | clean | clean | tie（正反序不一致，判tie） |
| p15 | Education: currently in grad school | clean | clean | tie（正反序不一致，判tie） |
| p16 | Job: project manager | **fell_back_to_m0**（触发同一guard，且重写也没救回来，整个兜底成8个词的通用回复） | clean | **B（不带MP更好，正反序一致，非"猜"出来的tie）** |
| p17 | Job: graduate student | clean | fixed_by_rewrite（触发`TRACE_REFERENCES_UNAUTHORIZED_EVIDENCE_ID`+`REQUIRED_EVIDENCE_NOT_USED`，跟MP无关，是MS那边的问题） | A（带MP更好，正反序一致） |

## 解读（不夸大，逐条讲清楚每个数字实际代表什么）

**核心、可复现的发现：MP事实类型是"Job:"的，比"Education:"的更容易触发
`POSSIBLE_ASSISTANT_SELF_ATTRIBUTION_OF_USER_FACT`兜底**——3个Job事实（p7/p16/p17）里2个
（p7/p16）在带MP时触发了这个guard，其中p16连重写都没救回来、整条回复退化成8个字的通用句子；
3个Education事实（p11/p13/p15）一个guard都没触发。样本小（3 vs 3），不能当成统计结论，但方向
是这次试点里唯一干净、可解释、有机制支撑的模式——"as a project manager, you've been..."这类
职业类表述，天然比"education: high school"更容易被模型带出第一人称口吻，从而撞上这道防止AI
冒充用户经历说话的guard。

**p16是这次试点里最干净的"MP拖累质量"的真实案例**：带MP直接兜底成通用句子，不带MP是一条具体、
贴合上下文的回复，judge正反序一致地更喜欢不带MP的版本——不是靠"正反不一致就判tie"这种弱证据，
是真的更喜欢。

**p17的"带MP更好"是假信号，不能算MP真的帮了忙**：not_MP arm之所以要重写，是`TRACE_REFERENCES_
UNAUTHORIZED_EVIDENCE_ID`+`REQUIRED_EVIDENCE_NOT_USED`——这两个guard都是关于MS证据处理的，
跟MP在不在完全无关。这次"A更好"更可能反映的是两次独立生成之间的自然波动（LLM本身是随机的），
不是"去掉MP让回复变差了"这个因果结论。

**4/6是tie，而且都是"正反序不一致才判的tie"**（`orders_disagree_resolved_to_tie`），不是
"两次都判tie"——说明这4个state上，judge本身对"带不带MP哪个更好"没有稳定、一致的偏好，这本身
就是一个真实信号：**大多数时候，一条准确匹配的MP事实对回复质量的影响小到连这个judge都测不
稳定**——跟这次试点开始前的猜测方向一致（"MP可能是惰性的"），但现在有真实配对judge数据支持，
不只是读回复文本的主观印象。

**风险judge意外发现了一个和MP无关、但很重要的judge基础设施缺口**：p11的with_mp风险判定里，
judge真的抓到一条`unsupported_personal_claim`+`stale_or_conflicting_evidence_use`
violation——但去读了逐字引用后发现这是个假阳性：证据原文是MS那条"Anna is stressed about
upcoming final exams and unable to study due to having the flu."（EvoEmo的第三人称叙事
习惯，"Anna"是p11的化名），生成器**正确地**把"Anna"解析回了"you"（这正是本session之前修的化名
指代bug的修复效果），但risk judge只看得到原始证据文本（还是"Anna"）和回复文本（说的是"you"），
没有拿到生成器用的那份`current_user_known_aliases`别名映射，没法验证这个指代关系，只能把它
当成"编造了一个证据里没有、还前后矛盾的说法"来判——**判断本身的逻辑没错，只是信息不全**。这
说明：以后要真的拿真实judge给EvoEmo数据打分（不管是这个pilot还是正式Step1训练），judge prompt
也需要拿到跟生成器同样的别名映射，否则每次正确的化名指代解析都会被系统性地误判成风险违规，
风险违规率会被人为抬高——这是MS的化名问题在judge侧的延伸，不是MP本身的问题，但会污染任何拿
这个risk judge在EvoEmo上测出来的`unsupported_personal_claim`基线数字。

## 结论：管线本身跑通了，可以作为往上加量的基础；MP组件本身没有强烈的正负信号，"Job:"类事实
是这次试点里唯一找到的、值得继续盯的具体风险点

1. **管线验证：通过**——6个真实state，36次基础调用（含重试更多），全部跑完，中途发现的2个真实
   bug（瞬时错误崩溃、格式修复重试失效）都已经在跑这批真实数据的过程中被发现并修复，不是靠
   猜的。这是这次试点最主要的目标，已经达成。
2. **MP组件本身**：这次n=6（MP会触发的state的真实上限）看不出稳定的整体质量提升或下降——4个
   tie，1个真实变差（p16，guard导致兜底），1个疑似变好但不能算数（p17，跟MP无关的MS问题）。
   跟之前`50`号脚本"MP修复后p13/p16/p17从失败变成功"这个发现放在一起看：**MP现在的问题不是
   "内容用错了"，是"某些MP事实类型（尤其Job类）比其他类型更容易把生成器带偏、撞上自我归因
   guard"**——这是一个更精确、比"MP整体有没有用"更小范围的问题，值得在正式Step1训练前针对
   "Job:"类MP事实单独看一下，但不构成"MP不能用"的结论。
3. **风险judge需要拿到别名映射才能在EvoEmo上准确工作**——这是继续往上加量（无论是扩大这个pilot
   还是启动正式Step1）之前，值得先修的一个真实、具体、有证据支持的judge基础设施缺口。
