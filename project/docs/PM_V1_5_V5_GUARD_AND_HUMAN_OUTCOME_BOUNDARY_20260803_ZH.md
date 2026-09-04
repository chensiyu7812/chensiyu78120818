# PM V1.5 V5：机器 Guard 与人工语义结果的最终边界

## 结论

本次 `me→ME` 与 `before/earlier→未授权记忆` 不是两个孤立 typo，而是项目第三次暴露同一类根因：
**把自然语言词面匹配升级成了语义硬裁决。** 失败账本里的 `V15-OBS-10` 与 `V15-GEN-09`
已经分别记录过 advice/listen 关键词和英文请求正则的误杀；`V15-IMPL-09` 是同一错误在 Step2
guard 中复发。

短组件名会与普通语言碰撞；时间词也不携带 provenance。`before` 可能是“回来前停一下”，
`earlier` 可能复述当前用户自己刚刚公开的说法，也可能真的是模型虚构历史。词本身无法区分三者。

## 三层责任

1. **机械实验有效性**只检查 assignment、candidate ID/version/owner、资源是否进入执行器、回复与
   usage 是否存在、executor/generator 身份是否冻结一致。失败才使 ITT 行无效。这里禁止任何自然
   语言语义正则。
2. **冻结运行时 fallback**只拦空回复、明确内部序列化/opaque ID 泄漏，以及已授权 MS/ME 却完全
   没有来源归因。触发 fallback 仍是有效 ITT outcome，不删除、不记 unknown。
3. **人工语义 outcome**判断陈旧/冲突证据、无依据个人断言、因果泛化、虚构回忆、边界违反、过度
   指令与资源是否真正做功。它用于训练 outcome 与机制分析，不是下一轮修 prompt 的许可证。

## 为什么完整人评不会重新形成循环

V2 后只做一个完整 FIT panel：256 个状态，每状态两个 seed-matched ON/OFF 比较；共 512 个质量
比较与 512 个 ON-arm grounding-risk 判断。20% 状态在看到标签前按协议 hash 固定为双评；只允许
一次分歧归并。禁止再拆出十几条“小包”，也禁止根据人评修改 guard、prompt、candidate、seed、
feature 或 threshold。

人评 `material risk=yes` 必须同时提供：回复中的具体 claim、它应当来自 current context 还是授权
resource、具体的 owner/time/entity/fact/cause/boundary mismatch，以及为何足以改变采用决定。仅出现
时间词、仅提到历史、回复通用或个人风格偏好都不够。证据不足填 uncertain；primary 保守记 nonpositive，
敏感性分析排除，但不能触发新版本修复。

## 本次修复为什么属于机械纠错，而不是看结果调方法

V1 在 1024 个计划调用中完成 346 行时出现 134 次 fallback，比例异常高。用修复后的 guard 对这 346
条**原始 primary response 原样离线重放**，预测 fallback 降为 16 条，且剩余原因全部是已授权 MS/ME
没有任何历史出处标记；没有读取 A/B 质量、风险或组件胜负。V1 与 V2 的 256 个状态、候选、arm、seed
和消息完全相同，`call_plan_sha256` 均为
`62bcd555b32f6c655b2886167345fdc2977feca83daa5374c165005081e0d37e`。因此本次只纠正了预先可证的
guard 实现错误，没有换题、换 seed 或根据科学结果挑样本。

## V2 之后的终局

V2 已启动后，剩余 guard false-positive、generator nonuse、material misuse 或人评不确定都作为当前
冻结执行器的真实表现报告。它们可以让某个 head 学不到、让系统主张缩小，不能再授权 V1.5 V3。
若未来研究要改 executor，应建立新的方法版本和新的 PM，而不是继续修到本论文通过。

机器合同：`data/pm_v1_5_contracts/v5_guard_and_human_outcome_boundary_v1.json`。
