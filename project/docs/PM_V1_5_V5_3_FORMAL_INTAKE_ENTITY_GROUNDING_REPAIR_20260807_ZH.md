# PM V1.5 V5.3 正式纵向 intake：实体时间线与去模板化修复

日期：2026-08-07

## 结论

首批 11 个正式纵向用户已经迁入受 Git 跟踪的 canonical intake。强化验收后，11/11 用户硬问题为 0，每个用户的 typed ME executable core 仍为 8/8；其中 10 个用户发生了可审计的 relationship/entity 修复，Claude u000 无需修改。

这不是把未通过内容“调到通过”。修复只处理可由当前用户文本客观确定的元数据错误：姓名或稳定角色称呼是否出现、首次出现在哪个 session、entity/event/candidate 是否提前引用，以及一个被绑定到错误 session 的事件。对话、MP/MS/ME 候选和 ME action/result span 均保留；另外仅对 3 处 GPT 模板句做了同步、同义的去模板化，以清除 4 个跨用户共享 8-gram。

## 发现与修复

- 84 条 relationship 全部重建为最小五字段 schema；删除 `note`、`status_after_session_*`、`first_mentioned_session`、`relation` 等可能携带未来轨迹或重复别名的字段。
- 29 条 relationship 的 `valid_from_session` 改为姓名/稳定角色在用户文本中的首次真实出现。
- 7 个从未由用户说出的亲属姓名改为用户实际使用的 `Dad`、`Mum`、`father` 或 `mother`。
- Claude u003 的 `u003_evt_14` 从无依据的 session 13 移到真实叙述该摩托车来源的 session 24，并改成该 session 可直接支持的事件描述。
- session、event、candidate 中所有 entity 引用均不得早于 relationship 的有效起点；event 必须且只能被其声明 session 引用。
- 3 处版本化 surface replacement 同步更新源 user turn 及全部 literal/candidate/preference 副本；语义、subtype 与功能不变。

## 验收结果

- 用户数：11；需修复用户：10。
- 硬验收：11/11 通过；typed exact action/result core：88/88。
- 被保护的 dialogue/candidate/profile/preference surface 非授权漂移：0。
- 1,845 个至少 8 token 的内部 surface：跨用户 exact duplicate group = 0；共享规范化 8-gram group = 0。
- 1,932 个内部 surface 对 48,959 个 ESConv + EvoEmo/ES-MemEval 外部 surface：exact overlap = 0；规范化 8-gram overlap = 0。
- relationship 未来状态字段或旧别名残留：0。
- API 调用：0；训练标签、quality、risk 或 outcome：未读取。

## 解释边界

当前状态仍是首批 intake，而不是完整 80 用户数据集已经完成。报告中的 `LEGACY_COMPILER_DIAGNOSTIC` 只说明旧正则编译器不覆盖这些自然表述；正式内容合同使用 typed exact action/result span，88/88 已通过，不能为了让旧正则变绿而改写自然用户文本。

剩余正式工作仍包括：继续接收并逐批验证其他作者用户、最终全 80 用户的全局配额/语义复核、state 生成后的 actual Rank-1 与严格过去检查，以及 paired effect 生成。当前修复不得被解释为 worth-opening 标签或 PM 训练 outcome。

## 产物

- canonical intake：`data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/canonical_users/`
- 修复前审计源（仅用于复现迁移，运行时不得读取）：`data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/pre_repair_audit_sources/`
- 修复清单：`data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/repair_manifest.json`
- 批次审计：`data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/batch_audit_report.json`
- 修复脚本：`scripts/v1_5/83l_repair_formal_longitudinal_entity_grounding_v1_5.py`
- 审计脚本：`scripts/v1_5/84l_audit_repaired_formal_longitudinal_intake_v1_5.py`
