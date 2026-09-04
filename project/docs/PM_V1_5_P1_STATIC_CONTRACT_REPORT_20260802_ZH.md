# PM V1.5 P1 静态合同完成报告

日期：2026-08-02  
状态：`PASS_P1_COMPLETE`  
API 调用：0  
新增人评：0

## 结论

P1 已完成。它没有宣称 PM 已经学会四组件，而是先使后续 256-state 数据成为一个定义一致、
不可错绑、不可把答案塞进特征的训练问题。P2 可以开始；旧 head、旧十几条小包和外部结果
仍不得用于晋级。

## 本轮真正修掉的结构问题

1. **候选绑定**：MP/MS/ME 的 gold 必须绑定真实 retriever Rank-1 的 `candidate_id +
   normalized text SHA-256`；RS 同样绑定真实 Rank-1 卡及完整 execution surface。
2. **ME 不再等于任意旧事件**：compiler 在任何标签和结果之前，仅抽取 action、result、
   mechanism、unresolved marker，并产生非 gold 的 subtype hint；H1 独立裁定最终 subtype。
3. **特征和标签分权**：feature builder protocol 与 labeler protocol 必须不同；每个 head 只有
   6–7 个冻结数值特征。label、action、response、judge、dataset/split/domain 和各类 ID 不能进入
   模型特征。
4. **动作链完整**：四 head 先给 requested bits；候选缺失、硬门、组件冲突和共同 token 预算
   形成 feasible action；Step 2 guard 后才是 realized action。任何后层只能删除 bit，不能新增。
5. **默认关闭合法但不预先阉割组件**：`M0+R0` 是无合格机会时的合法结果；完整运行仍保留
   16 个动作和四个 head，没有手动把 MP/MS/ME/RS 永久关闭。

## 冻结特征面

| Head | 特征数 | 主要可见构念 |
|---|---:|---|
| MP | 6 | 内容匹配、增量、偏好 scope、profile 相关性、实体 scope、当前冲突 |
| MS | 7 | 内容匹配、增量、当前目标、旧结果/区分、具体主线、连续性邀请、已解决标记 |
| ME | 7 | 内容匹配、增量、当前目标、action、result、mechanism、连续性邀请 |
| RS | 6 | 内容匹配、mode、goal、burden、boundary、nonredundancy |

这些是模型输入上限，不是 H1 人评字段上限。H1 可以看到足够证据作判断，但标签与理由不得
在训练时回流为特征。

## 验证

- Python 编译：通过；
- candidate binding、memory compiler、feature bridge、16-action runtime、Step 2 action
  contract 等定向回归：45/45 通过；
- whitespace/diff 静态检查：通过；
- 外部 lockbox：未读取；
- 生成 API：未调用。

机器报告见 `outputs/pm_v1_5_p1_static_contract_v1/report.json`。

## 唯一下一步

进入 P2，一次性建立 256 个 candidate-first state：128 FIT、64 FRESH_CONFIRMATION、64
SEALED_INTERNAL_TEST。先在冻结的内部 super-domain 中生成状态与候选，再制作一个 H1 bundle；
在 H1 冻结前不训练，在 P5 前不打开 sealed outcome，在 P6 前不消费 EvoEmo p7–p18。

P2 私有构造矩阵现已生成：
`data/pm_v1_5_final_candidate_first_v1/private/construction_blueprint.jsonl`。其 256 个 target
state 使用 256 个不同 user；每个 split 的 16 动作均衡；FIT 的 16 个判断逻辑族内，每个 head
都有 4 个 on 与 4 个 off 构造意图；confirmation/sealed 各使用 16 个 FIT 未见逻辑族。构造意图
明确不是 gold、不是模型输入，只有实际 Rank-1 加 H1 才能产生标签。
