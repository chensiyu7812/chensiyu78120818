# PM V1.5：ESConv train-only Strategy Card 归纳审计

状态：`INDUCTIVE_SOURCE_AUDIT_COMPLETE_OPEN_CODING_PENDING`

## 1. 对现有五族与 50 卡的纠正

现有五个 family 不是从数据聚类得到的。它们是 ESConv 原生八类：

`Question / Restatement / Reflection / Self-disclosure / Affirmation /
Suggestions / Information / Others`

中人为保留的五类：

`Question / Restatement / Reflection / Affirmation / Suggestions`。

排除 `Self-disclosure / Information / Others` 的理由是第一篇优先采用可泛化、低风险的
technique，而不是证据显示这三类在数据中自然消失。这个安全边界可以作为设计选择，
但不能写成“ESConv 数据自动浓缩出了五类”。

当前 50 个 submove 也是先由人工 taxonomy 定义，再用 BGE 尝试把五族内部来源弱映射
到这些 submove。固定样例已经证明 BGE 会发生功能错配。因此 50 卡现在降级为：

> top-down safety reference，等待与独立归纳出的 codebook 比较。

暂停继续扩卡，也暂停把 50 卡人审当作当前下一门。已有定义不删除，归纳结果与其一致
的部分可以复用；不一致部分必须合并、删除或补充。

## 2. 本次真正使用的数据

分析只使用现有 raw Strategy Bank 中的 ESConv train 来源：

- 冻结 custom 70/15/15 dialogue split；
- non-overlap train 共 875 个 dialogue；
- raw bank 已提前排除 52 个 development seed dialogue，剩 823 个来源；
- 再排除 75 个 SupportNeed packet source，最终 748 个独立 dialogue；
- validation/test 行读取为 0；
- EvoEmo-overlap source 为 0；
- external response、gold outcome、judge outcome 读取为 0。

分析粒度是一条有 prior visible seeker context 的 supporter turn。原生八类全部保留，
原标签只用于事后诊断和隐藏分层，不作为 card gold。

## 3. 客观清洗

从 11,590 条 raw rows 中，按第一个命中原因排除：

| 原因 | 行数 |
|---|---:|
| SupportNeed packet source | 1,094 |
| 没有 prior visible seeker turn | 413 |
| 不足 5 词，无法支持 card 归纳 | 581 |
| normalized duplicate | 168 |
| greeting-only | 45 |
| closing-only | 78 |
| platform/survey meta | 47 |
| 明确低信息 acknowledgement | 16 |

最终保留 9,148 条、748 个独立 dialogue。少于 5 词的门只表示该句信息不足以定义一张
可复用卡，不表示它在原对话中一定没有任何交流作用。

清洗后原生标签行数：

| ESConv native label | 行数 | 独立 dialogue |
|---|---:|---:|
| Question | 1,629 | 650 |
| Restatement or Paraphrasing | 562 | 412 |
| Reflection of feelings | 767 | 488 |
| Self-disclosure | 942 | 528 |
| Affirmation and Reassurance | 1,572 | 654 |
| Providing Suggestions | 1,632 | 618 |
| Information | 652 | 371 |
| Others | 1,392 | 580 |

清洗后仍发现 48 条 normalized duplicate 跨不同
native labels，其中 `Others | Question` 21 条、`Information | Question` 11 条。这是
原标签边界不稳定的直接证据。

## 4. 原始文本是否自然聚类

对 9,148 条清洗文本做 TF-IDF、cosine KMeans，比较 `k=5/8/12/20` 和三个固定 seed：

| k | silhouette | seed stability ARI | 与 ESConv 原标签 NMI |
|---:|---:|---:|---:|
| 5 | 0.0098 | 0.1693 | 0.0178 |
| 8 | 0.0137 | 0.2740 | 0.0291 |
| 12 | 0.0139 | 0.1755 | 0.0311 |
| 20 | -0.0332 | -0.0004 | 0.0051 |

这些结果不支持“选一个 k，聚类中心就是 card”。silhouette 接近 0、seed stability
低，而且 cluster 与 native label 几乎无关。raw response cluster 主要受话题、措辞和
具体内容驱动，而不是稳定的 support move。

## 5. ESConv 原生八类本身有多可靠

仅用 supporter response 文本，按 source dialogue 做五折 grouped OOF，预测原生八类：

- macro-F1：`0.3699`
- weighted-F1：`0.4022`

主要类别 F1：

- Question：0.5160
- Suggestions：0.4611
- Others：0.4403
- Self-disclosure：0.3935
- Affirmation：0.3752
- Reflection：0.2806
- Restatement：0.2628
- Information：0.2301

该诊断不是为了训练分类器。它说明原标签有部分表面信号，但类别重叠和噪声很强；
不能直接把八类当八张 card，也不能把其中五类各机械扩成十张。

## 6. 正确的归纳流程

下一步不是换一种 embedding 再自动聚类，而是低成本的 blinded open coding：

1. 从八个 native label 各隐藏分层抽 20 条文本多样样本，共 160 条；
2. 页面隐藏 native label、现有五族和 50 个 submove；
3. 编码者自由描述 response 实际执行的 primary/secondary atomic support action；
4. 同时判断是否主要为 information/self-disclosure、是否能抽象成通用卡、是否有
   risk/boundary 问题；
5. 合并同义动作，保留分歧和多动作，不强迫单标签；
6. 用归纳 codebook 回看全量 9,148 条，计算每个动作的独立 dialogue 支持、跨
   topic/emotion 覆盖和混淆；
7. 与现有 50 卡做 blind alignment：一致则复用，数据没有支持则删除，数据出现新动作
   则补充；
8. 最终 card 数由证据决定，不再预设 5、50、80 或 100。

为了控制人评量，可先让两个独立的语义编码器对 160 条做开放编码；人工只审核全部
分歧、所有 risk/self-disclosure/information 样本，以及固定随机 20% 的一致样本。
但编码器输出只是候选 code，最终 codebook、合并规则和可复用边界必须可见、可修改，
不能像整体 judge 分数一样不透明。

## 7. 研究主张影响

这次纠正不改变 PM 的核心研究问题，也不要求重做 external split。它只修复 RS
treatment 的资源定义：

> 同一冻结、由 train-only 证据支持的 Strategy Bank，是否在部分状态产生足以抵消
> risk 与 cost 的实质回复收益，使 PM 能学会选择性开启。

如果归纳结果只支持 15–30 张清楚的通用卡，少于 50 反而更可信。若
Self-disclosure 或 Information 中存在稳定、低风险、可泛化的原子动作，也允许进入
候选；它们是否保留由 source support、risk boundary 和 clean-pair utility 决定，而
不是先验排除。

## 8. 可复跑产物

- 分析脚本：
  `scripts/v1_5/23l_audit_esconv_train_strategy_taxonomy_v1_5.py`
- 完整报告：
  `outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/audit_report.json`
- 9,148 条清洗来源：
  `outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/clean_train_strategy_universe.jsonl`
- 160 条 label-blind 开放编码包：
  `outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/open_coding_packet.jsonl`
- 人工页面：
  `outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/human_open_coding.html`
- 原标签与来源映射（编码者不可见）：
  `outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/private_lineage.jsonl`
