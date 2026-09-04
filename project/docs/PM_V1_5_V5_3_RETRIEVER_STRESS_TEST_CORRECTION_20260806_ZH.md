# 更正：`57_retriever_scale_stress_test_v1_5.py`的"检索器崩溃"结论无效（2026-08-06）

## 状态

```
INVALID_FOR_V5_3_RETRIEVER_QUALIFICATION
CONTRACT_MISMATCH_AND_MISSING_TYPED_METADATA
```

`PM_V1_5_V5_3_RETRIEVER_SCALE_STRESS_TEST_20260806_ZH.md`（commit `5613921`）里"检索器在真实
规模下命中率从90-100%跌到0-10%"这个结论**不成立**，原因是测试脚本本身构造错误，不是检索器
真的坏了。commit不撤销（保留真实修改历史），但从现在起，主状态文档和这份原始发现文档都不能
再被当作"已确认事实"引用。

## 另一个Codex指出的问题，逐条用真实代码核实过，全部属实

1. **ME的"正确答案"模板本身不通过项目自己的编译器**。原脚本signal模板："In one prior
   episode involving {topic}, wrote down the hardest moment before responding and later
   felt less overwhelmed."——直接跑`compile_atomic_reusable_outcome()`，**返回None**。
   根源：这句话没有明确的"I/we"第一人称主语（"wrote down"前面没有主语词），不满足今天刚
   修过的编译器合同。
2. **这意味着我之前"即使接入me_subtype_hint,ME在109条下仍然1/10"这个验证是无意义的**——
   `me_subtype_hint`会把这条不合格的"signal"正确地标成非REUSABLE_OUTCOME（tier=0），
   跟所有干扰项一样，二级排序键根本没有机会发挥作用，1/10完全是词面排序在打平后随机选择
   的结果，不是"hint也救不回来"。
3. **换成一个真正合同有效的signal模板**（"When dealing with {topic}, I wrote down the
   hardest moment before responding, and it helped me feel calmer."，已验证通过编译器）
   **+ 正确传入`me_subtype_hints()`**，用同样的干扰项构造、同样的随机种子重新跑：
   **ME在2到109全部7个规模上都是10/10**，跟另一个Codex给出的数字完全吻合（直接复算验证，
   不是照抄对方结论）。
4. **MS的signal模板用的是"跨多个session的模式"（"Across multiple prior sessions, the
   user has repeatedly felt worse about..."），不是V1.5冻结的`MS_SESSION`构念**（"一个
   严格已完成的具体旧session的具体观察"）——历史修复记录`V15-MEM-50`明确写过"跨session的
   `MS_PATTERN`留给V2"，我这次的MS测试用错了构念，测的不是当前该测的东西。
5. **MS测试从头到尾没有传`ms_semantic_encoder`**，也就是没有用今天已经决定的BGE-M3默认
   路径，测的是已经不是生产默认的旧词面排序。
6. **MP的干扰项里包含"The user previously preferred not to discuss {topic}."**——如果
   用户当前正在谈这个话题，这条历史边界可能真的很重要，不能不假思索地当成"错误答案"的
   干扰噪音，这一条正确答案本身有歧义，不能算清楚的反例。
7. **MP和MS被一路测到109条，但两者在真实EvoEmo数据里的真实规模分别只有约7条（MP，固定）
   和13-33条（MS）**——今天早些时候直接从真实数据数出来的。把三个组件全部推到109再说
   "三组件在真实规模都崩了"，规模设定本身就不对。
8. **干扰项池因为主题+模板数量太少，被迫大量重复**：24个主题×每组件2个干扰模板，最多
   只有48种不同的干扰文本，但要有放回地抽出上百条——这制造的是"极端模板碰撞压力"，
   比真实EvoEmo数据（真实、各不相同的对话内容）人为更难，不能代表真实规模下的自然多样性。

## 现在的诚实结论

不是"V5.3完了"，也不是"检索器在真实规模下没问题"——是**这次测试本身不合格，没有资格
回答这个问题**。真正的答案还没有一次被正确测出来。

已知的、可信的部分：合同正确、规模正确、正确接入typed metadata后，ME在测试自己的干扰
结构下能做到10/10——这是一个真实、正面的信号，但另一个Codex的提醒也对：这还不是正式
资格赛（真实EvoEmo内容比模板碰撞更自然多样，MS还没用BGE-M3测过，MP的边界歧义还没处理），
不能就此宣布"检索没问题"。

## 下一步：正在按对方建议的方向重做一版尊重各组件真实合同的V2测试

- ME：signal必须先过`compile_atomic_reusable_outcome()`才能进入测试；必须传
  `me_subtype_hints()`；规模只测到真实上限109，但用真实通过编译器的多样化signal（不是
  单一模板复制）。
- MS：改用具体的"一个已完成旧session的具体观察/区分/未完成问题"构念；接入
  `ms_semantic_encoder`（BGE-M3）；规模只测13、22、33，不测到109。
- MP：规模只测到7；把"previously preferred not to discuss"这类边界类干扰项单独处理，
  不跟纯粹话题无关的干扰项混在一起统计。
- 每条测试跑之前加真实的gold有效性断言（模板通过对应编译器/构念检查），不满足直接跳过，
  不能产出一个基于无效gold的命中率。
