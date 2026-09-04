# RS域对比审计：训练构造在真实eligibility runtime下命中率为0（2026-08-06）

## 为什么RS要用不同方法

RS没有记忆目录、没有`compile_atomic_*`合同，它的"资格"完全由
`v1_5_strategy_rag_runtime.observable_flags()`+`eligible_moves()`这套对最近对话文本
的确定性正则判断决定——这套函数今天全程被复用、验证过（ME/MP的`explicit_advice_welcome`
就是从这里借来的）。所以RS真正的域对比问题不是"候选通不通过编译器"，是"同一套真实、已
验证的eligibility函数，在训练构造的语言风格上，跟在真实自然对话上，触发率是否可比"。

脚本：`65_rs_eligibility_domain_comparison_v1_5.py`。三个域：训练域（V3 blueprint，160个
RS state的`current_user_text`）、真实ESConv（`ESConv.json`，4947个真实seeker轮次）、
真实EvoEmo（138个state面板，跟全程使用的同一批）。全部调用同一个`observable_flags()`/
`eligible_moves()`，零API。

**一个真实踩坑记录在案**：V3导出自带的`private_retrieval_audit_not_model_input.
observable_flags`用的是另一套更丰富、字段完全不同的schema（`legal_occupational_
misconduct`、`ordinary_rag_hard_off`等，明显是另一个hard-off门模块产出的），直接拿来调
`eligible_moves()`会因为缺字段报`KeyError`。没有将错就错混用两套不同的资格实现，改成
用`current_user_text`重新调同一个`observable_flags()`，保证三个域用的是完全同一套函数。

## 结果：训练域在真实eligibility gate下命中率是0/160，真实两个域都不是0

| 域 | n | 至少一个move eligible的比例 |
|---|---:|---:|
| 训练域（V3构造） | 160 | **0.0%** |
| 真实ESConv | 4947 | 3.4% |
| 真实EvoEmo | 138 | 10.1% |

## 人工核实：不是脚本bug，是真实的模板-正则不匹配

抽读8条训练域`current_user_text`，逐条核对：

> "...Please ask exactly one focused question to clarify the feeling I already indicated;
> do not add another support move."
> （对应意图候选：`support_move: Ask one focused question to clarify a feeling...`）

这条文本按语义读，**明显是在要求"问一个聚焦澄清问题"**，人类读者会认为这应该触发
`focused_clarification_opportunity`。但`_CLARIFICATION_RE`实际匹配的是"hard to
explain"、"not sure how to explain/describe/say"、"it's complicated"、"depends on"这
类表达——训练构造用的是**直接向生成器下指令的口吻**（"Please ask exactly one focused
question...do not add another support move"），不是这套正则设计时参照的、更自然的
情绪化表达方式。8条样本全部是这个模式：**训练构造的"user turn"读起来更像是给生成器的
构造说明书，不是自然的求助者语言，所以从头到尾没有一条会触发这套已验证的真实eligibility
正则**。

## 这比已知的ME问题更严重的地方

ME的问题是"候选被检索到、但没通过编译器"（候选层面的问题）。RS这次发现的是**更前置的一层**：
如果训练数据照这个构造方式生成，RS在这160个state里**连"要不要考虑RS"这道最外层的资格门
都从未打开过**——不是候选选错，是RS从一开始就不会被视为候选来源。如果拿这批数据训练RS的
value head，模型在"当前轮是否存在RS机会"这个最基础的判断上，训练时看到的永远是"没有机会"
（因为eligibility gate从不开），这跟ME的`past_action_result`训练域恒为0是同一种机制，
但发生在更早的一环。

## 结论与建议

1. **这不是"RS表示不足"（V5.2历史结论），是这次构造的语言风格系统性绕开了真实eligibility
   正则**——根因层面和ME一样：内部构造语料自己的表达方式，没有对齐生产环境实际使用的检测
   逻辑；
2. 真实ESConv（3.4%）和真实EvoEmo（10.1%）本身命中率也不高——这提示`eligible_moves()`
   这套正则在两个真实域里都偏保守，RS机会本来就相对稀疏，这是需要单独关注的另一个问题，
   但至少不是0；
3. **P2生成正式RS数据前，必须先把RS的"current_user_text"构造方式改成会触发真实
   `observable_flags()`正则的自然表达**，不能沿用这次V3 blueprint的指令式措辞——这跟
   ME正例模板需要修正是同一类问题，应该一起解决，不要分别在两个组件上各自踩一遍同样的坑；
4. 建议在正式FIT数据生成前，对全部四个组件都跑一次"用真实生产验证过的判定函数
   （compiler/eligibility/observable_flags）自检训练构造"，作为P2冻结前的通用检查项，
   而不是等每个组件各自暴露问题才发现。
