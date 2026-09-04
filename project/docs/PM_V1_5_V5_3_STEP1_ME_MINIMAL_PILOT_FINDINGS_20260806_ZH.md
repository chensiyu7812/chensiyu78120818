# Step1最小试点：ME单组件——无法测试，且发现了一个更根本的真实gap（2026-08-06）

延续MP/MS/RS同样的"扩展到全部组件"的授权，检查ME是否也能做同样的真实配对生成+judge打分
试点时，第一步（找ME会真实命中的state）就发现ME在这批真实数据上完全打不出候选，原因比
"召回率低"更深一层。

## 检查结果：138个真实state，0个ME真实命中（无论有没有严格编译器把关）

用真实的`discover_final_typed_memory_candidates()`（生产检索/排序层）在全部138个真实
state上跑了一遍：

- **不做编译器把关，只看检索层选中了什么**：ME在138/138个state上"命中"了候选——但选中的
  内容都是原始对话片段（比如"Hey, do you have a moment to chat? Honestly, not great. My
  health has taken a tu..."），不是`ME_REUSABLE_OUTCOME`要求的、拆好的"动作span+结果span"。
- **加上严格编译器把关**（`compile_atomic_reusable_outcome()`，也就是这次session前面
  决定要继续用作ME生产路径的那个编译器）：**0/138**——检索层选出来的top-1候选，没有一个能
  通过编译器验证。
- **更深一层**：连编译器唯一真实通过的3条memory（p9的两条hiking记忆、p12的"engaging
  one-on-one"记忆，人工核实过语义完全站得住，见`PM_V1_5_V5_3_MS_MP_ME_EFFECT_TEST_
  20260806_ZH.md`3.2节）也**从来没有在它们自己用户的任何真实state上被检索层选为top-1
  候选过**——不是"编译器把好东西挡在外面"，是检索排序本身就没有把这3条排到第一位。

## 这意味着什么：不是"ME暂时没测"，是"按目前的决定，ME在这批真实数据上完全打不出候选"

这次session前面已经决定ME的生产路径继续用严格正则编译器（不上语义抽取pilot，见
`PM_V1_5_V5_3_MS_MP_ME_EFFECT_TEST_20260806_ZH.md`第四节）。这个决定本身没有问题——但这次
检查发现，**光有这个决定还不够：检索层和编译器验证层之间根本没有连起来**。`discover_final_
typed_memory_candidates()`对ME的"selected_items"从来没有经过`compile_atomic_reusable_
outcome()`过滤，如果照搬MP/MS现成的写法（直接把`item.text`塞进一个typed candidate字段）
去接入ME，要么在`TypedResourceCandidate.__post_init__`的必填字段检查上直接崩溃（因为ME需要
分开的`past_action`/`observed_outcome`两个字段，原始文本给不出来），要么如果有人不小心把
原始文本硬塞进这两个字段，就会把未经验证、owner/时态都没检查过的原始对话当成"已验证的可复用
记忆"注入进去——这正是这次session反复验证、反复强调必须挡住的那类错误。

**这不是ME试点跑不动的借口，这是试点本身的真实结果**：在"保留严格编译器"这个已经做出的决定下，
ME组件在这批真实数据上就是完全打不出候选，配对生成测试无从谈起（没有state可以配对）。

## 结论

1. **ME的Step1试点结果就是这个检查本身**：0/138真实命中，不需要（也无法）花真实API调用去
   测配对生成——没有能配对的state。
2. **发现了一个必须在ME真正接入V5.3之前修的真实gap**：检索层和编译器验证层未连接。修复
   方向很明确（ME候选构建时必须先过`compile_atomic_reusable_outcome()`，只用编译通过的
   `past_action_span`/`observed_outcome_span`去构造typed candidate），但由于当前ME在这批
   数据上命中率是0，这个修复目前不会改变任何真实行为，可以作为一个明确记录、暂不阻塞V5.3
   上线的已知工程债务，而不是紧急修复项。
3. **对V5.3的实际影响是安全的**：ME组件目前对这批真实数据完全是no-op（不会被真实调用触发），
   一个不会触发的组件不会让PM变差，只是学不到任何ME相关的信号——这跟"ME保守处理"这个决定
   的精神是一致的，只是这次把"保守"的真实后果（完全不触发，不是偶尔触发）说清楚了。
