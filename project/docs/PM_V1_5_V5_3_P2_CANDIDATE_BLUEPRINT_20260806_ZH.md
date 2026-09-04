# W7：正式P2候选蓝图物化（2026-08-06）

## 范围声明

只构造候选与split，**不生成任何ON/OFF回复，不训练head，不读取quality/risk**。全程绑定
`v1_5_v5_3_release_bindings.build_static_release_bindings()`产出的release_identity
（`v53static_d45cfb6508f90030460da7b6`），确保6-card Bank、BGE-M3 snapshot、ME
production方法跟当前冻结的静态release完全一致，不是脚本自己另选检索器。零API。

脚本：`scripts/v1_5/71w_materialize_p2_candidate_blueprint_v1_5.py`
output：`outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v1/{summary,mp_states,ms_states,
me_states,rs_states,interaction_states}.json`

## 四组件同栈细节，按运行手册W7要求逐条落实

- **MP**：保留`preference`/`profile`两个内部子域，`source_metadata`携带`mp_subtype`，
  验证`exact_rank1_subtype`跟设计意图的子域精确匹配（24个state，24/24精确匹配）；
- **MS**：通过`ms_semantic_encoder=BgeM3Encoder()`使用release绑定的BGE-M3 snapshot和
  真实全因果同用户候选池；
- **ME**：使用production lexical+typed_tier（`discover_final_typed_memory_candidates`
  不传BGE reranker），Rank-1编译失败记`me_available=False`、`hard_off_reason=
  rank1_compiler_invalid_me_unavailable_no_rank2`，**没有降级取Rank-2**；
- **RS**：`rs_mechanical_candidate_pool()`形成机械安全池，`rs_shared_candidate_top1()`
  产生所有policy共享的唯一候选（`transparent_priority`或`lexical_fallback`两种
  selection_mode之一）。

每个state保存：可见对话（`visible_dialogue`）、`current_goal`、完整Top-k候选ID、
exact Rank-1 ID、owner（`owner_id`，本轮均为单用户单state，天然不跨用户）、
`session_index`（time）、`exact_rank1_subtype`（version/subtype合一，本轮无版本冲突
构造）、score/margin（`score_top1_lexical_relevance`/`score_top1_top2_margin`）、
候选数（`n_candidates_in_catalog`）、增量token（`incremental_injected_tokens`）、
`model_visible_surface`、`hard_off_reason`、Step1特征（`step1_features`，直接复用
`mp_contribution_slots`/`ms_contribution_slots`/`me_subtype_hints`/RS语义flags，不是
另起一套）。

## 真实、唯一的规模（不复制模板、不冻结最终N）

| | 数量 |
|---|---:|
| MP state | 24（8族×preference/profile/redundant 3变体） |
| MS state | 16（8族×continuity/redundant 2变体） |
| ME state | 32（复用脚本68已验证的种子，未重新构造） |
| RS state | 7（5个AM家族自然语句+1个无信号对照+1个explicit_stop边界） |
| **单组件state合计** | **79** |
| interaction state（多组件） | 3（job_transition/chronic_pain/parenting_conflict） |
| 唯一counterfactual_group | 31 |
| 唯一family | 15（MP/MS/ME共享的8个主题族 + RS自己的7个动作/边界族） |
| 唯一user | 58 |

## 候选可用率（描述性，不是显著性检验）

| 组件 | 候选可用率 |
|---|---:|
| MP | 100%（24/24，修正查询措辞对齐后） |
| MS | 100%（16/16） |
| ME（`me_available`，需候选存在**且**通过编译器） | 96.9%（31/32，跟W6独立测得的
  compiler-valid率96.9%一致） |
| RS | 85.7%（6/7，唯一缺失的是刻意构造的explicit_stop边界，符合预期） |

**`me_available`不是"命中意图target"**：这里测的是"production方法在这个state上有没有
产出一个可执行的ME候选"，跟W6/脚本68测的"Rank-1是不是设计意图的那一条"是两个不同指标——
后者只有46.9%（W6结果），本文档不重复计算，避免把两个不同粒度的数字混着引用。

## 一个构造过程中真实发生、如实记录的问题：MP查询对齐

第一版MP"profile_positive"/"preference_positive"查询用的是泛泛话术（"I need advice on
how to arrange things given my situation with X"），跟目标MP条目文本零词面重叠，
候选可用率只有54.2%（11/24缺失）。**这是构造脚本自己的问题，不是MP组件的真实发现**——
跟脚本68早期ME查询未对齐target措辞时0/8的教训是同一类错误。已改成查询措辞直接引用
目标条目自身的具体用词（例如profile fact是"Occupation: currently between roles"，
查询就提到"my situation with...currently between roles"），修正后100%可用。

**副作用（如实披露、未进一步处理）**：修正后的"preference_positive"查询因为跟目标文本
共享了3个以上内容词，被`_is_redundant()`启发式误判成`current_redundant=True`——这个
启发式设计用来检测"当前话语已经说过候选要补充的具体内容"，但这次是"查询措辞刻意贴合
target来保证被检索到"和"当前话语本身就在重复目标内容"两种情况在文字层面难以区分。
留给leader决定：是接受这个已知局限（Step1特征层面的噪声），还是需要更细致的查询设计。

## Interaction state：3个，MS/ME/RS三个组件在同一state上都真实可用，MP系统性缺失

| state | MP | MS | ME | RS |
|---|---|---|---|---|
| job_transition | ✗ | ✓ | ✓ | ✓（lexical_fallback→AM14） |
| chronic_pain | ✗ | ✓ | ✓ | ✓（lexical_fallback→AM14） |
| parenting_conflict | ✗ | ✓ | ✓ | ✓（transparent_priority→AM10） |

MP在全部3个interaction state里都没有候选——这次interaction查询的措辞是围绕ME/MS内容
构造的（"could really use some advice...like I said, {MS目标}"），没有专门带上MP
preference/profile的具体用词，所以MP的`match_level`门槛没通过。**如实报告为真实的
3-component（不是4-component）interaction覆盖，没有为了凑满4组件而勉强对齐MP查询**——
按运行手册"只报告真实唯一group数量"的要求，这就是这次真实做到的范围。

## 未做/留给leader的事

1. 只测了3个family的interaction state，另外5个族没有构造对应的多组件版本；
2. 没有为interaction state专门设计能同时命中全部4组件的查询——如果leader认为
   4-component interaction覆盖是必需的，需要专门的查询/内容设计，不是简单拼接；
   已单组件生成的79个state加3个interaction state没有做shortcut/duplicate/同源
   overlap的独立复核——按运行手册7.4节，这是leader收到蓝图后的下一步；
3. `proposed_split`字段全部占位为`TBD_by_leader`，没有自行决定user/family/group
   如何分配train/fit/confirmation/sealed；
4. RS的"family"用的是AM动作家族，MP/MS/ME用的是话题家族，两套family体系目前是
   并列而非统一命名空间，留给leader决定是否需要对齐。
