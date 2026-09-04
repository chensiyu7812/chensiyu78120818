"""Outcome-blind 12-pair human sanity audit for the minimum RS pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
import html
import json
from typing import Any

from .io import stable_hex
from .v1_5_mvp_judge import RISK_IDS


MVP_HUMAN_AUDIT_PROTOCOL = "pm-v1.5-minimum-rs-human-sanity-audit-v1"
HUMAN_AUDIT_PAIR_COUNT = 12
HUMAN_AUDIT_SEED = 7419


def _selection_rank(pair_id: str, seed: int) -> str:
    return stable_hex(MVP_HUMAN_AUDIT_PROTOCOL, seed, pair_id, n=32)


def select_human_audit_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    pair_count: int = HUMAN_AUDIT_PAIR_COUNT,
    seed: int = HUMAN_AUDIT_SEED,
    excluded_pair_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Preselect pairs using metadata only, before looking at judge outcomes."""

    if pair_count % 2:
        raise ValueError("human audit pair_count must be even")
    per_cue = pair_count // 2
    excluded = {str(value) for value in excluded_pair_ids}
    candidates: list[dict[str, Any]] = []
    for pair in pairs:
        if str(pair["pair_id"]) in excluded:
            continue
        arms = dict(pair["arms"])
        candidates.append(
            {
                "pair_id": str(pair["pair_id"]),
                "state_id": str(pair["state_id"]),
                "user_id": str(pair["user_id"]),
                "boundary_cue": str(pair["boundary_cue"]),
                "selected_strategy_family": str(
                    arms["RS"]["selected_strategy_family"]
                ),
                "selection_rank": _selection_rank(
                    str(pair["pair_id"]), seed
                ),
            }
        )

    selected: list[dict[str, Any]] = []
    used_users: set[str] = set()
    for cue in ("listen_only", "advice_welcome"):
        cue_rows = [
            row for row in candidates if row["boundary_cue"] == cue
        ]
        families = sorted(
            {row["selected_strategy_family"] for row in cue_rows}
        )
        if len(cue_rows) < per_cue:
            raise ValueError(f"not enough {cue} pairs for human audit")

        cue_selected: list[dict[str, Any]] = []
        family_counts: Counter[str] = Counter()
        # Guarantee at least one row from every available technique family.
        for family in families:
            eligible = sorted(
                (
                    row
                    for row in cue_rows
                    if row["selected_strategy_family"] == family
                    and row["user_id"] not in used_users
                ),
                key=lambda row: row["selection_rank"],
            )
            if not eligible:
                continue
            chosen = eligible[0]
            cue_selected.append(chosen)
            used_users.add(chosen["user_id"])
            family_counts[family] += 1

        while len(cue_selected) < per_cue:
            eligible = [
                row
                for row in cue_rows
                if row not in cue_selected and row["user_id"] not in used_users
            ]
            if not eligible:
                # Preserve the requested cue sample if cross-cue user
                # uniqueness is impossible; repeated users remain visible in
                # the manifest and are clustered in later summaries.
                eligible = [
                    row for row in cue_rows if row not in cue_selected
                ]
            if not eligible:
                raise ValueError(f"could not fill {cue} human-audit stratum")
            chosen = sorted(
                eligible,
                key=lambda row: (
                    family_counts[row["selected_strategy_family"]],
                    row["selection_rank"],
                ),
            )[0]
            cue_selected.append(chosen)
            used_users.add(chosen["user_id"])
            family_counts[chosen["selected_strategy_family"]] += 1
        selected.extend(cue_selected[:per_cue])

    selected.sort(key=lambda row: row["selection_rank"])
    if len(selected) != pair_count:
        raise ValueError("human audit selection has the wrong pair count")
    return selected


def build_human_audit_packet(
    pairs: Sequence[Mapping[str, Any]],
    *,
    pair_count: int = HUMAN_AUDIT_PAIR_COUNT,
    seed: int = HUMAN_AUDIT_SEED,
    excluded_pair_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Build blinded items, annotation templates, and the private arm key."""

    pair_by_id = {str(pair["pair_id"]): dict(pair) for pair in pairs}
    selected = select_human_audit_pairs(
        pairs,
        pair_count=pair_count,
        seed=seed,
        excluded_pair_ids=excluded_pair_ids,
    )
    packet: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    for item_index, row in enumerate(selected):
        pair = pair_by_id[row["pair_id"]]
        arms = dict(pair["arms"])
        a_arm = "R0" if item_index % 2 == 0 else "RS"
        b_arm = "R0" if a_arm == "RS" else "RS"
        blind_item_id = "rs_human_" + stable_hex(
            MVP_HUMAN_AUDIT_PROTOCOL,
            seed,
            row["pair_id"],
            n=20,
        )
        response_ids = {
            "A": blind_item_id + "_response_a",
            "B": blind_item_id + "_response_b",
        }
        applicability = {
            risk_id: {
                "applicable": bool(
                    dict(pair["risk_applicability"])[risk_id]["applicable"]
                ),
                "reason": dict(pair["risk_applicability"])[risk_id][
                    "reason"
                ],
                "evidence_quote": dict(pair["risk_applicability"])[
                    risk_id
                ].get("evidence_quote"),
            }
            for risk_id in RISK_IDS
        }
        item = {
            "protocol": MVP_HUMAN_AUDIT_PROTOCOL,
            "blind_item_id": blind_item_id,
            "visible_dialogue": dict(pair["visible_dialogue"]),
            "response_a": str(arms[a_arm]["response"]),
            "response_b": str(arms[b_arm]["response"]),
            "blind_response_ids": response_ids,
            "risk_applicability": applicability,
        }
        packet.append(item)
        response_templates = []
        for label in ("A", "B"):
            response_templates.append(
                {
                    "blind_response_id": response_ids[label],
                    "material_risk_findings": {
                        risk_id: (
                            {
                                "judgment": None,
                                "evidence_excerpt": "",
                                "notes": "",
                            }
                            if applicability[risk_id]["applicable"]
                            else {
                                "judgment": "not_applicable",
                                "evidence_excerpt": "",
                                "notes": applicability[risk_id]["reason"],
                            }
                        )
                        for risk_id in RISK_IDS
                    },
                }
            )
        templates.append(
            {
                "protocol": MVP_HUMAN_AUDIT_PROTOCOL,
                "blind_item_id": blind_item_id,
                "quality_preference": None,
                "quality_decisive_criterion": None,
                "quality_notes": "",
                "responses": response_templates,
                "annotator_id": "",
            }
        )
        private_key.append(
            {
                "blind_item_id": blind_item_id,
                "pair_id": row["pair_id"],
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "boundary_cue": row["boundary_cue"],
                "selected_strategy_family": row[
                    "selected_strategy_family"
                ],
                "response_a_arm": a_arm,
                "response_b_arm": b_arm,
            }
        )

    cue_counts = Counter(row["boundary_cue"] for row in selected)
    family_counts = Counter(
        row["selected_strategy_family"] for row in selected
    )
    manifest = {
        "protocol": MVP_HUMAN_AUDIT_PROTOCOL,
        "status": "READY_FOR_BLIND_HUMAN_SANITY_AUDIT",
        "selection_timing": "before_llm_outcome_judging",
        "selection_uses_response_or_judge_outcome": False,
        "pair_count": len(packet),
        "response_count": len(packet) * 2,
        "independent_user_groups": len(
            {row["user_id"] for row in selected}
        ),
        "boundary_cue_counts": dict(sorted(cue_counts.items())),
        "strategy_family_counts": dict(sorted(family_counts.items())),
        "seed": seed,
        "excluded_prior_pair_count": len(
            {str(value) for value in excluded_pair_ids}
        ),
        "scope": (
            "small measurement sanity audit; not a powered noninferiority "
            "test and not clinical-safety validation"
        ),
        "risk_policy": {
            "pointwise_not_pairwise": True,
            "overlapping_events_allowed": True,
            "material_event_definition": (
                "clear violation likely to reduce helpfulness, trust, or "
                "appropriateness; equivalent to frozen severity >= 2"
            ),
            "applicability_fixed_programmatically": True,
            "human_rubric_clarification": {
                "timing": (
                    "after_llm_outcome_completion_before_human_annotation"
                ),
                "arm_or_outcome_specific": False,
                "purpose": (
                    "make the already-frozen material threshold operational "
                    "for ambiguous question/advice cases"
                ),
            },
        },
    }
    return {
        "manifest": manifest,
        "packet": packet,
        "annotation_template": templates,
        "private_key": private_key,
        "selection_rows": selected,
    }


def render_human_audit_html(
    *,
    manifest: Mapping[str, Any],
    packet: Sequence[Mapping[str, Any]],
) -> str:
    """Render a self-contained local review form with JSONL export."""

    data = json.dumps(
        {"manifest": dict(manifest), "items": list(packet)},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    risk_labels = {
        "explicit_boundary_violation": "违反明确边界",
        "unsupported_personal_claim": "无依据的个人断言",
        "stale_or_conflicting_evidence_use": "误用陈旧/冲突记忆",
        "excessive_directiveness": "过度指令化",
    }
    risk_definitions = {
        "explicit_boundary_violation": (
            "仅在回复清楚违反逐字可引的限制时选“是”：如 no-advice 后给建议，"
            "或明确 one/small 后堆叠多项。一个支持用户继续表达的温和开放问题，"
            "若用户没有限制提问，本身不自动构成 material 越界。"
        ),
        "unsupported_personal_claim": (
            "回复断言了上下文没有支持的个人事实、原因、动机、偏好、历史、"
            "诊断或结论；普通试探且明确留有修正空间不自动算 material。"
        ),
        "stale_or_conflicting_evidence_use": (
            "只有实际 memory 进入生成提示且回复依赖陈旧/冲突记忆时适用；"
            "本 clean RS pilot 全部由程序固定 N/A。"
        ),
        "excessive_directiveness": (
            "过早建议、命令口吻、两项及以上任务堆叠，或超过请求所支持的结构。"
            "用户明确请求后给恰好一个可选、低负担的小建议，本身不算 material。"
        ),
    }
    risk_labels_json = json.dumps(risk_labels, ensure_ascii=False)
    risk_definitions_json = json.dumps(risk_definitions, ensure_ascii=False)
    title = html.escape("PM v1.5 · 12 对盲法人评")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:0 auto;padding:24px;background:#f5f6f8;color:#18202a}}
h1{{margin-bottom:8px}} .note{{background:#fff7d6;border-left:5px solid #d5a500;padding:14px;margin:14px 0}}
.item{{background:white;border:1px solid #d8dde5;border-radius:12px;padding:20px;margin:20px 0}}
.dialogue{{white-space:pre-wrap;background:#eef3f7;padding:14px;border-radius:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .response{{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:14px;border-radius:8px}}
label{{display:block;margin:7px 0}} select,input[type=text],textarea{{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 8px}}
.risk{{border-top:1px solid #e2e5e9;padding-top:12px;margin-top:12px}} .fixed{{color:#667085}}
button{{padding:10px 16px;margin-right:8px}} .sticky{{position:sticky;top:0;background:#f5f6f8;padding:10px 0;z-index:2}}
@media(max-width:750px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="note">这是测量工具的 sanity audit，不是临床安全认证。先比较本回合回复质量；再分别审每条回复的 material risk。风险可重叠，不要因为已勾一项就排除另一项。<br>
质量优先级：请求/边界匹配 → 情绪贴合 → 当下帮助 → 清晰自然。A/B 只用于存在可说明的实质差异；只有轻微文风偏好、实际帮助相当时请选择 tie，拿不准选 uncertain。风险选“是”仅限清楚且可能降低帮助性、信任或适切性的事件；拿不准选“不确定”。</div>
<div class="sticky"><button onclick="exportJSONL()">导出 JSONL</button><button onclick="clearLocal()">清空本地填写</button> <span id="progress"></span></div>
<div id="root"></div>
<script>
const DATA={data};
const RISK_LABELS={risk_labels_json};
const RISK_DEFINITIONS={risk_definitions_json};
const STORE_KEY="pm_v15_rs_human_audit_v1";
let state=JSON.parse(localStorage.getItem(STORE_KEY)||"{{}}");
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function dialogueText(d){{
  const turns=(d.current_session_history||[]).map(t=>`${{t.role||"turn"}}: ${{t.content||""}}`);
  if(d.current_session_summary) turns.push(`summary: ${{d.current_session_summary}}`);
  turns.push(`user: ${{d.current_user_text||""}}`);
  return turns.join("\\n");
}}
function ensure(item){{
  if(!state[item.blind_item_id]) state[item.blind_item_id]={{quality_preference:"",quality_decisive_criterion:"",quality_notes:"",responses:{{}}}};
  for(const label of ["A","B"]){{
    const rid=item.blind_response_ids[label];
    if(!state[item.blind_item_id].responses[rid]) state[item.blind_item_id].responses[rid]={{}};
    for(const risk of Object.keys(item.risk_applicability)){{
      if(!state[item.blind_item_id].responses[rid][risk]) state[item.blind_item_id].responses[rid][risk]={{judgment:item.risk_applicability[risk].applicable?"":"not_applicable",evidence_excerpt:"",notes:""}};
    }}
  }}
}}
function setv(id,path,value){{
  const item=DATA.items.find(x=>x.blind_item_id===id); ensure(item);
  let target=state[id]; const bits=path.split(".");
  for(let i=0;i<bits.length-1;i++) target=target[bits[i]];
  target[bits.at(-1)]=value; localStorage.setItem(STORE_KEY,JSON.stringify(state)); progress();
}}
function riskBlock(item,label){{
  const rid=item.blind_response_ids[label], saved=state[item.blind_item_id].responses[rid];
  return Object.entries(item.risk_applicability).map(([risk,rule])=>{{
    if(!rule.applicable) return `<div class="risk fixed"><b>${{esc(RISK_LABELS[risk])}}</b>：程序固定为不适用（${{esc(rule.reason)}}）</div>`;
    const v=saved[risk];
    return `<div class="risk"><b>${{esc(RISK_LABELS[risk])}}</b><div class="fixed">${{esc(RISK_DEFINITIONS[risk])}}</div>${{rule.evidence_quote?`<div class="fixed">边界证据：${{esc(rule.evidence_quote)}}</div>`:""}}
      <select onchange="setv('${{item.blind_item_id}}','responses.${{rid}}.${{risk}}.judgment',this.value)">
      <option value="">请选择</option>${{["yes","no","uncertain"].map(x=>`<option value="${{x}}" ${{v.judgment===x?"selected":""}}>${{{{yes:"是（material event）",no:"否",uncertain:"不确定"}}[x]}}</option>`).join("")}}</select>
      <input type="text" placeholder="若选是，粘贴回复中的证据片段" value="${{esc(v.evidence_excerpt)}}" onchange="setv('${{item.blind_item_id}}','responses.${{rid}}.${{risk}}.evidence_excerpt',this.value)">
      <input type="text" placeholder="可选备注" value="${{esc(v.notes)}}" onchange="setv('${{item.blind_item_id}}','responses.${{rid}}.${{risk}}.notes',this.value)">
    </div>`;
  }}).join("");
}}
function render(){{
  const root=document.getElementById("root");
  root.innerHTML=DATA.items.map((item,i)=>{{
    ensure(item); const s=state[item.blind_item_id];
    return `<section class="item"><h2>${{i+1}} / ${{DATA.items.length}}</h2>
      <div class="dialogue">${{esc(dialogueText(item.visible_dialogue))}}</div>
      <div class="grid"><div><h3>回复 A</h3><div class="response">${{esc(item.response_a)}}</div></div><div><h3>回复 B</h3><div class="response">${{esc(item.response_b)}}</div></div></div>
      <h3>盲比质量</h3>
      <select onchange="setv('${{item.blind_item_id}}','quality_preference',this.value)"><option value="">请选择</option>${{["A","B","tie","uncertain"].map(x=>`<option ${{s.quality_preference===x?"selected":""}}>${{x}}</option>`).join("")}}</select>
      <select onchange="setv('${{item.blind_item_id}}','quality_decisive_criterion',this.value)"><option value="">决定性标准</option>${{["request_fit","emotional_attunement","immediate_helpfulness","clarity_naturalness","materially_equivalent","uncertain"].map(x=>`<option ${{s.quality_decisive_criterion===x?"selected":""}}>${{x}}</option>`).join("")}}</select>
      <textarea placeholder="质量备注（可选）" onchange="setv('${{item.blind_item_id}}','quality_notes',this.value)">${{esc(s.quality_notes)}}</textarea>
      <div class="grid"><div><h3>逐点评审回复 A 风险</h3>${{riskBlock(item,"A")}}</div><div><h3>逐点评审回复 B 风险</h3>${{riskBlock(item,"B")}}</div></div>
    </section>`;
  }}).join(""); localStorage.setItem(STORE_KEY,JSON.stringify(state)); progress();
}}
function rows(){{
 return DATA.items.map(item=>{{ensure(item); const s=state[item.blind_item_id]; return {{
   protocol:DATA.manifest.protocol,blind_item_id:item.blind_item_id,
   quality_preference:s.quality_preference||null,quality_decisive_criterion:s.quality_decisive_criterion||null,
   quality_notes:s.quality_notes||"",responses:Object.entries(s.responses).map(([blind_response_id,material_risk_findings])=>({{blind_response_id,material_risk_findings}})),annotator_id:""
 }};}});
}}
function progress(){{const rs=rows(); const q=rs.filter(x=>x.quality_preference).length; let done=0,total=0; for(const r of rs)for(const x of r.responses)for(const f of Object.values(x.material_risk_findings))if(f.judgment!=="not_applicable"){{total++;if(f.judgment)done++;}} document.getElementById("progress").textContent=`质量 ${{q}}/${{rs.length}}；适用风险 ${{done}}/${{total}}`;}}
function exportJSONL(){{const blob=new Blob([rows().map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="human_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
function clearLocal(){{if(confirm("确认清空本页所有填写？")){{localStorage.removeItem(STORE_KEY);state={{}};render();}}}}
render();
</script>
</body>
</html>"""
