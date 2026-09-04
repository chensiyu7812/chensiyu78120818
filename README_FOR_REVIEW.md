# MetaCom V3.3 GitHub/GPT Review Package 2026-06-28

> **Current PM-v1.5 review:** start with
> [`README_PM_V1_5_REVIEW_ZH.md`](README_PM_V1_5_REVIEW_ZH.md). The historical
> `pm-v1.5-supplemental` label meant a no-retraining PM-v1 diagnostic; the
> current conference-track PM-v1.5 is a separate retrained fast-track design.
> Do not mix their claims or results.

This package is for external code/science audit before final ESConv/EvoEmo evaluation. It contains current V3.3 code, tests, docs, internal development data/labels, PM CV checkpoints, validation selections, confirmatory-ready ESConv/EvoEmo inputs, generated ESConv action outcomes, fixed EvoEmo seeker tracks, and audit reports.

Important boundary:
- Full judging and M2b labels are complete and attested.
- Internal development selection/CI results are included.
- ESConv response generation and GPT-4o final pairwise judging are complete for the strategy-only comparison (`M0+RS` vs `M0+R0`).
- EvoEmo fixed seeker tracks are complete, but EvoEmo policy generation / final judging has not been run yet.
- Raw API call logs are intentionally excluded. `project/outputs/esconv_sweep/artifact_attestation.json` is regenerated as a package-local attestation so the GitHub package can be verified after relocation; it binds the same action outcome hashes and marks raw calls as excluded.
- Do not interpret this package as final paper evidence or `CONFIRMATORY_READY`.

Recommended audit entry points:
0. `snap_reports/METACOM_V33_EXTERNAL_EVALUATION_PLAN_FOR_GPT55_ZH.md`
0b. `snap_reports/METACOM_V33_ESCONV_STRATEGY_EVAL_RESULT_ZH.md`
1. `snap_reports/METACOM_V33_GITHUB_REVIEW_FIX_REPORT_20260628_ZH.md`
2. `snap_reports/METACOM_V33_PRE_GITHUB_AUDIT_20260628_ZH.md`
3. `project/docs/CLAIM_BOUNDARIES_CN.md`
4. `project/docs/RUNBOOK_CN.md`
5. `project/scripts/14a_eval_esconv_strategy_only.py`
6. `project/src/metacom_pm/training.py`, `selection.py`, `evoemo.py`, `esconv.py`, `evo_metrics.py`
7. `project/tests/`

Verification commands from `project/`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m pytest -q
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/99_release_preflight.py --skip-tests --out release_preflight_review.json
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/14a_eval_esconv_strategy_only.py --dry-run --overwrite
```
