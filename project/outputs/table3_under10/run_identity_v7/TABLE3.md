# Table III — frozen six-arm rerun

Main judge: gpt-4.1-mini-2025-04-14; n=204 per arm. Rates are fractions.

| arm | overall | misuse_clear | misuse_severity | omission_clear | memory_tokens_est | source_invocations |
| --- | --- | --- | --- | --- | --- | --- |
| pm_off | 3.4461 | 0.0000 | 0.0098 | 0.0343 | 683.2549 | 2.7157 |
| pm_on | 3.2647 | 0.0000 | 0.0245 | 0.0441 | 71.4510 | 2.7157 |
| rule_off | 3.4363 | 0.0000 | 0.0147 | 0.0490 | 1024.5539 | 3.0000 |
| rule_on | 3.2696 | 0.0000 | 0.0147 | 0.0196 | 91.5196 | 3.0000 |
| fixed_off | 3.4804 | 0.0000 | 0.0049 | 0.0245 | 1037.8186 | 4.0000 |
| fixed_on | 3.2843 | 0.0000 | 0.0294 | 0.0294 | 92.9902 | 4.0000 |

All planned arms and states retained. Clear misuse/omission means severity ≥2; any misuse (≥1) is also exported in CSV.
Paired confidence intervals use a state-weighted mean and cluster resampling (user primary; scenario sensitivity).
This run changes the main judge and does not claim identical historical Table III numbers.
