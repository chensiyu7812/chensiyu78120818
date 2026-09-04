# Report source and visualization notes

- Audience: technical.
- Delivery: one self-contained portable HTML report.
- Question: whether the 24-item repair delta can close H-Eligibility and what it may be used for.
- Decision-useful answer: freeze the 128-row Eligibility reviewer-proxy reference; do not use it as Step 1 `worth_opening` gold.
- Denominators: delta agreement uses 24 changed decision surfaces; final shape uses 128 states, 32 per component.
- Comparison basis: field-level pre-adjudication dual-review agreement and exact decision-surface lineage.
- Chart map: section `完美的最终标签一致性仍掩盖了门级错误`; analytical question `which review field contains disagreement`; family/type `comparison / bar`; fields `field, raw_agreement`; takeaway `derived eligibility is 100% agreed while specific increment is only 83.3%`; palette `single-root preferred`; destination `report.html`.
- Table rationale: component label shape and lineage are exact audit lookups; tables preserve counts and semantics better than additional charts.
- Limitation: browser QA was structural-only because no compatible installed Chromium was available; payload equality, runtime roots, semantic fallback, one chart and three tables passed packaging verification.
