# Report source and visualization notes

- Audience: technical.
- Delivery: one portable HTML report.
- Decision: whether any frozen Observation candidate can be fitted and whether H-Step2 may start.
- Grain: 128 state-component Rank-1 candidates; four factor labels; 64 counterfactual groups.
- Chart map: section `BGE-small 提高了局部识别，但未证明跨语义族可用`; comparison/grouped bar; x=factor, y=grouped OOF balanced accuracy, color=candidate; supports the claim that BGE improves some aggregate factors while no candidate passes every gate; relaxed multi-category palette with direct factor labels and candidate legend.
- Exact audit detail remains in two tables because qualification depends on BA, recall, specificity, counterfactual direction and leave-family-out jointly.
- The report contains no external outcome, ESConv test/validation, or EvoEmo p7-p18 data.
