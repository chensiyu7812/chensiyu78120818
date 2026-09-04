from __future__ import annotations

from dataclasses import dataclass

from .io import canonical_json, sha256_text
from .pm_v2_contracts import PMV2State, PolicyDecision
from .pm_v2_model import PMV2Model


@dataclass
class FixedActionPMV2Model(PMV2Model):
    """Serializable fixed-action adapter using the identical PM-v2 runtime path.

    It exists only to guarantee that learned PM-v2 and fixed baselines share state
    construction, retrieval, prompt assembly, generator, logging, and fixed tracks.
    """

    fixed_action: str = "M0+R0"
    baseline_name: str = "fixed_action"
    source_checkpoint_sha256: str = ""

    def choose(self, state: PMV2State) -> PolicyDecision:
        if self.fixed_action not in state.allowed_actions:
            raise RuntimeError(
                f"fixed action {self.fixed_action} is illegal for state {state.state_id}"
            )
        config_hash = sha256_text(
            canonical_json(
                {
                    "selection_config_hash": self.selection_config.digest(),
                    "fixed_action": self.fixed_action,
                    "baseline_name": self.baseline_name,
                    "source_checkpoint_sha256": self.source_checkpoint_sha256,
                }
            )
        )
        return PolicyDecision(
            state_id=state.state_id,
            chosen_action=self.fixed_action,
            # A fixed policy neither observes Step-0 nor needs counterfactual model
            # predictions.  Keeping this empty also prevents diagnostic-only model
            # calls from silently consuming the directory representation.
            predictions={},
            semantic_ood_score=0.0,
            metadata_ood_score=0.0,
            ood_fallback_used=False,
            decision_reason=(
                f"frozen fixed-action baseline {self.baseline_name}: "
                f"always execute {self.fixed_action}"
            ),
            config_hash=config_hash,
        )
