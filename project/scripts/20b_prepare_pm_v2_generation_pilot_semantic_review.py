#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review import (
    prepare_generation_pilot_semantic_review,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = (
    ROOT
    / "outputs"
    / "pm_v2_generation_compatibility_pilot_deterministic_evidence_v7"
    / "artifact_attestation.json"
)
DEFAULT_OUT = ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v7"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the no-API two-annotator semantic review packet for the "
            "PM-v2 generation compatibility pilot."
        )
    )
    parser.add_argument(
        "--generation-pilot-attestation", type=Path, default=DEFAULT_PILOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare_generation_pilot_semantic_review(
        pilot_attestation_path=args.generation_pilot_attestation,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
