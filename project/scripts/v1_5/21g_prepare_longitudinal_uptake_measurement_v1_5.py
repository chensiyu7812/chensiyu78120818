from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.io import read_json, write_json
from metacom_pm.pm_v1_5_semantic import semantic_encoder_spec_from_config
from metacom_pm.v1_5_response_mechanism_uptake import (
    uptake_measurement_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-config", type=Path, required=True)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.pm_config)
    pilot_contract = read_json(args.pilot_contract)
    encoder_spec = semantic_encoder_spec_from_config(config)
    contract = uptake_measurement_contract(
        pilot_contract_sha256=str(pilot_contract["contract_sha256"]),
        semantic_encoder_spec_sha256=encoder_spec.digest(),
    )
    write_json(args.out, contract)
    print(contract)


if __name__ == "__main__":
    main()
