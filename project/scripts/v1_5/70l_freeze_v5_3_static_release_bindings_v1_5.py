#!/usr/bin/env python3
"""Freeze decided V5.3 static assets without reading outcomes or using APIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_release_bindings import (  # noqa: E402
    build_static_release_bindings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_3_static_release_bindings_v1",
    )
    args = parser.parse_args()
    binding = build_static_release_bindings(ROOT)
    write_json(args.out_dir / "release_bindings.json", binding.model_dump(mode="json"))
    print(binding.model_dump(mode="json"))


if __name__ == "__main__":
    main()
