#!/usr/bin/env python3
"""Fit CuRobo link spheres from pybullet-tree-sim collision STLs (BSD-3-Clause)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.baselines.curobo import (  # noqa: E402
    default_spheres_path,
    fit_spheres_from_stls,
    spheres_payload,
    ur5e_pruner_oracle_status,
)
from isaaclab_pruning.robot import imported_usd_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    destination = args.json_out or default_spheres_path()
    payload = spheres_payload(fit_spheres_from_stls())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    usd = imported_usd_path()
    status = ur5e_pruner_oracle_status(
        urdf_usd_path=str(usd) if usd.is_file() else None,
        spheres_path=destination,
    )
    payload["status"] = {"configured": status.configured, "reason": status.reason}
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["status"], indent=2))
    print("links", sorted(payload["spheres"]))
    return 0 if status.configured else 1


if __name__ == "__main__":
    raise SystemExit(main())
