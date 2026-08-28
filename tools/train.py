#!/usr/bin/env python3
"""Train a pruning variant. Requires Isaac Lab; refuses to start without baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.policies.protocol import assert_ready_for_policy_claim, load_training_protocol  # noqa: E402
from isaaclab_pruning.sim.pruning_env import require_isaaclab  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="B_tof")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scripted-tof-complete", action="store_true")
    parser.add_argument("--curobo-oracle-complete", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_training_protocol()
    if args.variant not in {variant.value for variant in protocol.variants}:
        raise SystemExit(f"Unknown variant {args.variant}.")
    if args.seed not in protocol.seeds:
        raise SystemExit(f"Seed {args.seed} is outside the five-seed protocol {protocol.seeds}.")
    assert_ready_for_policy_claim(
        {
            "scripted_tof": args.scripted_tof_complete,
            "curobo_oracle": args.curobo_oracle_complete,
        }
    )
    require_isaaclab()
    raise SystemExit(
        "Gate 0 and URDF import passed. Next gate is hpc/slurm/env_smoke.sbatch "
        "(one v60 slot: trainer import, A–D obs asserts, step, PhysX). PPO stays "
        "blocked until both baselines have Isaac job logs. See docs/ISAAC_STACK.md."
    )


if __name__ == "__main__":
    raise SystemExit(main())
