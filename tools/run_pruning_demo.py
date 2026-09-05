#!/usr/bin/env python3
"""Generate the portable pruning demo: GIF, poster, HTML replay, and measured JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/isaaclab_pruning"))

from isaaclab_pruning.demo.render import write_outputs
from isaaclab_pruning.demo.simulation import DemoConfig, run_demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo-output"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    report = run_demo(DemoConfig(seed=args.seed))
    paths = write_outputs(report, args.output_dir)
    print("Analytic CPU simulation; ideal tool motion; no learned policy or arm dynamics.")
    for episode in report["episodes"]:
        metrics = episode["metrics"]
        print(
            f"{episode['scenario']:18s} {metrics['outcome']:24s} {metrics['steps']:3d} frames "
            f"final distance {metrics['final_target_distance_m'] * 1000:.1f} mm"
        )
    for key, path in paths.items():
        print(f"{key}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
