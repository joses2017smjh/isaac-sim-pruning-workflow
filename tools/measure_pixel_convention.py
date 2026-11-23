"""How large the tracker's half-pixel convention mismatch is, measured on the registered runs.

Two conventions are in the code:

* The jaw self-mask and the depth-aware check treat pixel index u as the ray through continuous
  image coordinate u + 0.5 (``PIXEL_CENTRE = 0.5`` in ``perception/jaw_self_mask.py`` and
  ``perception/depth_appearance.py``). The renderer's camera agrees: its principal point is the
  image centre (240, 160) of a 480 x 320 image, and the mask-fidelity checks built on this
  convention found no depth violation on any recorded frame (the hold sweep's S6, every frame of
  30 runs).
* The point tracker uses OpenCV coordinates, where index u is centred at u, and back-projects its
  tracked pixel through K with no offset (``project_depth_to_world`` in ``perception/visual_servo.py``).

So the tracker's 3D target sits on a ray half a pixel from the one the image content came from, in
each image axis. At depth z and focal length f that is at most 0.5 z / f per axis, and 0.5 sqrt(2) z / f
in all. This tool reports that bound over every tracking frame of the 129 registered runs, so the
mismatch can be weighed against the controller's limits (8 mm mouth tolerance, 4 mm step). It
changes nothing: making the tracker use u + 0.5 alters every baseline replay, so it needs a
labelled arm.

CPU only. Reads recorded files; writes one new JSON file outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
REGISTERED_RUNS_EVIDENCE = ROOT / "docs/evidence/depth_regression_replay_2026-10-01.json"
MASK_FIDELITY_EVIDENCE = ROOT / "docs/evidence/hold_sweep_verdicts_2026-11-22.json"
PERCENTILES = (0, 5, 50, 95, 100)


def percentile(sorted_values, q):
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q / 100.0
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def half_pixel_offsets(depths_m, focal_px):
    """Per-axis and total 3D offset (mm) of a half-pixel ray shift at each depth."""
    per_axis = sorted(500.0 * z / focal_px for z in depths_m)
    total = sorted(500.0 * math.sqrt(2.0) * z / focal_px for z in depths_m)
    return per_axis, total


def run_depths(run):
    """The tracker's accepted depth on every frame it was tracking, and the run's wrist intrinsics."""
    report = json.loads((run / "report.json").read_text())
    matrix = report["camera"]["wrist_intrinsics"]
    frames = json.loads((run / "frames.json").read_text())["frames"]
    depths = []
    for frame in frames:
        measurement = (frame.get("live_vision") or {}).get("measurement") or {}
        if measurement.get("state") == "tracking" and isinstance(measurement.get("depth_m"), (int, float)):
            depths.append(float(measurement["depth_m"]))
    return matrix, depths


def mask_fidelity():
    """What the hold sweep's S6 found with the u + 0.5 convention, read from its committed verdicts."""
    verdicts = json.loads(MASK_FIDELITY_EVIDENCE.read_text())
    s6 = next(p for p in verdicts["predictions"] if p["id"] == "S6")
    counts, depth = s6["clauses"][0]["observed"], s6["clauses"][1]["observed"]["runs"]
    return {
        "source": str(MASK_FIDELITY_EVIDENCE.relative_to(ROOT)),
        "verdict": s6["verdict"],
        "runs": len(counts),
        "frames": sum(run["frames_checked"] for run in counts.values()),
        "count_mismatches": sum(len(run["count_mismatches"]) for run in counts.values()),
        "depth_violations": sum(run["violations"] for run in depth.values()),
    }


def build(base=VISION_ROBUSTNESS, evidence=REGISTERED_RUNS_EVIDENCE):
    names = [row["run"] for row in json.loads(Path(evidence).read_text())["runs"]]
    focal, principal, depths, runs_with_tracking = set(), set(), [], 0
    for name in names:
        matrix, run_depths_m = run_depths(Path(base) / name)
        focal.add((matrix[0][0], matrix[1][1]))
        principal.add((matrix[0][2], matrix[1][2]))
        depths.extend(run_depths_m)
        runs_with_tracking += bool(run_depths_m)
    if len(focal) != 1 or len({fx for fx, fy in focal} | {fy for fx, fy in focal}) != 1:
        raise ValueError(f"the runs do not share one square-pixel focal length: {sorted(focal)}")
    focal_px = next(iter(focal))[0]
    per_axis, total = half_pixel_offsets(depths, focal_px)
    ordered = sorted(depths)
    return {
        "schema_version": 1,
        "tool": "tools/measure_pixel_convention.py",
        "scope": (
            "A bound on the tracker's half-pixel convention mismatch over every tracking frame of the 129 registered "
            "runs. Simulation with exact depth. Nothing is changed or replayed."
        ),
        "runs": len(names),
        "runs_with_tracking_frames": runs_with_tracking,
        "tracking_frames": len(depths),
        "focal_px": focal_px,
        "principal_points_px": sorted(principal),
        "depth_m": {f"p{q}": percentile(ordered, q) for q in PERCENTILES},
        "offset_per_axis_mm": {f"p{q}": percentile(per_axis, q) for q in PERCENTILES},
        "offset_total_mm": {f"p{q}": percentile(total, q) for q in PERCENTILES},
        "mask_fidelity_with_u_plus_half": mask_fidelity(),
        "conventions": {
            "jaw_self_mask": "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py PIXEL_CENTRE = 0.5",
            "depth_appearance": "source/isaaclab_pruning/isaaclab_pruning/perception/depth_appearance.py "
            "PIXEL_CENTRE = 0.5",
            "tracker": "source/isaaclab_pruning/isaaclab_pruning/perception/visual_servo.py project_depth_to_world "
            "back-projects the OpenCV pixel with no offset",
            "seed": "hpc/inner/render_pruning_workflow.py seeds the tracker with the continuous projection K X / Z",
        },
        "reading": (
            "The offsets are upper bounds of a fixed half-pixel ray shift, not measured tracking errors. Changing the "
            "tracker to u + 0.5 would change every baseline replay, so it belongs in a labelled, registered arm."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    document = build()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document["provenance"] = {
        "code_revision": head.stdout.strip(),
        "tracked_tree_clean": dirty.stdout.strip() == "",
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
    }
    output.write_text(json.dumps(document, indent=1) + "\n")
    print(
        f"{document['tracking_frames']} tracking frames in {document['runs_with_tracking_frames']} runs; "
        f"depth p5-p95 {document['depth_m']['p5']:.3f}-{document['depth_m']['p95']:.3f} m; "
        f"half-pixel offset per axis p5-p95 {document['offset_per_axis_mm']['p5']:.2f}-"
        f"{document['offset_per_axis_mm']['p95']:.2f} mm (max {document['offset_per_axis_mm']['p100']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
