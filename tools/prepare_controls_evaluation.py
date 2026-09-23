#!/usr/bin/env python3
"""Validate the controls renders and materialize the read-only evaluation plan.

Reads the frozen batch plan, checks every registered tree's render manifest,
records what is missing (a missing condition stays in the denominator), derives
the test-time photometric variants from the plain ``evening`` cell, and writes
one plan that ``depth_generalization.py`` scores unchanged. Nothing here looks
at a prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generalization_controls as gc  # noqa: E402
import photometric_normalization as pn  # noqa: E402
from depth_generalization import sha256  # noqa: E402

DEPTH_CONVENTION = "Cycles optical-Z metres verified by 2m frontoparallel plane at center and off axis"


def manifest_path(batch, plan, tree_id):
    index = [t["tree_id"] for t in plan["trees"]].index(tree_id)
    return batch / f"controls_{index}_{tree_id}" / "render_manifest.json"


def check_frame(frame):
    rgb = np.asarray(Image.open(frame["rgb"]).convert("RGB"))
    depth = np.load(frame["depth"], allow_pickle=False)
    mask = np.asarray(Image.open(frame["mask"])) > 0
    if not (rgb.shape[:2] == depth.shape == mask.shape):
        raise ValueError(f"RGB, depth and mask disagree in shape: {frame['rgb']}")
    if rgb.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError(f"Rendered size differs from the recorded camera model: {frame['rgb']}")
    if rgb.std() <= 3 or not mask.any():
        raise ValueError(f"Blank render or empty tree mask: {frame['rgb']}")
    if not np.isfinite(np.asarray(frame["K"])).all() or frame["K"][0][0] <= 0:
        raise ValueError(f"Bad intrinsics: {frame['rgb']}")
    return float(pn.luma(np.ascontiguousarray(rgb[:, :, ::-1])))


def validate_tree(data, conditions, lumas):
    """Check one manifest against the registered conditions; return frames and missing conditions."""
    frames, missing = [], []
    by_condition = {}
    for frame in data["frames"]:
        by_condition.setdefault(frame["condition"], []).append(frame)
    skipped = {s["condition"]: s["reason"] for s in data.get("skipped", [])}
    for cond in conditions:
        expected = gc.pose_ids(gc.POSE_SETS[cond["pose_set"]])
        got = by_condition.get(cond["id"], [])
        if [f["view_id"] for f in got] != expected:
            missing.append(
                {
                    "tree_id": data["tree_id"],
                    "condition": cond["id"],
                    "expected_frames": len(expected),
                    "present_frames": len(got),
                    "reason": skipped.get(cond["id"], "frames_absent_or_out_of_order"),
                }
            )
            continue
        for frame in got:
            lumas.setdefault(cond["id"], []).append(check_frame(frame))
        frames.extend(got)
    # Lighting cells share geometry with their baseline and differ in RGB bytes.
    lighting = [c for c in conditions if c["group"] == "lighting" and c["id"] in by_condition]
    for cond in lighting:
        for a, b in zip(by_condition[cond["id"]], by_condition.get("source", [])):
            if a["view_id"] == b["view_id"] and (a["depth"] != b["depth"] or a["mask"] != b["mask"]):
                raise ValueError(f"Lighting cell {cond['id']} does not share geometry with source: {a['view_id']}")
    for view in {f["view_id"] for c in lighting for f in by_condition[c["id"]]}:
        hashes = {sha256(f["rgb"]) for c in lighting for f in by_condition[c["id"]] if f["view_id"] == view}
        if len(hashes) != len(lighting):
            raise ValueError(f"Two lighting cells rendered identical bytes for view {view}")
    return frames, missing


def build(batch, output, *, variants=gc.PHOTOMETRIC_VARIANTS):
    batch, output = Path(batch), Path(output)
    plan = json.loads((batch / "plan.json").read_text())
    conditions = plan["conditions"]
    per_tree = sum(len(gc.pose_ids(gc.POSE_SETS[c["pose_set"]])) for c in conditions)
    status, frames, missing, sources, lumas = [], [], [], [], {}
    for tree in plan["trees"]:
        path = manifest_path(batch, plan, tree["tree_id"])
        present = path.is_file()
        data = json.loads(path.read_text()) if present else None
        ok = bool(data and data.get("ok"))
        status.append(
            {
                "tree_id": tree["tree_id"],
                "family": tree["family"],
                "manifest": str(path),
                "present": present,
                "render_ok": ok,
            }
        )
        if not data:
            missing.append(
                {
                    "tree_id": tree["tree_id"],
                    "condition": "*",
                    "expected_frames": per_tree,
                    "present_frames": 0,
                    "reason": "manifest_absent",
                }
            )
            continue
        tree_frames, tree_missing = validate_tree(data, conditions, lumas)
        frames.extend(tree_frames)
        missing.extend(tree_missing)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "tree_id": data["tree_id"],
                "geometry_sha256": data["geometry_sha256"],
                "render_ok": ok,
                "lighting": data.get("lighting"),
            }
        )
    (output / "render_status.json").write_text(
        json.dumps(
            {"registered": len(plan["trees"]), "with_manifest": len(sources), "trees": status, "missing": missing},
            indent=2,
        )
        + "\n"
    )
    render_plan = {"schema_version": 1, "frames": frames, "depth_convention": DEPTH_CONVENTION}
    (output / "plan_renders.json").write_text(json.dumps(render_plan, indent=2) + "\n")

    photometric = None
    if any(pn.is_plain_cell(f, "evening") for f in frames):
        photometric = pn.export(
            output / "plan_renders.json", output / "photometric", lighting="evening", variants=tuple(variants)
        )
        for variant in variants:
            variant_plan = json.loads(Path(photometric["variants"][variant]["plan"]).read_text())
            frames.extend(variant_plan["frames"])
            lumas[f"evening/{variant}"] = [p["luma_after"] for p in photometric["variants"][variant]["per_frame"]]
    else:
        missing.append(
            {
                "tree_id": "*",
                "condition": "evening/*",
                "expected_frames": 0,
                "present_frames": 0,
                "reason": "no_plain_evening_frames_to_normalize",
            }
        )

    registered_frames = per_tree * len(plan["trees"]) + len(variants) * len(gc.pose_ids(gc.POSE_SETS["far"])) * len(
        plan["trees"]
    )
    final = {
        "schema_version": 1,
        "scope": plan["scope"],
        "protocol": plan.get("protocol"),
        "batch_plan_sha256": sha256(batch / "plan.json"),
        "source_renders": sources,
        "conditions": conditions,
        "baseline_of": gc.BASELINE_OF,
        "registered_frames": registered_frames,
        "scored_frames": len(frames),
        "missing": missing,
        "luma_by_condition": {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(lumas.items())},
        "photometric_manifest": str(output / "photometric" / "manifest.json") if photometric else None,
        "frames": frames,
        "depth_convention": DEPTH_CONVENTION,
        "target_scope": (
            "Fixed geometric spur segment, no outcome-driven replacement; invisible projections are explicit failures"
        ),
        "temporal_scope": "Static paired views; cross-view changes are not temporal stability measurements",
    }
    with (output / "plan.json").open("x") as stream:
        json.dump(final, stream, indent=2)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True, help="Frozen batch directory holding plan.json")
    parser.add_argument("--output", type=Path, required=True, help="Evaluation directory; plan.json must not exist")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    final = build(args.batch, args.output)
    print(
        json.dumps(
            {
                "ok": not final["missing"],
                "trees": len(final["source_renders"]),
                "registered_frames": final["registered_frames"],
                "scored_frames": final["scored_frames"],
                "missing": len(final["missing"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
