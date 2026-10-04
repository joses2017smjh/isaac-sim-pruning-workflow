#!/usr/bin/env python3
"""Check the closure-hold generalization P0 gate (protocol of October 4) before any GPU submission.

The test runs the jaw self-mask and the closure hold, unchanged, on new targets with new planned orientations. P0
shows that the code the runs will use is the code that produced the jaw-in-view result, and that the registers
hold exactly what the planner selected:

A. The controller with every arm off reproduces the 141 recordings of the published held-out and regression
   replays through each recorded stop.
B. The controller with the jaw self-mask and the closure hold reproduces the 6 live jaw-in-view recordings
   through each recorded stop, holds included.
C. Each registered target's planned orientation and geometry equal the committed search and pool evidence, its
   planner variant meets (a)-(c) at the strategy's standoff with no shortening, and the selected set is exactly
   the qualifying targets with no checked jaw-free plan.
D. The two strategies are planned_pose_jaw_hold under their own names, with only 18143's standoff differing.

"Equal" in A and B uses the depth closed-loop C0 rule (discrete fields exact, floats within 1e-6). Any mismatch
blocks submission. Offline replay of recorded images and register arithmetic only; nothing here grades a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/isaaclab_pruning"))
import check_agreement_c0 as agreement_c0  # noqa: E402 - the committed agreement gate's live-recording checks
import check_depth_loop_c0 as depth_c0  # noqa: E402 - the committed depth closed-loop gate's comparison rules
import queue_vision_robustness as launcher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_HOLD_GENERALIZATION_2026-10-04.md"
EVIDENCE = ROOT / "docs/evidence"
SEARCH = "search_known_map_planner_2026-10-04.json"
POOLS = {0: "screened_pool_tree0_2026-10-04.json", 1: "screened_pool_tree1_2026-10-04.json"}
#: register file -> (strategy, standoff the planner variant must have)
REGISTERS = {
    "eval_targets_hold_gen_a_2026-10-04.json": ("planned_pose_jaw_hold_gen", 0.06),
    "eval_targets_hold_gen_18143_2026-10-04.json": ("planned_pose_jaw_hold_gen_s100", 0.10),
}
SELECTED = {"tree0_3721", "tree1_18143", "tree1_36196"}
GEOMETRY_KEYS = ("target_tree_index", "component_first_vertex", "max_radius_m", "length_m", "axis", "source_center_m")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jaw_free(unit):
    """A checked variant with a clear path on which the open jaw keeps the patch and closure does not lose it."""
    return any(
        v["conditions"]["a_clear_path"]
        and v["conditions"]["b_approach_keeps_patch"]
        and not v["conditions"]["c_closure_loses_patch"]
        for v in unit.get("variants", [])
    )


def check_registers(search, pools):
    """C: the registers against the search and pool evidence."""
    problems, detail = [], {}
    candidates = {c["target"]: c for c in search["candidates"]}
    needs_hold = sorted(key for key in candidates if not jaw_free(search["units"][key]))
    if set(needs_hold) != SELECTED:
        problems.append({"kind": "selected_set", "needs_hold": needs_hold, "registered": sorted(SELECTED)})
    registered = set()
    for name, (strategy, standoff) in REGISTERS.items():
        document = json.loads((EVIDENCE / name).read_text())
        for source in document["derived_from"]:
            if sha256(ROOT / source["path"]) != source["sha256"]:
                problems.append({"kind": "derived_from_hash", "register": name, "path": source["path"]})
        for target in document["targets"]:
            key = f"tree{target['target_tree_index']}_{target['component_first_vertex']}"
            registered.add(key)
            candidate = candidates.get(key)
            pool_entry = pools[int(target["target_tree_index"])].get(int(target["component_first_vertex"]))
            row = {"register": name, "strategy": strategy}
            if candidate is None or pool_entry is None:
                problems.append({"kind": "unknown_target", **row, "target": key})
                continue
            if any(target[k] != pool_entry[k] for k in GEOMETRY_KEYS):
                problems.append({"kind": "geometry", **row, "target": key})
            if target.get("planned_final_tool_quat_wxyz") != [float(q) for q in candidate["quat_wxyz"]]:
                problems.append({"kind": "planned_orientation", **row, "target": key})
            if candidate["standoff_m"] != standoff or candidate["short_m"] != 0.0:
                problems.append({"kind": "variant", **row, "target": key, "standoff_m": candidate["standoff_m"]})
            if not candidate["conditions"]["qualifies"]:
                problems.append({"kind": "not_qualifying", **row, "target": key})
            detail[key] = {**row, "total_rotation_deg": candidate["total_rotation_deg"], **candidate["conditions"]}
    if registered != SELECTED:
        problems.append({"kind": "registered_set", "registered": sorted(registered)})
    return problems, detail


def check_strategies():
    """D: the two strategies are the jaw-in-view arm, differing at most in 18143's standoff."""
    problems = []
    reference = {k: v for k, v in launcher.STRATEGIES["planned_pose_jaw_hold"].items() if k != "name"}
    for strategy, standoff in REGISTERS.values():
        mine = {k: v for k, v in launcher.STRATEGIES[strategy].items() if k != "name"}
        if mine != {**reference, "standoff_m": standoff}:
            problems.append({"kind": "strategy", "strategy": strategy})
        if launcher.STRATEGY_PROTOCOLS.get(strategy) != PROTOCOL:
            problems.append({"kind": "protocol", "strategy": strategy})
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    import cv2

    cv2.setNumThreads(1)
    search = json.loads((EVIDENCE / SEARCH).read_text())
    pools = {
        tree: {e["component_first_vertex"]: e for e in json.loads((EVIDENCE / name).read_text())["targets"]}
        for tree, name in POOLS.items()
    }
    problems = {key: [] for key in "ABCD"}
    detail = {key: {} for key in "ABCD"}
    problems["C"], detail["C"] = check_registers(search, pools)
    problems["D"] = check_strategies()
    for selection, relative in depth_c0.PUBLISHED.items():
        document = json.loads((ROOT / relative).read_text())
        if document["selection"] != selection or len(document["runs"]) != depth_c0.EXPECTED_RUNS[selection]:
            parser.error(f"{relative}: unexpected selection or run count")
        for run in document["runs"]:
            name = run["run"]
            off = depth_c0.check_controller_off(depth_c0.depth_tool.VISION_ROBUSTNESS / name)
            detail["A"][name] = off
            if not (off["reproduces"] and off["tracker_config_matches"]):
                problems["A"].append({"run": name, **off})
            print(f"A {name}: {'ok' if off['reproduces'] else 'MISMATCH'}", flush=True)
    jaw_hold = agreement_c0.live_runs(agreement_c0.JAW_HOLD_BATCHES)
    if len(jaw_hold) != agreement_c0.EXPECTED_LIVE_RUNS["jaw_hold"]:
        parser.error(f"expected 6 jaw-in-view recordings, found {len(jaw_hold)}")
    for path in jaw_hold:
        name = f"{path.parent.name}/{path.name}"
        result = agreement_c0.check_live_reproduced(path, jaw_self_mask=True, closure_hold=True)
        detail["B"][name] = result
        if not (result["reproduces"] and result["tracker_config_matches"]):
            problems["B"].append({"run": name, **result})
        print(f"B {name}: {'ok' if result['reproduces'] else 'MISMATCH'} held={result['held_frames']}", flush=True)
    statements = {
        "A": "The controller with every arm off reproduces the 141 published recordings through each recorded stop.",
        "B": "The controller with the jaw self-mask and the closure hold reproduces the 6 live jaw-in-view "
        "recordings through each recorded stop, holds included.",
        "C": "Each registered target equals the search and pool evidence, its variant meets (a)-(c) at the "
        "strategy's standoff, and the registered set is exactly the qualifying targets with no checked jaw-free plan.",
        "D": "The two strategies are planned_pose_jaw_hold under their own names and protocol; only 18143's standoff "
        "differs.",
    }
    checks = [
        {
            "id": k,
            "statement": statements[k],
            "passed": not problems[k],
            "problems": problems[k][:100],
            "detail": detail[k],
        }
        for k in "ABCD"
    ]
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    inputs = {relative: sha256(ROOT / relative) for relative in depth_c0.PUBLISHED.values()}
    for name in (SEARCH, *POOLS.values(), *REGISTERS):
        inputs[f"docs/evidence/{name}"] = sha256(EVIDENCE / name)
    for path in jaw_hold:
        for name in ("frames.json", "report.json"):
            inputs[f"{path.parent.name}/{path.name}/{name}"] = sha256(path / name)
    document = {
        "schema_version": 1,
        "scope": (
            "P0 pre-submission gate of the closure-hold generalization test: offline CPU replays of 141 published "
            "recordings and the 6 live jaw-in-view recordings through the repository controller, and the registers "
            "against the committed planner evidence. Recorded images and arithmetic only; no live outcome or grade."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": bool(dirty),
        "inputs_sha256": inputs,
        "float_tolerance": depth_c0.FLOAT_TOLERANCE,
        "max_float_difference_accepted": depth_c0.COMPARE.max_float_difference,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }
    serialized = json.dumps(depth_c0.normalized(document), indent=1, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({c["id"]: c["passed"] for c in checks} | {"passed": document["passed"]}, indent=1))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
