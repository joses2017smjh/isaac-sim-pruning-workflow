#!/usr/bin/env python3
"""Check the closure-hold sweep P0 gate (protocol of October 6) before any GPU submission.

A. The controller with every arm off reproduces the 141 recordings of the published held-out and regression
   replays through each recorded stop.
B. The controller with the jaw self-mask and the closure hold reproduces the 6 live jaw-in-view recordings through
   each recorded stop, holds included.
C. The sweep register holds exactly the qualifying search targets minus the registered exclusions (the three tested
   on October 4 and the three that qualify only with a shortened plan), each with its pool geometry and its
   qualifying variant's planned orientation at a 60 mm standoff with no shortening.
D. The sweep strategy is planned_pose_jaw_hold under its own name and protocol.

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
import check_agreement_c0 as agreement_c0  # noqa: E402 - the committed live-recording checks
import check_depth_loop_c0 as depth_c0  # noqa: E402 - the committed comparison rules
import queue_vision_robustness as launcher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_HOLD_SWEEP_2026-10-06.md"
EVIDENCE = ROOT / "docs/evidence"
SEARCH = "search_known_map_planner_2026-10-04.json"
POOLS = {0: "screened_pool_tree0_2026-10-04.json", 1: "screened_pool_tree1_2026-10-04.json"}
REGISTER = "eval_targets_hold_sweep_2026-10-06.json"
STRATEGY = "planned_pose_jaw_hold_sweep"
TESTED = {"tree0_3721", "tree1_36196", "tree1_18143"}
GEOMETRY_KEYS = ("target_tree_index", "component_first_vertex", "max_radius_m", "length_m", "axis", "source_center_m")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def executable(candidate):
    return candidate["standoff_m"] == 0.06 and candidate["short_m"] == 0.0


def check_register(search, pools, document):
    """C: the sweep register against the search and pool evidence."""
    problems = []
    candidates = {c["target"]: c for c in search["candidates"]}
    expected = sorted(k for k, c in candidates.items() if k not in TESTED and executable(c))
    shortened = sorted(k for k, c in candidates.items() if k not in TESTED and not executable(c))
    if sorted(document.get("excluded", {})) != sorted(TESTED | set(shortened)):
        problems.append({"kind": "exclusions", "registered": sorted(document.get("excluded", {}))})
    registered = []
    for target in document["targets"]:
        key = f"tree{target['target_tree_index']}_{target['component_first_vertex']}"
        registered.append(key)
        candidate = candidates.get(key)
        pool_entry = pools[int(target["target_tree_index"])].get(int(target["component_first_vertex"]))
        if candidate is None or pool_entry is None:
            problems.append({"kind": "unknown_target", "target": key})
            continue
        if any(target[k] != pool_entry[k] for k in GEOMETRY_KEYS):
            problems.append({"kind": "geometry", "target": key})
        if target.get("planned_final_tool_quat_wxyz") != [float(q) for q in candidate["quat_wxyz"]]:
            problems.append({"kind": "planned_orientation", "target": key})
        if not executable(candidate) or not candidate["conditions"]["qualifies"]:
            problems.append({"kind": "variant", "target": key})
    if sorted(registered) != expected:
        problems.append({"kind": "registered_set", "missing": sorted(set(expected) - set(registered))})
    for source in document["derived_from"]:
        if sha256(ROOT / source["path"]) != source["sha256"]:
            problems.append({"kind": "derived_from_hash", "path": source["path"]})
    return problems, {"registered": len(registered), "shortened_excluded": shortened}


def check_strategy():
    """D: the sweep strategy is the jaw-in-view arm under its own protocol."""
    reference = {k: v for k, v in launcher.STRATEGIES["planned_pose_jaw_hold"].items() if k != "name"}
    mine = {k: v for k, v in launcher.STRATEGIES[STRATEGY].items() if k != "name"}
    problems = [] if mine == reference else [{"kind": "strategy"}]
    if launcher.STRATEGY_PROTOCOLS.get(STRATEGY) != PROTOCOL:
        problems.append({"kind": "protocol"})
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
    problems["C"], detail["C"] = check_register(search, pools, json.loads((EVIDENCE / REGISTER).read_text()))
    problems["D"] = check_strategy()
    for selection, relative in depth_c0.PUBLISHED.items():
        document = json.loads((ROOT / relative).read_text())
        if document["selection"] != selection or len(document["runs"]) != depth_c0.EXPECTED_RUNS[selection]:
            parser.error(f"{relative}: unexpected selection or run count")
        for run in document["runs"]:
            off = depth_c0.check_controller_off(depth_c0.depth_tool.VISION_ROBUSTNESS / run["run"])
            detail["A"][run["run"]] = off
            if not (off["reproduces"] and off["tracker_config_matches"]):
                problems["A"].append({"run": run["run"], **off})
    jaw_hold = agreement_c0.live_runs(agreement_c0.JAW_HOLD_BATCHES)
    if len(jaw_hold) != agreement_c0.EXPECTED_LIVE_RUNS["jaw_hold"]:
        parser.error(f"expected 6 jaw-in-view recordings, found {len(jaw_hold)}")
    for path in jaw_hold:
        result = agreement_c0.check_live_reproduced(path, jaw_self_mask=True, closure_hold=True)
        detail["B"][f"{path.parent.name}/{path.name}"] = result
        if not (result["reproduces"] and result["tracker_config_matches"]):
            problems["B"].append({"run": f"{path.parent.name}/{path.name}", **result})
    statements = {
        "A": "The controller with every arm off reproduces the 141 published recordings through each recorded stop.",
        "B": "The controller with the jaw self-mask and the closure hold reproduces the 6 live jaw-in-view "
        "recordings through each recorded stop, holds included.",
        "C": "The sweep register is exactly the qualifying search targets minus the registered exclusions, with pool "
        "geometry and the qualifying variant's orientation at a 60 mm standoff and no shortening.",
        "D": "The sweep strategy is planned_pose_jaw_hold under its own name and protocol.",
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
    for name in (SEARCH, *POOLS.values(), REGISTER):
        inputs[f"docs/evidence/{name}"] = sha256(EVIDENCE / name)
    document = {
        "schema_version": 1,
        "scope": "P0 pre-submission gate of the closure-hold sweep: controller replays of 141 published and 6 live "
        "jaw-in-view recordings, and the sweep register against the committed planner evidence. No live outcome.",
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": bool(dirty),
        "inputs_sha256": inputs,
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
