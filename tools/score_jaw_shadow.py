#!/usr/bin/env python3
"""Score the jaw-shadow counterfactual (protocol of September 30) against J1-J5.

Outcomes come from the per-batch evidence that ``aggregate_eval.py`` wrote (the
grader's classes, never re-graded here). Each run's own records supply what a
clause names: the no-shadow readback, the tracker's patch correlation and
reason per frame, and the patch darkening, measured with the tracker's own
sampling (13 x 13 ``cv2.getRectSubPix`` of RGB2GRAY at the recorded pixel).
Each run is labelled with its GPU model from its node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402
from score_planned_approach import gpu_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md"
EVIDENCE_DIR = "docs/evidence/jaw_shadow_2026-10-01"
#: Frames between which the jaw shadow darkens the patch in the recorded arm A runs (protocol, Measures).
DARKENING_FRAMES = {"evening": (56, 59), "morning": (72, 75)}
EVENING_CHECK_FRAMES = (57, 58, 59, 67)
MORNING_CHECK_FRAMES = (73, 74, 75)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_mean(run, index, pixel):
    image = cv2.imread(str(run / f"frames/wrist_{index:05d}.png"))
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.getRectSubPix(gray, (13, 13), (float(pixel[0]), float(pixel[1]))).mean())


def run_record(root, batch_dir, row):
    """The fields J1-J5 need for one run, from its own records."""
    run = root / batch_dir / row["run_directory"]
    report = json.loads((run / "report.json").read_text())
    frames = json.loads((run / "frames.json").read_text())["frames"]
    measurement = {f["index"]: (f.get("live_vision") or {}).get("measurement") or {} for f in frames}
    stop = next(
        (f for f in frames if ((f.get("live_vision") or {}).get("cut") or {}).get("stopped_reason")),
        None,
    )
    light = row["daylight"]
    first, last = DARKENING_FRAMES[light]
    darkening = None
    if first in measurement and last in measurement and measurement[first].get("pixel_xy"):
        before = patch_mean(run, first, measurement[first]["pixel_xy"])
        after = patch_mean(run, last, measurement[last]["pixel_xy"])
        darkening = {"frames": [first, last], "before": before, "after": after, "change": after / before - 1}
    checks = EVENING_CHECK_FRAMES if light == "evening" else MORNING_CHECK_FRAMES
    shadow = (report.get("blender_scene") or {}).get("jaw_proxy_shadow") or {}
    return {
        "batch": batch_dir.split("/")[-1],
        "target": int(row["component_first_vertex"]),
        "daylight": light,
        "arm": "B" if row.get("strategy") == "jaw_no_shadow" else "A",
        "outcome": row["outcome"],
        "checks": f"{row['checks_passed']}/{row['checks_total']}",
        "node": report.get("node"),
        "gpu_model": gpu_model(report.get("node")),
        "do_not_cast_shadows_readback": shadow.get("do_not_cast_shadows_readback"),
        "stop": None
        if stop is None
        else {
            "decision_frame": stop["index"],
            "phase": stop.get("phase"),
            "tracker_reason": ((stop.get("live_vision") or {}).get("measurement") or {}).get("reason"),
            "stopped_reason": stop["live_vision"]["cut"]["stopped_reason"],
        },
        "patch_correlation": {k: measurement.get(k, {}).get("patch_correlation") for k in checks},
        "patch_darkening": darkening,
    }


def load(root):
    records = []
    for path in sorted((root / EVIDENCE_DIR).glob("jaw-shadow-*.json")):
        document = json.loads(path.read_text())
        batch_dir = document["batches"][0]["batch_dir"]
        plan = json.loads((root / batch_dir / "plan.json").read_text())
        strategies = {
            (r["target_tree_index"], r["component_first_vertex"]): r.get("strategy", {}).get("name")
            for r in plan["runs"]
        }
        for row in document["runs"]:
            row = {**row, "strategy": strategies[(row["target_tree_index"], row["component_first_vertex"])]}
            records.append(run_record(root, batch_dir, row))
    return records


def appearance_loss(record):
    return record["stop"] is not None and record["stop"]["tracker_reason"] == "appearance_changed_or_occluded"


def score(records):
    arm_b = [r for r in records if r["arm"] == "B"]
    arm_a = [r for r in records if r["arm"] == "A"]
    eve_b = [r for r in arm_b if r["daylight"] == "evening"]
    mor_b = [r for r in arm_b if r["daylight"] == "morning"]
    j1_readback = all(r["do_not_cast_shadows_readback"] == [True, True] for r in arm_b)
    changes = {f"{r['batch']} {r['target']}": r["patch_darkening"]["change"] for r in arm_b if r["patch_darkening"]}
    # Judged on the list, not the display dict, so two runs can never share a key.
    measured = [r["patch_darkening"]["change"] for r in arm_b if r["patch_darkening"]]
    j1 = prediction(
        "J1",
        "Every arm-B run reads back doNotCastShadows on both cubes and its patch darkens by less than 15%.",
        [
            clause("readback true on both cubes in every arm-B run", j1_readback, j1_readback, refutes=not j1_readback),
            clause(
                "arm-B patch darkening under 15% over the shadow frames",
                {
                    **changes,
                    "arm_A_for_comparison": {
                        f"{r['batch']} {r['target']}": r["patch_darkening"]["change"]
                        for r in arm_a
                        if r["patch_darkening"]
                    },
                },
                len(measured) == len(arm_b) and all(abs(c) < 0.15 for c in measured),
                refutes=any(c <= -0.30 for c in measured),
            ),
        ],
    )
    eve_corr_ok = all(all(v is not None and v >= 0.85 for v in r["patch_correlation"].values()) for r in eve_b)
    j2 = prediction(
        "J2",
        "Arm B evening 14944 and 15004: no appearance loss during the approach, correlation >= 0.85 at 57-59 and 67.",
        [
            clause(
                "no arm-B evening appearance stop during the approach",
                {f"{r['batch']} {r['target']}": r["stop"] for r in eve_b},
                not any(appearance_loss(r) for r in eve_b),
                refutes=any(appearance_loss(r) and r["stop"]["phase"] != "retreat" for r in eve_b),
            ),
            clause(
                "correlation >= 0.85 at 57-59 and 67",
                {f"{r['batch']} {r['target']}": r["patch_correlation"] for r in eve_b},
                eve_corr_ok,
            ),
        ],
    )
    mor_ok = all(all(v is not None and v >= 0.9 for v in r["patch_correlation"].values()) for r in mor_b)
    j3 = prediction(
        "J3",
        "Arm B morning 15004 completes closure without an appearance loss, correlation >= 0.9 at 73-75.",
        [
            clause(
                "no arm-B morning appearance loss during closure, correlation >= 0.9 at 73-75",
                {f"{r['batch']} {r['target']}": {"stop": r["stop"], "corr": r["patch_correlation"]} for r in mor_b},
                mor_ok and not any(appearance_loss(r) for r in mor_b),
                refutes=any(appearance_loss(r) for r in mor_b),
            )
        ],
    )
    a14944 = [r for r in arm_a if r["daylight"] == "evening" and r["target"] == 14944]
    j4_hold = all((r["patch_correlation"].get(67) or 1.0) < 0.5 and appearance_loss(r) for r in a14944)
    j4 = prediction(
        "J4",
        "Arm A evening 14944 loses appearance at the last approach frame (correlation < 0.5 at 67) in 2 of 2.",
        [
            clause(
                "arm A evening 14944 loses appearance with correlation < 0.5 at 67",
                {f"{r['batch']}": {"stop": r["stop"], "corr_67": r["patch_correlation"].get(67)} for r in a14944},
                j4_hold,
                refutes=all(r["outcome"] == "pass" for r in a14944),
            ),
            clause(
                "measurements (not predicted): arm A evening 15004 and morning 15004",
                {
                    f"{r['batch']} {r['target']}": {
                        "outcome": r["outcome"],
                        "stop": r["stop"],
                        "corr": r["patch_correlation"],
                    }
                    for r in arm_a
                    if r["target"] == 15004
                },
                True,
            ),
        ],
    )
    passes = sum(1 for r in eve_b if r["outcome"] == "pass")
    j5 = prediction(
        "J5",
        "Arm B evening 14944 and 15004 pass 17 of 17 in at least 3 of their 4 runs.",
        [clause("arm-B evening passes", f"{passes} of {len(eve_b)}", passes >= 3, refutes=passes < 2)],
    )
    return [j1, j2, j3, j4, j5]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    records = load(args.root)
    head = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain", "--untracked-files=no"])
    )
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the jaw-shadow counterfactual (J1-J5): 12 simulator runs, arm A unchanged and arm B with "
            "the visual jaw surrogate casting no shadow (still visible in RGB and depth). Simulator renders; the "
            "surrogate is not the pruner CAD; removing the shadow isolates a cause and is not a remedy."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "inputs_sha256": {
            str(p.relative_to(args.root)): sha256(p) for p in sorted((args.root / EVIDENCE_DIR).glob("*.json"))
        },
        "runs": records,
        "predictions": score(records),
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
