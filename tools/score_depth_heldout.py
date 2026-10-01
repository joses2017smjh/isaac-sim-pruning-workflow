#!/usr/bin/env python3
"""Score the depth-aware appearance check's held-out replay (protocol of October 1) against H1-H4.

Inputs are the two outputs of ``tools/replay_depth_appearance.py``: ``--heldout`` (the 12 jaw-shadow
recordings) and ``--regression`` (the 129 earlier runs). Recorded facts (stop frame, stop reason, the tracker's
reason and depth reason at the stop) come from the replay's ``recorded`` block, which it read from each run's
own files. The regression run names are checked against the jaw-in-view P0 inventory of the same 129 runs.

Where the protocol leaves a detail open it is fixed here and recorded in ``interpretations``. Offline replay
of recorded images only: no outcome here is a live result or a grade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md"
P0_INVENTORY = "docs/evidence/jaw_in_view_p0_replay_2026-10-01.json"
MIN_CONFIDENCE = 0.15
APPEARANCE_LOSS = "appearance_changed_or_occluded"
HELDOUT_RUNS = 12
REGRESSION_RUNS = 129
#: The earlier low-sun appearance events (the study's three continued events), where H4 allows a divergence.
LOW_SUN_EVENTS = {
    ("tree1-listed-evening-20260926/run_01_evening_tree1_v14944_baseline", 67),
    ("tree1-listed-evening-20260926/run_02_evening_tree1_v15004_baseline", 67),
    ("tree1-listed-morning-20260926/run_02_morning_tree1_v15004_baseline", 75),
}
JAW_STOPS = tuple(f"planned-pose-gpu-r{r}-20260929/run_01_source_tree1_v19444_planned_pose" for r in (1, 2, 3))
WIRE_TARGET = "_v12142_"
INTERPRETATIONS = [
    "arm A and arm B are read from the jaw-shadow batch names (-a- unchanged scene, -b- jaw casting no shadow)",
    "an arm-A appearance stop is a recorded stop whose recorded tracker reason is appearance_changed_or_occluded; "
    "H1 judges the depth test's decision on the replayed event at that frame",
    "H2 judges every accepted event of the held-out runs: the strict arm is consistent when it continues with a "
    "strict confidence >= 0.15 or stops at the confidence gate below it; a stop at another gate (depth window, "
    "calibration, world jump) neither supports nor refutes the rule and is listed",
    "H2's agreement clause is refuted by any accepted event at which the agreement arm does not continue",
    "H2's expectation for evening 14944 (a low-confidence stop) is reported as a clause that cannot refute",
    "H3 and H4 count a divergence as any frame where an arm's replayed measurement differs from the recording "
    "(state, reason and pixel exactly, correlation within 1e-6), through each run's recorded stop",
    "H4's kept stops are the 19444 r1-r3 jaw stops, every run of the wire target 12142 with a recorded stop, and "
    "every recorded stop whose depth reason is mixed_surfaces; kept means the strict arm is not tracking at the "
    "recorded stop frame",
    "variant results count only where the base tracker reproduces the recording; a run where it does not makes "
    "its prediction unable to hold",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm(run_name):
    batch = run_name.split("/")[0]
    return "B" if "-b-" in batch else "A"


def joined_at(run, frame):
    return next((row for row in run.get("events_joined") or [] if row["frame"] == frame), None)


def stop_record(run):
    recorded = run["recorded"]
    measurement = recorded.get("stop_measurement") or {}
    return {
        "frame": recorded.get("stop_frame"),
        "stop_reason": recorded.get("stop_reason"),
        "tracker_reason": measurement.get("reason"),
        "depth_reason": measurement.get("depth_reason"),
    }


def event_summary(row):
    keys = (
        "ncc",
        "accept",
        "reasons",
        "failed_conditions",
        "not_evaluated_conditions",
        "strict_confidence",
        "agreement_confidence",
        "verified_fraction",
        "n_same_surface",
        "near_fraction",
        "median_abs_m",
        "pixel_disagreement_px",
        "strict_decision",
        "agreement_decision",
    )
    summary = {key: row.get(key) for key in keys}
    summary["jaw_touches"] = (row.get("jaw") or {}).get("touches")
    return summary


def score_h1(heldout):
    stops, rejected, missing = {}, [], []
    for run in heldout:
        if arm(run["run"]) != "A" or "error" in run:
            continue
        stop = stop_record(run)
        if stop["frame"] is None or stop["tracker_reason"] != APPEARANCE_LOSS:
            continue
        row = joined_at(run, stop["frame"])
        stops[f"{run['run']}@{stop['frame']}"] = None if row is None else event_summary(row)
        if row is None:
            missing.append(run["run"])
        elif not row["accept"]:
            rejected.append(run["run"])
    exact = all(r.get("base_reproduces_recording") for r in heldout if "error" not in r)
    return prediction(
        "H1",
        "At each of the 6 arm-A appearance-stop frames the depth test accepts (silhouette clear, no nearer pixels "
        "beyond the limit, median within 3 mm).",
        [
            clause(
                "the depth test accepts at all 6 arm-A appearance stops",
                {"events": stops, "rejected": rejected, "no_event_at_stop": missing, "base_exact": exact},
                len(stops) == 6 and not rejected and not missing and exact,
                refutes=bool(rejected),
            )
        ],
    )


def strict_consistency(row):
    """'consistent', 'inconsistent' or 'other_gate' for the strict arm at an accepted event."""
    decision = row.get("strict_decision") or {}
    confidence = row.get("strict_confidence")
    if confidence is None:
        return "other_gate"
    if decision.get("continues"):
        return "consistent" if confidence >= MIN_CONFIDENCE else "inconsistent"
    if decision.get("reason") == "low_confidence":
        return "consistent" if confidence < MIN_CONFIDENCE else "inconsistent"
    return "other_gate"


def score_h2(heldout):
    events, agreement_stops, inconsistent, other_gate = {}, [], [], []
    evening_14944 = {}
    for run in heldout:
        if "error" in run:
            continue
        for row in run.get("events_joined") or []:
            if not row["accept"]:
                continue
            key = f"{run['run']}@{row['frame']}"
            status = strict_consistency(row)
            events[key] = {**event_summary(row), "strict_rule": status}
            if status == "inconsistent":
                inconsistent.append(key)
            elif status == "other_gate":
                other_gate.append(key)
            agreement = row.get("agreement_decision")
            if agreement is not None and not agreement.get("continues"):
                agreement_stops.append(key)
            if "evening" in run["run"] and "_v14944_" in run["run"]:
                evening_14944[key] = (row.get("strict_decision") or {}).get("reason")
    return prediction(
        "H2",
        "At each accepted event D_strict continues if and only if its strict confidence is at least 0.15, and the "
        "agreement arm continues at every accepted event.",
        [
            clause(
                "D_strict's decision agrees with its own strict confidence at every accepted event",
                {"events": events, "inconsistent": inconsistent, "ended_by_another_gate": other_gate},
                bool(events) and not inconsistent and not other_gate,
                refutes=bool(inconsistent),
            ),
            clause(
                "the agreement arm continues at every accepted event",
                {"agreement_stops": agreement_stops},
                bool(events) and not agreement_stops,
                refutes=bool(agreement_stops),
            ),
            clause(
                "expected (cannot refute): evening 14944's accepted events end in a low-confidence stop",
                evening_14944,
                bool(evening_14944) and all(reason == "low_confidence" for reason in evening_14944.values()),
            ),
        ],
    )


def divergence_frames(run, name):
    return (run.get("divergence") or {}).get(name, {}).get("mismatch_frames") or []


def score_h3(heldout):
    arm_b = [run for run in heldout if arm(run["run"]) == "B"]
    table = {}
    for run in arm_b:
        if "error" in run:
            table[run["run"]] = {"error": run["error"]}
            continue
        table[run["run"]] = {
            "grey_failures": run["grey_failures"],
            "base_exact": run["base_reproduces_recording"],
            "strict_divergence": divergence_frames(run, "strict"),
            "agreement_divergence": divergence_frames(run, "agreement"),
        }
    diverged = [
        name for name, r in table.items() if "error" in r or r["strict_divergence"] or r["agreement_divergence"]
    ]
    clean = [name for name, r in table.items() if "error" not in r and not r["grey_failures"] and r["base_exact"]]
    return prediction(
        "H3",
        "In all 6 arm-B runs no grey check fails, D never acts, and the replay equals the recording on every frame.",
        [
            clause(
                "6 of 6 arm-B runs: no grey failure and no divergence in either arm",
                {"runs": table, "diverged": diverged},
                len(arm_b) == 6 and len(clean) == 6 and not diverged,
                refutes=bool(diverged),
            )
        ],
    )


def score_h4(regression, inventory):
    names = sorted(run["run"] for run in regression)
    unexpected, kept, not_kept, low_sun = [], {}, [], {}
    errors = [run["run"] for run in regression if "error" in run]
    not_exact = [run["run"] for run in regression if "error" not in run and not run["base_reproduces_recording"]]
    for run in regression:
        if "error" in run:
            continue
        for frame in divergence_frames(run, "strict"):
            if (run["run"], frame) in LOW_SUN_EVENTS:
                low_sun[f"{run['run']}@{frame}"] = (run["measurement_at_recorded_stop"].get("strict") or {}).get(
                    "reason"
                )
            else:
                unexpected.append(f"{run['run']}@{frame}")
        stop = stop_record(run)
        real = (
            run["run"] in JAW_STOPS
            or (WIRE_TARGET in run["run"] and stop["frame"] is not None)
            or stop["depth_reason"] == "mixed_surfaces"
        )
        if real and stop["frame"] is not None:
            at_stop = run["measurement_at_recorded_stop"].get("strict") or {}
            kept[f"{run['run']}@{stop['frame']}"] = {**stop, "strict_state_at_stop": at_stop.get("state")}
            if at_stop.get("state") == "tracking":
                not_kept.append(run["run"])
    missing_jaw = [name for name in JAW_STOPS if not any(key.startswith(name + "@") for key in kept)]
    pinned = names == sorted(inventory) and len(names) == REGRESSION_RUNS
    return prediction(
        "H4",
        "On the 129 earlier runs D_strict + J diverges from the recording only at the earlier low-sun appearance "
        "events, and every real occlusion or wrong-surface stop is kept.",
        [
            clause(
                "the 129 registered runs replayed, the base exact on each",
                {"runs": len(names), "equal_to_p0_inventory": pinned, "errors": errors, "base_not_exact": not_exact},
                pinned and not errors and not not_exact,
            ),
            clause(
                "D_strict diverges only at the earlier low-sun events",
                {"low_sun_divergences": low_sun, "other_divergences": unexpected},
                not unexpected,
                refutes=bool(unexpected),
            ),
            clause(
                "every jaw, wire and mixed-surfaces stop is kept",
                {"stops": kept, "continued": not_kept, "jaw_stops_missing": missing_jaw},
                bool(kept) and not not_kept and not missing_jaw,
                refutes=bool(not_kept),
            ),
        ],
    )


def agreement_regression(regression):
    """Measurement only: where the agreement arm diverges on the regression runs."""
    return {
        run["run"]: divergence_frames(run, "agreement")
        for run in regression
        if "error" not in run and divergence_frames(run, "agreement")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", type=Path, required=True, help="replay_depth_appearance.py --heldout output")
    parser.add_argument("--regression", type=Path, required=True, help="replay_depth_appearance.py --regression output")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    heldout_doc = json.loads(args.heldout.read_text())
    regression_doc = json.loads(args.regression.read_text())
    for name, doc, expected in (
        ("heldout", heldout_doc, HELDOUT_RUNS),
        ("regression", regression_doc, REGRESSION_RUNS),
    ):
        if doc.get("selection") != name or len(doc["runs"]) != expected:
            parser.error(f"--{name}: selection {doc.get('selection')!r} with {len(doc['runs'])} runs")
    p0 = json.loads((args.root / P0_INVENTORY).read_text())
    inventory = [row[0] for row in next(c for c in p0["checks"] if c["id"] == "flags_off_exact")["detail"]["per_run"]]
    heldout, regression = heldout_doc["runs"], regression_doc["runs"]
    head = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain", "--untracked-files=no"])
    )
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the depth-aware appearance check's held-out offline replay (H1-H4): D_strict + J and the "
            "agreement arm replayed on the 12 jaw-shadow recordings (held out) and the 129 earlier runs "
            "(regression). Simulator depth: RTX optical-Z ground truth, noise-free, perfectly registered, exact "
            "camera motion; not a sensor. Offline replay of recorded images, not a closed loop."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "replay_code_revision": {
            "heldout": heldout_doc.get("code_revision"),
            "regression": regression_doc.get("code_revision"),
        },
        "replay_source_status": {
            "heldout": heldout_doc.get("source_files_status_porcelain"),
            "regression": regression_doc.get("source_files_status_porcelain"),
        },
        "inputs_sha256": {
            "heldout": {"path": str(args.heldout), "sha256": sha256(args.heldout)},
            "regression": {"path": str(args.regression), "sha256": sha256(args.regression)},
            P0_INVENTORY: sha256(args.root / P0_INVENTORY),
        },
        "interpretations": INTERPRETATIONS,
        "predictions": [score_h1(heldout), score_h2(heldout), score_h3(heldout), score_h4(regression, inventory)],
        "agreement_arm_regression_divergences": agreement_regression(regression),
        "heldout_runs": {
            run["run"]: {
                "arm": arm(run["run"]),
                "recorded_stop": stop_record(run),
                "grey_failures": run.get("grey_failures"),
                "base_exact": run.get("base_reproduces_recording"),
            }
            for run in heldout
            if "error" not in run
        },
    }
    serialized = json.dumps(document, indent=1, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
