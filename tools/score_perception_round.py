#!/usr/bin/env python3
"""Score the perception and tree1 round against the predictions registered before it ran.

Inputs are the twelve per-batch evidence files written by ``aggregate_eval.py``
(the grader's outcomes, never re-graded here), the September 24 evidence that
the protocol compares against, the two known-map prediction registers, and each
run's own records for the per-frame details a clause names: the tracker's
reason at the stop decision, the first frame without tracking, the drop, the
startup contact force and the contacting body.

Every clause carries the observed value, whether the expectation ``holds``, and
whether the protocol's own "refuted if" condition is met (``refutes``). A
prediction is refuted when any clause refutes it, supported when every clause
holds, and partly supported otherwise. Sections marked post hoc are diagnoses
made after the runs, not predictions.

Frame numbers: ``stop_frame`` is the aggregation's index of the first frame in
the ``stopped_failure`` phase; the controller's stop decision is recorded one
frame earlier (``decision_frame``). Both are 0-based ``frames.json`` indices.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_PERCEPTION_2026-09-26.md"
EVIDENCE_DIR = "docs/evidence/perception_round_2026-09-28"
REFERENCE_TREE0 = "docs/evidence/strategies_2026-09-24.json"
REFERENCE_TREE1 = "docs/evidence/tree1_listed_2026-09-24.json"
PRESENTABILITY = "docs/evidence/tree1_presentability_predictions_2026-09-26.json"
SWEPT = "docs/evidence/tree1_swept_path_predictions_2026-09-27.json"

BATCHES = {
    "similarity_tree0": "perc-similarity-tree0-20260926",
    "similarity_tree1": "perc-similarity-tree1-20260926",
    "mount_flip": "perc-mount-flip-20260926",
    "both_tree0": "perc-both-tree0-20260926",
    "both_tree1": "perc-both-tree1-20260926",
    "repeat_r1": "tree1-listed-repeat-r1-20260926",
    "repeat_r2": "tree1-listed-repeat-r2-20260926",
    "repeat_r3": "tree1-listed-repeat-r3-20260926",
    "morning": "tree1-listed-morning-20260926",
    "evening": "tree1-listed-evening-20260926",
    "seeded_ten": "tree1-seeded-ten-20260926",
    "seeded30": "tree1-seeded30-20260926",
}

LAYOUT = "rejected_layout_startup_contact"
NOT_VISIBLE = "stopped_initial_target_not_visible"
VISION = "stopped_vision_invalid"
HAZARD = "stopped_hazard_contact"
TOF = "stopped_tof_minimum_clearance"
GEOMETRY_CLASSES = (LAYOUT, HAZARD, TOF)
LOSS_REASONS = ("world_target_jump_or_wrong_surface", "optical_flow_failed", "appearance_changed_or_occluded")
FORCE = re.compile(r"startup robot contact ([0-9]+\.[0-9]+) N")

#: The swept model's classes and the recorded outcome each one names.
SWEPT_OUTCOME = {
    "home_overlap": LAYOUT,
    "not_visible": NOT_VISIBLE,
    "hazard_contact": HAZARD,
    "tof_minimum_clearance": TOF,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clause(text, observed, holds, refutes=False):
    return {"clause": text, "observed": observed, "holds": bool(holds), "refutes": bool(refutes)}


def within(value, low, high):
    return value is not None and low <= value <= high


def verdict(clauses):
    if any(c["refutes"] for c in clauses):
        return "refuted"
    return "supported" if all(c["holds"] for c in clauses) else "partly supported"


def prediction(pid, text, clauses):
    return {"id": pid, "prediction": text, "clauses": clauses, "verdict": verdict(clauses)}


def by_target(rows):
    return {int(r["component_first_vertex"]): r for r in rows}


def summary(row):
    """The fields a verdict cites for one run."""
    keys = (
        "outcome",
        "stop_frame",
        "decision_frame",
        "tracker_reason_at_stop",
        "depth_reason_at_stop",
        "first_non_tracking_frame",
        "first_non_tracking_reason",
    )
    return {k: row.get(k) for k in keys if row.get(k) is not None}


def run_details(batch_dir, row):
    """Per-frame details from one run's own records; refused runs contribute their log's force only."""
    batch_dir = Path(batch_dir)
    index = int(row["run_directory"].split("_")[1])
    details = {}
    for log in sorted((batch_dir / "logs").glob(f"*_{index}.out")):
        match = FORCE.search(log.read_text(encoding="utf-8", errors="replace"))
        if match:
            details["startup_contact_n"] = float(match.group(1))
    run = batch_dir / row["run_directory"]
    if row.get("status") != "graded" or not (run / "frames.json").is_file():
        return details
    frames = read_json(run / "frames.json")["frames"]
    for frame in frames:
        live = frame.get("live_vision") or {}
        measurement = live.get("measurement") or {}
        if measurement.get("state") != "tracking" and "first_non_tracking_frame" not in details:
            details["first_non_tracking_frame"] = int(frame["index"])
            details["first_non_tracking_reason"] = measurement.get("reason")
            details["first_non_tracking_phase"] = frame.get("phase")
        cut = live.get("cut") or {}
        if cut.get("stopped_reason") and "decision_frame" not in details:
            details["decision_frame"] = int(frame["index"])
            details["tracker_state_at_stop"] = measurement.get("state")
            details["tracker_reason_at_stop"] = measurement.get("reason")
            details["depth_reason_at_stop"] = measurement.get("depth_reason")
            details["patch_correlation_at_stop"] = measurement.get("patch_correlation")
            forces = ((frame.get("contact") or {}).get("forces_w_n") or [[]])[0]
            names = (((read_json(run / "report.json").get("contact_coverage") or {}).get("coverage")) or {}).get(
                "actual_body_names"
            ) or []
            touching = [i for i, f in enumerate(forces) if any(abs(v) > 0 for v in f)]
            if touching:
                details["contact_bodies_at_stop"] = [names[i] if i < len(names) else i for i in touching]
    phases = []
    for frame in frames:
        phase = ((frame.get("live_vision") or {}).get("cut") or {}).get("phase")
        if phase and phase not in phases:
            phases.append(phase)
    details["cut_phases_seen"] = phases
    grade = run / "sequence_grade.json"
    if grade.is_file():
        drop = (read_json(grade).get("metrics") or {}).get("maximum_post_detach_drop_m")
        if drop is not None:
            details["maximum_post_detach_drop_m"] = drop
    report = read_json(run / "report.json")
    initialization = report.get("vision_initialization") or {}
    if initialization:
        details["initial_visibility"] = (initialization.get("visibility") or {}).get("visible")
        details["initial_tracker"] = {
            k: (initialization.get("tracker") or {}).get(k) for k in ("state", "reason", "feature_count")
        }
    return details


def load_round(root, evidence_dir):
    rounds = {}
    for key, batch in BATCHES.items():
        document = read_json(root / evidence_dir / f"{batch}.json")
        rows = []
        for row in document["runs"]:
            rows.append({**row, **run_details(root / document["batches"][0]["batch_dir"], row)})
        rounds[key] = rows
    return rounds


def reference_classes(root):
    tree0 = [r for r in read_json(root / REFERENCE_TREE0)["runs"] if r.get("strategy", "baseline") == "baseline"]
    tree1 = read_json(root / REFERENCE_TREE1)["runs"]
    return {int(r["component_first_vertex"]): r["outcome"] for r in tree0 + tree1}


def class_changes(rows, reference, only_classes=None):
    changes = {}
    for vertex, row in by_target(rows).items():
        before = reference.get(vertex)
        if before is None or (only_classes and before not in only_classes):
            continue
        if row["outcome"] != before:
            changes[vertex] = {"before": before, "after": row["outcome"], **summary(row)}
    return changes


def score_p1(rows, reference):
    t = by_target(rows)
    r590, r8353, r12142 = t[590], t[8353], t[12142]
    mixed_before_align = (
        r590["outcome"] == VISION
        and r590.get("depth_reason_at_stop") == "mixed_surfaces"
        and "align" not in r590.get("cut_phases_seen", [])
    )
    kept = class_changes(rows, reference, only_classes=GEOMETRY_CLASSES)
    return prediction(
        "P1",
        "Similarity tracker, tree0: 590 no longer stops mixed_surfaces before align; 8353 and 12142 keep "
        "their vision_invalid stops; contact, ToF and layout targets keep their class.",
        [
            clause(
                "590 does not stop mixed_surfaces before align",
                {**summary(r590), "cut_phases_seen": r590.get("cut_phases_seen")},
                not mixed_before_align,
                refutes=mixed_before_align,
            ),
            clause(
                "8353 still stops vision_invalid",
                summary(r8353),
                r8353["outcome"] == VISION,
                r8353["outcome"] != VISION,
            ),
            clause(
                "8353 stops within its first 10 frames", r8353.get("stop_frame"), within(r8353.get("stop_frame"), 0, 10)
            ),
            clause(
                "12142 still stops vision_invalid",
                summary(r12142),
                r12142["outcome"] == VISION,
                r12142["outcome"] != VISION,
            ),
            clause(
                "12142 stops at frames 27-33 on a world jump, flow or appearance loss",
                summary(r12142),
                within(r12142.get("stop_frame"), 27, 33) and r12142.get("tracker_reason_at_stop") in LOSS_REASONS,
            ),
            clause("every contact, ToF and layout target keeps its class", kept, not kept),
        ],
    )


def score_p2(rows, reference):
    t = by_target(rows)
    passes = {v: t[v]["outcome"] for v in (14944, 15004)}
    others = {v: c for v, c in class_changes(rows, reference).items() if v not in passes}
    both_pass = all(o == "pass" for o in passes.values())
    return prediction(
        "P2",
        "Similarity tracker, tree1 listed: 14944 and 15004 pass; every other listed spur keeps its class.",
        [
            clause("14944 and 15004 pass", passes, both_pass, refutes=not both_pass),
            clause("every other listed spur keeps its class", others, not others),
        ],
    )


def score_p3(rows):
    t = by_target(rows)
    r19264, r8353 = t[19264], t[8353]
    refused = r19264["outcome"] == NOT_VISIBLE
    first = r8353.get("first_non_tracking_frame")
    stop = r8353.get("stop_frame")
    early_stop = stop is not None and stop <= 7
    mixed = r8353.get("depth_reason_at_stop") == "mixed_surfaces"
    return prediction(
        "P3",
        "Mount side: 19264 initializes; 8353's first non-tracking frame is later than frame 7, and a later "
        "mixed_surfaces stop would hold pixels off the spur.",
        [
            clause("19264 initializes (not refused as not visible)", summary(r19264), not refused, refutes=refused),
            clause(
                "8353's first non-tracking frame is later than frame 7",
                {"first_non_tracking_frame": first, "stop_frame": stop, "vacuous": first is None},
                first is None or first > 7,
            ),
            clause("8353 does not stop within 7 frames", summary(r8353), not early_stop, refutes=early_stop),
            clause(
                "8353 does not stop with a mixed_surfaces window (the off-spur test applies only if it does)",
                {"depth_reason_at_stop": r8353.get("depth_reason_at_stop"), "vacuous": not mixed},
                not mixed,
                refutes=False,
            ),
        ],
    )


def score_p4(rows_tree0, rows_tree1, reference):
    t0, t1 = by_target(rows_tree0), by_target(rows_tree1)
    vision = sorted(v for v, r in t0.items() if r["outcome"] == VISION)
    passes = {v: t1[v]["outcome"] for v in (14944, 15004)}
    kept = class_changes(rows_tree0, reference, only_classes=GEOMETRY_CLASSES)
    tree1_passes = sum(1 for r in rows_tree1 if r["outcome"] == "pass")
    return prediction(
        "P4",
        "Similarity tracker and mount rule together: tree0 vision_invalid falls to at most 12142; contact, ToF "
        "and layout classes unchanged; tree1 listed passes at least 2 of 7 with 14944 and 15004.",
        [
            clause(
                "590 does not stop vision_invalid",
                summary(t0[590]),
                t0[590]["outcome"] != VISION,
                t0[590]["outcome"] == VISION,
            ),
            clause(
                "8353 does not stop vision_invalid",
                summary(t0[8353]),
                t0[8353]["outcome"] != VISION,
                t0[8353]["outcome"] == VISION,
            ),
            clause(
                "14944 and 15004 pass",
                passes,
                all(o == "pass" for o in passes.values()),
                refutes=any(o != "pass" for o in passes.values()),
            ),
            clause(
                "tree0 vision_invalid targets are at most 12142",
                {"vision_invalid_targets": vision},
                vision in ([], [12142]),
            ),
            clause("contact, ToF and layout classes unchanged", kept, not kept),
            clause("tree1 listed passes at least 2 of 7", tree1_passes, tree1_passes >= 2),
        ],
    )


def score_p5(repeats, reference):
    changes = {name: class_changes(rows, reference) for name, rows in repeats.items()}
    any_change = any(changes.values())
    detail = {}
    for name, rows in repeats.items():
        t = by_target(rows)
        detail[name] = {
            "14884_failed_checks": t[14884].get("failed_checks"),
            "14884_drop_m": t[14884].get("maximum_post_detach_drop_m"),
            "19145_startup_contact_n": t[19145].get("startup_contact_n"),
            "19384_stop_frame": t[19384].get("stop_frame"),
            "19444_stop_frame": t[19444].get("stop_frame"),
        }
    drop_ok = all(
        d["14884_failed_checks"] == "piece_dropped_after_release" and within(d["14884_drop_m"], 0.0, 0.002)
        for d in detail.values()
    )
    force_ok = all(d["19145_startup_contact_n"] == 60.554 for d in detail.values())
    frames_ok = all(
        within(d["19384_stop_frame"], 44, 46) and within(d["19444_stop_frame"], 43, 45) for d in detail.values()
    )
    return prediction(
        "P5",
        "Three baseline repeats of the tree1 listed seven: every outcome class repeats; 14944 and 15004 pass 3 of 3.",
        [
            clause("no outcome class changes in any repeat", changes, not any_change, refutes=any_change),
            clause("14884 fails only the drop check, drop under 2 mm", detail, drop_ok),
            clause(
                "19145 is refused at 60.554 N", {k: v["19145_startup_contact_n"] for k, v in detail.items()}, force_ok
            ),
            clause(
                "19384 and 19444 stop within one frame of 45 and 44",
                {k: (v["19384_stop_frame"], v["19444_stop_frame"]) for k, v in detail.items()},
                frames_ok,
            ),
        ],
    )


def score_p6(light_rows, reference, source_rows):
    changes = {name: class_changes(rows, reference) for name, rows in light_rows.items()}
    source = by_target(source_rows)
    for name, change in changes.items():
        rows = by_target(light_rows[name])
        for vertex, entry in change.items():
            entry["patch_correlation_at_stop"] = rows[vertex].get("patch_correlation_at_stop")
            entry["source_repeat_r1_outcome"] = source[vertex]["outcome"]
    any_change = any(changes.values())
    return prediction(
        "P6",
        "Morning and evening light: every listed spur keeps its source-light outcome class.",
        [clause("no class change under morning or evening", changes, not any_change, refutes=any_change)],
    )


def score_p7(rows):
    t = by_target(rows)
    presented = {v: t[v]["outcome"] for v in (15599, 37023, 37796)}
    not_presented = [v for v, o in presented.items() if o in (LAYOUT, NOT_VISIBLE)]
    others = [r for v, r in t.items() if v not in presented]
    refused = sum(1 for r in others if r["outcome"] == LAYOUT)
    passes = sum(1 for r in rows if r["outcome"] == "pass")
    return prediction(
        "P7",
        "Seeded ten, first presentation: 15599, 37023 and 37796 are presented; at least 4 of the other 7 are "
        "refused at startup; at most 3 of 10 pass.",
        [
            clause("15599, 37023, 37796 presented", presented, not not_presented, refutes=bool(not_presented)),
            clause(
                "at least 4 of the other 7 refused", f"{refused} of {len(others)}", refused >= 4, refutes=refused < 4
            ),
            clause("at most 3 of 10 pass", passes, passes <= 3),
        ],
    )


def score_p8(rows, presentability):
    predicted = {int(t["component_first_vertex"]): t["prediction"] for t in presentability}
    n = len(rows)
    refused_or_hidden = sum(1 for r in rows if r["outcome"] in (LAYOUT, NOT_VISIBLE))
    initialization_failed = sum(
        1 for r in rows if r["outcome"] == VISION and r.get("stop_frame") == 0 and r.get("initial_visibility") is True
    )
    bad_pass = [
        int(r["component_first_vertex"])
        for r in rows
        if r["outcome"] == "pass" and predicted.get(int(r["component_first_vertex"])) == "predicted_layout_overlap"
    ]
    fraction = refused_or_hidden / n
    return prediction(
        "P8",
        "Seeded thirty: layout refusals plus not-visible exceed 40%; no pass comes from a target predicted to "
        "overlap at home.",
        [
            clause(
                "layout refusals plus not-visible exceed 40% of 30",
                {
                    "refused_or_not_visible": refused_or_hidden,
                    "of": n,
                    "fraction": round(fraction, 4),
                    "also_failed_tracker_initialization_at_frame_0": initialization_failed,
                },
                fraction > 0.40,
                refutes=fraction < 0.30,
            ),
            clause(
                "no pass from a target predicted to overlap at home",
                {
                    "passes": [int(r["component_first_vertex"]) for r in rows if r["outcome"] == "pass"],
                    "predicted_overlap_passes": bad_pass,
                },
                not bad_pass,
                refutes=bool(bad_pass),
            ),
        ],
    )


def score_presentability(rows, targets):
    """Known-map presentability calls against the recorded outcome."""
    t = by_target(rows)
    table, agree = [], 0
    for target in targets:
        vertex = int(target["component_first_vertex"])
        outcome = t[vertex]["outcome"]
        call = target["prediction"]
        expected = {"predicted_layout_overlap": (LAYOUT,), "predicted_not_visible": (NOT_VISIBLE,)}.get(call)
        match = outcome in expected if expected else outcome not in (LAYOUT, NOT_VISIBLE)
        agree += match
        table.append({"target": vertex, "call": call, "outcome": outcome, "agrees": match})
    refused = [r for r in table if r["outcome"] == LAYOUT]
    flagged = [r for r in table if r["call"] == "predicted_layout_overlap"]
    return {
        "agreeing": agree,
        "of": len(table),
        "refusals_flagged": f"{sum(1 for r in refused if r['call'] == 'predicted_layout_overlap')} of {len(refused)}",
        "flags_refused": f"{sum(1 for r in flagged if r['outcome'] == LAYOUT)} of {len(flagged)}",
        "targets": table,
    }


def score_swept(rows, register):
    """Known-map swept-path classes against the recorded outcome, with the event window where one was given."""
    t = by_target(rows)
    table = []
    for target in register["targets"]:
        vertex = int(target["component_first_vertex"])
        row = t[vertex]
        call = target["class"]
        outcome = row["outcome"]
        if call in SWEPT_OUTCOME:
            status = "agrees" if outcome == SWEPT_OUTCOME[call] else "disagrees"
            if status == "disagrees" and call in ("hazard_contact", "tof_minimum_clearance"):
                window = target.get("recorded_frame_window") or [None, None]
                decided = row.get("decision_frame")
                if outcome == VISION and decided is not None and window[0] is not None and decided < window[0]:
                    status = "untested: perception stopped before the predicted event"
        elif call == "geometry_clear":
            status = "agrees" if outcome not in GEOMETRY_CLASSES else "disagrees"
        else:
            status = "no definite call"
        entry = {"target": vertex, "call": call, "outcome": outcome, "status": status, "detail": target.get("detail")}
        if target.get("recorded_frame_window"):
            window = target["recorded_frame_window"]
            decided = row.get("decision_frame")
            entry["predicted_decision_window"] = window
            entry["decision_frame"] = decided
            entry["in_window"] = decided is not None and window[0] <= decided <= window[1]
        if row.get("contact_bodies_at_stop"):
            entry["contact_bodies_at_stop"] = row["contact_bodies_at_stop"]
        if row.get("startup_contact_n") is not None:
            entry["startup_contact_n"] = row["startup_contact_n"]
        table.append(entry)
    counts = {}
    for entry in table:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    passes = sum(1 for r in rows if r["outcome"] == "pass")
    return {
        "counts": counts,
        "passes": passes,
        "predicted_passes_at_most": register.get("predicted_passes_at_most"),
        "targets": table,
    }


def render_repeatability(root, batch_dirs):
    """Post hoc: frame-0 images and depth of runs that share revision, target, light and camera side."""
    import cv2

    groups = {}
    for batch in batch_dirs:
        batch = root / batch
        revision = read_json(batch / "plan.json")["code_revision"]
        for run in sorted(batch.glob("run_*")):
            if not (run / "frames/wrist_00000.png").is_file():
                continue
            match = re.search(r"run_\d+_([a-z]+)_tree(\d)_v(\d+)", run.name)
            report = read_json(run / "report.json")
            frame0 = read_json(run / "frames.json")["frames"][0]
            side = report["camera_mount_selection"].get("side", "default")
            key = (int(match.group(2)), int(match.group(3)), match.group(1), side, revision)
            groups.setdefault(key, []).append(
                (run, tuple(frame0["tool_pose_wxyz"]), tuple(report["initial_joint_position_rad"]))
            )
    result, totals = [], {"pairs": 0, "rgb_identical_pairs": 0, "depth_identical_pairs": 0, "pose_identical_groups": 0}
    for key, runs in sorted(groups.items()):
        if len(runs) < 2:
            continue
        entry = {
            "tree": key[0],
            "target": key[1],
            "daylight": key[2],
            "camera_side": key[3],
            "runs": [str(r[0].relative_to(root)) for r in runs],
            "tool_pose_identical": len({r[1] for r in runs}) == 1,
            "joints_identical": len({r[2] for r in runs}) == 1,
        }
        totals["pose_identical_groups"] += entry["tool_pose_identical"] and entry["joints_identical"]
        for camera in ("wrist", "close", "overview"):
            images = [cv2.imread(str(r[0] / f"frames/{camera}_00000.png")).astype(np.int16) for r in runs]
            pairs = []
            for a, b in itertools.combinations(images, 2):
                difference = np.abs(a - b).max(axis=2)
                pairs.append((int((difference > 0).sum()), int(difference.max())))
            entry[camera] = {
                "pairs": len(pairs),
                "identical_pairs": sum(1 for p in pairs if p[0] == 0),
                "max_pixels_differing": max(p[0] for p in pairs),
                "pixels": int(images[0].shape[0] * images[0].shape[1]),
                "max_grey_level_difference": max(p[1] for p in pairs),
            }
        depths = [np.load(r[0] / "frames/depth_00000.npy") for r in runs]
        same = [bool(np.array_equal(a, b, equal_nan=True)) for a, b in itertools.combinations(depths, 2)]
        entry["depth"] = {"pairs": len(same), "identical_pairs": sum(same)}
        totals["pairs"] += entry["wrist"]["pairs"]
        totals["rgb_identical_pairs"] += entry["wrist"]["identical_pairs"]
        totals["depth_identical_pairs"] += entry["depth"]["identical_pairs"]
        result.append(entry)
    totals["groups"] = len(result)
    return {"totals": totals, "groups": result}


def same_scene_pair(root, first, second):
    """Post hoc: two runs whose camera, pose and target are the same, compared at frame 0."""
    import cv2

    record = {"runs": [first, second]}
    reports = [read_json(root / run / "report.json") for run in (first, second)]
    frames = [read_json(root / run / "frames.json")["frames"][0] for run in (first, second)]
    record["camera_side"] = [r["camera_mount_selection"].get("side") for r in reports]
    record["mount_rule"] = [r["camera_mount_selection"].get("rule") for r in reports]
    record["tool_pose_identical"] = frames[0]["tool_pose_wxyz"] == frames[1]["tool_pose_wxyz"]
    record["seed_pixel_identical"] = (
        reports[0]["vision_initialization"]["pixel_xy"] == reports[1]["vision_initialization"]["pixel_xy"]
    )
    record["nodes"] = [r.get("node") for r in reports]
    a, b = (cv2.imread(str(root / run / "frames/wrist_00000.png")).astype(np.int16) for run in (first, second))
    difference = np.abs(a - b).max(axis=2)
    record["wrist_frame0_pixels_differing"] = int((difference > 0).sum())
    record["wrist_frame0_max_grey_level_difference"] = int(difference.max())
    record["initial_feature_count"] = [
        ((r.get("initial_live_vision") or {}).get("measurement") or {}).get("feature_count") for r in reports
    ]
    record["frame0_tracker"] = [
        {
            k: ((f.get("live_vision") or {}).get("measurement") or {}).get(k)
            for k in ("state", "reason", "feature_count", "roundtrip_inlier_count")
        }
        for f in frames
    ]
    return record


def git_revision(root):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]).strip()
    )
    return head, dirty


def score(root):
    root = Path(root)
    rounds = load_round(root, EVIDENCE_DIR)
    reference = reference_classes(root)
    presentability = read_json(root / PRESENTABILITY)["registers"]
    swept = read_json(root / SWEPT)["registers"]
    predictions = [
        score_p1(rounds["similarity_tree0"], reference),
        score_p2(rounds["similarity_tree1"], reference),
        score_p3(rounds["mount_flip"]),
        score_p4(rounds["both_tree0"], rounds["both_tree1"], reference),
        score_p5({k: rounds[k] for k in ("repeat_r1", "repeat_r2", "repeat_r3")}, reference),
        score_p6({k: rounds[k] for k in ("morning", "evening")}, reference, rounds["repeat_r1"]),
        score_p7(rounds["seeded_ten"]),
        score_p8(rounds["seeded30"], presentability["tree1_seeded30"]["targets"]),
    ]
    return {
        "predictions": predictions,
        "known_map_presentability": {
            "label": "known-map CPU prediction, registered 2026-09-26 before the runs; never a filter",
            "seeded_ten": score_presentability(rounds["seeded_ten"], presentability["tree1_seeded_ten"]["targets"]),
            "seeded30": score_presentability(rounds["seeded30"], presentability["tree1_seeded30"]["targets"]),
        },
        "known_map_swept_path": {
            "label": (
                "known-map CPU prediction, registered 2026-09-27 before the seeded runs; never a filter. Its "
                "validation section cites run 21442150, which finished before the file was committed; only the "
                "seeded-register calls are predictions relative to their own runs."
            ),
            "seeded_ten": score_swept(rounds["seeded_ten"], swept["eval_targets_tree1_seeded_2026-09-26.json"]),
            "seeded30": score_swept(rounds["seeded30"], swept["eval_targets_tree1_seeded30_2026-09-26.json"]),
        },
        "post_hoc_22988": {
            "label": "post hoc diagnosis, not a prediction",
            **same_scene_pair(
                root,
                "artifacts/vision_robustness/perc-similarity-tree0-20260926/run_08_source_tree0_v22988_similarity_tracker",
                "artifacts/vision_robustness/perc-both-tree0-20260926/run_08_source_tree0_v22988_similarity_mount",
            ),
        },
        "post_hoc_render_repeatability": {
            "label": "post hoc diagnosis, not a prediction",
            **render_repeatability(root, [f"artifacts/vision_robustness/{b}" for b in BATCHES.values()]),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    revision, dirty = git_revision(args.root)
    inputs = [
        PROTOCOL,
        REFERENCE_TREE0,
        REFERENCE_TREE1,
        PRESENTABILITY,
        SWEPT,
        *[f"{EVIDENCE_DIR}/{b}.json" for b in BATCHES.values()],
    ]
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the September 26 perception and tree1 protocol (P1-P8) from the grader's outcomes of 111 "
            "planned simulator runs in twelve batches, plus the two known-map prediction registers and a post hoc "
            "render-repeatability diagnosis. Simulator ground truth at a canonicalized approach pose; not a field "
            "success rate. Outcomes come from tools/validate_vision_sequence.py via tools/aggregate_eval.py."
        ),
        "protocol": PROTOCOL,
        "code_revision": revision,
        "code_tree_dirty": dirty,
        "inputs_sha256": {path: sha256(args.root / path) for path in inputs},
        "frame_convention": (
            "stop_frame is the index of the first stopped_failure frame; decision_frame is the frame whose "
            "controller output first carries the stop, one earlier; both 0-based frames.json indices"
        ),
        **score(args.root),
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
