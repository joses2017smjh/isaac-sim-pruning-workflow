#!/usr/bin/env python3
"""Offline readings of the jaw-gap beam grid at every recorded control cycle. CPU only.

The mock pruner has no sensor in its jaws. ``isaaclab_pruning.sensors.jaw_beam_grid`` defines one, nine beams
across the jaw gap, and this tool evaluates it after the fact: for every recorded capture that carries the
orchard's placement it casts the nine beams at each cycle's recorded tool pose and drawn jaw closure against
the exported orchard, trees and ground relief. The offline caster is the one ``tools/study_mouth_evidence.py``
validated against the recorded ToF grids and the recorded depth images.

Per cycle it records
  beams          nine characters, one per beam: '.' clear, 'T' the target, 'W' other wood, 'G' the ground
  target_beams   nine 0/1 flags from a cast against the target alone: the geometric truth, never the tracker
  back_beams     the same reading with the whole tool BACKOFF_M short of its recorded pose along its own axis,
                 the pose of a jaw that stopped before reaching the spur
  beneath        three 0/1 flags: tool point, jaw mouth and wrist camera beneath the ground relief
and, at the first cycle whose certificate was ready to close, the reading at each displacement the study used.

A capture in which the tool point is beneath the ground at any cycle is not a physical scene. It is listed
and kept out of every count about the sensor.

Writes two new files into a new directory outside the repository: ``jaw_beams.jsonl.gz`` (one line per capture,
the per-cycle readings) and ``jaw_beams_summary.json`` (numbers only).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "source/isaaclab_pruning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import export_gate_parity as egp  # noqa: E402
import known_map_planner as kmp  # noqa: E402
import study_mouth_evidence as study  # noqa: E402

from isaaclab_pruning.perception import jaw_self_mask as jsm  # noqa: E402
from isaaclab_pruning.sensors import jaw_beam_grid as jbg  # noqa: E402

SCHEMA_VERSION = 1
#: How far short of the recorded pose the "jaw stopped before the spur" reading is taken.
BACKOFF_M = 0.020
CLEAR, TARGET, WOOD, GROUND = ".", "T", "W", "G"


def pose_digest(pose):
    return hashlib.sha256(json.dumps([float(v) for v in pose]).encode()).hexdigest()[:16]


def castable(report):
    b = report.get("blender_scene") or {}
    component = (b.get("target") or {}).get("source_component") or {}
    return all(key in b for key in ("translation_w_m", "yaw_degrees", "visual_proxy_roll_rad")) and bool(
        component.get("source_vertex_indices")
    )


def scene_key(report):
    b = report["blender_scene"]
    vertices = np.asarray(b["target"]["source_component"]["source_vertex_indices"], np.int64)
    tree = b.get("target_tree_index", 1 if b.get("tree_path", "").endswith("Tree1") else 0)
    return (
        tuple(round(float(v), 9) for v in b["translation_w_m"]),
        round(float(b["yaw_degrees"]), 9),
        int(tree),
        hashlib.sha256(vertices.tobytes()).hexdigest(),
    )


def read_beams(rig, gap, scene):
    """``(codes, target flags)``: what each beam hits first within the gap, and whether it crosses the target."""
    origins = jbg.beam_origins(rig.p, rig.RJ, gap)
    directions = np.repeat(rig.RJ[:, 0][None], len(origins), 0)
    t_all, g_all, _ = scene.all.cast(origins, directions)
    t_target, _, _ = scene.target.cast(origins, directions)
    codes = []
    for hit, index in zip(jbg.blocked_beams(t_all, gap), g_all):
        label = scene.labels[int(index)] if hit else None
        codes.append(CLEAR if not hit else TARGET if label == "TARGET" else GROUND if label == study.GROUND else WOOD)
    return "".join(codes), "".join("1" if flag else "0" for flag in jbg.blocked_beams(t_target, gap))


def beneath_flags(rig, scene, has_camera):
    relief = study.ground_relief(rig, scene)
    if relief is None:
        return "---"
    flags = [relief["tool_point"]["beneath_the_surface"], relief["mouth"]["beneath_the_surface"]]
    return "".join("1" if flag else "0" for flag in flags) + (
        ("1" if relief["camera"]["beneath_the_surface"] else "0") if has_camera else "-"
    )


def cast_capture(records, frames, report, scene):
    """Per-cycle readings of one capture, as lists aligned with ``records``."""
    b = report["blender_scene"]
    roll, radius = float(b["visual_proxy_roll_rad"]), float(b["target"]["radius_m"])
    back = np.array([0.0, 0.0, -BACKOFF_M])
    out = {key: [] for key in ("index", "stamp_ns", "pose_digest", "gap_mm", "beams", "target_beams")}
    out.update({key: [] for key in ("back_beams", "back_target_beams", "beneath")})
    first_ready, displaced = None, []
    for record, frame in zip(records, frames):
        pose = frame["tool_pose_wxyz"]
        if int(frame["index"]) != record["index"] or [float(v) for v in pose] != record["tool_pose_wxyz"]:
            raise RuntimeError(f"cycle {record['index']}: the record and the frame are not the same cycle")
        has_camera = frame.get("wrist_position_w_m") is not None and frame.get("wrist_rotation_w_ros") is not None
        rig = study.Rig(
            pose,
            roll,
            frame["wrist_position_w_m"] if has_camera else np.zeros(3),
            frame["wrist_rotation_w_ros"] if has_camera else np.eye(3),
        )
        progress = frame.get("visual_jaw_closure_progress")
        progress = float(record["proposal"]["closure_progress"] if progress is None else progress)
        gap = jsm.jaw_gap_m(progress, radius)
        beams, target = read_beams(rig, gap, scene)
        back_beams, back_target = read_beams(rig.shifted(back), gap, scene)
        out["index"].append(record["index"])
        out["stamp_ns"].append(record["stamp_ns"])
        out["pose_digest"].append(pose_digest(record["tool_pose_wxyz"]))
        out["gap_mm"].append(round(1000 * gap, 3))
        out["beams"].append(beams)
        out["target_beams"].append(target)
        out["back_beams"].append(back_beams)
        out["back_target_beams"].append(back_target)
        out["beneath"].append(beneath_flags(rig, scene, has_camera))
        if first_ready is None and record["python"]["certificate"]["ready_to_close"]:
            first_ready = record["index"]
            for axis, label in enumerate(study.AXES):
                for sign in (-1, 1):
                    for offset in study.OFFSETS_M:
                        vector = np.zeros(3)
                        vector[axis] = sign * offset
                        codes, flags = read_beams(rig.shifted(vector), gap, scene)
                        displaced.append(
                            {
                                "axis": label,
                                "offset_mm": round(1000 * sign * offset, 1),
                                "beams": codes,
                                "target": flags,
                            }
                        )
    out["first_ready"], out["displaced_at_first_ready"] = first_ready, displaced
    return out


def blocked(codes):
    return any(code != CLEAR for code in codes)


def on_target(flags):
    return "1" in flags


def summarise_capture(label, records, reading):
    """Numbers only: how the reading relates to the recorded cut, cycle by cycle."""
    phases = [record["proposal"]["cut_phase"] for record in records]
    ready = [i for i, record in enumerate(records) if record["python"]["certificate"]["ready_to_close"]]
    closing = [i for i, phase in enumerate(phases) if phase == "closing"]
    high, advancing = 0.0, []
    for i, record in enumerate(records):
        progress = float(record["proposal"]["closure_progress"])
        if progress > high:
            advancing.append(i)
            high = progress
    window = sorted(set(ready) | set(closing))

    def tally(indices, beams_key, target_key):
        return {
            "cycles": len(indices),
            "any_beam_blocked": sum(blocked(reading[beams_key][i]) for i in indices),
            "target_between_jaws": sum(on_target(reading[target_key][i]) for i in indices),
            "blocked_by": dict(Counter(code for i in indices for code in reading[beams_key][i] if code != CLEAR)),
        }

    displaced = reading["displaced_at_first_ready"]
    return {
        "capture": label,
        "target": records[0]["target"]["id"],
        "cycles": len(records),
        "first_ready_index": reading["first_ready"],
        "first_closing_index": records[closing[0]]["index"] if closing else None,
        "detached": any(record["proposal"]["cut_detached"] for record in records),
        "beneath_the_ground": {
            "tool_point_cycles": sum(flags[0] == "1" for flags in reading["beneath"]),
            "mouth_cycles": sum(flags[1] == "1" for flags in reading["beneath"]),
            "camera_cycles": sum(flags[2] == "1" for flags in reading["beneath"]),
        },
        "tool_beneath_the_ground": any(flags[0] == "1" for flags in reading["beneath"]),
        "ready": tally(ready, "beams", "target_beams"),
        "closing": tally(closing, "beams", "target_beams"),
        "jaw_advancing": tally(advancing, "beams", "target_beams"),
        "tool_short_ready_or_closing": tally(window, "back_beams", "back_target_beams"),
        "displaced_at_first_ready": {
            "poses": len(displaced),
            "target_between_jaws": sum(on_target(row["target"]) for row in displaced),
            "any_beam_blocked": sum(blocked(row["beams"]) for row in displaced),
            "blocked_without_the_target": sum(
                blocked(row["beams"]) and not on_target(row["target"]) for row in displaced
            ),
            "target_without_a_blocked_beam": sum(
                on_target(row["target"]) and not blocked(row["beams"]) for row in displaced
            ),
        },
    }


def totals(rows):
    """Sums over captures, for one group of them."""
    out = {"captures": len(rows), "cycles": sum(row["cycles"] for row in rows)}
    out["reach_closing"] = sum(row["first_closing_index"] is not None for row in rows)
    for key in ("ready", "closing", "jaw_advancing", "tool_short_ready_or_closing"):
        out[key] = {
            "cycles": sum(row[key]["cycles"] for row in rows),
            "any_beam_blocked": sum(row[key]["any_beam_blocked"] for row in rows),
            "target_between_jaws": sum(row[key]["target_between_jaws"] for row in rows),
            "blocked_by": dict(sum((Counter(row[key]["blocked_by"]) for row in rows), Counter())),
            "captures_with_a_cycle_without_a_blocked_beam": sum(
                row[key]["any_beam_blocked"] < row[key]["cycles"] for row in rows
            ),
            "captures_with_a_blocked_beam": sum(row[key]["any_beam_blocked"] > 0 for row in rows),
        }
    out["displaced_at_first_ready"] = {
        key: sum(row["displaced_at_first_ready"][key] for row in rows)
        for key in (
            "poses",
            "target_between_jaws",
            "any_beam_blocked",
            "blocked_without_the_target",
            "target_without_a_blocked_beam",
        )
    }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside the repository")
    parser.add_argument("--roots", nargs="+", default=[str(ROOT / "artifacts")])
    parser.add_argument("--only", nargs="*", help="Limit to captures whose path contains one of these strings")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    t0 = time.time()
    inputs = {}
    export = kmp.stream_export(kmp.EXPORT_DEFAULT, kmp.ISAAC_PYTHON_DEFAULT, inputs)
    names = [str(name) for name in export["names"]]
    ground = study.stream_ground()
    scenes, rows, skipped = {}, [], Counter()
    output.mkdir(parents=True)
    with gzip.open(output / "jaw_beams.jsonl.gz", "wt", encoding="utf-8") as sidecar:
        for directory in egp.discover(args.roots):
            label = str(directory.relative_to(ROOT)) if ROOT in directory.parents else str(directory)
            if args.only and not any(part in label for part in args.only):
                continue
            try:
                records, _, _ = egp.capture_cycles(directory)
            except egp.Skip as skip:
                skipped[str(skip)] += 1
                continue
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            if (
                not castable(report)
                or Path(report["blender_scene"].get("export_dir", "")).name != kmp.EXPORT_DEFAULT.name
            ):
                skipped["no_orchard_placement"] += 1
                continue
            document = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
            frames = document["frames"] if isinstance(document, dict) else document
            key = scene_key(report)
            if key not in scenes:
                scenes[key] = study.build_scene(export, names, report, ground=ground)
            reading = cast_capture(records, frames, report, scenes[key])
            row = summarise_capture(label, records, reading)
            rows.append(row)
            sidecar.write(json.dumps({"capture": label, **reading}, separators=(",", ":")) + "\n")
            print(
                f"{label}: closing {row['closing']['any_beam_blocked']}/{row['closing']['cycles']} blocked, "
                f"beneath ground {row['tool_beneath_the_ground']}",
                flush=True,
            )
    above = [row for row in rows if not row["tool_beneath_the_ground"]]
    beneath = [row for row in rows if row["tool_beneath_the_ground"]]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    sidecar_path = output / "jaw_beams.jsonl.gz"
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/cast_jaw_beams.py",
        "scope": (
            "Offline readings of a sensor the simulated tool does not have: a nine-beam grid across the jaw gap, cast "
            "at recorded tool poses against the exported orchard (trees and ground relief). Simulator geometry."
        ),
        "definitions": {
            "beams_along_jaw_m": list(jbg.BEAMS_ALONG_JAW_M),
            "beams_along_tool_m": list(jbg.BEAMS_ALONG_TOOL_M),
            "codes": {CLEAR: "clear", TARGET: "the target", WOOD: "other wood", GROUND: "the ground relief"},
            "tool_short_m": BACKOFF_M,
            "displacements_m": list(study.OFFSETS_M),
            "jaw_gap": "the drawn jaw's closure in that frame (visual_jaw_closure_progress)",
            "ground_triangles": int(len(ground[1])),
        },
        "captures_cast": len(rows),
        "skipped": dict(skipped),
        "scenes_built": len(scenes),
        "tool_above_the_ground": totals(above),
        "tool_beneath_the_ground": {
            **totals(beneath),
            "captures_by_target": dict(Counter(row["target"] for row in beneath)),
            "list": [row["capture"] for row in beneath],
        },
        "targets_above_the_ground": dict(Counter(row["target"] for row in above)),
        "captures": rows,
        "sidecar": {"file": sidecar_path.name, "sha256": kmp.sha256(sidecar_path)},
        "inputs_sha256": inputs,
        "provenance": {
            "code_revision": head.stdout.strip(),
            "tracked_tree_clean": dirty.stdout.strip() == "",
            "host": socket.gethostname(),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    (output / "jaw_beams_summary.json").write_text(json.dumps(document, indent=1, default=float) + "\n")
    print(json.dumps({k: document[k] for k in ("captures_cast", "skipped", "scenes_built")}))
    print("tool beneath the ground:", document["tool_beneath_the_ground"]["captures_by_target"])
    print(
        "above the ground:",
        json.dumps({k: v for k, v in document["tool_above_the_ground"].items() if k != "displaced"}),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
