#!/usr/bin/env python3
"""Score the jaw-presence replay against docs/EVAL_PROTOCOL_JAW_PRESENCE_2026-11-25.md. CPU only, no ROS.

Inputs, all produced from the registered commit:
  --cycles            today's rebuilt cycles (tools/export_gate_parity.py)
  --beams             jaw_beams.jsonl.gz (tools/cast_jaw_beams.py)
  --baseline-cycles   the November 5 cycles file, whose stored verdicts came from the gate before this change
  --baseline          docs/evidence/cpp_safety_gate_parity_2026-11-05.json
  --harness-off       tools/validate_safety_gate.py on the November 5 cycles, jaw presence off
  --harness-jaw       tools/validate_safety_gate.py on today's cycles with --jaw-beams

A capture whose tool point is beneath the ground relief in any cycle is not a physical scene. It is listed and
kept out of J2 to J5. Writes one new JSON file; numbers and capture names only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import gate_replay  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL = "docs/EVAL_PROTOCOL_JAW_PRESENCE_2026-11-25.md"
MIN_BEAMS = 1
ALL_CLEAR = "........."
TOLERANCE = 1e-9
DISCRETE = ("passed", "reason", "latched_reason", "tof_reason", "tof_missing_frames", "closure_authorized")
CERTIFICATE_DISCRETE = ("safe_to_approach", "ready_to_close", "reasons", "vision_source", "waived_checks")
CERTIFICATE_CONTINUOUS = ("mouth_distance_m", "perpendicularity_error_deg", "vision_age_s")


def load_cycles(path):
    """``({capture: records}, digest)``; the digest is over each line without its newline, as the harness takes it."""
    captures, digest = defaultdict(list), hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            digest.update(line.rstrip("\n").encode())
            record = json.loads(line)
            captures[record["capture"]].append(record)
    return dict(captures), digest.hexdigest()


def load_beams(path):
    readings = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            reading = json.loads(line)
            readings[reading["capture"]] = reading
    return readings, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right):
    left = None if left is None or (isinstance(left, float) and math.isnan(left)) else left
    right = None if right is None or (isinstance(right, float) and math.isnan(right)) else right
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE * max(1.0, abs(float(right)))


def differing_fields(new, stored):
    """Fields where a replayed verdict differs from a stored one."""
    out = [key for key in DISCRETE if new[key] != stored[key]]
    if not close(new["tof_min_range_m"], stored["tof_min_range_m"]):
        out.append("tof_min_range_m")
    out += [
        f"certificate.{key}" for key in CERTIFICATE_DISCRETE if new["certificate"][key] != stored["certificate"][key]
    ]
    out += [
        f"certificate.{key}"
        for key in CERTIFICATE_CONTINUOUS
        if not close(new["certificate"][key], stored["certificate"][key])
    ]
    return out


def verdict_of(holds, checked):
    """supported when every condition holds; untested when nothing could be checked."""
    if not checked:
        return "untested"
    return "supported" if all(holds.values()) else "refuted"


def score_j1(baseline, harness_off, baseline_captures, baseline_digest):
    """Jaw presence off changes nothing: the node, the faults and the reference on the November 5 cycles."""
    differing = Counter()
    cycles = 0
    for records in baseline_captures.values():
        for record, verdict in zip(records, gate_replay.replay(records)):
            cycles += 1
            for key in differing_fields(verdict, record["python"]):
                differing[key] += 1
    old, new = baseline["parity"]["totals"], harness_off["parity"]["totals"]
    old_faults, new_faults = baseline["faults"]["scenarios"], harness_off["faults"]["scenarios"]
    changed_faults = sorted(
        name
        for name in set(old_faults) | set(new_faults)
        if name not in old_faults
        or name not in new_faults
        or any(
            old_faults[name][key] != new_faults[name][key] for key in ("ok", "first_hold_index", "first_hold_reason")
        )
    )
    holds = {
        "same_cycles_file": harness_off["cycles_sha256"] == baseline["cycles_sha256"] == baseline_digest,
        "node_matches_stored_verdicts_on_every_cycle": new.get("mismatched_cycles", 0) == 0
        and new.get("matched_cycles") == old.get("matched_cycles") == new.get("cycles"),
        "node_holds_as_before": new.get("cpp_holds") == old.get("cpp_holds"),
        "no_forwarding_problem": not any(key.startswith("forwarding_") and value for key, value in new.items()),
        "every_fault_as_before": not changed_faults
        and harness_off["faults"]["stopping_faults_held"] == harness_off["faults"]["stopping_faults"],
        "reference_matches_stored_verdicts_on_every_cycle": not differing and cycles == old.get("cycles"),
    }
    return {
        "prediction": "J1",
        "verdict": verdict_of(holds, cycles),
        "holds": holds,
        "cycles": cycles,
        "captures": len(baseline_captures),
        "node": {
            key: new.get(key) for key in ("cycles", "matched_cycles", "mismatched_cycles", "cpp_holds", "captures")
        },
        "node_on_november_5": {key: old.get(key) for key in ("cycles", "matched_cycles", "cpp_holds", "captures")},
        "faults": {
            "stopping": harness_off["faults"]["stopping_faults"],
            "held": harness_off["faults"]["stopping_faults_held"],
            "changed": changed_faults,
        },
        "reference_fields_differing": dict(differing),
    }


def usable(captures, readings):
    """Captures that have a reading and a tool above the ground, and the ones set aside."""
    above, beneath, unread = {}, [], []
    for label, records in sorted(captures.items()):
        reading = readings.get(label)
        if reading is None:
            unread.append(label)
        elif gate_replay.tool_above_the_ground(reading):
            above[label] = (records, reading)
        else:
            beneath.append(label)
    return above, beneath, unread


def reaches_closing(records):
    return any(record["proposal"]["cut_phase"] == "closing" for record in records)


def score_j2(above, harness_jaw):
    """With the recorded-pose reading, jaw presence on changes no recorded cut."""
    changed, ready_unblocked, closures, cycles = [], 0, 0, 0
    for label, (records, reading) in above.items():
        joined = gate_replay.attach_beams(records, reading, "beams")
        off = gate_replay.replay(joined, node_rules=True)
        on = gate_replay.replay(joined, jaw_min_beams=MIN_BEAMS, jaw_key="jaw_beams", node_rules=True)
        cycles += len(records)
        closures += reaches_closing(records)
        differs = [
            record["index"]
            for record, a, b in zip(records, off, on)
            if any(a[key] != b[key] for key in ("passed", "reason", "latched_reason", "gated_jaw"))
        ]
        ready_unblocked += sum(
            1 for a, b in zip(off, on) if a["certificate"]["ready_to_close"] and b["jaw_presence_ok"] is False
        )
        if differs:
            index, reason = gate_replay.first_hold(on, records)
            changed.append({"capture": label, "first_changed_index": differs[0], "first_hold": [index, reason]})
    node = {label: item["recorded"] for label, item in harness_jaw["captures"].items()}
    node_mismatches = {label: s["mismatched_fields"] for label, s in node.items() if s["mismatched_fields"]}
    node_jaw_differs = sorted(label for label, s in node.items() if s["max_gated_jaw"] != s["recorded_max_jaw"])
    expected_node = sorted(label for label, (records, _) in above.items() if reaches_closing(records))
    holds = {
        "no_capture_changes": not changed,
        "node_matches_the_reference_on_every_cycle": not node_mismatches
        and all(s["missing_outputs"] == 0 for s in node.values()),
        "node_ran_every_closure_capture": sorted(node) == expected_node,
        "node_jaw_output_equals_the_recorded_jaw": not node_jaw_differs,
    }
    return {
        "prediction": "J2",
        "verdict": verdict_of(holds, closures),
        "holds": holds,
        "captures": len(above),
        "cycles": cycles,
        "captures_reaching_closing": closures,
        "captures_changed": changed,
        "ready_cycles_without_a_blocked_beam": ready_unblocked,
        "node": {
            "captures": len(node),
            "cycles": sum(s["cycles"] for s in node.values()),
            "captures_with_mismatched_fields": node_mismatches,
            "captures_whose_jaw_output_differs": node_jaw_differs,
        },
    }


def refusal(harness_jaw, scenario, above):
    """Per capture, whether the node kept the jaw where it was and agreed with the reference."""
    rows = {}
    for label, item in sorted(harness_jaw["captures"].items()):
        s = item[scenario]
        records = above[label][0]
        closing = next(r["index"] for r in records if r["proposal"]["cut_phase"] == "closing")
        rows[label] = {
            "first_closing_index": closing,
            "first_hold_index": s["first_hold_index"],
            "first_hold_reason": s["first_hold_reason"],
            "as_the_reference_expects": s["first_hold_index"] == s["expected_first_hold_index"]
            and s["first_hold_reason"] == s["expected_first_hold_reason"]
            and not s["mismatched_fields"],
            "max_gated_jaw": s["max_gated_jaw"],
            "recorded_max_jaw": s["recorded_max_jaw"],
            "jaw_never_advanced": s["max_gated_jaw"] == 0.0,
            "held": s["first_hold_index"] is not None,
            "blocked_cycles": s["blocked_cycles"],
            "clean": not s["missing_outputs"]
            and not s["forwarding_problems"]
            and s["held_cycles_commanding_motion"] == 0,
        }
    return rows


def score_j3(harness_jaw, above):
    """Every beam reported clear: the node refuses every recorded closure before the jaw moves."""
    rows = refusal(harness_jaw, "all_clear", above)
    moved = sorted(label for label, row in rows.items() if not row["jaw_never_advanced"])
    unheld = sorted(label for label, row in rows.items() if row["recorded_max_jaw"] > 0.0 and not row["held"])
    holds = {
        "jaw_never_advanced": not moved,
        "held_wherever_the_recorded_jaw_advanced": not unheld,
        "node_as_the_reference_expects": all(row["as_the_reference_expects"] for row in rows.values()),
        "outputs_clean": all(row["clean"] for row in rows.values()),
    }
    return {
        "prediction": "J3",
        "verdict": verdict_of(holds, len(rows)),
        "holds": holds,
        "captures": len(rows),
        "captures_whose_jaw_advanced": moved,
        "captures_not_held": unheld,
        "first_hold_reasons": dict(Counter(str(row["first_hold_reason"]) for row in rows.values())),
        "cycles_from_first_closing_to_hold": dict(
            Counter(
                str(row["first_hold_index"] - row["first_closing_index"])
                for row in rows.values()
                if row["first_hold_index"] is not None
            )
        ),
    }


def score_j4(harness_jaw, above):
    """The tool 20 mm short of its recorded pose: no beam blocked, and every closure refused."""
    rows = refusal(harness_jaw, "tool_short", above)
    blocked, target_in, window_cycles = {}, {}, 0
    for label in rows:
        records, reading = above[label]
        window = [
            i
            for i, record in enumerate(records)
            if record["python"]["certificate"]["ready_to_close"] or record["proposal"]["cut_phase"] == "closing"
        ]
        window_cycles += len(window)
        hits = sum(gate_replay.blocked_count(reading["back_beams"][i]) > 0 for i in window)
        truth = sum("1" in reading["back_target_beams"][i] for i in window)
        if hits:
            blocked[label] = {
                "cycles": hits,
                "by": dict(Counter(c for i in window for c in reading["back_beams"][i] if c != ".")),
            }
        if truth:
            target_in[label] = truth
    accepted = sorted(label for label, row in rows.items() if not row["jaw_never_advanced"])
    holds = {
        "no_beam_blocked_in_a_ready_or_closing_cycle": not blocked,
        "every_closure_refused": not accepted,
        "node_as_the_reference_expects": all(row["as_the_reference_expects"] for row in rows.values()),
    }
    return {
        "prediction": "J4",
        "verdict": verdict_of(holds, len(rows)),
        "holds": holds,
        "captures": len(rows),
        "ready_or_closing_cycles": window_cycles,
        "captures_with_a_blocked_beam": blocked,
        "captures_with_the_target_still_between_the_jaws": target_in,
        "closures_accepted": accepted,
        "first_hold_reasons": dict(Counter(str(row["first_hold_reason"]) for row in rows.values())),
    }


def describe_j5(above):
    """Counts only: the study's displacements at each capture's first ready cycle."""
    table, captures = defaultdict(Counter), 0
    for _, reading in above.values():
        rows = reading["displaced_at_first_ready"]
        captures += bool(rows)
        for row in rows:
            cell = table[(row["axis"], row["offset_mm"])]
            target, beam = "1" in row["target"], gate_replay.blocked_count(row["beams"]) > 0
            cell["poses"] += 1
            cell["target_between_jaws"] += target
            cell["any_beam_blocked"] += beam
            cell["blocked_without_the_target"] += beam and not target
            cell["target_without_a_blocked_beam"] += target and not beam
            for code in row["beams"]:
                if code != "." and not target:
                    cell[f"blocked_by_{code}_without_the_target"] += 1
    total = Counter()
    for cell in table.values():
        total.update(cell)
    return {
        "captures_with_a_ready_cycle": captures,
        "total": dict(total),
        "by_displacement": [
            {"axis": axis, "offset_mm": offset, **dict(table[(axis, offset)])} for axis, offset in sorted(table)
        ],
    }


def score_j6(harness_jaw):
    faults = harness_jaw.get("input_faults") or {}
    holds = {name: bool(item["ok"]) for name, item in faults.items()}
    return {
        "prediction": "J6",
        "verdict": verdict_of(holds, len(faults)),
        "holds": holds,
        "capture": harness_jaw.get("input_fault_capture"),
        "scenarios": {
            name: {
                key: item[key]
                for key in ("expected_reason", "first_hold_reason", "first_hold_index", "held_through_end")
            }
            for name, item in faults.items()
        },
    }


def ground_audit(captures, readings, beneath):
    rows = []
    for label in beneath:
        records, reading = captures[label], readings[label]
        rows.append(
            {
                "capture": label,
                "target": records[0]["target"]["id"],
                "cycles": len(records),
                "tool_point_beneath_cycles": sum(flags[0] == "1" for flags in reading["beneath"]),
                "mouth_beneath_cycles": sum(flags[1] == "1" for flags in reading["beneath"]),
                "camera_beneath_cycles": sum(flags[2] == "1" for flags in reading["beneath"]),
                "reaches_closing": reaches_closing(records),
                "detached": any(record["proposal"]["cut_detached"] for record in records),
            }
        )
    return {
        "captures_with_the_tool_beneath_the_ground": len(rows),
        "by_target": dict(Counter(row["target"] for row in rows)),
        "reaching_closing": sum(row["reaches_closing"] for row in rows),
        "captures": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("cycles", "beams", "baseline-cycles", "baseline", "harness-off", "harness-jaw", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    captures, cycles_digest = load_cycles(args.cycles)
    readings, beams_digest = load_beams(args.beams)
    baseline_captures, baseline_digest = load_cycles(args.baseline_cycles)
    baseline = json.loads(args.baseline.read_text())
    harness_off = json.loads(args.harness_off.read_text())
    harness_jaw_document = json.loads(args.harness_jaw.read_text())
    harness_jaw = harness_jaw_document["jaw_presence"]
    above, beneath, unread = usable(captures, readings)
    consistent = {
        "jaw_harness_used_these_cycles": harness_jaw_document["cycles_sha256"] == cycles_digest,
        "jaw_harness_used_these_beams": harness_jaw_document.get("jaw_beams_sha256") == beams_digest,
        "jaw_harness_min_beams": harness_jaw["min_beams"] == MIN_BEAMS,
        "jaw_harness_captures_are_above_the_ground": all(label in above for label in harness_jaw["captures"]),
    }
    verdicts = {}
    verdicts["J1"] = score_j1(baseline, harness_off, baseline_captures, baseline_digest)
    if all(consistent.values()):
        verdicts["J2"] = score_j2(above, harness_jaw)
        verdicts["J3"] = score_j3(harness_jaw, above)
        verdicts["J4"] = score_j4(harness_jaw, above)
        verdicts["J6"] = score_j6(harness_jaw)
    else:
        for name in ("J2", "J3", "J4", "J6"):
            verdicts[name] = {"prediction": name, "verdict": "untested", "holds": {}, "why": "inputs do not match"}
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/score_jaw_presence.py",
        "protocol": PROTOCOL,
        "min_beams": MIN_BEAMS,
        "inputs_consistent": consistent,
        "captures": {
            "rebuilt": len(captures),
            "cycles": sum(len(records) for records in captures.values()),
            "with_a_beam_reading": len(captures) - len(unread),
            "tool_above_the_ground": len(above),
            "tool_beneath_the_ground": len(beneath),
            "above_the_ground_reaching_closing": sum(reaches_closing(records) for records, _ in above.values()),
            "above_the_ground_targets": dict(Counter(records[0]["target"]["id"] for records, _ in above.values())),
            "without_a_beam_reading": unread,
        },
        "verdicts": verdicts,
        "descriptive": {
            "J5_displaced_at_first_ready": describe_j5(above),
            "ground_audit": ground_audit(captures, readings, beneath),
        },
        "inputs_sha256": {
            "cycles": cycles_digest,
            "beams": beams_digest,
            "baseline_cycles": baseline_digest,
            "baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "harness_off": hashlib.sha256(args.harness_off.read_bytes()).hexdigest(),
            "harness_jaw": hashlib.sha256(args.harness_jaw.read_bytes()).hexdigest(),
        },
        "provenance": {"code_revision": head.stdout.strip(), "tracked_tree_clean": dirty.stdout.strip() == ""},
    }
    output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(json.dumps({name: item["verdict"] for name, item in verdicts.items()}, indent=2))
    print(json.dumps(document["captures"] | {"without_a_beam_reading": len(unread), "above_the_ground_targets": None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
