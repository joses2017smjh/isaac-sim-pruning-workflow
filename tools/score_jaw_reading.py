#!/usr/bin/env python3
"""Score the centred-spur jaw reading against docs/EVAL_PROTOCOL_JAW_READING_2026-11-25.md. CPU only, Python only.

For each capture of one side of tools/jaw_reading_split.py it turns every ranged reading into accept or refuse with
tools/jaw_reading_rules.py, feeds that to the reference gate as its jaw-presence input, and replays the capture
with the node's jaw rule (tools/gate_replay.py), exactly as the one-beam rule was replayed. No C++ and no ROS: the
rule is tested as a rule before anything is built around it.

Measured per side, for the centred-spur rule and for the one-beam rule on the same captures:
  recorded   closures whose replay is unchanged with the recorded-pose reading
  short      closures refused with the tool 20 mm short of its recorded pose (the jaw output never moves)
  displaced  at each capture's first ready cycle, the 36 displaced poses: accepted with and without the target
Captures with the tool point beneath the ground are left out, as in the one-beam replay.

Writes one new JSON file; numbers and capture names only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import gate_replay  # noqa: E402
import jaw_reading_rules as rules  # noqa: E402
import jaw_reading_split as split  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL = "docs/EVAL_PROTOCOL_JAW_READING_2026-11-25.md"
YES, NO = "T........", "........."
OPEN_GAP_MM = 32.0
RULES = ("one_beam", "centred_spur")


def load_cycles(path, wanted):
    captures = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["capture"] in wanted:
                captures[record["capture"]].append(record)
    return dict(captures)


def load_beams(paths):
    readings = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                reading = json.loads(line)
                if reading["capture"] in readings:
                    raise ValueError(f"{reading['capture']} is in more than one beam file")
                readings[reading["capture"]] = reading
    return readings


def decide(rule, codes, ranges, gap_mm, radius_mm):
    """True when ``rule`` accepts one reading."""
    if rule == "one_beam":
        return gate_replay.blocked_count(codes) > 0
    return rules.accepts(ranges, gap_mm, radius_mm)


def decisions(rule, reading, codes_key, ranges_key, radius_mm):
    """One accept/refuse per cycle, as the nine-character reading the gate replay expects."""
    return [
        YES if decide(rule, codes, ranges, gap, radius_mm) else NO
        for codes, ranges, gap in zip(reading[codes_key], reading[ranges_key], reading["gap_mm"])
    ]


def replay_with(records, reading, codes):
    joined = gate_replay.attach_beams(records, {**reading, "decisions": codes}, "decisions")
    return gate_replay.replay(joined, jaw_min_beams=1, jaw_key="jaw_beams", node_rules=True)


def score_capture(records, reading):
    """How each rule treats one capture: its recorded closure, the short jaw, and the displaced poses."""
    radius_mm = 1000.0 * float(records[0]["target"]["radius_m"])
    off = gate_replay.replay(records, node_rules=True)
    closes = any(record["proposal"]["cut_phase"] == "closing" for record in records)
    out = {"reaches_closing": closes, "rules": {}}
    for rule in RULES:
        row = {}
        if closes:
            on = replay_with(records, reading, decisions(rule, reading, "beams", "ranges", radius_mm))
            row["recorded_kept"] = all(
                a[key] == b[key] for a, b in zip(off, on) for key in ("passed", "reason", "latched_reason", "gated_jaw")
            )
            row["recorded_first_hold"] = list(gate_replay.first_hold(on, records))
            short = replay_with(records, reading, decisions(rule, reading, "back_beams", "back_ranges", radius_mm))
            row["short_refused"] = max(verdict["gated_jaw"] for verdict in short) == 0.0
        table = Counter()
        for pose in reading["displaced_at_first_ready"]:
            target = "1" in pose["target"]
            accepted = decide(rule, pose["beams"], pose["ranges"], OPEN_GAP_MM, radius_mm)
            table[("with_target" if target else "without_target") + ("_accepted" if accepted else "_refused")] += 1
        row["displaced"] = dict(table)
        out["rules"][rule] = row
    return out


def summarise(rows):
    """Counts over the captures of one side, per rule and per target group."""
    out = {}
    for rule in RULES:
        groups = {}
        for name, keep in (
            ("all", lambda r: True),
            ("target_8235", lambda r: r["divided"]),
            ("other_targets", lambda r: not r["divided"]),
        ):
            closing = [row for row in rows if row["reaches_closing"] and keep(row)]
            displaced = Counter()
            for row in rows:
                if keep(row):
                    displaced.update(row["rules"][rule]["displaced"])
            groups[name] = {
                "closure_captures": len(closing),
                "recorded_kept": sum(row["rules"][rule]["recorded_kept"] for row in closing),
                "recorded_refused": sorted(
                    row["capture"] for row in closing if not row["rules"][rule]["recorded_kept"]
                ),
                "short_refused": sum(row["rules"][rule]["short_refused"] for row in closing),
                "short_accepted": sorted(row["capture"] for row in closing if not row["rules"][rule]["short_refused"]),
                "displaced": dict(displaced),
            }
        out[rule] = groups
    return out


def verdicts(test):
    """The registered predictions, on the test side only."""
    rule, one = test["centred_spur"], test["one_beam"]
    others, divided, everything = rule["other_targets"], rule["target_8235"], rule["all"]

    def judge(holds, checked):
        return "untested" if not checked else ("supported" if holds else "refuted")

    false_in = everything["displaced"].get("without_target_accepted", 0)
    false_in_one = one["all"]["displaced"].get("without_target_accepted", 0)
    return {
        "K1": {
            "prediction": "every recorded closure on a target other than 8235 is unchanged",
            "verdict": judge(others["recorded_kept"] == others["closure_captures"], others["closure_captures"]),
            "kept": others["recorded_kept"],
            "of": others["closure_captures"],
            "changed": others["recorded_refused"],
        },
        "K2": {
            "prediction": "with the tool 20 mm short every closure is refused",
            "verdict": judge(
                everything["short_refused"] == everything["closure_captures"], everything["closure_captures"]
            ),
            "refused": everything["short_refused"],
            "of": everything["closure_captures"],
            "accepted": everything["short_accepted"],
            "one_beam_accepted": one["all"]["short_accepted"],
        },
        "K3": {
            "prediction": "every recorded closure on target 8235, which has other wood in the gap, is refused",
            "verdict": judge(divided["recorded_kept"] == 0, divided["closure_captures"]),
            "refused": divided["closure_captures"] - divided["recorded_kept"],
            "of": divided["closure_captures"],
        },
        "K4": {
            "prediction": "fewer displaced poses without the target are accepted than under the one-beam rule",
            "verdict": judge(false_in < false_in_one, false_in_one),
            "accepted": false_in,
            "one_beam_accepted": false_in_one,
            "poses_without_the_target": false_in + everything["displaced"].get("without_target_refused", 0),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=Path, required=True)
    parser.add_argument(
        "--beams", type=Path, nargs="+", required=True, help="jaw_beams.jsonl.gz with ranges (schema 2)"
    )
    parser.add_argument("--side", choices=split.SIDES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    readings = load_beams(args.beams)
    captures = load_cycles(args.cycles, set(readings))
    rows, skipped = [], Counter()
    for label, reading in sorted(readings.items()):
        records = captures.get(label)
        if records is None:
            skipped["no_cycles"] += 1
            continue
        target = records[0]["target"]["id"]
        if split.side(label, target) != args.side:
            skipped["other_side_of_the_split"] += 1
            continue
        if not gate_replay.tool_above_the_ground(reading):
            skipped["tool_beneath_the_ground"] += 1
            continue
        row = score_capture(records, reading)
        rows.append(
            {"capture": label, "target": target, "divided": split.component(target) == split.DIVIDED_TARGET, **row}
        )
    summary = summarise(rows)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/score_jaw_reading.py",
        "protocol": PROTOCOL,
        "side": args.side,
        "rule": {
            "centre_max_mm": rules.CENTRE_MAX_MM,
            "thickness_slack_mm": rules.THICKNESS_SLACK_MM,
            "min_spur_beams": rules.MIN_SPUR_BEAMS,
        },
        "captures": len(rows),
        "closure_captures": sum(row["reaches_closing"] for row in rows),
        "targets": dict(Counter(row["target"] for row in rows if row["reaches_closing"])),
        "skipped": dict(skipped),
        "summary": summary,
        "verdicts": verdicts(summary) if args.side == "test" else None,
        "rows": rows,
        "inputs_sha256": {
            "cycles": hashlib.sha256(args.cycles.read_bytes()).hexdigest(),
            **{f"beams:{path.parent.name}": hashlib.sha256(path.read_bytes()).hexdigest() for path in args.beams},
        },
        "provenance": {"code_revision": head.stdout.strip(), "tracked_tree_clean": dirty.stdout.strip() == ""},
    }
    output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    for rule in RULES:
        for group in ("other_targets", "target_8235"):
            g = summary[rule][group]
            print(
                f"{args.side} {rule:12} {group:13}: recorded kept {g['recorded_kept']}/{g['closure_captures']}, "
                f"short refused {g['short_refused']}/{g['closure_captures']}, displaced {g['displaced']}"
            )
    if document["verdicts"]:
        print(json.dumps({name: item["verdict"] for name, item in document["verdicts"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
