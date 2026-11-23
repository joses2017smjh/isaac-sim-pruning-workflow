"""Three things the November 5 safety-gate results leave out, computed from the recorded data.

1. **What the parity count is made of.** ``docs/evidence/cpp_safety_gate_parity_2026-11-05.json``
   reports 43,623 matching cycles. A hold latches for the rest of a run, so every cycle after a
   run's first hold is a hold whose outcome was already fixed. This splits the cycles the C++
   gate was compared on into: passing cycles before any hold, the first hold of each run, and
   cycles after a latch. The first two are the cycles where the gate still had a decision to make.
   The cycles file is the one the parity run used: its digest, computed as tools/validate_safety_gate.py
   does (sha256 over each decompressed line without its newline), must equal the evidence's.

2. **The vision age the gate saw in the live loop.** The gate holds when vision is older than
   0.25 s. This collects every ``vision_age_s`` the gate logged in the registered live-loop batch
   (``ros2_loop_decisions.jsonl`` in each run). The controller stamps its vision with the frame's
   own capture time, so the age is zero in every decision: the delay test delayed commands, not
   sensor data, and the freshness check never fired in closed loop.

3. **How delay slowed the arm.** Per registered live-loop run: the modelled delay, how many frames
   the approach took, the median tool motion per frame during it, and the frame of the cut, beside a
   post-hoc model fitted after the result and not registered. Each command is the measured pose plus
   one capped step, so a command that lands late overlaps the next: progress per frame is
   step x 6 / (6 + delay steps). This reports the measurements beside the model; it does not test it.

None changes a registered verdict; each is a limit on what those verdicts mean.

CPU only. Reads files; writes one new JSON file outside the repository and never overwrites.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARE = Path("/nfs/hpc/share/sanchej7")
PARITY_EVIDENCE = "docs/evidence/cpp_safety_gate_parity_2026-11-05.json"
LIVE_LOOP_VERDICTS = "docs/evidence/ros2_live_loop_verdicts_2026-11-05.json"
VISION_MAX_AGE_S = 0.25


class AuditError(RuntimeError):
    pass


def parity_breakdown(rows):
    """Split ``(capture, python_verdict)`` rows, in recorded order, by what each cycle could still decide."""
    by_capture = defaultdict(list)
    for capture, verdict in rows:
        by_capture[capture].append(verdict)
    passing = post_latch = never_latched = 0
    first_holds = Counter()
    for verdicts in by_capture.values():
        latched = False
        for verdict in verdicts:
            if verdict["passed"]:
                if latched:
                    raise AuditError("a cycle passed after a hold had latched; the latch rule is broken")
                passing += 1
            elif not latched:
                latched = True
                first_holds[verdict["reason"]] += 1
            else:
                post_latch += 1
        never_latched += 0 if latched else 1
    total = passing + sum(first_holds.values()) + post_latch
    return {
        "captures": len(by_capture),
        "cycles": total,
        "passing_before_any_hold": passing,
        "first_holds": sum(first_holds.values()),
        "first_hold_reasons": dict(sorted(first_holds.items())),
        "cycles_with_a_decision": passing + sum(first_holds.values()),
        "cycles_after_a_latch": post_latch,
        "captures_that_never_held": never_latched,
    }


def cycles_digest(path):
    """The parity run's digest of a cycles file: sha256 over each decompressed line without its newline."""
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            digest.update(line.rstrip("\n").encode())
    return digest.hexdigest()


def read_cycles(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield row["capture"], row["python"]


def vision_ages(batch_dir):
    """Every vision_age_s the gate logged, per run. None and NaN (fault cycles) are counted, not dropped."""
    runs = {}
    for run in sorted(Path(batch_dir).glob("run_*")):
        log = run / "ros2_loop_decisions.jsonl"
        if not log.is_file():
            raise AuditError(f"{run} has no ros2_loop_decisions.jsonl")
        ages, missing = Counter(), 0
        with open(log) as handle:
            for line in handle:
                age = json.loads(line).get("vision_age_s")
                if age is None or (isinstance(age, float) and math.isnan(age)):
                    missing += 1
                else:
                    ages[float(age)] += 1
        runs[run.name] = {"decisions": sum(ages.values()) + missing, "ages_s": dict(ages), "without_age": missing}
    if not runs:
        raise AuditError(f"no runs under {batch_dir}")
    return runs


def delay_mechanism(batch_dir, step_mm=4.0, steps_per_frame=6):
    """Per run: delay, approach length, median tool motion per approach frame, cut frame, and the post-hoc model."""
    rows = []
    for run in sorted(Path(batch_dir).glob("run_*")):
        frames = json.loads((run / "frames.json").read_text())["frames"]
        delay = int(json.loads((run / "report.json").read_text())["ros2_loop"]["delay_steps"])
        positions = [frame["tool_position_m"] for frame in frames]
        approach = [i for i, f in enumerate(frames) if f["phase"] == "vision_approach" and i + 1 < len(frames)]
        steps = sorted(1000.0 * math.dist(positions[i], positions[i + 1]) for i in approach)
        middle = len(steps) // 2
        median = None
        if steps:
            median = steps[middle] if len(steps) % 2 else (steps[middle - 1] + steps[middle]) / 2
        cut = next(
            (i for i, f in enumerate(frames) if ((f.get("live_vision") or {}).get("cut") or {}).get("detach_event")),
            None,
        )
        rows.append(
            {
                "run": run.name,
                "delay_steps": delay,
                "delay_ms": round(delay * 1000.0 / 60.0, 1),
                "approach_frames": len(approach),
                "median_approach_mm_per_frame": None if median is None else round(median, 3),
                "post_hoc_model_mm_per_frame": round(step_mm * steps_per_frame / (steps_per_frame + delay), 3),
                "cut_frame": cut,
                "frames": len(frames),
            }
        )
    return rows


def revision():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    return head.stdout.strip(), dirty.stdout.strip() == ""


def build(root=ROOT, share=SHARE):
    parity = json.loads((root / PARITY_EVIDENCE).read_text())
    cycles_file = share / parity["provenance"]["cycles_file"].split(" ")[0]
    if not cycles_file.is_file():
        raise AuditError(f"{cycles_file} is missing")
    digest = cycles_digest(cycles_file)
    if digest != parity["cycles_sha256"]:
        raise AuditError(f"{cycles_file} digest {digest} is not the parity run's {parity['cycles_sha256']}")
    breakdown = parity_breakdown(read_cycles(cycles_file))
    totals = parity["parity"]["totals"]
    if breakdown["cycles"] != totals["cycles"] or breakdown["captures"] != totals["captures"]:
        raise AuditError(f"breakdown covers {breakdown['cycles']} cycles, the evidence {totals['cycles']}")
    if breakdown["first_holds"] + breakdown["cycles_after_a_latch"] != totals["python_holds"]:
        raise AuditError("first holds plus post-latch cycles do not equal the evidence's hold count")

    verdicts = json.loads((root / LIVE_LOOP_VERDICTS).read_text())
    batch = verdicts["provenance"]["batch"].split(" ")[0]
    runs = vision_ages(root / batch)
    every_age = Counter()
    for run in runs.values():
        every_age.update(run["ages_s"])
    over_limit = sum(count for age, count in every_age.items() if age > VISION_MAX_AGE_S)
    mechanism = delay_mechanism(root / batch)
    return {
        "schema_version": 1,
        "tool": "tools/audit_gate_evidence.py",
        "scope": (
            "Addendum to the November 5 gate evidence. It recounts recorded data and changes no registered verdict. "
            "Simulation only."
        ),
        "parity_composition": {
            "source": PARITY_EVIDENCE,
            "cycles_file": str(cycles_file.relative_to(share)),
            "cycles_sha256": digest,
            "cycles_sha256_convention": "sha256 over each decompressed line without its newline, as the parity run",
            **breakdown,
            "reading": (
                "A hold latches for the run, so cycles after a run's first hold were already decided. The cycles "
                "where the gate still had a decision to make are the passing cycles plus each run's first hold."
            ),
        },
        "live_loop_vision_age": {
            "source": LIVE_LOOP_VERDICTS,
            "batch": batch,
            "runs": len(runs),
            "decisions": sum(run["decisions"] for run in runs.values()),
            "decisions_without_age": sum(run["without_age"] for run in runs.values()),
            "distinct_ages_s": sorted(every_age),
            "decisions_over_limit": over_limit,
            "limit_s": VISION_MAX_AGE_S,
            "per_run": runs,
            "reading": (
                "The controller stamps its vision with the frame's capture time, so the age the gate checks is zero. "
                "The registered delay held back arm commands; images reached the controller undelayed. The 0.25 s "
                "freshness check is covered by injected faults (vision_stale_300ms, vision_from_future_50ms) and has "
                "not fired in closed loop."
            ),
        },
        "delay_mechanism": {
            "source": LIVE_LOOP_VERDICTS,
            "batch": batch,
            "runs": mechanism,
            "post_hoc_model": (
                "Fitted after the registered result and not registered: each command is the measured pose plus a step "
                "capped at 4 mm. With d of the 6 physics steps per frame still applying the previous command, the arm "
                "is short of the last command when the next pose is measured, so progress per frame is 4 x 6 / (6 + d) "
                "mm. It predicts, untested, that the 66.7 and 100 ms runs would pass with a longer episode."
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    code_revision, clean = revision()
    document = build()
    document["provenance"] = {
        "code_revision": code_revision,
        "tracked_tree_clean": clean,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
    }
    output.write_text(json.dumps(document, indent=1) + "\n")
    composition = document["parity_composition"]
    ages = document["live_loop_vision_age"]
    print(
        f"parity: {composition['cycles']} cycles = {composition['passing_before_any_hold']} passing + "
        f"{composition['first_holds']} first holds + {composition['cycles_after_a_latch']} after a latch; "
        f"live loop: {ages['decisions']} decisions in {ages['runs']} runs, ages {ages['distinct_ages_s']}"
    )
    for row in document["delay_mechanism"]["runs"]:
        print(
            f"  {row['run']}: delay {row['delay_ms']} ms, approach {row['approach_frames']} frames at "
            f"{row['median_approach_mm_per_frame']} mm/frame (model {row['post_hoc_model_mm_per_frame']}), "
            f"cut at {row['cut_frame']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
