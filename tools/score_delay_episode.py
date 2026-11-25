"""Score the long-episode delay batch against docs/EVAL_PROTOCOL_DELAY_EPISODE_2026-11-25.md (D0-D3).

On November 5 the 66.7 and 100 ms command-delay runs made the cut every time but failed the return-home check
when the 20 s episode ended. A model fitted afterwards says the delay only slowed the arm, so a 30 s episode should
pass. This scores the six registered runs (3 at 66.7 ms, 3 at 100 ms, 300 frames each) against that prediction.

Reads only what the runs recorded: experiment result, report, frames and sequence grade. The per-run integrity
check (ROS 2 drove the arm on every frame) is the November 5 scorer's R1, imported unchanged. Committed with the
protocol before any registered run exists. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vision_experiment import run_label  # noqa: E402
from score_ros2_live_loop import _first_stop, _load, score_r1  # noqa: E402

PROTOCOL = "docs/EVAL_PROTOCOL_DELAY_EPISODE_2026-11-25.md"
FRAMES = 300
CHECKS = 17
#: Registered per delay (control steps of 1/60 s): the frame the arm reaches home (first frame in phase
#: "complete") and the frame of the cut (detach event). Home = approach + align + closing + retreat frames, with
#: the approach and align lengths measured on November 5 and the retreat as long as the approach.
EXPECTED = {4: {"home_frame": 233, "cut_frame": 120}, 6: {"home_frame": 277, "cut_frame": 142}}
HOME_TOLERANCE_FRAMES = 10
CUT_TOLERANCE_FRAMES = 2
RUNS_PER_DELAY = 3


def first_frame(frames, test):
    return next((frame["index"] for frame in frames or [] if test(frame)), None)


def cut_frame(frames):
    return first_frame(frames, lambda f: bool(((f.get("live_vision") or {}).get("cut") or {}).get("detach_event")))


def home_frame(frames):
    return first_frame(frames, lambda f: f.get("phase") == "complete")


def score_run(row, run_dir):
    result = _load(run_dir / "experiment_result.json", {})
    report = _load(run_dir / "report.json", {})
    frames_doc = _load(run_dir / "frames.json", {})
    frames = frames_doc.get("frames") if isinstance(frames_doc, dict) else frames_doc
    grade = _load(run_dir / "sequence_grade.json", {})
    r1 = score_r1(row, report, frames)
    unscored = []
    if not r1["ok"]:
        unscored.append("ros2_did_not_drive_every_frame: " + ",".join(r1["problems"]))
    if result.get("configuration_matches") is not True:
        unscored.append("configuration_does_not_match_the_plan")
    if len(frames or []) != FRAMES or [f.get("index") for f in frames or []] != list(range(FRAMES)):
        unscored.append(f"capture_is_not_{FRAMES}_frames_in_order")
    delay = row["ros2_loop"]["delay_steps"]
    checks = grade.get("checks") or {}
    return {
        "index": row["index"],
        "run": run_dir.name,
        "delay_steps": delay,
        "delay_ms": row["ros2_loop"]["delay_ms"],
        "scored": not unscored,
        "unscored_reasons": unscored,
        "pass": result.get("ok") is True and len(checks) == CHECKS and all(checks.values()),
        "failed_checks": grade.get("failed_checks"),
        "task_outcome": report.get("task_outcome"),
        "first_stop": _first_stop(report, frames),
        "cut_frame": cut_frame(frames),
        "home_frame": home_frame(frames),
        "expected": EXPECTED.get(delay),
        "final_home_error_m": (grade.get("metrics") or {}).get("final_home_error_m"),
        "r1": r1,
    }


def judge(runs, holds, name):
    """Supported only if every registered run is scored and holds; refuted by any scored run that does not."""
    scored = [run for run in runs if run["scored"]]
    departures = [run["run"] for run in scored if not holds(run)]
    if departures:
        verdict = "refuted"
    elif len(scored) == len(runs) == RUNS_PER_DELAY * len(EXPECTED):
        verdict = "supported"
    else:
        verdict = "not supported: unscored runs"
    return {"prediction": name, "verdict": verdict, "departures": departures, "scored": len(scored), "runs": len(runs)}


def score(batch):
    batch = Path(batch)
    plan = _load(batch / "plan.json")
    if plan.get("protocol") != PROTOCOL or plan.get("frames") != FRAMES:
        raise ValueError(f"{batch} is not a {FRAMES}-frame batch registered to {PROTOCOL}")
    delays = sorted(row["ros2_loop"]["delay_steps"] for row in plan["runs"])
    if delays != sorted(list(EXPECTED) * RUNS_PER_DELAY):
        raise ValueError(f"the plan's delays {delays} are not the registered ones")
    runs = [score_run(row, batch / run_label(row["index"], row)) for row in plan["runs"]]

    def near(value, target, tolerance):
        return value is not None and abs(value - target) <= tolerance

    verdicts = {
        "D0_ros2_drove_the_arm": {
            "verdict": "supported" if all(run["r1"]["ok"] for run in runs) else "refuted",
            "runs_ok": sum(run["r1"]["ok"] for run in runs),
            "runs": len(runs),
        },
        "D1_all_pass_with_a_30_s_episode": judge(runs, lambda r: r["pass"], "every run passes all 17 checks"),
        "D2_home_frame": judge(
            runs,
            lambda r: near(r["home_frame"], r["expected"]["home_frame"], HOME_TOLERANCE_FRAMES),
            f"the arm is home within {HOME_TOLERANCE_FRAMES} frames of 233 (66.7 ms) or 277 (100 ms)",
        ),
        "D3_first_200_frames_unchanged": judge(
            runs,
            lambda r: near(r["cut_frame"], r["expected"]["cut_frame"], CUT_TOLERANCE_FRAMES),
            f"the cut comes within {CUT_TOLERANCE_FRAMES} frames of November 5's (120 at 66.7 ms, 142 at 100 ms)",
        ),
    }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "batch": batch.name,
        "code_revision": plan.get("code_revision"),
        "registered": {
            "expected": EXPECTED,
            "home_tolerance_frames": HOME_TOLERANCE_FRAMES,
            "cut_tolerance_frames": CUT_TOLERANCE_FRAMES,
            "frames": FRAMES,
        },
        "verdicts": verdicts,
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = score(args.batch)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: v["verdict"] for name, v in result["verdicts"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
