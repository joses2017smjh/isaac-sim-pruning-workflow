#!/usr/bin/env python3
"""Score the depth-aware appearance closed-loop test (protocol of October 1) against C1-C6.

Outcomes come from the per-batch evidence that ``aggregate_eval.py`` wrote (the grader's classes, never
re-graded here). Each file must be the registered batch's grade under this protocol, and the 10 registered runs
must all be present; otherwise nothing is scored. Failed, ungraded and refused runs stay in every table.

Everything a clause names comes from each run's own records:
- the depth event the live tracker attached to every grey-check failure
  (``live_vision.measurement.depth_appearance``);
- the run's stop: the first frame whose cut phase is 'stopped';
- the tracker's reason at that stop;
- the arm's readback in the report.

C6 replays each comparable run's own frames offline through the strict arm of ``tools/replay_depth_appearance.py``,
after checking that the replay code equals the code the runs were frozen with. Each run is labelled with its GPU
model.

Where the protocol leaves a detail open it is fixed in ``INTERPRETATIONS`` and recorded in the output, with the
scorer's construction notes. Results of this arm change the 0.35 appearance gate's rule and are never pooled with
unchanged-gate results.
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
from score_planned_approach import gpu_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md"
EVIDENCE_DIR = "docs/evidence/depth_loop_2026-10-01"
VISION_ROBUSTNESS = "artifacts/vision_robustness"
GRADER = "tools/validate_vision_sequence.py:grade_sequence"
CHECKS = 17
LABEL = (
    "depth-aware appearance check D_strict + J; changes the 0.35 appearance gate's rule; simulator depth (RTX "
    "optical-Z ground truth, noise-free, perfectly registered); not a sensor"
)
#: The registered batches (protocol, Runs): scene kind, strategy and targets in plan order.
REGISTRY = {
    "depth-loop-eve-r1-20261001": ("evening", "baseline_depth_appearance", (14944, 15004)),
    "depth-loop-mor-r1-20261001": ("morning", "baseline_depth_appearance", (15004,)),
    "depth-loop-eve-r2-20261001": ("evening", "baseline_depth_appearance", (14944, 15004)),
    "depth-loop-mor-r2-20261001": ("morning", "baseline_depth_appearance", (15004,)),
    "depth-loop-src-20261001": ("source", "baseline_depth_appearance", (14944, 15004)),
    "depth-loop-ctl-19444-20261001": ("control", "planned_pose_depth_appearance", (19444,)),
    "depth-loop-ctl-12142-20261001": ("control", "tool_axis_standoff_depth_appearance", (12142,)),
}
DAYLIGHT = {"evening": "evening", "morning": "morning", "source": "source", "control": "source"}
EXPECTED_RUNS = 10
#: Registered shadow windows (inclusive frame indexes) of the low-sun scenes.
SHADOW_WINDOW = {"evening": (56, 67), "morning": (72, 75)}
MIN_CONFIDENCE = 0.15
APPEARANCE_LOSS = "appearance_changed_or_occluded"
LOW_CONFIDENCE = "low_confidence"
#: C4's registered stop mechanisms: the condition a rejecting event at the control's stop names.
CONTROL_MECHANISM = {
    19444: ("J_jaw_silhouette", "19444 should stop when the open jaw reaches the patch (recorded 68 / 68 / 70)"),
    12142: ("3_near_fraction", "12142 should stop at the wire (recorded 29)"),
}
#: Code the runs were frozen with that C6 executes (besides the depth tool's own SOURCE_FILES).
FROZEN_EXTRA = (
    "source/isaaclab_pruning/isaaclab_pruning/sim/vision_demo_controller.py",
    "hpc/inner/render_pruning_workflow.py",
    "tools/check_depth_loop_c0.py",
)
#: This scorer was amended after the batches were frozen, so it is the one repository module not compared.
SELF = "tools/score_depth_loop.py"
CONSTRUCTION_NOTES = [
    "The scorer was committed at 1bd1ea1 before any depth-loop run was submitted.",
    "It was amended after an independent blind review (15 confirmed findings), before any depth-loop grade file, "
    "run directory or log was opened by anyone; the amendment is in this file's history and code_revision.",
    "It was amended a second time (8acbf19) to close five gaps the amendment's blind verifier found, again before "
    "any grade file, run directory or log was opened.",
    "Before the amendments the main session and the reviewer had seen only the runs' final Slurm states and elapsed "
    "times (four FAILED exit 1: evening 14944 r1 and r2 and both controls; six COMPLETED; all on cn-gpu5).",
    "Those states disclosed outcomes. Each task ran tools/run_vision_experiment.py, whose execute() exits 0 only when "
    "the capture completed, the unchanged grader passed all 17 checks (grade_sequence ok) and configuration_matches "
    "is true. So COMPLETED meant that the four 15004 low-sun runs and both source-light runs passed 17/17 with "
    "configuration_matches true, and exit 1 meant that the two evening 14944 runs and both controls did not. An "
    "earlier version of this note said 'A Slurm state is not a grade', which understated this; the correction "
    "changed no scoring code. Outcomes here are still read only from the per-batch grade files.",
    "What the states bear on. They revealed C2's pass count (4 of 4) and that no 15004 run stopped anywhere, so none "
    "stopped on appearance in its window; C5's outcome (both source runs passed); C4's pass clause (neither control "
    "passed); and that neither evening 14944 run passed, which C3 predicts, without saying where or why it stopped. "
    "A rejected grey failure in a shadow window would most likely have stopped a 15004 run, so the four passes also "
    "suggest that any grey failure in those runs' windows was accepted (C1 for 4 of its 6 runs). They did not reveal "
    "C3's stop frame or reason, C4's depth clause, C1 for the evening 14944 runs, or C6. No clause reads any of "
    "this.",
]
INTERPRETATIONS = [
    "The grade files must be exactly the 7 registered batches (any other depth-loop grade file refuses), each "
    "graded under this protocol by tools/validate_vision_sequence.py with 17 checks, holding the registered "
    "targets, light and strategy in plan order under their registered run directories (10 runs in all); otherwise "
    "nothing is scored.",
    "A run is scored when it was graded (status 'graded', 17 checks), its capture completed (report stage "
    "'complete', and the runner's experiment_result.json with configuration_matches true), and its report shows "
    "the arm on with the registered constants and the jaw self-mask and closure hold off. Any other run is listed "
    "in every table and contributes nothing: it is neither a pass nor a failure.",
    "A grey-check failure is a frame whose live measurement carries a depth_appearance event (the tracker attaches "
    "one exactly when the grey NCC fails 0.35). A run's stop is its first frame whose recorded cut phase is "
    "'stopped'; the grade files' stop_frame is the next frame (the first 'stopped_failure' command) and is not "
    "used. 'In the shadow window' means the frame index lies in the registered inclusive window.",
    "Only grey failures up to and including the stop are judged (the frames C6 verifies); later ones are listed as "
    "post_stop and never decide a clause.",
    "C1 judges the first grey failure inside each low-sun run's window. It is supported only when all 6 runs have "
    "one and every one is accepted; a run without one is untested.",
    "C2 counts a pass only from the grader's class 'pass' (all 17 checks). It is supported when all 4 runs are "
    "scored, at least 3 pass and none stops on appearance in its window. It is refuted when at least 2 stop in "
    "the window for any reason, or when even counting every unscored run as a pass fewer than 2 would pass.",
    "C3 judges the first grey failure in each evening 14944 run's window. Accepted, then a stop at that frame on "
    "low_confidence with strict confidence below 0.15, holds; accepted and no stop at or before it refutes "
    "('continued past the event'); rejected with the stop at that frame refutes; any other case is labelled and "
    "neither holds nor refutes.",
    "C4 judges each scored control: a graded pass refutes; any accepted grey failure refutes; a control without a "
    "grey failure leaves the second clause untested. The registered expectations (19444 stops when the open jaw "
    "reaches the patch, 12142 at the wire) are one clause per control that can only hold: the event at the stop "
    "is rejected naming J_jaw_silhouette (19444) or 3_near_fraction (12142). The recorded stop frames are context, "
    "not thresholds.",
    "C5 is refuted only by a scored source run that fails a check. An unscored run contributes nothing: one "
    "unscored run leaves C5 at most partly supported, and two leave it untested.",
    "C6 compares each scored run whose recording is complete with the offline strict arm on its own frames, "
    "through its stop: the strict arm must equal the recording (state, reason and pixel exactly; pixels are "
    "float32 in both trackers, so any cross-CPU difference exceeds 1e-6 anyway; correlation within 1e-6), and the "
    "live and offline depth events, preview included, must have the same frames, the same keys and equal values "
    "(floats within 1e-6, as the C0 gate). A run the offline tool cannot replay (an unscored run, frame indexes "
    "not contiguous, a preview or frame file missing) is not comparable: listed, and C6 cannot "
    "be supported, but it is not refuted. Any other replay error aborts the scorer.",
    "A prediction is 'untested' when no run contributed evidence to any of its clauses, 'refuted' when any clause "
    "refutes, 'supported' when every clause holds, and 'partly supported' otherwise.",
]


class InputError(ValueError):
    """An input this scorer refuses (the file and field are in the message)."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def judged(text, observed, holds, refutes, tested):
    return {"clause": text, "observed": observed, "holds": bool(holds), "refutes": bool(refutes), "tested": tested}


def verdict(clauses):
    if any(c["refutes"] for c in clauses):
        return "refuted"
    if not any(c["tested"] for c in clauses):
        return "untested"
    return "supported" if all(c["holds"] for c in clauses) else "partly supported"


def prediction(pid, text, clauses):
    return {"id": pid, "prediction": text, "clauses": clauses, "verdict": verdict(clauses)}


# ------------------------------------------------------------------------------------------------ inputs
def frame_facts(frames):
    """Per frame: index, cut phase, stop reason, the tracker's state and reason, and the depth event if any."""
    facts = []
    for frame in frames:
        live = frame.get("live_vision") or {}
        measurement = live.get("measurement") or {}
        cut = live.get("cut") or {}
        event = measurement.get("depth_appearance")
        if event is not None and not isinstance(event.get("accept"), bool):
            raise InputError(f"frame {frame.get('index')}: depth event accept is not a boolean")
        facts.append(
            {
                "index": frame["index"],
                "cut_phase": cut.get("phase"),
                "stopped_reason": cut.get("stopped_reason"),
                "state": measurement.get("state"),
                "reason": measurement.get("reason"),
                "event": None
                if event is None
                else {
                    key: event.get(key)
                    for key in (
                        "accept",
                        "failed_conditions",
                        "not_evaluated_conditions",
                        "reasons",
                        "strict_confidence",
                        "decision",
                        "ended_by",
                        "ncc",
                    )
                },
            }
        )
    return facts


def configuration(run, report):
    """Whether the capture ran the registered arm: readback, constants, the jaw arms off, configuration_matches."""
    from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance

    block = report.get("depth_appearance") or {}
    registered = json.loads(json.dumps(registered_depth_appearance()))
    result_path = Path(run) / "experiment_result.json"
    matches = json.loads(result_path.read_text()).get("configuration_matches") if result_path.is_file() else None
    checks = {
        "capture_complete": report.get("stage") == "complete",
        "experiment_result_recorded": result_path.is_file(),
        "enabled": block.get("enabled") is True,
        "constants_registered": block.get("constants") == registered,
        "jaw_arms_off": all(
            (report.get(arm) or {}).get("enabled") is not True for arm in ("jaw_self_mask", "closure_hold")
        ),
        "configuration_matches": matches,
    }
    checks["ok"] = (
        checks["capture_complete"]
        and checks["enabled"]
        and checks["constants_registered"]
        and checks["jaw_arms_off"]
        and matches is True
    )
    return checks


def run_record(root, batch, row):
    """The fields C1-C6 need for one run, from its own records; an unscored run keeps its grade row."""
    kind, _, _ = REGISTRY[batch]
    run = Path(root) / VISION_ROBUSTNESS / batch / row["run_directory"]
    graded = row.get("status") == "graded" and row.get("checks_total") == CHECKS
    record = {
        "batch": batch,
        "kind": kind,
        "run_directory": row["run_directory"],
        "target": int(row["component_first_vertex"]),
        "daylight": row["daylight"],
        "status": row.get("status"),
        "outcome": row["outcome"],
        "checks": f"{row.get('checks_passed')}/{row.get('checks_total')}",
        "graded": graded,
        "captured": False,
        "configuration": None,
        "scored": False,
        "unscored_reason": None,
        "stop": None,
        "grey_failures": [],
        "post_stop_grey_failures": [],
        "events": {},
        "post_stop_events": {},
        "preview_event": None,
        "node": None,
        "gpu_model": None,
        "report_sha256": None,
        "frames_sha256": None,
        "path": str(run),
    }
    if not graded:
        record["unscored_reason"] = f"not graded ({row.get('status')}, checks {record['checks']})"
        return record
    report_path, frames_path = run / "report.json", run / "frames.json"
    if not (report_path.is_file() and frames_path.is_file()):
        record["unscored_reason"] = "graded but its report or frames are missing"
        return record
    report = json.loads(report_path.read_text())
    frames = json.loads(frames_path.read_text())["frames"]
    facts = frame_facts(frames)
    stop = next((f for f in facts if f["cut_phase"] == "stopped"), None)
    stop_frame = None if stop is None else stop["index"]
    events = {f["index"]: f["event"] for f in facts if f["event"] is not None}
    through = [i for i in sorted(events) if stop_frame is None or i <= stop_frame]
    after = [i for i in sorted(events) if stop_frame is not None and i > stop_frame]
    preview = ((report.get("initial_live_vision") or {}).get("measurement") or {}).get("depth_appearance")
    record.update(
        {
            "captured": True,
            "configuration": configuration(run, report),
            "stop": None
            if stop is None
            else {"frame": stop["index"], "stopped_reason": stop["stopped_reason"], "tracker_reason": stop["reason"]},
            "grey_failures": through,
            "post_stop_grey_failures": after,
            "events": {i: events[i] for i in through},
            "post_stop_events": {i: events[i] for i in after},
            "preview_event": None if preview is None else {"accept": preview.get("accept")},
            "node": report.get("node"),
            "gpu_model": gpu_model(report.get("node")),
            "report_sha256": sha256(report_path),
            "frames_sha256": sha256(frames_path),
        }
    )
    checks = record["configuration"]
    record["scored"] = checks["ok"]
    if not checks["capture_complete"] or not checks["experiment_result_recorded"]:
        record["unscored_reason"] = "capture incomplete (killed or crashed before the runner finished)"
    elif not record["scored"]:
        record["unscored_reason"] = "configuration refused"
    return record


def expected_run_directory(index, row, strategy):
    """run_vision_experiment.run_label for a registered row."""
    return (
        f"run_{index:02d}_{row['daylight']}_tree{row['target_tree_index']}_v{row['component_first_vertex']}_{strategy}"
    )


def load(root):
    """The registered grade files, checked, and one record per registered run; refuses anything else."""
    try:
        return _load(Path(root))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(f"malformed grade input: {error!r}") from error


def _load(root):
    records, files = [], {}
    extra = sorted(path.name for path in (root / EVIDENCE_DIR).glob("depth-loop-*.json") if path.stem not in REGISTRY)
    if extra:
        raise InputError(f"{EVIDENCE_DIR}: grade files outside the registered batches: {extra}")
    for batch, (kind, strategy, targets) in REGISTRY.items():
        path = root / EVIDENCE_DIR / f"{batch}.json"
        if not path.is_file():
            raise InputError(f"{path}: the registered batch's grade file is missing")
        document = json.loads(path.read_text())
        files[str(path.relative_to(root))] = {
            "sha256": sha256(path),
            "protocol": document.get("protocol"),
            "grader": document.get("grader"),
        }
        if (
            document.get("schema_version") != 1
            or document.get("protocol") != PROTOCOL
            or document.get("grader") != GRADER
        ):
            raise InputError(f"{path}: schema_version, protocol or grader is not the registered one")
        batches = document.get("batches") or []
        if len(batches) != 1 or str(batches[0]["batch_dir"]).rstrip("/") != f"{VISION_ROBUSTNESS}/{batch}":
            raise InputError(f"{path}: must grade exactly {VISION_ROBUSTNESS}/{batch}")
        rows = document.get("runs") or []
        wanted = [(target, DAYLIGHT[kind], strategy) for target in targets]
        found = [(int(r["component_first_vertex"]), r.get("daylight"), r.get("strategy")) for r in rows]
        if found != wanted:
            raise InputError(f"{path}: rows {found} are not the registered {wanted}")
        for index, row in enumerate(rows):
            if row["run_directory"] != expected_run_directory(index, row, strategy):
                raise InputError(f"{path}: run directory {row['run_directory']} is not the registered run {index}")
            if row.get("status") == "graded" and row.get("checks_total") != CHECKS:
                raise InputError(f"{path}: {row['run_directory']} was graded with {row.get('checks_total')} checks")
        records += [run_record(root, batch, row) for row in rows]
    if len(records) != EXPECTED_RUNS:
        raise InputError(f"{len(records)} runs, registered {EXPECTED_RUNS}")
    return records, files


def check_frozen_sources(root):
    """The code C6 executes must equal the code every batch was frozen with (plan.json source_sha256).

    Covers the depth tool's SOURCE_FILES, the live controller and runner, and every repository module that the
    replay imports (found in sys.modules after importing it), except this scorer itself.
    """
    import check_depth_loop_c0  # noqa: F401 - imported for its module set
    import replay_depth_appearance as depth_tool

    root = Path(root).resolve()
    imported = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file or not Path(file).is_absolute():
            continue  # extension and namespace modules carry relative or no file names
        try:
            relative = Path(file).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if (
            relative.endswith(".py")
            and relative.split("/")[0] in ("source", "tools", "hpc")
            and relative != SELF
            and (root / relative).is_file()
        ):
            imported.add(relative)
    paths = sorted(set(depth_tool.SOURCE_FILES) | set(FROZEN_EXTRA) | imported)
    current = {path: sha256(root / path) for path in paths}
    for batch in REGISTRY:
        plan_path = root / VISION_ROBUSTNESS / batch / "plan.json"
        try:
            plan = json.loads(plan_path.read_text())
        except (OSError, ValueError) as error:
            raise InputError(f"{plan_path}: unreadable ({error!r})") from error
        frozen = plan.get("source_sha256") or {}
        differing = sorted(path for path in paths if frozen.get(path) != current[path])
        if differing:
            raise InputError(f"{batch}/plan.json: the checkout differs from the frozen code in {differing}")
    return current


# ------------------------------------------------------------------------------------------------ predictions
def name(record):
    return f"{record['batch']}/{record['run_directory']}"


def first_in_window(record):
    """The first judged grey failure inside the run's registered shadow window, as (frame, event), or None."""
    first, last = SHADOW_WINDOW[record["kind"]]
    frame = next((index for index in record["grey_failures"] if first <= index <= last), None)
    return None if frame is None else (frame, record["events"][frame])


def stops_in_window(record):
    first, last = SHADOW_WINDOW[record["kind"]]
    return record["stop"] is not None and first <= record["stop"]["frame"] <= last


def unscored(runs):
    return {name(r): r["unscored_reason"] for r in runs if not r["scored"]}


def score_c1(records):
    low_sun = [r for r in records if r["kind"] in SHADOW_WINDOW]
    observed, rejected, untested, tested = {}, [], [], 0
    for record in low_sun:
        if not record["scored"]:
            untested.append(name(record))
            observed[name(record)] = {"untested": record["unscored_reason"]}
            continue
        first = first_in_window(record)
        if first is None:
            untested.append(name(record))
            observed[name(record)] = {"untested": "no grey failure in the window", "stop": record["stop"]}
            continue
        tested += 1
        frame, event = first
        observed[name(record)] = {"frame": frame, **event, "stop": record["stop"]}
        if event["accept"] is False:
            rejected.append(name(record))
    return prediction(
        "C1",
        "In each of the 6 low-sun runs the depth test accepts the first grey-check failure inside the shadow window.",
        [
            judged(
                "the first in-window grey failure is accepted in all 6 low-sun runs",
                {"first_in_window": observed, "rejected": rejected, "untested": untested},
                len(low_sun) == 6 and tested == 6 and not rejected,
                bool(rejected),
                tested,
            )
        ],
    )


def score_c2(records):
    runs = [r for r in records if r["kind"] in SHADOW_WINDOW and r["target"] == 15004]
    scored = [r for r in runs if r["scored"]]
    passes = sum(r["outcome"] == "pass" for r in scored)
    not_scored = len(runs) - len(scored)
    in_window = [name(r) for r in scored if stops_in_window(r)]
    on_appearance = [name(r) for r in scored if stops_in_window(r) and r["stop"]["tracker_reason"] == APPEARANCE_LOSS]
    return prediction(
        "C2",
        "At least 3 of the 4 15004 runs pass all 17 checks, and none stops on appearance in its shadow window.",
        [
            judged(
                "15004: passes, and stops in the shadow window",
                {
                    "runs": {
                        name(r): {"outcome": r["outcome"], "checks": r["checks"], "stop": r["stop"]} for r in scored
                    },
                    "passes": f"{passes} of {len(runs)}",
                    "stops_in_window": in_window,
                    "appearance_stops_in_window": on_appearance,
                    "unscored": unscored(runs),
                },
                len(runs) == 4 and len(scored) == 4 and passes >= 3 and not on_appearance,
                len(in_window) >= 2 or passes + not_scored < 2,
                len(scored),
            )
        ],
    )


def score_c3(records):
    runs = [r for r in records if r["kind"] == "evening" and r["target"] == 14944]
    observed, held, refutes, tested = {}, 0, False, 0
    for record in runs:
        if not record["scored"]:
            observed[name(record)] = {"untested": record["unscored_reason"]}
            continue
        first, stop = first_in_window(record), record["stop"]
        if first is None:
            observed[name(record)] = {"untested": "no grey failure in the window", "stop": stop}
            continue
        tested += 1
        frame, event = first
        entry = {"first_in_window": frame, "accept": event["accept"], "strict_confidence": event["strict_confidence"]}
        entry["stop"] = stop
        if event["accept"] is False:
            if stop is not None and stop["frame"] == frame:
                refutes = True
                entry["case"] = "stopped at the event with the depth test rejecting"
            else:
                entry["case"] = "rejected without a stop at the event"
        elif stop is None or stop["frame"] > frame:
            refutes = True
            entry["case"] = "continued past the event"
        elif stop["frame"] == frame:
            low = event["strict_confidence"] is not None and event["strict_confidence"] < MIN_CONFIDENCE
            if stop["tracker_reason"] == LOW_CONFIDENCE and low:
                held += 1
                entry["case"] = "low-confidence stop at the event"
            else:
                entry["case"] = f"stopped at the event on {stop['tracker_reason']}"
        else:
            entry["case"] = "stopped before the event"
        observed[name(record)] = entry
    return prediction(
        "C3",
        "Both evening 14944 runs stop at the accepted shadow event on low_confidence (strict confidence below 0.15).",
        [
            judged(
                "evening 14944: low-confidence stop at the accepted event",
                observed,
                len(runs) == 2 and held == 2,
                refutes,
                tested,
            )
        ],
    )


def score_c4(records):
    controls = [r for r in records if r["kind"] == "control"]
    scored = [r for r in controls if r["scored"]]
    passed = [name(r) for r in scored if r["outcome"] == "pass"]
    accepted = {name(r): [f for f in r["grey_failures"] if r["events"][f]["accept"] is True] for r in scored}
    accepted = {key: value for key, value in accepted.items() if value}
    with_failures = [r for r in scored if r["grey_failures"]]
    table = {
        name(r): {
            "outcome": r["outcome"],
            "stop": r["stop"],
            "grey_failures": r["events"],
            "post_stop": r["post_stop_events"],
        }
        for r in scored
    }
    clauses = [
        judged(
            "neither control passes",
            {"runs": table, "passed": passed, "unscored": unscored(controls)},
            len(controls) == 2 and len(scored) == 2 and not passed,
            bool(passed),
            len(scored),
        ),
        judged(
            "the depth test rejects every grey failure in both controls",
            {
                "accepted": accepted,
                "untested_no_grey_failure": [name(r) for r in scored if not r["grey_failures"]],
                "unscored": unscored(controls),
            },
            len(controls) == 2 and len(with_failures) == 2 and not accepted,
            bool(accepted),
            len(with_failures),
        ),
    ]
    for target, (condition, context) in CONTROL_MECHANISM.items():
        record = next((r for r in controls if r["target"] == target), None)
        event = None
        if record is not None and record["scored"] and record["stop"] is not None:
            event = record["events"].get(record["stop"]["frame"])
        holds = event is not None and event["accept"] is False and condition in (event.get("failed_conditions") or [])
        clauses.append(
            judged(
                f"{target}: the event at its stop is rejected naming {condition} (expectation; cannot refute)",
                {
                    "context": context,
                    "stop": None if record is None else record["stop"],
                    "event_at_stop": event,
                    "unscored": None if record is None else record["unscored_reason"],
                },
                holds,
                False,
                int(event is not None),
            )
        )
    return prediction(
        "C4", "Neither control passes, and the depth test rejects every grey-check failure in both.", clauses
    )


def score_c5(records):
    runs = [r for r in records if r["kind"] == "source"]
    scored = [r for r in runs if r["scored"]]
    failed = [name(r) for r in scored if r["outcome"] != "pass"]
    return prediction(
        "C5",
        "Source-light 14944 and 15004 both pass 17/17.",
        [
            judged(
                "both source-light runs pass all 17 checks",
                {
                    "runs": {name(r): {"outcome": r["outcome"], "checks": r["checks"]} for r in scored},
                    "unscored": unscored(runs),
                    "grey_failures_reported": {
                        name(r): {**r["events"], **{f"post_stop_{k}": v for k, v in r["post_stop_events"].items()}}
                        for r in scored
                        if r["events"] or r["post_stop_events"]
                    },
                },
                len(runs) == 2 and len(scored) == 2 and not failed,
                bool(failed),
                len(scored),
            )
        ],
    )


# ------------------------------------------------------------------------------------------------ C6
def comparability(record):
    """Why the offline tool cannot replay this run, or None (checked here, never by catching its errors)."""
    if not record["scored"]:
        return record["unscored_reason"]
    run = Path(record["path"])
    report = json.loads((run / "report.json").read_text())
    if report.get("stage") != "complete":
        return f"recording not complete (stage {report.get('stage')!r})"
    frames = json.loads((run / "frames.json").read_text())["frames"]
    if [f["index"] for f in frames] != list(range(len(frames))):
        return "frame indexes are not contiguous"
    if not ((run / "preview_wrist.png").is_file() and (run / "preview_depth.npy").is_file()):
        return "preview files missing"
    last = record["stop"]["frame"] if record["stop"] is not None else len(frames) - 1
    missing = [
        i
        for i in range(last + 1)
        if not ((run / f"frames/wrist_{i:05d}.png").is_file() and (run / f"frames/depth_{i:05d}.npy").is_file())
    ]
    return f"frame files missing: {missing[:5]}" if missing else None


def event_differences(offline, live, compare, context):
    """Keys present on one side only, and keys whose values differ, ignoring the offline record's context."""
    if live is None:
        return ["no live event"]
    offline_keys, live_keys = set(offline) - set(context), set(live) - set(context)
    problems = [f"offline only: {k}" for k in sorted(offline_keys - live_keys)]
    problems += [f"live only: {k}" for k in sorted(live_keys - offline_keys)]
    problems += [k for k in sorted(offline_keys & live_keys) if not compare.equal(live[k], offline[k])]
    return problems


def offline_comparison(record):
    """C6 for one comparable run: the offline strict arm on the run's own frames against the live record."""
    import check_depth_loop_c0 as c0
    import replay_depth_appearance as depth_tool

    result = c0.normalized(depth_tool.replay_run(Path(record["path"])))
    strict = result["divergence"]["strict"]
    problems = []
    if not strict["equals_recording"]:
        problems.append(
            {
                "kind": "strict_differs_from_recording",
                "frames": strict["mismatch_frames"],
                "preview_fields": strict["preview_mismatch_fields"],
            }
        )
    run = Path(record["path"])
    report = json.loads((run / "report.json").read_text())
    frames = json.loads((run / "frames.json").read_text())["frames"]
    live = {
        str(f["index"]): c0.flat_event(((f.get("live_vision") or {}).get("measurement") or {}).get("depth_appearance"))
        for f in frames[: result["replayed_through_frame"] + 1]
    }
    live["preview"] = c0.flat_event(
        ((report.get("initial_live_vision") or {}).get("measurement") or {}).get("depth_appearance")
    )
    live = {frame: event for frame, event in live.items() if event is not None}
    offline = {str(event["frame"]): event for event in result["events"]["strict"]}
    if sorted(live) != sorted(offline):
        problems.append({"kind": "event_frames", "live": sorted(live), "offline": sorted(offline)})
    for frame, event in offline.items():
        differing = event_differences(event, live.get(frame), c0.COMPARE, c0.EVENT_CONTEXT)
        if differing:
            problems.append({"kind": "event", "frame": frame, "fields": differing})
    return {
        "status": "mismatch" if problems else "equal",
        "problems": problems,
        "replayed_through_frame": result["replayed_through_frame"],
        "recording_sha256": result["source_sha256"],
        "max_strict_correlation_difference": strict["max_abs_patch_correlation_difference"],
    }


def score_c6(records, comparisons):
    mismatched = {key: value for key, value in comparisons.items() if value["status"] == "mismatch"}
    not_comparable = {key: value["why"] for key, value in comparisons.items() if value["status"] == "not_comparable"}
    compared = sum(value["status"] in ("equal", "mismatch") for value in comparisons.values())
    return prediction(
        "C6",
        "Every live run equals the offline strict arm replayed on its own frames, event for event.",
        [
            judged(
                "live = offline in every run",
                {"runs_compared": compared, "mismatches": mismatched, "not_comparable": not_comparable},
                len(records) == EXPECTED_RUNS and compared == EXPECTED_RUNS and not mismatched,
                bool(mismatched),
                compared,
            )
        ],
    )


def score(records, comparisons):
    return [
        score_c1(records),
        score_c2(records),
        score_c3(records),
        score_c4(records),
        score_c5(records),
        score_c6(records, comparisons),
    ]


def run_summary(record):
    return {key: value for key, value in record.items() if key != "path"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    import cv2

    cv2.setNumThreads(1)
    try:
        records, files = load(args.root)
        frozen = check_frozen_sources(args.root)
    except InputError as error:
        parser.error(str(error))
    import check_depth_loop_c0 as c0
    import replay_depth_appearance as depth_tool

    comparisons = {}
    for record in records:
        why = comparability(record)
        comparisons[name(record)] = (
            {"status": "not_comparable", "why": why} if why is not None else offline_comparison(record)
        )
    head = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain", "--untracked-files=no"])
    )
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the depth-aware appearance check's closed-loop test (C1-C6): 10 simulator runs with "
            "D_strict + J in the live controller, the jaw surrogate casting its shadow. Simulator renders and depth; "
            "two targets and two low suns; results never pooled with unchanged-gate runs."
        ),
        "label": LABEL,
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "construction_notes": CONSTRUCTION_NOTES,
        "grade_files": files,
        "frozen_sources_sha256": frozen,
        "offline_tool_provenance": depth_tool._provenance(),
        "float_tolerance": c0.FLOAT_TOLERANCE,
        "max_event_float_difference_accepted": c0.COMPARE.max_float_difference,
        "max_strict_correlation_difference": max(
            (c.get("max_strict_correlation_difference") or 0.0 for c in comparisons.values()), default=0.0
        ),
        "interpretations": INTERPRETATIONS,
        "runs": [run_summary(record) for record in records],
        "c6_comparisons": comparisons,
        "predictions": score(records, comparisons),
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
