#!/usr/bin/env python3
"""Score the agreement-arm closed-loop test (protocol of October 3) against A1-A5.

Outcomes come from the per-batch evidence that ``aggregate_eval.py --protocol`` wrote (the grader's classes, never
re-graded here). Each file must be the registered batch's grade under this protocol, and the 6 registered runs must
all be present; otherwise nothing is scored. Failed, ungraded and refused runs stay in every table.

Everything a clause names comes from each run's own records:
- the depth event the live tracker attached to every grey-check failure
  (``live_vision.measurement.depth_appearance``);
- the run's stop: the first frame whose cut phase is 'stopped';
- the tracker's state and reason at that stop;
- the frames during closure (cut phase 'closing' on the frame or on the frame before it);
- the arm's readback in the report.

A5 replays each comparable run's own frames offline through the agreement arm of
``tools/replay_depth_appearance.py``, after checking that the replay code equals the code the runs were frozen with.
Each run is labelled with its GPU model. A0 (the pre-submission C0 gate) is not scored here.

The scorer scores only its own checkout, and only when it is tracked and identical to HEAD; its hash goes into the
output with each batch's frozen copy of it.

Where the protocol leaves a detail open it is fixed in ``INTERPRETATIONS`` and recorded in the output, with the
scorer's construction notes. Results of this arm change the confidence gate's input and are reported apart from
D_strict + J and from every unchanged-gate result, never pooled.
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
from run_vision_experiment import run_label  # noqa: E402
from score_planned_approach import gpu_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_AGREEMENT_ARM_CLOSED_LOOP_2026-10-03.md"
EVIDENCE_DIR = "docs/evidence/agreement_loop_2026-10-03"
VISION_ROBUSTNESS = "artifacts/vision_robustness"
GRADER = "tools/validate_vision_sequence.py:grade_sequence"
CHECKS = 17
ARM = "agreement"
LABEL = (
    "agreement arm of the depth-aware appearance check (+ J): on acceptance the 0.35 gate passes and the "
    "depth-agreement fraction replaces the NCC in the confidence, which changes the confidence gate's input; "
    "simulator depth (RTX optical-Z ground truth, noise-free, perfectly registered); not a sensor"
)


def _registered(kind, daylight, tree, target, strategy, run_directory):
    return {
        "kind": kind,
        "daylight": daylight,
        "target_tree_index": tree,
        "target": target,
        "strategy": strategy,
        "run_directory": run_directory,
    }


_EVENING = ("evening", "evening", 1, 14944, "baseline_depth_agreement")
#: The registered batches (protocol, Runs): one run each, with its light, target, strategy and run directory
#: (tools/run_vision_experiment.run_label of plan row 0).
REGISTRY = {
    **{
        f"agree-eve-r{repeat}-20261003": _registered(*_EVENING, "run_00_evening_tree1_v14944_baseline_depth_agreement")
        for repeat in (1, 2, 3, 4)
    },
    "agree-ctl-19444-20261003": _registered(
        "control",
        "source",
        1,
        19444,
        "planned_pose_depth_agreement",
        "run_00_source_tree1_v19444_planned_pose_depth_agreement",
    ),
    "agree-ctl-12142-20261003": _registered(
        "control",
        "source",
        0,
        12142,
        "tool_axis_standoff_depth_agreement",
        "run_00_source_tree0_v12142_tool_axis_standoff_depth_agreement",
    ),
}
EXPECTED_RUNS = 6
EVENING_RUNS = 4
#: A1's and A2's registered window (inclusive frame indexes): the evening 14944 shadow event.
SHADOW_WINDOW = (56, 67)
APPEARANCE_LOSS = "appearance_changed_or_occluded"
LOW_CONFIDENCE = "low_confidence"
CLOSING = "closing"
JAW_CONDITION = "J_jaw_silhouette"
#: A4's registered stop mechanisms: the condition a rejecting event at the control's stop names.
CONTROL_MECHANISM = {
    19444: (JAW_CONDITION, "19444 should stop when the open jaw reaches the patch"),
    12142: ("3_near_fraction", "12142 should stop at the wire"),
}
#: Code the runs were frozen with that A5 executes or that decided the live run (besides the depth tool's own
#: SOURCE_FILES and every repository module the scorer and the replay import).
FROZEN_EXTRA = (
    "source/isaaclab_pruning/isaaclab_pruning/sim/vision_demo_controller.py",
    "hpc/inner/render_pruning_workflow.py",
    "tools/run_vision_experiment.py",
    "tools/check_depth_loop_c0.py",
)
#: This scorer must be tracked and equal to HEAD when it scores (main refuses otherwise). Every batch's frozen code
#: must contain it; it may be amended after the batches are frozen (each amendment disclosed below), so a
#: difference from the frozen copy is reported per batch (amended_since_freeze), never refused.
SELF = "tools/score_agreement_loop.py"
CONSTRUCTION_NOTES = [
    "This scorer was first written, and is committed before submission: before any agreement-loop run was "
    "submitted. When it was first written the checkout held no agreement-loop batch directory under "
    "artifacts/vision_robustness, no grade file under docs/evidence/agreement_loop_2026-10-03, no log of this "
    "experiment, and no design-study or C0 evidence in docs/evidence; its author did not query Slurm: it is written "
    "blind, and no outcome of this experiment could shape it.",
    "It was amended after commits 6cd4e75 and 2cbdd93, following an independent blind review "
    "(pruning_work/scorer_review_20261003: 68 one-line mutations of the first version, 26 surviving its tests, 16 "
    "of them judged real). No agreement-loop run, batch directory, grade file or log existed. The amendment adds "
    "self-verification (the scorer must be tracked and equal to HEAD, its hash is in the output, and every batch's "
    "frozen code must contain it, with any post-freeze amendment reported), refuses another --root, turns a hashing "
    "error into a refusal, runs the git calls before any replay, supports A3 only when all 4 evening runs reached "
    "closure, gives a runner result without a configuration check its own unscored reason, and adds tests that "
    "kill the 16 real survivors. The amendment is in this file's history and code_revision.",
    "To check the A5 code path, its author replayed one depth closed-loop recording "
    "(depth-loop-ctl-12142-20261001, strict arm) through offline_comparison; it reported the one expected "
    "difference, the event's arm. No agreement-loop record was read or replayed.",
    "To learn the record schema its author read the depth closed-loop experiment's published records: the 10 "
    "depth-loop-*-20261001 recordings (cut phases, measurements and depth events of frames.json), their grade "
    "files, docs/evidence/depth_loop_verdicts_2026-10-02.json and tools/score_depth_loop.py, whose structure this "
    "scorer keeps. Those are another experiment's results, already cited by this protocol.",
    "The design-study evidence (docs/evidence/agreement_design_replay_2026-10-03.json, "
    "docs/evidence/agreement_design_closure_shadow_2026-10-03.json) and the C0 gate (A0: tools/check_agreement_c0.py, "
    "docs/evidence/agreement_c0_2026-10-03.json) are inputs to the protocol. No clause reads them. The scorer's "
    "author did not open the three evidence files, neither when first writing the scorer (none was in "
    "docs/evidence) nor when amending it after commits 6cd4e75 and 2cbdd93 added them, and did not read those two "
    "commits' messages, which summarize them. It read the committed code of tools/check_agreement_c0.py for its "
    "comparison rules. A0 is a pre-submission gate and is not scored here.",
    "The runner (tools/run_vision_experiment.py execute) exits 0 only when the capture completed, the unchanged "
    "grader passed all 17 checks (grade_sequence ok) and configuration_matches is true, so each task's Slurm state "
    "reveals whether that run passed 17/17 with a matching configuration: A2's pass count and A4's pass clause. "
    "Because this scorer is committed before submission, nothing seen afterwards, Slurm states included, can shape "
    "it. Any amendment made after submission must be added here with everything that had been seen by then.",
]
INTERPRETATIONS = [
    "The grade files must be exactly the 6 registered batches (any other JSON file in "
    "docs/evidence/agreement_loop_2026-10-03 refuses), each graded under this protocol by "
    "tools/validate_vision_sequence.py with 17 checks and naming exactly artifacts/vision_robustness/<batch>, "
    "holding one row with the registered target, light and strategy under the registered run directory (equal to "
    "tools/run_vision_experiment.run_label of that row, run 0); otherwise nothing is scored. The scorer scores only "
    "its own checkout (--root must be it), only when it is tracked and equal to HEAD, and only when every batch's "
    "plan.json froze it and the code the replay executes.",
    "A run is scored when it was graded (status 'graded', 17 checks), its capture completed (report stage "
    "'complete', and the runner's experiment_result.json with configuration_matches true), and its report shows "
    "the depth-aware check on with exactly registered_depth_appearance('agreement') (arm 'agreement') and the jaw "
    "self-mask and closure hold off. A capture with the strict arm's constants, or with the check off, is "
    "'configuration refused'; a runner result without configuration_matches (the runner failed before checking) "
    "is 'runner error: no configuration check'. Any other run is listed in every table and contributes nothing: it "
    "is neither a pass nor a failure.",
    "A grey-check failure is a frame whose live measurement carries a depth_appearance event (the tracker attaches "
    "one exactly when the grey NCC fails 0.35). A run's stop is its first frame whose recorded cut phase is "
    "'stopped'; the tracker state and reason at the stop are that frame's live measurement. The grade files' "
    "stop_frame is the next frame (the first 'stopped_failure' command) and is not used. 'In frames 56-67' means "
    "the frame index lies in that inclusive window.",
    "Only grey failures up to and including the stop are judged (the frames A5 verifies); later ones are listed as "
    "post_stop and never decide a clause.",
    "An accepted event ends in a low-confidence stop when its ended_by or result_reason names low_confidence, or the "
    "frame's tracker reason is low_confidence. A rejected event never does (its loss stands on appearance).",
    "A1 is one clause judged per evening 14944 run. The first grey failure in frames 56-67 is judged: accepted, the "
    "run is tested and holds; rejected refutes. Any accepted event through the stop, in any frame, that ends in a "
    "low-confidence stop refutes. A run without a grey failure in 56-67 is untested (listed). A stop at an accepted "
    "event on another gate is listed and counts against A2, not A1. A1 is supported only when all 4 runs are "
    "scored and all 4 have an accepted first in-window failure, with nothing refuting; untested when no scored run "
    "has an in-window grey failure and nothing refutes; partly supported otherwise.",
    "A2 counts a pass only from the grader's class 'pass' (all 17 checks). A stop on appearance in frames 56-67 is "
    "a stop inside the window whose tracker reason is appearance_changed_or_occluded, or low_confidence at an "
    "accepted depth event on the stop frame. A2 is supported when all 4 evening runs are scored, at least 3 pass "
    "and none stops on appearance in the window; refuted when at least 2 stop on appearance in the window, or when "
    "even counting every unscored run as a pass fewer than 2 would pass; partly supported otherwise. Each failed "
    "run is listed with its stop frame, tracker state and reason, depth events and the grader's failed checks.",
    "A3 is an expectation and cannot refute. A frame is during closure when its recorded cut phase is 'closing' or "
    "the previous frame's is: the cutter was closing when the frame's measurement was taken or consumed. This "
    "keeps a stop during closure (whose own cut phase is 'stopped') and the frame that completes closure (cut "
    "phase 'retreat'), the protocol's 'frames 73-77 (closure)'. Every grey failure during closure through the stop "
    "of a scored evening run is listed with its frame, depth-test decision, failed conditions, J's condition and "
    "both confidences; every closure frame's patch_correlation is listed. A run is tested when it reached closure "
    "through its stop. The expectation holds only when all 4 evening runs are scored, all 4 reached closure, and "
    "every such failure is accepted with J passed; a run that reached closure without a grey failure is reported "
    "as such and keeps it holding. With no run in closure A3 is untested.",
    "A4 judges each scored control: a graded pass refutes; any accepted grey failure refutes; a control without a "
    "grey failure leaves the second clause untested. The registered expectations (19444 stops when the open jaw "
    "reaches the patch, 12142 at the wire) are one clause per control that can only hold: the event at the stop "
    "is rejected naming J_jaw_silhouette (19444) or 3_near_fraction (12142).",
    "A5 compares each scored run whose recording is complete with the offline agreement arm on its own frames, "
    "through its stop: the agreement arm must equal the recording (state, reason and pixel exactly; pixels are "
    "float32 in both trackers, so any cross-CPU difference exceeds 1e-6 anyway; correlation within 1e-6), and the "
    "live and offline depth events, preview included, must have the same frames, the same keys and equal values "
    "(floats within 1e-6, as the C0 gate). The strict arm, replayed in lockstep, is not compared. A run the "
    "offline tool cannot replay (an unscored run, frame indexes not contiguous, a preview or frame file missing) "
    "is not comparable: listed, and A5 cannot be supported, but it is not refuted. Any other replay error aborts "
    "the scorer.",
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
EVENT_KEYS = (
    "accept",
    "failed_conditions",
    "not_evaluated_conditions",
    "reasons",
    "strict_confidence",
    "agreement_confidence",
    "appearance_gate_value",
    "decision",
    "ended_by",
    "result_state",
    "result_reason",
    "arm",
    "ncc",
)


def frame_facts(frames):
    """Per frame: index, cut phase, stop reason, the tracker's measurement, closure progress and the depth event."""
    facts = []
    for frame in frames:
        live = frame.get("live_vision") or {}
        measurement = live.get("measurement") or {}
        cut = live.get("cut") or {}
        event = measurement.get("depth_appearance")
        if event is not None and (not isinstance(event, dict) or not isinstance(event.get("accept"), bool)):
            raise InputError(f"frame {frame.get('index')}: depth event accept is not a boolean")
        fact = {
            "index": frame["index"],
            "cut_phase": cut.get("phase"),
            "stopped_reason": cut.get("stopped_reason"),
            "state": measurement.get("state"),
            "reason": measurement.get("reason"),
            "patch_correlation": measurement.get("patch_correlation"),
            "closure_progress": frame.get("visual_jaw_closure_progress"),
            "event": None,
        }
        if event is not None:
            fact["event"] = {key: event.get(key) for key in EVENT_KEYS}
            fact["event"]["jaw_condition"] = (event.get("conditions") or {}).get(JAW_CONDITION)
            fact["event"]["at_frame"] = {
                "tracker_state": fact["state"],
                "tracker_reason": fact["reason"],
                "cut_phase": fact["cut_phase"],
            }
        facts.append(fact)
    return facts


def closure_frames(facts, stop_frame):
    """Frames during closure through the stop: the frame's or the previous frame's recorded cut phase is 'closing'."""
    closure, previous = [], None
    for fact in facts:
        if stop_frame is not None and fact["index"] > stop_frame:
            break
        if fact["cut_phase"] == CLOSING or previous == CLOSING:
            closure.append(
                {
                    "frame": fact["index"],
                    "cut_phase": fact["cut_phase"],
                    "previous_cut_phase": previous,
                    "closure_progress": fact["closure_progress"],
                    "tracker_state": fact["state"],
                    "tracker_reason": fact["reason"],
                    "patch_correlation": fact["patch_correlation"],
                }
            )
        previous = fact["cut_phase"]
    return closure


def registered_constants():
    """registered_depth_appearance('agreement'), JSON-normalized like a report; it must name the agreement arm."""
    from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance

    registered = json.loads(json.dumps(registered_depth_appearance(ARM)))
    if registered.get("arm") != ARM:
        raise InputError(f"registered_depth_appearance({ARM!r}) names arm {registered.get('arm')!r}")
    return registered


def configuration(run, report):
    """Whether the capture ran the registered arm: readback, constants, the jaw arms off, configuration_matches."""
    block = report.get("depth_appearance") or {}
    registered = registered_constants()
    result_path = Path(run) / "experiment_result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    matches = result.get("configuration_matches")
    checks = {
        "capture_complete": report.get("stage") == "complete",
        "experiment_result_recorded": result_path.is_file(),
        "configuration_checked": "configuration_matches" in result,
        "enabled": block.get("enabled") is True,
        "arm_agreement": (block.get("constants") or {}).get("arm") == ARM,
        "constants_registered": block.get("constants") == registered,
        "jaw_arms_off": all(
            (report.get(arm) or {}).get("enabled") is not True for arm in ("jaw_self_mask", "closure_hold")
        ),
        "configuration_matches": matches,
    }
    checks["ok"] = (
        checks["capture_complete"]
        and checks["enabled"]
        and checks["arm_agreement"]
        and checks["constants_registered"]
        and checks["jaw_arms_off"]
        and matches is True
    )
    return checks


RUNNER_ERROR = "runner error: no configuration check (experiment_result.json has no configuration_matches)"


def _failed_checks(row):
    value = row.get("failed_checks")
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return value


def run_record(root, batch, row):
    """The fields A1-A5 need for one run, from its own records; an unscored run keeps its grade row."""
    registered = REGISTRY[batch]
    run = Path(root) / VISION_ROBUSTNESS / batch / row["run_directory"]
    graded = row.get("status") == "graded" and row.get("checks_total") == CHECKS
    record = {
        "batch": batch,
        "kind": registered["kind"],
        "run_directory": row["run_directory"],
        "target": int(row["component_first_vertex"]),
        "daylight": row["daylight"],
        "status": row.get("status"),
        "outcome": row["outcome"],
        "checks": f"{row.get('checks_passed')}/{row.get('checks_total')}",
        "failed_checks": _failed_checks(row),
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
        "closure_frames": [],
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
            else {
                "frame": stop["index"],
                "stopped_reason": stop["stopped_reason"],
                "tracker_state": stop["state"],
                "tracker_reason": stop["reason"],
            },
            "grey_failures": through,
            "post_stop_grey_failures": after,
            "events": {i: events[i] for i in through},
            "post_stop_events": {i: events[i] for i in after},
            "closure_frames": closure_frames(facts, stop_frame),
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
    elif not checks["configuration_checked"]:
        record["unscored_reason"] = RUNNER_ERROR
    elif not record["scored"]:
        record["unscored_reason"] = "configuration refused"
    return record


def expected_run_directory(index, row, strategy):
    """tools/run_vision_experiment.run_label for a registered row."""
    plan_row = {
        "daylight": row["daylight"],
        "target_tree_index": row["target_tree_index"],
        "component_first_vertex": row["component_first_vertex"],
        "strategy": {"name": strategy},
    }
    return run_label(index, plan_row)


def load(root):
    """The registered grade files, checked, and one record per registered run; refuses anything else."""
    try:
        return _load(Path(root))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(f"malformed grade input: {error!r}") from error


def _load(root):
    records, files = [], {}
    extra = sorted(path.name for path in (root / EVIDENCE_DIR).glob("*.json") if path.stem not in REGISTRY)
    if extra:
        raise InputError(f"{EVIDENCE_DIR}: grade files outside the registered batches: {extra}")
    for batch, registered in REGISTRY.items():
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
        wanted = [(registered["target"], registered["daylight"], registered["strategy"])]
        found = [(int(r["component_first_vertex"]), r.get("daylight"), r.get("strategy")) for r in rows]
        if found != wanted:
            raise InputError(f"{path}: rows {found} are not the registered {wanted}")
        for index, row in enumerate(rows):
            label = expected_run_directory(index, row, registered["strategy"])
            if row["run_directory"] != registered["run_directory"] or label != registered["run_directory"]:
                raise InputError(
                    f"{path}: run directory {row['run_directory']} (run_label {label}) is not the registered "
                    f"{registered['run_directory']}"
                )
            if row.get("status") == "graded" and row.get("checks_total") != CHECKS:
                raise InputError(f"{path}: {row['run_directory']} was graded with {row.get('checks_total')} checks")
        records += [run_record(root, batch, row) for row in rows]
    if len(records) != EXPECTED_RUNS:
        raise InputError(f"{len(records)} runs, registered {EXPECTED_RUNS}")
    return records, files


def frozen_paths(root):
    """The code A5 executes and the live run used: the depth tool's SOURCE_FILES, the controller and runner, and
    every repository module the scorer and the replay import (found in sys.modules), except this scorer itself."""
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
    return sorted((set(depth_tool.SOURCE_FILES) | set(FROZEN_EXTRA) | imported) - {SELF})


def check_frozen_sources(root):
    """The code A5 executes must equal the code every batch was frozen with (plan.json source_sha256).

    Every batch must also have frozen this scorer. Its current hash may differ from the frozen one (a disclosed
    amendment after the freeze): that is reported per batch as amended_since_freeze, never refused. Returns the
    compared hashes and the scorer's freeze record.
    """
    root = Path(root).resolve()
    try:
        paths = frozen_paths(root)
        current = {path: sha256(root / path) for path in paths}
        scorer_now = sha256(root / SELF)
    except OSError as error:
        raise InputError(f"cannot hash the code to compare with the frozen code: {error!r}") from error
    scorer = {"path": SELF, "current_sha256": scorer_now, "batches": {}}
    for batch in REGISTRY:
        plan_path = root / VISION_ROBUSTNESS / batch / "plan.json"
        try:
            plan = json.loads(plan_path.read_text())
        except (OSError, ValueError) as error:
            raise InputError(f"{plan_path}: unreadable ({error!r})") from error
        frozen = plan.get("source_sha256") if isinstance(plan, dict) else None
        if not isinstance(frozen, dict):
            raise InputError(f"{batch}/plan.json: no source_sha256 mapping")
        if not frozen.get(SELF):
            raise InputError(f"{batch}/plan.json: the frozen code does not contain {SELF}")
        differing = sorted(path for path in paths if frozen.get(path) != current[path])
        if differing:
            raise InputError(f"{batch}/plan.json: the checkout differs from the frozen code in {differing}")
        scorer["batches"][batch] = {
            "frozen_sha256": frozen[SELF],
            "amended_since_freeze": frozen[SELF] != scorer_now,
        }
    return current, scorer


def verify_self(root):
    """This scorer must be tracked and identical to HEAD, so the code that scores is the committed code."""
    root = Path(root)
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", SELF], capture_output=True, check=False
        )
        unchanged = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", SELF], capture_output=True, check=False
        )
        digest = sha256(root / SELF)
    except OSError as error:
        raise InputError(f"cannot verify {SELF}: {error!r}") from error
    if tracked.returncode != 0:
        raise InputError(f"{SELF} is not tracked in {root}: commit it before scoring")
    if unchanged.returncode != 0:
        raise InputError(f"{SELF} differs from HEAD in {root}: commit it before scoring")
    return digest


# ------------------------------------------------------------------------------------------------ predictions
def name(record):
    return f"{record['batch']}/{record['run_directory']}"


def in_window(frame):
    first, last = SHADOW_WINDOW
    return first <= frame <= last


def first_in_window(record):
    """The first judged grey failure in frames 56-67, as (frame, event), or None."""
    frame = next((index for index in record["grey_failures"] if in_window(index)), None)
    return None if frame is None else (frame, record["events"][frame])


def ends_low_confidence(event):
    """An accepted event whose update ended in the low-confidence stop."""
    if event.get("accept") is not True:
        return False
    return (
        LOW_CONFIDENCE in (event.get("ended_by") or [])
        or event.get("result_reason") == LOW_CONFIDENCE
        or (event.get("at_frame") or {}).get("tracker_reason") == LOW_CONFIDENCE
    )


def event_at_stop(record):
    stop = record["stop"]
    return None if stop is None else record["events"].get(stop["frame"])


def unscored(runs):
    return {name(r): r["unscored_reason"] for r in runs if not r["scored"]}


def evening(records):
    return [r for r in records if r["kind"] == "evening"]


def _brief_event(frame, event):
    keys = ("accept", "failed_conditions", "reasons", "strict_confidence", "agreement_confidence", "decision")
    return {"frame": frame, **{key: event.get(key) for key in keys}, "ended_by": event.get("ended_by")}


def score_a1(records):
    runs = evening(records)
    observed, rejected, low, other_gate, untested, tested = {}, [], {}, {}, [], 0
    for record in runs:
        if not record["scored"]:
            untested.append(name(record))
            observed[name(record)] = {"untested": record["unscored_reason"]}
            continue
        lows = [frame for frame in record["grey_failures"] if ends_low_confidence(record["events"][frame])]
        if lows:
            low[name(record)] = lows
        at_stop = event_at_stop(record)
        if at_stop is not None and at_stop["accept"] is True and not ends_low_confidence(at_stop):
            other_gate[name(record)] = {"stop": record["stop"], **_brief_event(record["stop"]["frame"], at_stop)}
        entry = {"stop": record["stop"], "accepted_events_ending_low_confidence": lows}
        first = first_in_window(record)
        if first is None:
            untested.append(name(record))
            entry["untested"] = "no grey failure in frames 56-67 through its stop"
            observed[name(record)] = entry
            continue
        tested += 1
        frame, event = first
        entry["first_in_window"] = _brief_event(frame, event)
        if event["accept"] is False:
            rejected.append(name(record))
            entry["case"] = "the depth test rejected the first in-window grey failure"
        elif ends_low_confidence(event):
            entry["case"] = "accepted, then ended in a low-confidence stop"
        elif event.get("ended_by"):
            entry["case"] = f"accepted, then stopped by another gate {event['ended_by']}"
        else:
            entry["case"] = "accepted and the tracker continued"
        observed[name(record)] = entry
    all_scored = len(runs) == EVENING_RUNS and all(r["scored"] for r in runs)
    return prediction(
        "A1",
        "In each evening 14944 run the depth test accepts the first grey-check failure in frames 56-67, and no "
        "accepted event ends in a low-confidence stop.",
        [
            judged(
                "evening 14944: the first grey failure in frames 56-67 is accepted in all 4 runs, and no accepted "
                "event through the stop ends in a low-confidence stop",
                {
                    "runs": observed,
                    "rejected_first_in_window": rejected,
                    "accepted_events_ending_low_confidence": low,
                    "stops_at_accepted_events_on_another_gate_counted_against_a2": other_gate,
                    "untested": untested,
                },
                all_scored and tested == EVENING_RUNS and not rejected and not low,
                bool(rejected) or bool(low),
                tested,
            )
        ],
    )


def stops_on_appearance_in_window(record):
    stop = record["stop"]
    if stop is None or not in_window(stop["frame"]):
        return False
    if stop["tracker_reason"] == APPEARANCE_LOSS:
        return True
    event = event_at_stop(record)
    return stop["tracker_reason"] == LOW_CONFIDENCE and event is not None and event["accept"] is True


def score_a2(records):
    runs = evening(records)
    scored = [r for r in runs if r["scored"]]
    passes = sum(r["outcome"] == "pass" for r in scored)
    not_scored = len(runs) - len(scored)
    in_window_stops = [name(r) for r in scored if r["stop"] is not None and in_window(r["stop"]["frame"])]
    on_appearance = [name(r) for r in scored if stops_on_appearance_in_window(r)]
    failed = {
        name(r): {
            "outcome": r["outcome"],
            "checks": r["checks"],
            "stop_frame": None if r["stop"] is None else r["stop"]["frame"],
            "stop": r["stop"],
            "depth_events": r["events"],
            "post_stop_depth_events": r["post_stop_events"],
            "failed_checks": r["failed_checks"],
        }
        for r in scored
        if r["outcome"] != "pass"
    }
    return prediction(
        "A2",
        "At least 3 of the 4 evening 14944 runs pass all 17 checks of the unchanged grader, and none stops on "
        "appearance in frames 56-67.",
        [
            judged(
                "evening 14944: passes, and stops on appearance in frames 56-67",
                {
                    "runs": {
                        name(r): {"outcome": r["outcome"], "checks": r["checks"], "stop": r["stop"]} for r in scored
                    },
                    "passes": f"{passes} of {len(runs)}",
                    "stops_in_window": in_window_stops,
                    "appearance_stops_in_window": on_appearance,
                    "failed_runs": failed,
                    "unscored": unscored(runs),
                },
                len(runs) == EVENING_RUNS and len(scored) == EVENING_RUNS and passes >= 3 and not on_appearance,
                len(on_appearance) >= 2 or passes + not_scored < 2,
                len(scored),
            )
        ],
    )


def score_a3(records):
    runs = evening(records)
    observed, failures, not_clear, tested = {}, [], [], 0
    for record in runs:
        if not record["scored"]:
            observed[name(record)] = {"untested": record["unscored_reason"]}
            continue
        closure = record["closure_frames"]
        if not closure:
            observed[name(record)] = {"untested": "did not reach closure through its stop", "stop": record["stop"]}
            continue
        tested += 1
        frames = {item["frame"] for item in closure}
        rows = []
        for frame in (index for index in record["grey_failures"] if index in frames):
            event = record["events"][frame]
            jaw = event.get("jaw_condition")
            jaw_clear = isinstance(jaw, dict) and jaw.get("passed") is True
            row = {
                "run": name(record),
                "frame": frame,
                "cut_phase": (event.get("at_frame") or {}).get("cut_phase"),
                "accept": event["accept"],
                "failed_conditions": event.get("failed_conditions"),
                "not_evaluated_conditions": event.get("not_evaluated_conditions"),
                "J_jaw_silhouette": jaw,
                "J_clear": jaw_clear,
                "strict_confidence": event.get("strict_confidence"),
                "agreement_confidence": event.get("agreement_confidence"),
                "decision": event.get("decision"),
                "ended_by": event.get("ended_by"),
            }
            rows.append(row)
            if not (event["accept"] is True and jaw_clear):
                not_clear.append(f"{name(record)}@{frame}")
        failures += rows
        observed[name(record)] = {
            "stop": record["stop"],
            "closure_frames": closure,
            "grey_failures_in_closure": rows or "none: reached closure with no grey-check failure",
        }
    all_scored = len(runs) == EVENING_RUNS and all(r["scored"] for r in runs)
    return prediction(
        "A3",
        "Every grey-check failure during closure in the evening 14944 runs is reported; the expectation is that each "
        "is accepted with J clear.",
        [
            judged(
                "evening 14944: every grey failure during closure is accepted with J clear (expectation; cannot "
                "refute)",
                {
                    "runs": observed,
                    "runs_that_reached_closure": f"{tested} of {len(runs)}",
                    "grey_failures_in_closure": failures,
                    "not_accepted_with_j_clear": not_clear,
                    "unscored": unscored(runs),
                },
                all_scored and tested == EVENING_RUNS and not not_clear,
                False,
                tested,
            )
        ],
    )


def score_a4(records):
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
        event = event_at_stop(record) if record is not None and record["scored"] else None
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
        "A4", "Neither control passes, and the depth test rejects every grey-check failure in both.", clauses
    )


# ------------------------------------------------------------------------------------------------ A5
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
    """A5 for one comparable run: the offline agreement arm on the run's own frames against the live record."""
    import check_depth_loop_c0 as c0
    import replay_depth_appearance as depth_tool

    result = c0.normalized(depth_tool.replay_run(Path(record["path"])))
    agreement = result["divergence"][ARM]
    problems = []
    if not agreement["equals_recording"]:
        problems.append(
            {
                "kind": "agreement_differs_from_recording",
                "frames": agreement["mismatch_frames"],
                "preview_fields": agreement["preview_mismatch_fields"],
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
    offline = {str(event["frame"]): event for event in result["events"][ARM]}
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
        "max_agreement_correlation_difference": agreement["max_abs_patch_correlation_difference"],
    }


def score_a5(records, comparisons):
    mismatched = {key: value for key, value in comparisons.items() if value["status"] == "mismatch"}
    not_comparable = {key: value["why"] for key, value in comparisons.items() if value["status"] == "not_comparable"}
    compared = sum(value["status"] in ("equal", "mismatch") for value in comparisons.values())
    return prediction(
        "A5",
        "Every live run equals the offline agreement arm replayed on its own frames through its stop, with the same "
        "depth event at every grey failure.",
        [
            judged(
                "live = offline agreement arm in every run",
                {"runs_compared": compared, "mismatches": mismatched, "not_comparable": not_comparable},
                len(records) == EXPECTED_RUNS and compared == EXPECTED_RUNS and not mismatched,
                bool(mismatched),
                compared,
            )
        ],
    )


def score(records, comparisons):
    return [
        score_a1(records),
        score_a2(records),
        score_a3(records),
        score_a4(records),
        score_a5(records, comparisons),
    ]


def run_summary(record):
    return {key: value for key, value in record.items() if key != "path"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="Must be this scorer's own checkout (the default)")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    if args.root.resolve() != ROOT:
        parser.error(f"--root must be this scorer's own checkout {ROOT}, not {args.root.resolve()}")
    import cv2

    cv2.setNumThreads(1)
    try:
        scorer_sha256 = verify_self(ROOT)
        records, files = load(ROOT)
        frozen, frozen_scorer = check_frozen_sources(ROOT)
    except InputError as error:
        parser.error(str(error))
    import check_depth_loop_c0 as c0
    import replay_depth_appearance as depth_tool

    # Every git call runs before the first replay, so the revision recorded is the one that scored.
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    provenance = depth_tool._provenance()
    comparisons = {}
    for record in records:
        why = comparability(record)
        comparisons[name(record)] = (
            {"status": "not_comparable", "why": why} if why is not None else offline_comparison(record)
        )
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the agreement-arm closed-loop test (A1-A5): 6 simulator runs with the agreement arm of the "
            "depth-aware appearance check (+ J) in the live controller, the jaw surrogate casting its shadow: four "
            "repeats of evening 14944 and two source-light controls (19444, 12142). Simulator renders and depth; one "
            "target in one light; reported apart from D_strict + J and every unchanged-gate result, never pooled."
        ),
        "label": LABEL,
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "scorer_sha256": scorer_sha256,
        "frozen_scorer": frozen_scorer,
        "construction_notes": CONSTRUCTION_NOTES,
        "grade_files": files,
        "frozen_sources_sha256": frozen,
        "offline_tool_provenance": provenance,
        "float_tolerance": c0.FLOAT_TOLERANCE,
        "max_event_float_difference_accepted": c0.COMPARE.max_float_difference,
        "max_agreement_correlation_difference": max(
            (c.get("max_agreement_correlation_difference") or 0.0 for c in comparisons.values()), default=0.0
        ),
        "interpretations": INTERPRETATIONS,
        "runs": [run_summary(record) for record in records],
        "a5_comparisons": comparisons,
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
