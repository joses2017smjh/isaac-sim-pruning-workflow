#!/usr/bin/env python3
"""Score the closure-hold sweep (protocol of October 6) against S1-S7.

The sweep runs the jaw self-mask and closure hold, unchanged, once on each of 28 targets the rebuilt known-map planner
qualified, plus controls 530 and 14944. This scorer is tools/score_hold_generalization.py (reviewed blind and
mutation-tested before the October 4 runs) with its registry replaced: S2, S3, S5, S6 and S7 keep that scorer's H2,
H3, H5, H6 and H7 logic line for line over the sweep's targets; S1 (the home visibility check registered as a
prediction) and S4 (the outcome where the hold engaged) are new. Outcomes come from the per-batch evidence
aggregate_eval.py wrote (the grader's classes, never re-graded here); every planned run stays in every table. Each
run's records are read by the committed tools/score_jaw_in_view.py, as there. Every result carries the label:
known-map plan; jaw self-mask; closure hold with freshness and frame-reuse checks waived on held frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))
import score_jaw_in_view as jiv  # noqa: E402 - the committed jaw-in-view scorer; its per-run logic is reused
from score_perception_round import clause, prediction  # noqa: E402

PROTOCOL = "docs/EVAL_PROTOCOL_HOLD_SWEEP_2026-10-06.md"
EVIDENCE_DIR = "docs/evidence/hold_sweep_2026-10-06"
GRADER = "tools/validate_vision_sequence.py:grade_sequence"
SELF = "tools/score_hold_sweep.py"
#: The scoring code: this scorer and the committed modules it imports. Every plan must have frozen each one.
SCORING_FILES = (
    SELF,
    "tools/score_jaw_in_view.py",
    "tools/score_perception_round.py",
    "tools/score_planned_approach.py",
)
LABEL = jiv.LABEL
STRATEGY = "planned_pose_jaw_hold_sweep"
#: The sweep's targets as (tree, target), in register order.
SWEEP = (
    (0, 4438),
    (0, 7049),
    (0, 7644),
    (0, 7760),
    (0, 8175),
    (0, 8532),
    (0, 11607),
    (0, 11727),
    (0, 12261),
    (0, 12321),
    (0, 12381),
    (0, 13213),
    (0, 16660),
    (0, 21268),
    (0, 21386),
    (0, 28493),
    (0, 28610),
    (0, 29554),
    (1, 3433),
    (1, 3673),
    (1, 20698),
    (1, 20995),
    (1, 21648),
    (1, 26425),
    (1, 27362),
    (1, 33128),
    (1, 34125),
    (1, 36786),
)
SWEEP_TARGETS = tuple(target for _, target in SWEEP)
POSITIVE_CONTROL, HOLDLESS_CONTROL = 530, 14944
#: S1: the targets the committed home visibility check (docs/evidence/home_visibility_check_2026-10-05.json) rejects.
PREDICTED_REJECTED = (4438, 3433, 3673)
REGISTER_SWEEP = "docs/evidence/eval_targets_hold_sweep_2026-10-06.json"
REGISTER_CONTROLS = "docs/evidence/eval_targets_jaw_hold_b_2026-09-30.json"
#: batch -> (registered (tree, target) in run-index order, strategy)
REGISTRY = {
    "hold-sweep-20261006": (SWEEP, STRATEGY),
    "hold-sweep-ctl-20261006": (((0, POSITIVE_CONTROL), (1, HOLDLESS_CONTROL)), STRATEGY),
}
EXPECTED_RUNS = {**{target: 1 for target in SWEEP_TARGETS}, POSITIVE_CONTROL: 1, HOLDLESS_CONTROL: 1}
#: The registered strategy's fields (tools/queue_vision_robustness.py); a plan row must carry exactly these.
STRATEGY_FIELDS = {
    STRATEGY: {"mode": "planned_pose_standoff", "standoff_m": 0.06, "max_step_m": 0.004, "max_rotation_deg": 1.5},
}
#: The exact keys of a plan row's strategy (queue_vision_robustness._strategy_row of a planned-pose strategy).
STRATEGY_KEYS = {
    "name",
    "mode",
    "standoff_m",
    "max_step_m",
    "max_rotation_deg",
    "jaw_self_mask",
    "closure_hold",
    "planned_tool_quat_wxyz",
}
#: The modules this scorer imports from the checkout; each must equal its copy frozen in every plan.
IMPORTED_SCORING_MODULES = {
    "score_jaw_in_view": "tools/score_jaw_in_view.py",
    "score_perception_round": "tools/score_perception_round.py",
    "score_planned_approach": "tools/score_planned_approach.py",
}
S4_MIN_RUNS = 3  # S4: judged with at least 3 runs where the hold engaged

CONSTRUCTION_NOTES = [
    "This scorer is committed before submission: before any hold-sweep run was submitted. When it was written the "
    "checkout held no hold-sweep batch directory, grade file or log; its author did not query Slurm for this "
    "experiment. It is written blind, and no outcome of this experiment could shape it.",
    "It is tools/score_hold_generalization.py with the registry replaced; that scorer was reviewed blind before the "
    "October 4 runs (the review's fixes and a 35-of-35 mutation re-run are recorded in its construction notes). S2, "
    "S3, S5, S6 and S7 keep its H2, H3, H5, H6 and H7 logic; S1 and S4 are new and were not reviewed independently "
    "(the reviewer was unavailable: the account's subagent usage limit was reached until October 8). They were "
    "checked by their own tests and one-line mutations.",
    "S1's predictions come from an exploratory CPU check (docs/evidence/home_visibility_check_2026-10-05.json): it "
    "reproduced all 174 recorded initializations and rejects 4438, 3433 and 3673 among the sweep targets.",
    "The runner (tools/run_vision_experiment.py execute) exits 0 only on a 17/17 pass with configuration_matches "
    "true, so each task's Slurm state reveals that run's pass or fail. Because this scorer is committed before "
    "submission, nothing seen afterwards can shape it.",
]
INTERPRETATIONS = [
    "S1: a run is rejected at home when its cut stops at frame 0 with initial_target_not_visible (the runner's "
    "initialization rule). S1 holds when every scored sweep run of a target in PREDICTED_REJECTED is rejected at home "
    "and every other scored sweep run is not; any scored run that departs refutes.",
    "S4: the population is the sweep runs where S3 applies (S2's loss occurred and the cut still closed after k+1). "
    "With at least 3 such scored runs, S4 holds when at least half pass 17/17 and refutes when fewer than half do; "
    "with fewer than 3 it is untested. Every sweep run that never started closure is listed with its stop (reported, "
    "not predicted).",
    "Plans: each plan must have been made from its registered register as committed (path and sha256), every row's "
    "strategy must have exactly the registered keys and values, and its planned orientation must equal the "
    "register's (None, the home orientation, for 14944).",
    "Run set: the grade files must be exactly the 2 registered batches (any other file in the evidence directory "
    "refuses), each aggregated under this protocol by tools/validate_vision_sequence.py, with batch_dir exactly "
    "artifacts/vision_robustness/<batch>, holding the registered targets in plan order under their registered run "
    "labels and strategies; plans must be this protocol's, source light, 200 frames at 10 fps, with both arms on "
    "and the registered strategy fields. Otherwise nothing is scored.",
    "Scored runs: a run is scored when it was graded (17 checks), configuration_matches is true, the capture is "
    "complete (all 200 frames recorded, in order), it is not refused, and its report shows no departure from the "
    "registered configuration. Any other run is listed in every table and decides nothing.",
    "Scored code: as in tools/score_jaw_in_view.py, the scored modules are imported from the first batch's frozen "
    "snapshot, every batch's snapshot must equal its plan for the scored sources and the plans must freeze identical "
    "code; this scorer must itself be tracked, equal to HEAD and frozen in every plan.",
    "Unscored runs: in S1-S6 every unscored sweep run keeps the prediction from being supported without refuting it, "
    "because it might fall inside the population. A pass is graded with checks_passed == checks_total == 17; a row "
    "whose outcome disagrees refuses.",
    "S2: tools/score_jaw_in_view.py's P4 rule (mechanism, p4_category) over the sweep targets' scored runs: "
    "predicted for runs whose gate mouth distance at closure start is at most 4 mm; a run with its first loss at "
    "k+1 is judged; a run whose cut left closing at k+1 for a non-vision reason never rendered the jaw closing and "
    "is untested for S2; runs above 4 mm or without a closure start in a settled recording are reported.",
    "S3: tools/score_jaw_in_view.py's P5 rule (p5_category, hold_coverage) over the sweep targets' scored runs: "
    "the population is the runs where S2's loss occurs (first closure loss at k+2, mask-attributable, cut still "
    "closing after k+1); coverage, tool and mouth are registered claims that cannot refute; a held frame that stops "
    "refutes, and so does detachment at another frame or no detachment once frame k+6 is recorded or the cut "
    "stopped by k+6.",
    "S5: tools/score_jaw_in_view.py's P8 rule (static_branch) over the sweep targets' scored runs that "
    "detached: the piece's PhysX origin and the true mouth-to-spur distance over [k, detach], both limits strict.",
    "S6: tools/score_jaw_in_view.py's per-frame mask reconstruction over every scored run: the count clause refutes "
    "on any mismatch; the depth clause judges predicted jaw pixels as amended on October 2 (centre at least 0.01 px "
    "inside the undilated silhouette: depth_test's violations_outside_edge_band) and refutes on any; the literal "
    "count over every hit is reported; unscored runs are listed in a coverage clause that cannot refute.",
    "S7: 530 holds when scored, a pass, held exactly k+2..k+6 with no held frame stopping, and detached at k+6; "
    "14944 holds when scored, a pass, never held (tools/score_jaw_in_view.py control_window: closure_hold.held or a "
    "closure_hold certificate on any frame) and no patch element masked before detach (window 0..d). A scored "
    "control that failed a check refutes; 14944 holding or masking a patch element before detach refutes; a held "
    "frame of 530 that stops refutes. An unscored control withholds support.",
    "Verdicts: refuted when any clause meets its registered refutation; otherwise 'untested' when the condition a "
    "prediction needs never arose (named in untested_reason); otherwise supported when every clause holds and "
    "partly supported when one does not.",
]
TEXT = {
    "S1": "(home visibility, CPU-predicted) The 3 targets the committed home visibility check rejects (4438, 3433, "
    "3673) are rejected at home (frame-0 stop, initial_target_not_visible); the other 25 initialize. Refuted by any "
    "departure.",
    "S2": "(mechanism, as P4) In sweep runs whose closure starts at a gate mouth distance of at most 4 mm, the first "
    "closure loss comes at k+2 and is mask-attributable; the shadow controller stops there. Refuted if tracking "
    "continues at k+3, or the first loss comes at another frame or is not mask-attributable.",
    "S3": "(hold to detachment, as P5) Where S2's loss occurs, the hold covers it to the deadline, the tool stays "
    "within 0.5 mm and 0.25 deg, held mouth distances stay within 0.05 mm of the closure-start value, and "
    "detachment comes at k+6. Refuted if any held frame stops or detachment comes at another frame.",
    "S4": "(outcome where the hold engaged) Among sweep runs where S2's loss occurs (at least 3), at least half pass "
    "all 17 checks. Refuted if fewer than half pass.",
    "S5": "(static-branch premise, as P8) In every sweep run that detaches, the piece moves less than 0.1 mm and the "
    "true mouth-to-spur distance changes by less than 0.1 mm between closure start and detach.",
    "S6": "(mask fidelity, P1 as amended October 2) On every frame of all runs the recorded mask count equals the "
    "reconstruction, and no predicted jaw pixel (at least 0.01 px inside the silhouette) shows depth more than "
    "1 mm beyond the box. Refuted by any mismatch.",
    "S7": "(controls) 530 passes 17/17 with the hold over k+2..k+6 and detach at k+6; 14944 passes 17/17, never "
    "holds and has no patch element masked before detach. Refuted if either fails a check, 14944 holds or masks a "
    "patch element before detach, or a held frame of 530 stops.",
}
SCOPE = (
    "Scoring of the closure-hold sweep (S1-S7): thirty simulator runs of the jaw self-mask and closure hold in two "
    "batches (28 planner-qualified targets once each; controls 530 and 14944). Every result is labelled '"
    + LABEL
    + "' and is reported apart from every unchanged-gate result, never pooled. Outcomes are the grader's, read from "
    "the per-batch evidence. Simulator renders of a two-box jaw surrogate with exact poses; not a field result."
)


class InputError(ValueError):
    """The inputs are not the registered experiment, or this scorer is not the committed one; nothing is scored."""


# ------------------------------------------------------------------------------------------------ inputs
def sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise InputError(f"cannot read {path}: {error!r}") from error


def register_of(name):
    return REGISTER_SWEEP if name == "hold-sweep-20261006" else REGISTER_CONTROLS


def check_plan(name, plan, root=ROOT):
    targets, strategy = REGISTRY[name]
    if (
        plan.get("protocol") != PROTOCOL
        or plan.get("frames") != jiv.PLANNED_FRAMES
        or plan.get("fps") != jiv.PLANNED_FPS
    ):
        raise InputError(f"{name}: the plan is not this protocol's 200 frames at 10 fps")
    registered = [(row.get("target_tree_index"), row.get("component_first_vertex")) for row in plan.get("runs") or []]
    if registered != list(targets):
        raise InputError(f"{name}: the plan's targets {registered} are not the registered {list(targets)}")
    relative = register_of(name)
    recorded = plan.get("target_register") or {}
    if recorded.get("targets_file") != relative or recorded.get("targets_sha256") != sha256(Path(root) / relative):
        raise InputError(f"{name}: the plan was not made from {relative} as committed")
    planned = {
        (int(e["target_tree_index"]), int(e["component_first_vertex"])): e.get("planned_final_tool_quat_wxyz")
        for e in json.loads((Path(root) / relative).read_text())["targets"]
    }
    for index, row in enumerate(plan["runs"]):
        fields = row.get("strategy") or {}
        quat = planned.get((row.get("target_tree_index"), row.get("component_first_vertex")))
        wanted = None if quat is None else [float(q) for q in quat]
        if (
            row.get("index") != index
            or row.get("daylight") != jiv.DAYLIGHT
            or set(fields) != STRATEGY_KEYS
            or fields.get("name") != strategy
            or fields.get("jaw_self_mask") is not True
            or fields.get("closure_hold") is not True
            or any(fields.get(key) != value for key, value in STRATEGY_FIELDS[strategy].items())
            or fields.get("planned_tool_quat_wxyz") != wanted
        ):
            raise InputError(f"{name}: plan row {index} is not the registered source-light {strategy} run")


def pair_rows(name, plan, rows):
    by_directory = {}
    for row in rows:
        by_directory.setdefault(row.get("run_directory"), []).append(row)
    pairs = []
    for index, plan_row in enumerate(plan["runs"]):
        directory = jiv.run_label(index, plan_row)
        matches = by_directory.pop(directory, [])
        if len(matches) != 1:
            raise InputError(f"{name}: {len(matches)} evidence rows for planned run {directory}")
        row = matches[0]
        for key in ("target_tree_index", "component_first_vertex", "daylight", "photometric_normalization"):
            if row.get(key) != plan_row.get(key):
                raise InputError(f"{name}: {directory} has {key} {row.get(key)!r}, the plan {plan_row.get(key)!r}")
        if row.get("strategy") != plan_row["strategy"]["name"]:
            raise InputError(f"{name}: {directory} is not a {plan_row['strategy']['name']} row")
        if row.get("status") == "graded" and row.get("checks_total") != len(jiv.GRADER_CHECKS):
            raise InputError(f"{name}: {directory} was graded with {row.get('checks_total')} checks, not 17")
        complete = row.get("status") == "graded" and row.get("checks_passed") == row.get("checks_total") == 17
        if (row.get("outcome") == "pass") != complete:
            raise InputError(f"{name}: {directory}'s outcome {row.get('outcome')!r} disagrees with its checks")
        pairs.append((plan_row, row))
    if by_directory:
        raise InputError(f"{name}: evidence rows for runs the plan does not have: {sorted(map(str, by_directory))}")
    return pairs


def validated_evidence(root):
    """``[(path, document, batch_dir, plan, pairs)]`` in registry order, after checking the registered run set."""
    directory = Path(root) / EVIDENCE_DIR
    found = sorted(path.stem for path in directory.glob("*.json"))
    if found != sorted(REGISTRY):
        raise InputError(f"{EVIDENCE_DIR}: expected exactly {sorted(REGISTRY)}; found {found}")
    validated, targets = [], []
    for name in REGISTRY:
        path = directory / f"{name}.json"
        try:
            document = jiv.read_json(path)
        except (OSError, ValueError) as error:
            raise InputError(f"{path.name}: unreadable ({error!r})") from error
        batches = document.get("batches") or []
        if len(batches) != 1 or batches[0].get("batch_dir") != f"artifacts/vision_robustness/{name}":
            raise InputError(f"{path.name}: not exactly one batch at artifacts/vision_robustness/{name}")
        if document.get("protocol") != PROTOCOL or document.get("grader") != GRADER:
            raise InputError(f"{path.name}: not graded by {GRADER} under {PROTOCOL}")
        batch_dir = batches[0]["batch_dir"]
        try:
            plan = jiv.read_json(jiv.batch_path(root, batch_dir) / "plan.json")
        except (OSError, ValueError) as error:
            raise InputError(f"{name}: plan unreadable ({error!r})") from error
        check_plan(name, plan, root)
        rows = document.get("runs") or []
        if not batches[0].get("planned_runs_kept") == len(plan["runs"]) == len(rows):
            raise InputError(f"{path.name}: planned_runs_kept, planned runs and evidence rows disagree")
        pairs = pair_rows(path.name, plan, rows)
        targets += [int(plan_row["component_first_vertex"]) for plan_row, _ in pairs]
        validated.append((path, document, batch_dir, plan, pairs))
    counts = {target: targets.count(target) for target in set(targets)}
    if counts != EXPECTED_RUNS:
        raise InputError(f"the evidence holds targets {counts}, not {EXPECTED_RUNS}")
    return validated


def verify_self(root):
    """Every scoring file must be tracked and identical to HEAD, and the scoring modules this process imported must
    be the checkout's own files, so the code that scores is the committed code. Returns this scorer's sha256."""
    root = Path(root)
    for name in SCORING_FILES:
        try:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", name], capture_output=True
            )
            unchanged = subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", name], capture_output=True
            )
        except OSError as error:
            raise InputError(f"cannot verify {name}: {error!r}") from error
        if tracked.returncode != 0:
            raise InputError(f"{name} is not tracked in {root}: commit it before scoring")
        if unchanged.returncode != 0:
            raise InputError(f"{name} differs from HEAD in {root}: commit it before scoring")
    for module, relative in IMPORTED_SCORING_MODULES.items():
        loaded = sys.modules.get(module)
        if loaded is None or Path(loaded.__file__).resolve() != (root / relative).resolve():
            raise InputError(f"{module} was not imported from {root / relative}")
    return sha256(root / SELF)


def frozen_scorer(validated, digest, root=ROOT):
    """Every batch must have frozen the scoring code. The imported modules must equal their frozen copies; an
    amendment of this scorer after the freeze is reported, never refused (a disclosed amendment is allowed)."""
    current = {name: digest if name == SELF else sha256(Path(root) / name) for name in SCORING_FILES}
    batches = {}
    for _, _, batch_dir, plan, _ in validated:
        frozen = plan.get("source_sha256") or {}
        missing = [name for name in SCORING_FILES if name not in frozen]
        if missing:
            raise InputError(f"{Path(batch_dir).name}: its plan did not freeze {missing} (submitted before them)")
        changed = [name for name in IMPORTED_SCORING_MODULES.values() if frozen[name] != current[name]]
        if changed:
            raise InputError(f"{Path(batch_dir).name}: {changed} differ from the copies frozen at submission")
        batches[Path(batch_dir).name] = {
            name: {"frozen_sha256": frozen[name], "amended_since_freeze": frozen[name] != current[name]}
            for name in SCORING_FILES
        }
    return {"path": SELF, "current_sha256": digest, "current_sha256_by_file": current, "batches": batches}


# ------------------------------------------------------------------------------------------------ run records
def unscored_reason(record):
    """Why a run decides nothing, or None for a scored run (see interpretations)."""
    if record["status"] != "graded":
        return f"not graded (status {record['status']})"
    if record["checks_total"] != len(jiv.GRADER_CHECKS):
        return f"graded with {record['checks_total']} checks"
    if record["configuration_matches"] is not True:
        return f"configuration_matches is {record['configuration_matches']!r}"
    if record["refused"]:
        return "refused: " + "; ".join(record["refused_reasons"])
    if not record.get("frames"):
        return f"not recorded: {record.get('not_recorded_reason')}"
    if not jiv.recording_complete(record):
        return "capture incomplete"
    if record["configuration_problems"]:
        return "the report departs from the registered configuration: " + "; ".join(record["configuration_problems"])
    return None


def load(root, validated):
    records = []
    for _, _, batch_dir, _, pairs in validated:
        for plan_row, row in pairs:
            record = jiv.run_record(root, batch_dir, plan_row, row)
            record["unscored_reason"] = unscored_reason(record)
            records.append(record)
    return records


def key(record):
    return jiv.run_key(record)


def passed(record):
    """The grader's class 'pass': graded with all 17 checks (pair_rows refuses a row where outcome disagrees)."""
    return record["status"] == "graded" and record["checks_passed"] == record["checks_total"] == len(jiv.GRADER_CHECKS)


def scored(records, targets):
    return [r for r in records if r["target"] in targets and r["unscored_reason"] is None]


def unscored(records, targets):
    return {
        key(r): {"outcome": r["outcome"], "status": r["status"], "why": r["unscored_reason"]}
        for r in records
        if r["target"] in targets and r["unscored_reason"] is not None
    }


def judged(pid, clauses, untested=None):
    result = prediction(pid, TEXT[pid], clauses)
    if untested and result["verdict"] != "refuted":
        result["verdict"] = "untested"
        result["untested_reason"] = untested
    return result


def stop_view(record):
    c = jiv.closure_facts(record)
    frames = jiv.indexed(record)
    stop = c["cut_stop_frame"]
    measurement = (frames.get(stop) or {}).get("measurement") or {} if stop is not None else {}
    return {
        "outcome": record["outcome"],
        "checks": f"{record['checks_passed']}/{record['checks_total']}",
        "failed_checks": record["failed_checks"],
        "closure_start_frame": c["closure_start_frame"],
        "cut_stop": [stop, c["cut_stop_reason"]],
        "tracker_at_stop": [measurement.get("state"), measurement.get("reason")],
        "gpu_model": record["gpu_model"],
    }


# ------------------------------------------------------------------------------------------------ S1
def rejected_at_home(record):
    c = jiv.closure_facts(record)
    return c["cut_stop_frame"] == 0 and c["cut_stop_reason"] == "initial_target_not_visible"


def score_s1(records):
    rows, departures = {}, []
    for record in scored(records, SWEEP_TARGETS):
        predicted, observed = record["target"] in PREDICTED_REJECTED, rejected_at_home(record)
        rows[key(record)] = {"predicted_rejected": predicted, "rejected_at_home": observed, **stop_view(record)}
        if predicted != observed:
            departures.append(key(record))
    clauses = [
        clause(
            "the home visibility check's verdict matches every scored sweep run (rejected at home or initialized)",
            {"runs": rows, "departures": departures},
            bool(rows) and not departures,
            refutes=bool(departures),
        )
    ]
    missing = unscored(records, SWEEP_TARGETS)
    if missing:
        clauses.append(jiv.not_judged("sweep runs not scored", missing))
    return judged("S1", clauses, None if rows else "no sweep run was scored")


# ------------------------------------------------------------------------------------------------ S4
def score_s4(records):
    engaged, before, other = {}, {}, {}
    for record in scored(records, SWEEP_TARGETS):
        c = jiv.closure_facts(record)
        category, why = jiv.p5_category(record, c)
        if category == "applies":
            engaged[key(record)] = {**stop_view(record), "passed": passed(record)}
        elif c["closure_start_frame"] is None:
            before[key(record)] = stop_view(record)
        else:
            other[key(record)] = {**stop_view(record), "why": why}
    passes = sum(1 for v in engaged.values() if v["passed"])
    judgeable = len(engaged) >= S4_MIN_RUNS
    clauses = [
        clause(
            "at least half of the sweep runs where S2's loss occurred pass all 17 checks",
            {"runs": engaged, "passes": passes, "of": len(engaged)},
            judgeable and 2 * passes >= len(engaged),
            refutes=judgeable and 2 * passes < len(engaged),
        ),
        clause("reported, not predicted: sweep runs that never started closure", before or {"none": True}, True),
        clause(
            "reported, not predicted: sweep runs that started closure without S2's loss", other or {"none": True}, True
        ),
    ]
    missing = unscored(records, SWEEP_TARGETS)
    if missing:
        clauses.append(jiv.not_judged("sweep runs not scored", missing))
    untested = None if judgeable else f"only {len(engaged)} scored sweep runs had S2's loss (at least 3 are needed)"
    return judged("S4", clauses, untested)


# ------------------------------------------------------------------------------------------------ S2 (as P4)
def score_s2(records):
    groups = {name: {} for name in ("predicted", "reported", "premise", "not_evaluable")}
    for record in scored(records, SWEEP_TARGETS):
        view = jiv.mechanism(record)
        category, why = jiv.p4_category(record, view)
        groups[category][key(record)] = view if why is None else {**view, "why": why}
    predicted = groups["predicted"]
    on_time = {k: v["first_loss_offset"] == jiv.FIRST_LOSS_OFFSET for k, v in predicted.items()}
    tracking_on = {k: v["state_at_k_plus_3"] == "tracking" for k, v in predicted.items()}
    unexplained = {k: v["first_loss_frame"] is not None and not v["attributable"] for k, v in predicted.items()}

    def subset(*names):
        return {k: {name: v[name] for name in names if name in v} for k, v in predicted.items()}

    def shadow_ok(v):
        return v["shadow_stop_frame"] == v["first_loss_frame"]

    clauses = [
        clause(
            "the first closure loss comes at closure start + 2",
            subset("closure_start_frame", "mouth_distance_at_start_mm", "first_loss_frame", "first_loss_offset"),
            bool(predicted) and all(on_time.values()),
            refutes=not all(on_time.values()),
        ),
        clause(
            "tracking does not continue at closure start + 3",
            subset("state_at_k_plus_3", "frame_k_plus_3_recorded"),
            bool(predicted) and all(v["frame_k_plus_3_recorded"] and not tracking_on[k] for k, v in predicted.items()),
            refutes=any(tracking_on.values()),
        ),
        clause(
            "the first closure loss is mask-attributable",
            subset("state_at_loss", "reason_at_loss", "mask_explanation", "attributable"),
            bool(predicted) and all(v["attributable"] for v in predicted.values()),
            refutes=any(unexplained.values()),
        ),
        clause(
            "the shadow controller (cut_without_hold) stops at the loss; its reason is reported (not a refutation "
            "condition)",
            subset("first_loss_frame", "shadow_stop_frame", "shadow_stop_reason"),
            bool(predicted) and all(shadow_ok(v) for v in predicted.values()),
        ),
        clause(
            "reported, not predicted: sweep runs whose closure started above 4 mm or never started",
            groups["reported"] or {"none": True},
            True,
        ),
    ]
    if groups["premise"]:
        clauses.append(
            jiv.vacuous(
                "runs that left closing at k+1 for a non-vision reason, so the jaw was never rendered closing",
                groups["premise"],
                "S2's premise (a first frame rendered closing at k+2) never arose in these runs",
            )
        )
    missing = {**unscored(records, SWEEP_TARGETS), **groups["not_evaluable"]}
    if missing:
        clauses.append(jiv.not_judged("sweep runs not scored or not settled", missing))
    untested = None
    if not predicted:
        untested = "no scored sweep run started closure at a gate mouth distance of at most 4 mm"
        if groups["premise"]:
            untested += "; in every such run the cut left closing at k+1"
    return judged("S2", clauses, untested)


# ------------------------------------------------------------------------------------------------ S3 (as P5)
def score_s3(records):
    applies, groups = {}, {name: {} for name in ("reported", "not_evaluable")}
    for record in scored(records, SWEEP_TARGETS):
        c = jiv.closure_facts(record)
        category, why = jiv.p5_category(record, c)
        if category == "applies":
            applies[key(record)] = jiv.hold_coverage(record)
        else:
            groups[category][key(record)] = {**jiv.closure_brief(record, c), "why": why}

    def subset(*names):
        return {k: {name: v[name] for name in names if name in v} for k, v in applies.items()}

    def every(name):
        return bool(applies) and all(v[name] for v in applies.values())

    late = any(v["detach_other"] for v in applies.values())
    stopped = any(v["held_stop_frames"] for v in applies.values())
    clauses = [
        clause(
            "the hold covers every frame from S2's loss (k+2) to the deadline (not a refutation condition)",
            subset("first_loss_frame", "held_frames", "expected_held_frames", "recorded_through_deadline"),
            every("covered"),
        ),
        clause(
            "the tool stays within 0.5 mm and 0.25 deg over k+1..k+6 (not a refutation condition)",
            subset("tool_frames_checked", "tool_max_translation_mm", "tool_max_rotation_deg"),
            every("tool_within"),
        ),
        clause(
            "held mouth distances equal the closure-start value within 0.05 mm (not a refutation condition)",
            subset("mouth_distance_at_start_mm", "held_mouth_max_deviation_mm"),
            every("mouth_within"),
        ),
        clause(
            "detachment comes at closure start + 6",
            subset("closure_start_frame", "detach_frame", "cut_stop", "recorded_through_deadline"),
            every("detach_on_time"),
            refutes=late,
        ),
        clause(
            "no held frame stops",
            subset("held_stop_frames", "cut_stop"),
            every("recorded_through_deadline") and not stopped,
            refutes=stopped,
        ),
        clause(
            "every held frame carries the protocol's definition of a held frame (not a refutation condition)",
            subset("mislabelled_held_frames"),
            bool(applies) and not any(v["mislabelled_held_frames"] for v in applies.values()),
        ),
        clause(
            "reported, not predicted: runs where S2's loss did not occur", groups["reported"] or {"none": True}, True
        ),
    ]
    missing = {**unscored(records, SWEEP_TARGETS), **groups["not_evaluable"]}
    if missing:
        clauses.append(jiv.not_judged("sweep runs not scored or not settled", missing))
    untested = None if any(v["evaluable"] for v in applies.values()) else "no scored sweep run had S2's loss"
    if applies and untested:
        untested = "every run with S2's loss ends before the deadline with nothing settled"
    return judged("S3", clauses, untested)


# ------------------------------------------------------------------------------------------------ S5 (as P8)
def score_s5(records):
    views, reported = {}, {}
    for record in scored(records, SWEEP_TARGETS):
        if jiv.closure_facts(record)["detach_frame"] is not None:
            views[key(record)] = jiv.static_branch(record)
        else:
            reported[key(record)] = {**jiv.closure_brief(record), "why": "never detached"}
    tested = {k: v for k, v in views.items() if v["status"] == "tested"}

    def subset(*names):
        return {k: {name: v[name] for name in names if name in v} for k, v in tested.items()}

    piece_moved = any(not v["piece_static"] for v in tested.values())
    distance_moved = any(not v["distance_static"] for v in tested.values())
    clauses = [
        clause(
            "the selected piece moves less than 0.1 mm between closure start and detach",
            subset("window", "piece_motion_max_mm", "piece_rotation_max_deg"),
            bool(tested) and not piece_moved,
            refutes=piece_moved,
        ),
        clause(
            "the true mouth-to-spur distance changes by less than 0.1 mm between closure start and detach",
            subset("window", "true_distance_at_start_mm", "true_distance_change_max_mm", "true_distance_change_net_mm"),
            bool(tested) and not distance_moved,
            refutes=distance_moved,
        ),
        clause("reported, not predicted: runs that never detached", reported or {"none": True}, True),
    ]
    untested_runs = {k: v for k, v in views.items() if v["status"] != "tested"}
    if untested_runs:
        clauses.append(jiv.vacuous("runs that detached but cannot be measured", untested_runs, "see each run's why"))
    missing = unscored(records, SWEEP_TARGETS)
    if missing:
        clauses.append(jiv.not_judged("sweep runs not scored", missing))
    untested = None if tested else ("no detached run could be measured" if views else "no sweep run detached")
    return judged("S5", clauses, untested)


# ------------------------------------------------------------------------------------------------ S6 (P1 amended)
def score_s6(records):
    counts, depths, coverage = {}, {}, {}
    mismatched, violated, literal, uncovered = [], [], [], []
    for record in records:
        k, p1 = key(record), record.get("p1")
        if record["unscored_reason"] is not None or p1 is None:
            coverage[k] = {"judged": False, "why": record["unscored_reason"]}
            uncovered.append(k)
            continue
        counts[k] = jiv.p1_views(p1)[0]
        depths[k] = p1["depth"]
        coverage[k] = {
            "judged": True,
            "frames_recorded": p1["frames_recorded"],
            "missing_mask_record_frames": p1["missing_mask_record_frames"],
            "missing_depth_frames": p1["depth"]["missing_depth_frames"],
        }
        mismatched += [k] if p1["count_mismatches"] else []
        violated += [k] if p1["depth"]["edge_band_sensitivity"]["violations_outside_edge_band"] else []
        literal += [k] if p1["depth"]["violations"] else []
        if p1["missing_mask_record_frames"] or p1["depth"]["missing_depth_frames"]:
            uncovered.append(k)
    checked = sum(entry["frames_checked"] for entry in counts.values())
    depth_checked = sum(entry["frames_checked"] for entry in depths.values())
    clauses = [
        clause(
            "the recorded mask pixel count equals the offline reconstruction on every recorded frame",
            counts,
            checked > 0 and not mismatched,
            refutes=bool(mismatched),
        ),
        clause(
            "no predicted jaw pixel (centre at least 0.01 px inside the undilated silhouette) shows recorded depth "
            "more than 1 mm beyond the analytic box depth (as amended October 2)",
            {
                "amendment_label": "P1 amended October 2 (0.01 px outline band)",
                "edge_band_listing_note": "depth_test lists the band hits beyond 1 mm (signed distance and depths) "
                "and counts every band hit; band hits within 1 mm are counted, not listed",
                "runs": depths,
                "runs_with_violations": violated,
                "literal_reading_not_registered_here": {
                    "runs_with_violations": literal,
                    "violations": sum(entry["violations"] for entry in depths.values()),
                },
            },
            depth_checked > 0 and not violated,
            refutes=bool(violated),
        ),
        clause(
            "every frame of all thirty planned runs was recorded and checked (coverage; not a refutation condition)",
            coverage,
            not uncovered and len(records) == sum(EXPECTED_RUNS.values()),
        ),
    ]
    return judged("S6", clauses, None if counts else "no scored run has recorded frames")


# ------------------------------------------------------------------------------------------------ S7
def positive_control(record):
    c = jiv.closure_facts(record)
    k = c["closure_start_frame"]
    held, detach = c["held_frames"], c["detach_frame"]
    frames = jiv.indexed(record)
    held_stops = [i for i in held if frames[i]["cut_phase"] == "stopped"]
    expected = list(range(k + jiv.FIRST_LOSS_OFFSET, k + jiv.CLOSURE_FRAMES + 1)) if k is not None else None
    return {
        **stop_view(record),
        "held_frames": held,
        "expected_held_frames": expected,
        "detach_frame": detach,
        "held_stop_frames": held_stops,
        "holds": passed(record) and held == expected and detach == (None if k is None else k + 6) and not held_stops,
    }


def score_s7(records):
    observed, refutes, holds = {}, [], True
    for target in (POSITIVE_CONTROL, HOLDLESS_CONTROL):
        runs = [r for r in records if r["target"] == target]
        for record in runs:
            k = key(record)
            if record["unscored_reason"] is not None:
                observed[k] = {"scored": False, "why": record["unscored_reason"]}
                holds = False
                continue
            failed = not passed(record)
            if target == POSITIVE_CONTROL:
                view = positive_control(record)
                observed[k] = view
                holds = holds and view["holds"]
                if failed:
                    refutes.append(f"{k}: failed a check")
                if view["held_stop_frames"]:
                    refutes.append(f"{k}: held frames {view['held_stop_frames']} stopped")
            else:
                window = jiv.control_window(record)
                view = {
                    **stop_view(record),
                    **{n: window[n] for n in ("detach_frame", "window", "window_complete", "masked", "held_frames")},
                }
                observed[k] = view
                holds = holds and not failed and window["window_complete"] and not window["masked"]
                holds = holds and not window["held_frames"] and not window["patch_telemetry_missing_while_tracking"]
                if failed:
                    refutes.append(f"{k}: failed a check")
                if window["held_frames"]:
                    refutes.append(f"{k}: held frames {window['held_frames']}")
                if window["masked"]:
                    refutes.append(f"{k}: masked patch elements before detach")
    clauses = [
        clause(
            "530 passes with the hold over k+2..k+6 and detach at k+6; 14944 passes, never holds and masks no patch "
            "element before detach",
            {"controls": observed, "refuting": refutes},
            holds and len(observed) == 2,
            refutes=bool(refutes),
        )
    ]
    return judged("S7", clauses)


def score(records):
    jiv.check_rows(records)
    return [
        score_s1(records),
        score_s2(records),
        score_s3(records),
        score_s4(records),
        score_s5(records),
        score_s6(records),
        score_s7(records),
    ]


# ------------------------------------------------------------------------------------------------ document
def run_summary(record):
    c = jiv.closure_facts(record)
    return {
        "batch": record["batch"],
        "run_directory": record["run_directory"],
        "target": record["target"],
        "label": LABEL,
        "outcome": record["outcome"],
        "status": record["status"],
        "checks": f"{record['checks_passed']}/{record['checks_total']}",
        "failed_checks": record["failed_checks"],
        "configuration_matches": record["configuration_matches"],
        "scored": record["unscored_reason"] is None,
        "unscored_reason": record["unscored_reason"],
        "frames_recorded": None if record["frames"] is None else len(record["frames"]),
        "closure_start_frame": c["closure_start_frame"],
        "first_loss_frame": c["first_loss_frame"],
        "held_frames": c["held_frames"],
        "detach_frame": c["detach_frame"],
        "cut_stop": [c["cut_stop_frame"], c["cut_stop_reason"]],
        "shadow_stop": [c["shadow_stop_frame"], c["shadow_stop_reason"]],
        "node": record["node"],
        "gpu_model": record["gpu_model"],
    }


def build_document(records, scored_code, scorer, inputs, code_revision, code_tree_dirty):
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "label": LABEL,
        "protocol": PROTOCOL,
        "code_revision": code_revision,
        "code_tree_dirty": code_tree_dirty,
        "scorer_sha256": scorer["current_sha256"] if scorer else None,
        "frozen_scorer": scorer,
        "construction_notes": CONSTRUCTION_NOTES,
        "scored_code": scored_code,
        "inputs_sha256": inputs,
        "grader_checks": list(jiv.GRADER_CHECKS),
        "interpretations": INTERPRETATIONS,
        "runs": [run_summary(record) for record in records],
        "predictions": score(records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="The scorer's own checkout (no other root is accepted)")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    if args.root.resolve() != ROOT:
        parser.error(f"--root must be this scorer's own checkout ({ROOT})")
    try:
        digest = verify_self(ROOT)  # before any grade file is opened
        validated = validated_evidence(ROOT)
        scorer = frozen_scorer(validated, digest)
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
        scored_code = jiv.frozen_code(ROOT, validated)
    except (InputError, jiv.EvidenceError, jiv.FrozenCodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    import cv2

    cv2.setNumThreads(1)
    records = load(ROOT, validated)
    inputs = {}
    for path, _, batch_dir, _, _ in validated:
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        inputs[f"{batch_dir}/plan.json"] = sha256(jiv.batch_path(ROOT, batch_dir) / "plan.json")
    for record in records:
        inputs.update(record["inputs"])
    document = build_document(records, scored_code, scorer, inputs, head, dirty)
    serialized = json.dumps(jiv.json_safe(document), indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
