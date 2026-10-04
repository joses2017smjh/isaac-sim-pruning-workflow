#!/usr/bin/env python3
"""Score the closure-hold generalization experiment (protocol of October 4) against H1-H7.

Outcomes come from the per-batch evidence ``aggregate_eval.py`` wrote (the grader's classes, never re-graded
here); every planned run stays in every table. The evidence is checked against the registered seven batches and
eleven runs before anything is scored. Each run's own records are read by the committed jaw-in-view scorer
(``tools/score_jaw_in_view.py``), whose per-run logic this scorer reuses unchanged: the capture and mask
reconstruction (``run_record``), closure facts, the P4 mechanism view, the P5 hold coverage, the P8 static-branch
check and the P2 control window. As there, the scored modules are imported from the batches' frozen snapshot
(``<batch>/code``), never from the checkout.

Frame numbers are 0-based ``frames.json`` indices; closure start ``k`` is the first frame whose cut phase is
``closing``; the deadline is ``k + 6``; a stop is decided on the first frame whose cut phase is ``stopped``. Where
the protocol leaves a detail open it is fixed here and recorded in ``interpretations``. Every result carries the
label: known-map plan; jaw self-mask; closure hold with freshness and frame-reuse checks waived on held frames.
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

PROTOCOL = "docs/EVAL_PROTOCOL_HOLD_GENERALIZATION_2026-10-04.md"
EVIDENCE_DIR = "docs/evidence/hold_gen_2026-10-04"
GRADER = "tools/validate_vision_sequence.py:grade_sequence"
SELF = "tools/score_hold_generalization.py"
#: The scoring code: this scorer and the committed modules it imports. Every plan must have frozen each one.
SCORING_FILES = (
    SELF,
    "tools/score_jaw_in_view.py",
    "tools/score_perception_round.py",
    "tools/score_planned_approach.py",
)
LABEL = jiv.LABEL
STRATEGY = "planned_pose_jaw_hold_gen"
STRATEGY_S100 = "planned_pose_jaw_hold_gen_s100"
SELECTED = (3721, 36196, 18143)
POSITIVE_CONTROL, HOLDLESS_CONTROL = 530, 14944
#: The target register each batch was planned from (the controls reuse the jaw-in-view register unchanged).
REGISTER_A = "docs/evidence/eval_targets_hold_gen_a_2026-10-04.json"
REGISTER_18143 = "docs/evidence/eval_targets_hold_gen_18143_2026-10-04.json"
REGISTER_CONTROLS = "docs/evidence/eval_targets_jaw_hold_b_2026-09-30.json"
#: batch -> (registered (tree, target) in run-index order, strategy)
REGISTRY = {
    **{f"hold-gen-a-r{i}-20261004": (((0, 3721), (1, 36196)), STRATEGY) for i in (1, 2, 3)},
    **{f"hold-gen-18143-r{i}-20261004": (((1, 18143),), STRATEGY_S100) for i in (1, 2, 3)},
    "hold-gen-ctl-20261004": (((0, POSITIVE_CONTROL), (1, HOLDLESS_CONTROL)), STRATEGY),
}
EXPECTED_RUNS = {3721: 3, 36196: 3, 18143: 3, POSITIVE_CONTROL: 1, HOLDLESS_CONTROL: 1}
#: The registered strategies' fields (tools/queue_vision_robustness.py); a plan row must carry exactly these.
STRATEGY_FIELDS = {
    STRATEGY: {"mode": "planned_pose_standoff", "standoff_m": 0.06, "max_step_m": 0.004, "max_rotation_deg": 1.5},
    STRATEGY_S100: {"mode": "planned_pose_standoff", "standoff_m": 0.10, "max_step_m": 0.004, "max_rotation_deg": 1.5},
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
H1_MIN_STARTS = 2  # H1: closure starts in at least 2 of a target's 3 runs

CONSTRUCTION_NOTES = [
    "This scorer is committed before submission: before any hold-generalization run was submitted. When it was "
    "written the checkout held no hold-gen batch directory, grade file or log; its author did not query Slurm for "
    "this experiment. It is written blind, and no outcome of this experiment could shape it.",
    "It reuses the per-run logic of tools/score_jaw_in_view.py, the scorer committed blind for the jaw-in-view "
    "experiment and published with that experiment's verdicts; the clause logic here is adapted from that scorer's "
    "P2, P4, P5, P6 and P8 with the target sets of this protocol. The jaw-in-view recordings and verdicts are "
    "another experiment's published results.",
    "Before it was committed it was reviewed blind by an independent reviewer (no run of this experiment existed). "
    "The review found the control batch's plan would have been refused (14944's planned orientation is None, the "
    "home orientation), that the imported jaw-in-view modules were not required to equal their frozen copies, that "
    "unscored runs of targets that already hold changed H1 and H4, and 28 surviving one-line mutations; all were "
    "fixed, and a re-run of the mapped mutations (pruning_work/hold_gen_review_20261004/rerun) killed 35 of 35.",
    "The runner (tools/run_vision_experiment.py execute) exits 0 only on a 17/17 pass with configuration_matches "
    "true, so each task's Slurm state reveals that run's pass or fail (H4's and H7's pass clauses). Because this "
    "scorer is committed before submission, nothing seen afterwards can shape it.",
]
INTERPRETATIONS = [
    "Plans: each plan must have been made from its registered register as committed (path and sha256), every row's "
    "strategy must have exactly the registered keys and values, and its planned orientation must equal the "
    "register's (None, the home orientation, for 14944).",
    "Run set: the grade files must be exactly the 7 registered batches (any other file in the evidence directory "
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
    "Unscored runs: in H1 and H4 an unscored run is listed as not judged only for a target that does not already "
    "hold, since a target that holds on its scored runs holds whatever its unscored run would show; in H2, H3, H5 and "
    "H6 every unscored selected-target run keeps the prediction from being supported, because it might fall inside "
    "the population. A pass is graded with checks_passed == checks_total == 17; a row whose outcome disagrees refuses.",
    "H1: a run starts closure when it has a frame whose cut phase is 'closing'. A target holds with at least 2 "
    "starts among its scored runs; it refutes with at least 2 scored runs and no start; otherwise it neither holds "
    "nor refutes. Each run that stopped before closure is listed with its cut stop frame and reason and the tracker "
    "state there.",
    "H2: tools/score_jaw_in_view.py's P4 rule (mechanism, p4_category) over the selected targets' scored runs: "
    "predicted for runs whose gate mouth distance at closure start is at most 4 mm; a run with its first loss at "
    "k+1 is judged; a run whose cut left closing at k+1 for a non-vision reason never rendered the jaw closing and "
    "is untested for H2; runs above 4 mm or without a closure start in a settled recording are reported.",
    "H3: tools/score_jaw_in_view.py's P5 rule (p5_category, hold_coverage) over the selected targets' scored runs: "
    "the population is the runs where H2's loss occurs (first closure loss at k+2, mask-attributable, cut still "
    "closing after k+1); coverage, tool and mouth are registered claims that cannot refute; a held frame that stops "
    "refutes, and so does detachment at another frame or no detachment once frame k+6 is recorded or the cut "
    "stopped by k+6.",
    "H4: per selected target, a pass is the grader's class 'pass' (17/17) on a scored run. A target holds with at "
    "least one pass; it refutes when at least one of its scored runs started closure and none passed; a target "
    "whose scored runs never started closure is untested for H4. Supported when every target holds; untested when "
    "no target is judged.",
    "H5: tools/score_jaw_in_view.py's P8 rule (static_branch) over the selected targets' scored runs that "
    "detached: the piece's PhysX origin and the true mouth-to-spur distance over [k, detach], both limits strict.",
    "H6: tools/score_jaw_in_view.py's per-frame mask reconstruction over every scored run: the count clause refutes "
    "on any mismatch; the depth clause judges predicted jaw pixels as amended on October 2 (centre at least 0.01 px "
    "inside the undilated silhouette: depth_test's violations_outside_edge_band) and refutes on any; the literal "
    "count over every hit is reported; unscored runs are listed in a coverage clause that cannot refute.",
    "H7: 530 holds when scored, a pass, held exactly k+2..k+6 with no held frame stopping, and detached at k+6; "
    "14944 holds when scored, a pass, never held (tools/score_jaw_in_view.py control_window: closure_hold.held or a "
    "closure_hold certificate on any frame) and no patch element masked before detach (window 0..d). A scored "
    "control that failed a check refutes; 14944 holding or masking a patch element before detach refutes; a held "
    "frame of 530 that stops refutes. An unscored control withholds support.",
    "Verdicts: refuted when any clause meets its registered refutation; otherwise 'untested' when the condition a "
    "prediction needs never arose (named in untested_reason); otherwise supported when every clause holds and "
    "partly supported when one does not.",
]
TEXT = {
    "H1": "(the planned approach reaches closure) Each selected target starts closure in at least 2 of its 3 runs. "
    "Refuted if a target with at least 2 scored runs starts closure in none of them.",
    "H2": "(mechanism, as P4) In selected-target runs whose closure starts at a gate mouth distance of at most 4 mm, "
    "the first closure loss comes at k+2 and is mask-attributable; the shadow controller stops there. Refuted if "
    "tracking continues at k+3, or the first loss comes at another frame or is not mask-attributable.",
    "H3": "(hold to detachment, as P5) Where H2's loss occurs, the hold covers it to the deadline, the tool stays "
    "within 0.5 mm and 0.25 deg, held mouth distances stay within 0.05 mm of the closure-start value, and "
    "detachment comes at k+6. Refuted if any held frame stops or detachment comes at another frame.",
    "H4": "(outcome) Each selected target passes all 17 checks in at least 1 of its runs. Refuted if a target with "
    "at least one scored run that started closure has no pass.",
    "H5": "(static-branch premise, as P8) In every selected-target run that detaches, the piece moves less than "
    "0.1 mm and the true mouth-to-spur distance changes by less than 0.1 mm between closure start and detach.",
    "H6": "(mask fidelity, P1 as amended October 2) On every frame of all runs the recorded mask count equals the "
    "reconstruction, and no predicted jaw pixel (at least 0.01 px inside the silhouette) shows depth more than "
    "1 mm beyond the box. Refuted by any mismatch.",
    "H7": "(controls) 530 passes 17/17 with the hold over k+2..k+6 and detach at k+6; 14944 passes 17/17, never "
    "holds and has no patch element masked before detach. Refuted if either fails a check, 14944 holds or masks a "
    "patch element before detach, or a held frame of 530 stops.",
}
SCOPE = (
    "Scoring of the closure-hold generalization experiment (H1-H7): eleven simulator runs of the jaw self-mask and "
    "closure hold in seven batches (tree0 3721, tree1 36196 and tree1 18143 three times each; controls 530 and "
    "14944). Every result is labelled '" + LABEL + "' and is reported apart from every unchanged-gate result, never "
    "pooled. Outcomes are the grader's, read from the per-batch evidence. Simulator renders of a two-box jaw "
    "surrogate with exact poses; not a field result."
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
    if name.startswith("hold-gen-a-"):
        return REGISTER_A
    if name.startswith("hold-gen-18143-"):
        return REGISTER_18143
    return REGISTER_CONTROLS


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


# ------------------------------------------------------------------------------------------------ H1
def score_h1(records):
    per_target, refuting, short = {}, [], []
    for target in SELECTED:
        runs = scored(records, (target,))
        starts = [key(r) for r in runs if jiv.closure_facts(r)["closure_start_frame"] is not None]
        before = {key(r): stop_view(r) for r in runs if jiv.closure_facts(r)["closure_start_frame"] is None}
        per_target[str(target)] = {
            "scored_runs": len(runs),
            "closure_starts": len(starts),
            "stopped_before_closure": before,
        }
        if len(starts) < H1_MIN_STARTS:
            short.append(target)
        if len(runs) >= H1_MIN_STARTS and not starts:
            refuting.append(target)
    clauses = [
        clause(
            "each selected target starts closure in at least 2 of its scored runs",
            {"targets": per_target, "short": short, "refuting": refuting},
            not short,
            refutes=bool(refuting),
        )
    ]
    missing = unscored(records, tuple(short))  # an unscored run matters only where its target does not hold
    if missing:
        clauses.append(jiv.not_judged("runs not scored, of targets that do not hold", missing))
    clauses.append(
        clause("informational: every unscored selected-target run", unscored(records, SELECTED) or {"none": True}, True)
    )
    return judged("H1", clauses)


# ------------------------------------------------------------------------------------------------ H2 (as P4)
def score_h2(records):
    groups = {name: {} for name in ("predicted", "reported", "premise", "not_evaluable")}
    for record in scored(records, SELECTED):
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
            "reported, not predicted: selected-target runs whose closure started above 4 mm or never started",
            groups["reported"] or {"none": True},
            True,
        ),
    ]
    if groups["premise"]:
        clauses.append(
            jiv.vacuous(
                "runs that left closing at k+1 for a non-vision reason, so the jaw was never rendered closing",
                groups["premise"],
                "H2's premise (a first frame rendered closing at k+2) never arose in these runs",
            )
        )
    missing = {**unscored(records, SELECTED), **groups["not_evaluable"]}
    if missing:
        clauses.append(jiv.not_judged("selected-target runs not scored or not settled", missing))
    untested = None
    if not predicted:
        untested = "no scored selected-target run started closure at a gate mouth distance of at most 4 mm"
        if groups["premise"]:
            untested += "; in every such run the cut left closing at k+1"
    return judged("H2", clauses, untested)


# ------------------------------------------------------------------------------------------------ H3 (as P5)
def score_h3(records):
    applies, groups = {}, {name: {} for name in ("reported", "not_evaluable")}
    for record in scored(records, SELECTED):
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
            "the hold covers every frame from H2's loss (k+2) to the deadline (not a refutation condition)",
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
            "reported, not predicted: runs where H2's loss did not occur", groups["reported"] or {"none": True}, True
        ),
    ]
    missing = {**unscored(records, SELECTED), **groups["not_evaluable"]}
    if missing:
        clauses.append(jiv.not_judged("selected-target runs not scored or not settled", missing))
    untested = None if any(v["evaluable"] for v in applies.values()) else "no scored selected-target run had H2's loss"
    if applies and untested:
        untested = "every run with H2's loss ends before the deadline with nothing settled"
    return judged("H3", clauses, untested)


# ------------------------------------------------------------------------------------------------ H4
def score_h4(records):
    per_target, refuting, untested_targets, holding = {}, [], [], []
    for target in SELECTED:
        runs = scored(records, (target,))
        started = [r for r in runs if jiv.closure_facts(r)["closure_start_frame"] is not None]
        passes = [key(r) for r in runs if passed(r)]
        per_target[str(target)] = {
            "runs": {key(r): stop_view(r) for r in runs},
            "passes": passes,
            "scored_runs_that_started_closure": len(started),
        }
        if passes:
            holding.append(target)
        elif started:
            refuting.append(target)
        else:
            untested_targets.append(target)
    clauses = [
        clause(
            "each selected target passes all 17 checks in at least 1 of its runs",
            {"targets": per_target, "without_a_pass_after_closure": refuting, "untested": untested_targets},
            len(holding) == len(SELECTED),
            refutes=bool(refuting),
        )
    ]
    open_targets = tuple(t for t in SELECTED if t not in holding)
    missing = unscored(records, open_targets)  # an unscored run matters only where its target has no pass
    if missing:
        clauses.append(jiv.not_judged("runs not scored, of targets without a pass", missing))
    clauses.append(
        clause("informational: every unscored selected-target run", unscored(records, SELECTED) or {"none": True}, True)
    )
    untested = None
    if not holding and not refuting:
        untested = "no selected target had a scored run that passed or started closure"
    return judged("H4", clauses, untested)


# ------------------------------------------------------------------------------------------------ H5 (as P8)
def score_h5(records):
    views, reported = {}, {}
    for record in scored(records, SELECTED):
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
    missing = unscored(records, SELECTED)
    if missing:
        clauses.append(jiv.not_judged("selected-target runs not scored", missing))
    untested = None if tested else ("no detached run could be measured" if views else "no selected-target run detached")
    return judged("H5", clauses, untested)


# ------------------------------------------------------------------------------------------------ H6 (P1 amended)
def score_h6(records):
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
            "every frame of all eleven planned runs was recorded and checked (coverage; not a refutation condition)",
            coverage,
            not uncovered and len(records) == sum(EXPECTED_RUNS.values()),
        ),
    ]
    return judged("H6", clauses, None if counts else "no scored run has recorded frames")


# ------------------------------------------------------------------------------------------------ H7
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


def score_h7(records):
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
    return judged("H7", clauses)


def score(records):
    jiv.check_rows(records)
    return [
        score_h1(records),
        score_h2(records),
        score_h3(records),
        score_h4(records),
        score_h5(records),
        score_h6(records),
        score_h7(records),
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
