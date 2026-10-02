#!/usr/bin/env python3
"""Score the render-gap check (protocol of September 30) against G0-G3.

Model errors come from the aggregate evidence that the committed aggregation tools
(``aggregate_family_eval.py``, recompute on, as published; ``anchor_depth_analysis.py``) wrote for the Matched and
Native arms of the render-gap evaluation, with the published model labels. F's published source error (G3) comes
from the published rendered-lighting matrix. G0 re-applies the registered geometry thresholds to every view of the
evaluation's own record (``eval/pose_identity.json``). Nothing is re-evaluated here: no prediction, depth array,
image or checkpoint is read.

Before any prediction the inputs are checked (``check_inputs``, I1-I7): the published pins, the plan, the gate's
thresholds, every evaluation's provenance, the aggregates' links back to those evaluations, the tree sets of each
arm, and that every Matched frame keeps the published ground truth. A failed check raises ``InputError`` and no
output is written. A cell with a view count other than 6 (I8) is not a refusal; the tree becomes unscored for any
clause that uses it.

Where the protocol leaves a detail open it is fixed here and recorded in ``interpretations``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md"
BATCH = "artifacts/generalization/render-gap-20260930"
ARMS = ("matched", "native")
EVIDENCE = {
    "matched": "docs/evidence/render_gap_matched_matrix_2026-10-01.json",
    "native": "docs/evidence/render_gap_native_matrix_2026-10-01.json",
    "matched_anchoring": "docs/evidence/render_gap_matched_anchoring_2026-10-01.json",
    "native_anchoring": "docs/evidence/render_gap_native_anchoring_2026-10-01.json",
    "published": "docs/evidence/finetune_rendered_family_matrix_2026-09-30.json",
    "published_anchoring": "docs/evidence/finetune_rendered_anchoring_2026-09-30.json",
}
#: I1: the published evidence this check compares against, pinned by content.
PUBLISHED_PINS = {
    EVIDENCE["published"]: "e34a744987fc0b867d20f0a6477f711563c54ca657913e378144a923c323d0f0",
    EVIDENCE["published_anchoring"]: "d24032af5ebd16e2a8df7119e8d1350b979dd80a76d8ebb7fe9bf7b8a44f4ff4",
}
MODEL = {
    "F": "da2_frozen",
    "J": "da2_jitter",
    "C": "da2_control",
    "A": "da2_rendered_jitter",
    "B": "da2_rendered_control",
}
PRESETS = ("source", "morning", "noon", "evening")
TREES = 8
TREES_PER_FAMILY = 4
VIEWS = 6
FRAMES_PER_TREE = 24
FRAMES_PER_MODEL = TREES * FRAMES_PER_TREE
#: G0's registered gate (training_lighting.agreement thresholds, unchanged).
G0_TOLERANCE_M = 0.001
G0_MIN_WITHIN = 0.995
G0_MIN_IOU = 0.995
#: The aggregation job that wrote the four aggregates (SLURM_JOBS.md).
AGGREGATION = {
    "job": "21501978",
    "partition": "share",
    "clone_revision": "f3442dffce5480088fc5dd268359d8ef270a73cd",
}
INTERPRETATIONS = [
    "Per-tree errors are the published lighting_effect values (sql/family/03: the mean tree-mask MAE over 6 views, "
    "4 dp), as R3-R5 used them. Ratios are of those values. Unrounded means are recorded and never decide.",
    "'At least' is >= and 'at most' is <=. G3's 'at least as large' is a ratio >= 1.0.",
    "G1 is one clause per arm (A/C and B/C), as R5 scored \"A's and B's\". The protocol's 'A or B' refutation means "
    "any refuting clause refutes.",
    "A ratio between a hold threshold and a refutation threshold counts toward neither. A prediction that neither "
    "holds nor is refuted is 'partly supported'.",
    "G2 states a refutation for its Envy clause only. The UFO clause (A/J <= 0.5) cannot refute, as R5's third "
    "clause could not.",
    "Counts are absolute over the registered trees (6 of 8; 3 of a family's 4). An unscored tree counts toward "
    "neither and is listed with its reason; a clause with fewer scored trees than it needs is not decidable.",
    "G0 is scored from the gate's own record (pose_identity.json, job 21499599, training_lighting.agreement). The "
    "registered thresholds are re-applied to every view, and a view without values fails. A disagreement between "
    "the record and the re-applied gate is an input error, not a verdict.",
    "A registered tree the gate never checked (not_checked) counts as not passing G0, as "
    "prepare_render_gap_evaluation.py records it.",
    "The 1920x1080 point-sampled geometry is a measurement and never gates.",
    "G3's denominator is F's source lighting_effect value in finetune_rendered_family_matrix_2026-09-30.json.",
    "Native against Matched is Native / Matched of the per-tree values and of the family means. It confounds "
    "resolution, input filtering and ground truth, and is not attributed to any one of them.",
    "Matched against published, for every model and preset, is a measurement. Only F at source is registered (G3).",
]
#: The protocol's readings, keyed by (G1 verdict, G2 verdict); None stands for any G1 verdict.
READINGS = {
    ("supported", "supported"): "The source gain was renderer robustness, and the low-sun gain is lighting.",
    ("refuted", "supported"): "Both gains are real, beyond the render gap.",
    (None, "refuted"): "The lighting claim does not survive training-like renders.",
}


class InputError(ValueError):
    """An input that this check refuses to score (the file and field are in the message)."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count(values, test):
    return sum(1 for v in values if test(v))


def per_tree(rows, model, **match):
    """{tree: error} from lighting_effect rows of ``model`` matching every field in ``match``."""
    out = {}
    for row in rows:
        if row["model"] == model and all(row.get(k) == v for k, v in match.items()):
            if row["tree_id"] in out:
                raise InputError(f"duplicate lighting_effect row for {model} {row['tree_id']} {match}")
            out[row["tree_id"]] = row["mask_mae_m"]
    return dict(sorted(out.items()))


def per_tree_unrounded(frames, model, lighting):
    """{tree: unrounded mean mask MAE over the tree's frames}, never deciding."""
    groups = defaultdict(list)
    for row in frames:
        if row["model"] == model and row["lighting"] == lighting and row.get("mask_mae_m") is not None:
            groups[row["tree_id"]].append(row["mask_mae_m"])
    return {tree: statistics.fmean(values) for tree, values in sorted(groups.items())}


def cell_views(frames):
    """{(model, tree, lighting): number of views with a mask MAE} (I8)."""
    return Counter(
        (row["model"], row["tree_id"], row["lighting"]) for row in frames if row.get("mask_mae_m") is not None
    )


def ratios(numerator, denominator, trees):
    return {
        tree: numerator[tree] / denominator[tree]
        for tree in sorted(trees)
        if numerator.get(tree) is not None and denominator.get(tree)
    }


# ------------------------------------------------------------------------------------------------ inputs
def _relative(root, path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def load(root):
    """Every input the scorer reads, with the path of each file for inputs_sha256."""
    root = Path(root)
    batch = root / BATCH
    files = {}

    def read(path):
        path = Path(path)
        files[_relative(root, path)] = path
        return json.loads(path.read_text())

    data = {name: read(root / path) for name, path in EVIDENCE.items()}
    data["plan"] = read(batch / "plan.json")
    data["receipts"] = {
        kind: read(batch / f"submission_receipt_{kind}.json")["stdout"].strip() for kind in ("render", "eval")
    }
    data["render_status"] = read(batch / "eval/render_status.json")
    data["pose_identity"] = read(batch / "eval/pose_identity.json")
    data["arm_status"] = {arm: read(batch / f"eval/{arm}/status.json") for arm in ARMS}
    data["arm_plan_frames"] = {arm: read(batch / f"eval/{arm}/plan.json")["frames"] for arm in ARMS}
    data["arm_plan_sha256"] = {arm: sha256(batch / f"eval/{arm}/plan.json") for arm in ARMS}
    published_plan = Path(data["plan"]["matrix_evaluation_plan"])
    data["published_plan_frames"] = read(published_plan)["frames"]
    data["published_plan_sha256"] = sha256(published_plan)
    data["evaluations"] = {}
    for arm in ARMS:
        for label in MODEL:
            evaluation = read(batch / f"eval/{arm}/{label}/evaluation.json")
            data["evaluations"][(arm, label)] = {
                "ok": evaluation.get("ok"),
                "has_error": evaluation.get("error") is not None,
                "job_id": evaluation.get("job_id"),
                "manifest_sha256": evaluation.get("manifest_sha256"),
                "rows": len(evaluation.get("rows") or []),
                "checkpoint_sha256": (evaluation.get("model") or {}).get("checkpoint_sha256"),
            }
            del evaluation  # one evaluation in memory at a time
    data["manifests"] = {}
    for tree in data["plan"]["trees"]:
        path = batch / "render" / tree["tree_id"] / "render_manifest.json"
        if path.is_file():
            manifest = read(path)
            data["manifests"][tree["tree_id"]] = {
                "ok": manifest.get("ok"),
                "smoke": manifest.get("smoke"),
                "renderer_mode": manifest.get("renderer_mode"),
                "frames_per_light": dict(Counter(frame.get("lighting") for frame in manifest.get("frames") or [])),
                "grid_views": len((manifest.get("matrix_grid_geometry") or {}).get("views") or []),
            }
    data["files"] = files
    return data


def _require(condition, message):
    if not condition:
        raise InputError(message)


def render_ok_trees(data):
    """I6's R: trees present and rendered, whose manifest is a complete training-renderer render."""
    status = {tree["tree_id"]: tree for tree in data["render_status"]["trees"]}
    ok = set()
    for tree_id, manifest in data["manifests"].items():
        recorded = status.get(tree_id) or {}
        if not (recorded.get("present") and recorded.get("render_ok")):
            continue
        if manifest["ok"] is not True or manifest["smoke"] is not False or manifest["renderer_mode"] != "training":
            continue
        if manifest["frames_per_light"] != dict.fromkeys(PRESETS, VIEWS):
            continue
        if manifest["grid_views"] != VIEWS:
            continue
        ok.add(tree_id)
    return ok


def check_pins(root):
    """I1: the published evidence is the pinned published file."""
    for path, pinned in PUBLISHED_PINS.items():
        _require(sha256(Path(root) / path) == pinned, f"I1: {path} sha256 is not the pinned published file")


def check_plan(data):
    """I2: the plan registered this protocol, 8 trees (4 per family), the five models and the unchanged gate."""
    plan = data["plan"]
    _require(plan.get("protocol") == PROTOCOL, "I2: plan.json protocol")
    families = Counter(tree["family"] for tree in plan["trees"])
    _require(
        len(plan["trees"]) == TREES and families == {"envy": TREES_PER_FAMILY, "ufo": TREES_PER_FAMILY},
        "I2: plan.json trees",
    )
    _require(set(plan["models"]) == set(MODEL), "I2: plan.json model labels")
    gate = plan.get("pose_identity") or {}
    _require(
        (gate.get("tolerance_m"), gate.get("min_fraction"), gate.get("min_iou"))
        == (G0_TOLERANCE_M, G0_MIN_WITHIN, G0_MIN_IOU),
        "I2: plan.json pose_identity thresholds",
    )
    _require(
        data["published_plan_sha256"] == plan.get("matrix_evaluation_plan_sha256"),
        "I2: plan.json matrix_evaluation_plan_sha256",
    )


def check_gate(data):
    """I3: the gate's own record shows the registered thresholds."""
    gate = data["pose_identity"].get("gate") or {}
    _require(
        (gate.get("tolerance_m"), gate.get("min_fraction_within_tolerance"), gate.get("min_mask_iou"))
        == (G0_TOLERANCE_M, G0_MIN_WITHIN, G0_MIN_IOU),
        "I3: pose_identity.json gate thresholds",
    )


def check_evaluations(data, expected_trees):
    """I4: every evaluation is complete, from the evaluation job, on its arm's plan, with its model's checkpoint."""
    plan = data["plan"]
    for arm in ARMS:
        status = data["arm_status"][arm]
        _require(status.get("plan_sha256") == data["arm_plan_sha256"][arm], f"I4: eval/{arm}/status.json plan_sha256")
        frames = data["arm_plan_frames"][arm]
        _require(len(frames) == FRAMES_PER_TREE * len(expected_trees[arm]), f"I4: eval/{arm}/plan.json frame count")
        for label in MODEL:
            evaluation = data["evaluations"][(arm, label)]
            where = f"eval/{arm}/{label}/evaluation.json"
            _require(evaluation["ok"] is True and not evaluation["has_error"], f"I4: {where} ok/error")
            _require(evaluation["job_id"] == data["receipts"]["eval"], f"I4: {where} job_id")
            _require(evaluation["manifest_sha256"] == status.get("plan_sha256"), f"I4: {where} manifest_sha256")
            _require(evaluation["rows"] == len(frames), f"I4: {where} rows")
            _require(
                evaluation["checkpoint_sha256"] == plan["models"][label]["checkpoint_sha256"],
                f"I4: {where} checkpoint_sha256",
            )


def check_aggregates(root, data):
    """I5: each aggregate links back to its arm's five evaluations by path, content and checkpoint."""
    plan = data["plan"]
    labels = {model: label for label, model in MODEL.items()}
    expected_queries = {"matrix": data["published"]["queries"], "anchoring": data["published_anchoring"]["queries"]}
    for arm in ARMS:
        for kind, name in (("matrix", arm), ("anchoring", f"{arm}_anchoring")):
            document = data[name]
            paths = {row["model"]: Path(row["path"]).as_posix() for row in document["inputs"]}
            wanted = {MODEL[label]: f"{BATCH}/eval/{arm}/{label}/evaluation.json" for label in MODEL}
            _require(
                set(paths) == set(wanted) and all(paths[m].endswith(wanted[m]) for m in wanted),
                f"I5: {EVIDENCE[name]} inputs paths",
            )
            for row in document["inputs"]:
                path = Path(row["path"])
                path = path if path.is_absolute() else Path(root) / path
                _require(sha256(path) == row["sha256"], f"I5: {EVIDENCE[name]} input sha256 for {row['model']}")
                _require(
                    row["checkpoint_sha256"] == plan["models"][labels[row["model"]]]["checkpoint_sha256"],
                    f"I5: {EVIDENCE[name]} checkpoint_sha256 for {row['model']}",
                )
                _require(
                    row.get("frames", row.get("frames_scored")) == FRAMES_PER_MODEL,
                    f"I5: {EVIDENCE[name]} frames for {row['model']}",
                )
            _require(document.get("queries") == expected_queries[kind], f"I5: {EVIDENCE[name]} queries")


def check_tree_sets(data, render_ok, g0_passing):
    """I6: Native holds the render-ok trees; Matched holds those that also pass G0, by every record."""
    native_trees = {row["tree_id"] for row in data["native"]["frames"]}
    matched_trees = {row["tree_id"] for row in data["matched"]["frames"]}
    _require(native_trees == set(render_ok), "I6: native aggregate trees are not the render-ok set")
    _require(
        matched_trees == set(render_ok) & set(g0_passing), "I6: matched aggregate trees are not G0-pass and render-ok"
    )
    flagged = {tree["tree_id"] for tree in data["render_status"]["trees"] if tree.get("matched")}
    _require(flagged == matched_trees, "I6: render_status matched flags")
    manifests = {Path(path).parent.name for path in data["arm_status"]["matched"].get("render_manifests") or []}
    _require(manifests == matched_trees, "I6: eval/matched/status.json render_manifests")


def check_ground_truth(data):
    """I7: Matched keeps the published depth, mask, K and target with a new RGB; Native uses its own render."""

    def key(frame):
        return frame["tree_id"], frame["lighting"], frame["view_id"]

    published = {key(frame): frame for frame in data["published_plan_frames"]}
    for frame in data["arm_plan_frames"]["matched"]:
        reference = published.get(key(frame))
        _require(reference is not None, f"I7: no published frame for {key(frame)}")
        for field in ("depth", "mask", "K", "target_pixel_xy"):
            _require(frame.get(field) == reference.get(field), f"I7: matched {key(frame)} {field}")
        _require(frame.get("rgb") != reference.get("rgb"), f"I7: matched {key(frame)} rgb is the published rgb")
    for frame in data["arm_plan_frames"]["native"]:
        prefix = f"render-gap-20260930/render/{frame['tree_id']}/"
        for field in ("depth", "mask", "rgb"):
            _require(prefix in str(frame.get(field)), f"I7: native {key(frame)} {field} outside its render")


def check_inputs(root, data, g0_passing):
    """I1-I7 in order; raises InputError naming the file and field. Returns the render-ok tree set R."""
    check_pins(root)
    check_plan(data)
    check_gate(data)
    render_ok = render_ok_trees(data)
    check_evaluations(data, {"native": render_ok, "matched": render_ok & set(g0_passing)})
    check_aggregates(root, data)
    check_tree_sets(data, render_ok, g0_passing)
    check_ground_truth(data)
    return render_ok


# ------------------------------------------------------------------------------------------------ G0
def view_passes(view):
    """(passes, reason) for one view's recorded grid-pass agreement under the registered gate."""
    gate = view.get("gate") or {}
    iou, within = gate.get("mask_iou"), gate.get("tree_depth_within_tolerance")
    if iou is None or within is None:
        return False, gate.get("reason") or view.get("reason") or "no gate values recorded"
    if gate.get("tolerance_m") != G0_TOLERANCE_M:
        return False, f"tolerance {gate.get('tolerance_m')} is not the registered {G0_TOLERANCE_M}"
    if iou >= G0_MIN_IOU and within >= G0_MIN_WITHIN:
        return True, None
    return False, f"mask IoU {iou}, within 1 mm {within}"


def score_g0(pose_identity, families, registered=None):
    """G0 per tree; returns (prediction, passing trees, table). A record/gate disagreement raises InputError."""
    recorded = {tree["tree_id"]: tree for tree in pose_identity.get("trees") or []}
    not_checked = {
        (entry.get("tree_id") if isinstance(entry, dict) else entry): (
            entry.get("reason") if isinstance(entry, dict) else "never checked"
        )
        for entry in pose_identity.get("not_checked") or []
    }
    universe = sorted(set(registered or ()) | set(recorded) | set(not_checked))
    table, failing = {}, {}
    for tree_id in universe:
        tree = recorded.get(tree_id)
        if tree is None or tree_id in not_checked:
            failing[tree_id] = not_checked.get(tree_id) or "absent from the gate's record"
            table[tree_id] = {"family": families.get(tree_id), "recorded_ok": None, "reason": failing[tree_id]}
            continue
        views = tree.get("views") or []
        results = {view.get("view_id"): view_passes(view) for view in views}
        failing_views = {view: reason for view, (ok, reason) in results.items() if not ok}
        complete = (
            tree.get("matrix_grid_pass") is True
            and len(views) == tree.get("views_checked") == VIEWS
            and not tree.get("missing_views")
            and not tree.get("extra_views")
        )
        ok = complete and not failing_views
        gates = [view.get("gate") or {} for view in views]
        table[tree_id] = {
            "family": families.get(tree_id),
            "views_checked": len(views),
            "views_passing": count(results.values(), lambda result: result[0]),
            "failing_views": failing_views,
            "complete": complete,
            "recorded_ok": tree.get("ok"),
            "min_mask_iou": min((g["mask_iou"] for g in gates if g.get("mask_iou") is not None), default=None),
            "min_within_1mm": min(
                (g["tree_depth_within_tolerance"] for g in gates if g.get("tree_depth_within_tolerance") is not None),
                default=None,
            ),
        }
        if not ok:
            failing[tree_id] = "incomplete record" if not complete else "; ".join(sorted(map(str, failing_views)))
        if tree.get("ok") is not ok:
            raise InputError(f"G0: pose_identity.json tree {tree_id} records ok={tree.get('ok')}, re-applied {ok}")
    passing = sorted(set(universe) - set(failing))
    if sorted(pose_identity.get("matched_trees") or []) != passing:
        raise InputError("G0: pose_identity.json matched_trees differs from the re-applied gate")
    excluded = set(pose_identity.get("excluded_trees") or []) | set(not_checked)
    if excluded != set(failing):
        raise InputError("G0: pose_identity.json excluded_trees and not_checked differ from the re-applied failures")
    views_passing = sum(row.get("views_passing", 0) for row in table.values())
    g0 = prediction(
        "G0",
        "Every tree's grid-pass depth and mask reproduce the published matrix in every view.",
        [
            clause(
                "all 8 registered trees pass the registered gate in all 6 views (>= 99.5% of tree-mask pixels "
                "within 1 mm, mask IoU >= 0.995)",
                {
                    "passing": f"{len(passing)} of {TREES}",
                    "views_passing": f"{views_passing} of {TREES * VIEWS}",
                    "failing": failing,
                    "min_mask_iou": min(
                        (r["min_mask_iou"] for r in table.values() if r.get("min_mask_iou") is not None), default=None
                    ),
                    "min_within_1mm": min(
                        (r["min_within_1mm"] for r in table.values() if r.get("min_within_1mm") is not None),
                        default=None,
                    ),
                },
                len(passing) == TREES and len(universe) == TREES,
                refutes=len(passing) < TREES,
            )
        ],
    )
    return g0, passing, table


def cross_grid(pose_identity):
    """The 1920x1080 geometry point-sampled at matrix pixel centres: a measurement, never a gate."""
    per_tree_values, would_pass, total = {}, 0, 0
    for tree in pose_identity.get("trees") or []:
        measurements = [view.get("cross_grid_measurement") or {} for view in tree.get("views") or []]
        total += len(measurements)
        would_pass += count(measurements, lambda m: m.get("would_pass") is True)
        entry = {}
        for field in (
            "mask_iou",
            "tree_depth_within_tolerance",
            "interior_within_tolerance",
            "median_abs_on_matrix_tree_m",
        ):
            values = [m[field] for m in measurements if m.get(field) is not None]
            entry[field] = [min(values), max(values)] if values else None
        entry["diagnostics"] = tree.get("diagnostics")
        per_tree_values[tree["tree_id"]] = entry
    return {
        "label": "1920x1080 geometry point-sampled at matrix pixel centres; measurement, never gates",
        "views_would_pass": f"{would_pass} of {total}",
        "per_tree_min_max": per_tree_values,
    }


# ------------------------------------------------------------------------------------------------ predicates
def ratio_clause(
    text,
    numerator,
    denominator,
    universe,
    *,
    hold,
    refute=None,
    need,
    unscored=None,
    unrounded=None,
    ratio_key="ratio",
):
    """A count clause over the registered ``universe``; unscored trees count toward neither side."""
    unscored = dict(unscored or {})
    for tree in universe:
        if tree not in unscored and (numerator.get(tree) is None or not denominator.get(tree)):
            unscored[tree] = "no value"
    scored = [tree for tree in sorted(universe) if tree not in unscored]
    ratio = ratios(numerator, denominator, scored)
    meets_hold = sorted(tree for tree, r in ratio.items() if hold(r))
    meets_refutation = sorted(tree for tree, r in ratio.items() if refute is not None and refute(r))
    observed = {
        "numerator": {tree: numerator.get(tree) for tree in sorted(universe)},
        "denominator": {tree: denominator.get(tree) for tree in sorted(universe)},
        ratio_key: ratio,
        "meets_hold": meets_hold,
        "meets_refutation": meets_refutation,
        "scored": len(scored),
        "of": len(universe),
        "needed": need,
        "decidable": len(scored) >= need,
        "unscored": {tree: unscored[tree] for tree in sorted(unscored) if tree in universe},
    }
    if unrounded is not None:
        num_u, den_u = unrounded
        unrounded_ratio = ratios(num_u, den_u, scored)
        observed["unrounded_ratio"] = unrounded_ratio
        observed["rounding_sensitive"] = sorted(
            tree
            for tree in ratio
            if tree in unrounded_ratio
            and (
                hold(ratio[tree]) != hold(unrounded_ratio[tree])
                or (refute is not None and refute(ratio[tree]) != refute(unrounded_ratio[tree]))
            )
        )
    return clause(text, observed, len(meets_hold) >= need, refutes=refute is not None and len(meets_refutation) >= need)


def _unscored_for(cells, trees, *cells_needed):
    """Trees missing from the arm or with a needed cell whose view count is not 6 (I8)."""
    out = {}
    for tree in trees:
        for model, lighting in cells_needed:
            views = cells.get((model, tree, lighting), 0)
            if views != VIEWS:
                out[tree] = f"{model} {lighting} has {views} views"
                break
    return out


def score_g1(matched, trees, unscored=None, cells=None, universe=None):
    rows = matched["lighting_effect"]
    universe = sorted(universe or trees)
    excluded = {tree: "not in Matched" for tree in universe if tree not in trees}
    excluded.update(unscored or {})
    c = per_tree(rows, MODEL["C"], lighting="source")
    clauses = []
    for arm in ("A", "B"):
        values = per_tree(rows, MODEL[arm], lighting="source")
        cell_gaps = (
            _unscored_for(cells, trees, (MODEL[arm], "source"), (MODEL["C"], "source")) if cells is not None else {}
        )
        unrounded = None
        if "frames" in matched:
            unrounded = (
                per_tree_unrounded(matched["frames"], MODEL[arm], "source"),
                per_tree_unrounded(matched["frames"], MODEL["C"], "source"),
            )
        clauses.append(
            ratio_clause(
                f"{arm} >= 0.85 x C on Matched source in at least 6 of 8 trees",
                values,
                c,
                universe,
                hold=lambda r: r >= 0.85,
                refute=lambda r: r <= 0.70,
                need=6,
                unscored={**excluded, **cell_gaps},
                unrounded=unrounded,
                ratio_key="ratios",
            )
        )
    return prediction(
        "G1",
        "On Matched, A's and B's source error is at least 0.85 x C's in at least 6 of 8 trees "
        "(the source gain is render-gap robustness).",
        clauses,
    )


def score_g2(matched, trees, families, unscored=None, cells=None, universe=None):
    rows = matched["lighting_effect"]
    universe = sorted(universe or trees)
    excluded = {tree: "not in Matched" for tree in universe if tree not in trees}
    excluded.update(unscored or {})
    envy = sorted(t for t in universe if families.get(t) == "envy")
    ufo = sorted(t for t in universe if families.get(t) == "ufo")
    b = per_tree(rows, MODEL["B"], lighting="evening")
    c = per_tree(rows, MODEL["C"], lighting="evening")
    a = per_tree(rows, MODEL["A"], lighting="evening")
    j = per_tree(rows, MODEL["J"], lighting="evening")

    def gaps(*needed):
        return _unscored_for(cells, trees, *needed) if cells is not None else {}

    def unrounded(x, y):
        if "frames" not in matched:
            return None
        return per_tree_unrounded(matched["frames"], x, "evening"), per_tree_unrounded(matched["frames"], y, "evening")

    return prediction(
        "G2",
        "On Matched, B's evening error is at most 0.3 x C's in at least 3 of 4 Envy trees, and A's UFO evening "
        "error is at most 0.5 x J's in at least 3 of 4 UFO trees (the low-sun gain is lighting).",
        [
            ratio_clause(
                "B <= 0.3 x C on Matched evening in at least 3 of 4 Envy trees",
                b,
                c,
                envy,
                hold=lambda r: r <= 0.3,
                refute=lambda r: r >= 0.6,
                need=3,
                unscored={**excluded, **gaps((MODEL["B"], "evening"), (MODEL["C"], "evening"))},
                unrounded=unrounded(MODEL["B"], MODEL["C"]),
                ratio_key="B_over_C",
            ),
            ratio_clause(
                "A <= 0.5 x J on Matched evening in at least 3 of 4 UFO trees (no refutation registered)",
                a,
                j,
                ufo,
                hold=lambda r: r <= 0.5,
                refute=None,
                need=3,
                unscored={**excluded, **gaps((MODEL["A"], "evening"), (MODEL["J"], "evening"))},
                unrounded=unrounded(MODEL["A"], MODEL["J"]),
                ratio_key="A_over_J",
            ),
        ],
    )


def score_g3(matched, published, trees, unscored=None, cells=None, universe=None):
    universe = sorted(universe or trees)
    excluded = {tree: "not in Matched" for tree in universe if tree not in trees}
    excluded.update(unscored or {})
    new = per_tree(matched["lighting_effect"], MODEL["F"], lighting="source")
    old = per_tree(published["lighting_effect"], MODEL["F"], lighting="source")
    gaps = _unscored_for(cells, trees, (MODEL["F"], "source")) if cells is not None else {}
    unrounded = None
    if "frames" in matched and "frames" in published:
        unrounded = (
            per_tree_unrounded(matched["frames"], MODEL["F"], "source"),
            per_tree_unrounded(published["frames"], MODEL["F"], "source"),
        )
    return prediction(
        "G3",
        "F's source error on Matched is at most 0.85 x its error on the published matrix in at least 6 of 8 trees.",
        [
            ratio_clause(
                "F Matched / F published <= 0.85 on source in at least 6 of 8 trees",
                new,
                old,
                universe,
                hold=lambda r: r <= 0.85,
                refute=lambda r: r >= 1.0,
                need=6,
                unscored={**excluded, **gaps},
                unrounded=unrounded,
                ratio_key="ratio",
            )
        ],
    )


def reading(g1, g2):
    """The protocol's reading for the G1 and G2 verdicts, or a note that it names none for this case."""
    text = READINGS.get((g1, g2)) or READINGS.get((None, g2))
    return {
        "g1": g1,
        "g2": g2,
        "text": text,
        "note": None if text else "The protocol names no reading for this combination of verdicts.",
    }


# ------------------------------------------------------------------------------------------------ measurements
def per_tree_tables(documents):
    """per_tree_mask_mae_m, its unrounded twin and the view counts, for every arm, model and preset."""
    rounded, unrounded, views = {}, {}, {}
    for arm, document in documents.items():
        cells = cell_views(document["frames"])
        rounded[arm], unrounded[arm], views[arm] = {}, {}, {}
        for label, model in MODEL.items():
            rounded[arm][label] = {p: per_tree(document["lighting_effect"], model, lighting=p) for p in PRESETS}
            unrounded[arm][label] = {p: per_tree_unrounded(document["frames"], model, p) for p in PRESETS}
            views[arm][label] = {
                p: {tree: n for (m, tree, lighting), n in sorted(cells.items()) if m == model and lighting == p}
                for p in PRESETS
            }
    return rounded, unrounded, views


def family_means(documents):
    out = {}
    for arm, document in documents.items():
        out[arm] = {label: {} for label in MODEL}
        labels = {model: label for label, model in MODEL.items()}
        for row in document["family_comparison"]:
            label = labels.get(row["model"])
            if label is None:
                continue
            for family in ("envy", "ufo"):
                out[arm][label][f"{family}_{row['lighting']}"] = row.get(f"{family}_mask_mae_m")
    return out


def native_vs_matched(matched, native, anchoring=None):
    """Measurement only: Native / Matched per model, preset and tree, family means, mask pixels, affine ceilings."""
    per_tree_table, family = {}, {}
    for label, model in MODEL.items():
        per_tree_table[label], family[label] = {}, {}
        for lighting in PRESETS:
            m = per_tree(matched["lighting_effect"], model, lighting=lighting)
            n = per_tree(native["lighting_effect"], model, lighting=lighting)
            per_tree_table[label][lighting] = {
                tree: {
                    "matched": m.get(tree),
                    "native": n.get(tree),
                    "native_over_matched": n[tree] / m[tree] if m.get(tree) and n.get(tree) is not None else None,
                }
                for tree in sorted(set(m) | set(n))
            }
            family[label][lighting] = {}
            rows = {
                arm: next(
                    (r for r in doc["family_comparison"] if r["model"] == model and r["lighting"] == lighting), {}
                )
                for arm, doc in (("matched", matched), ("native", native))
            }
            for fam in ("envy", "ufo"):
                mv, nv = rows["matched"].get(f"{fam}_mask_mae_m"), rows["native"].get(f"{fam}_mask_mae_m")
                family[label][lighting][fam] = {
                    "matched": mv,
                    "native": nv,
                    "native_over_matched": nv / mv if mv and nv is not None else None,
                }
    pixels = {}
    for arm, document in (("matched", matched), ("native", native)):
        groups = defaultdict(list)
        for row in document["frames"]:
            if row.get("mask_pixels") is not None and row["model"] == MODEL["F"]:
                groups[row["family"]].append(row["mask_pixels"])
        pixels[arm] = {fam: statistics.fmean(values) for fam, values in sorted(groups.items())}
    ceilings = {}
    for arm, document in (anchoring or {}).items():
        ceilings[arm] = {label: {} for label in MODEL}
        labels = {model: label for label, model in MODEL.items()}
        for cell in document["cells"]:
            label = labels.get(cell["model"])
            if label is not None:
                ceilings[arm][label].setdefault(cell["family"], {})[cell["condition"]] = cell.get(
                    "affine_ceiling_mae_m"
                )
    return {
        "label": (
            "measurement, not predicted: Native differs from Matched in resolution, input filtering and ground truth "
            "at once"
        ),
        "per_tree": per_tree_table,
        "family": family,
        "mask_pixels_mean": pixels,
        "affine_ceiling": {"label": "diagnostic ceiling, never a result", **ceilings},
    }


def matched_vs_published(matched, published):
    """Measurement, not registered (G3 is its F x source slice)."""
    table = {}
    for label, model in MODEL.items():
        table[label] = {}
        for lighting in PRESETS:
            m = per_tree(matched["lighting_effect"], model, lighting=lighting)
            p = per_tree(published["lighting_effect"], model, lighting=lighting)
            table[label][lighting] = {
                tree: {
                    "published": p.get(tree),
                    "matched": m.get(tree),
                    "matched_over_published": m[tree] / p[tree] if p.get(tree) and m.get(tree) is not None else None,
                }
                for tree in sorted(set(m) | set(p))
            }
    return {"label": "measurement, not registered", "per_tree": table}


def checkpoint_hashes(document):
    return {row["model"]: row.get("checkpoint_sha256") for row in document["inputs"]}


def git_revision(root):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]))
    return head, dirty


def score(root, data):
    """Every prediction and table, after the integrity checks."""
    plan = data["plan"]
    families = {tree["tree_id"]: tree["family"] for tree in plan["trees"]}
    registered = sorted(families)
    g0, passing, g0_table = score_g0(data["pose_identity"], families, registered)
    render_ok = check_inputs(root, data, passing)
    matched_trees = sorted(set(passing) & render_ok)
    stalled = {tree: "render stalled or incomplete" for tree in registered if tree not in render_ok}
    unscored = {**stalled, **{tree: "failed G0" for tree in registered if tree not in passing}}
    cells = cell_views(data["matched"]["frames"])
    predictions = [
        g0,
        score_g1(data["matched"], matched_trees, unscored, cells, registered),
        score_g2(data["matched"], matched_trees, families, unscored, cells, registered),
        score_g3(data["matched"], data["published"], matched_trees, unscored, cells, registered),
    ]
    arms = {"published": data["published"], "matched": data["matched"], "native": data["native"]}
    rounded, unrounded, views = per_tree_tables(arms)
    return {
        "predictions": predictions,
        "g0_table": g0_table,
        "matched_trees": matched_trees,
        "render_ok": sorted(render_ok),
        "per_tree_mask_mae_m": rounded,
        "per_tree_mask_mae_unrounded_m": unrounded,
        "cell_views": views,
        "family_means": family_means(arms),
        "native_vs_matched": native_vs_matched(
            data["matched"],
            data["native"],
            {
                "published": data["published_anchoring"],
                "matched": data["matched_anchoring"],
                "native": data["native_anchoring"],
            },
        ),
        "matched_vs_published": matched_vs_published(data["matched"], data["published"]),
        "reading": reading(predictions[1]["verdict"], predictions[2]["verdict"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    root = args.root
    data = load(root)
    try:
        scored = score(root, data)
    except InputError as error:
        parser.error(str(error))
    head, dirty = git_revision(root)
    files = dict(sorted(data["files"].items()))
    files.setdefault(PROTOCOL, root / PROTOCOL)
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the render-gap check (G0-G3): the 8 matrix trees re-rendered with the training renderer, "
            "five checkpoints scored by the frozen depth_generalization.py on Matched (512x288 area-downsampled RGB "
            "against the published matrix depth and mask) and Native (1920x1080 against the new render's own depth "
            "and mask). Blender only, one bark, one render per frame; the Isaac camera is not touched."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "models": MODEL,
        "interpretations": INTERPRETATIONS,
        "inputs_sha256": {path: sha256(file) for path, file in sorted(files.items())},
        "batch": {
            "dir": BATCH,
            "plan_code_revision": data["plan"].get("code_revision"),
            "render_job": data["receipts"]["render"],
            "evaluation_job": data["receipts"]["eval"],
            "aggregation": {
                **AGGREGATION,
                "recomputed_from_saved_predictions": {
                    arm: data[arm].get("recomputed_from_saved_predictions") for arm in ARMS
                },
            },
        },
        "checkpoints_sha256": {name: checkpoint_hashes(data[name]) for name in ("matched", "native", "published")},
        "render_status": {k: data["render_status"].get(k) for k in ("registered", "native", "matched")},
        "render_ok_trees": scored["render_ok"],
        "matched_trees": scored["matched_trees"],
        "g0_geometry": scored["g0_table"],
        "g0_recomputed": False,
        "cross_grid_measurement": cross_grid(data["pose_identity"]),
        "per_tree_mask_mae_m": scored["per_tree_mask_mae_m"],
        "per_tree_mask_mae_unrounded_m": scored["per_tree_mask_mae_unrounded_m"],
        "cell_views": scored["cell_views"],
        "family_means": scored["family_means"],
        "native_vs_matched": scored["native_vs_matched"],
        "matched_vs_published": scored["matched_vs_published"],
        "predictions": scored["predictions"],
        "reading": scored["reading"],
    }
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
