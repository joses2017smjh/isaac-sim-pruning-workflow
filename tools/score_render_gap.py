#!/usr/bin/env python3
"""Score the render-gap check (protocol of September 30) against G0-G3.

Model errors come from the aggregate evidence that the committed aggregation tools
(``aggregate_family_eval.py``, recompute on, as published) wrote for the Matched and Native
arms of the render-gap evaluation, with the published model labels; F's published source error
(G3) comes from the published rendered-lighting matrix. G0 re-applies the registered geometry
thresholds to every view of the evaluation's own record (``eval/pose_identity.json``). Nothing
is re-evaluated here.

Where the protocol leaves a detail open it is fixed here and recorded in ``interpretations``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md"
BATCH = "artifacts/generalization/render-gap-20260930"
EVIDENCE = {
    "matched": "docs/evidence/render_gap_matched_matrix_2026-10-01.json",
    "native": "docs/evidence/render_gap_native_matrix_2026-10-01.json",
    "matched_anchoring": "docs/evidence/render_gap_matched_anchoring_2026-10-01.json",
    "native_anchoring": "docs/evidence/render_gap_native_anchoring_2026-10-01.json",
    "published": "docs/evidence/finetune_rendered_family_matrix_2026-09-30.json",
}
MODEL = {
    "F": "da2_frozen",
    "J": "da2_jitter",
    "C": "da2_control",
    "A": "da2_rendered_jitter",
    "B": "da2_rendered_control",
}
TREES = 8
#: G0's registered gate (training_lighting.agreement thresholds, unchanged).
G0_TOLERANCE_M = 0.001
G0_MIN_WITHIN = 0.995
G0_MIN_IOU = 0.995
G0_VIEWS = 6
INTERPRETATIONS = [
    "per-tree errors are the published lighting_effect values: the mean tree-mask MAE over the tree's 6 views, "
    "rounded to 4 dp, from aggregate_family_eval.py (as the published scorer used them); ratios are of those values",
    "G0 re-applies the registered gate to every view's recorded grid-pass agreement (IoU >= 0.995 and >= 99.5% of "
    "tree pixels within 1 mm); a tree passes only with all 6 views passing; the cross-grid point samples are "
    "reported, never gating",
    "a tree that fails G0 is left out of Matched; the counts stay 'of 8' (or 'of 4'), so a missing tree counts "
    "toward neither holding nor refuting",
    "G1 is one clause per arm (A/C and B/C), each holding at >= 0.85 in >= 6 trees and refuting at <= 0.70 in >= 6; "
    "the prediction is refuted if either arm refutes and partly supported if neither holds nor refutes",
    "G2's UFO clause (A/J <= 0.5 in >= 3 of 4) states no refutation",
    "G3's 'at least as large' is a ratio >= 1.0",
    "Native against Matched is a measurement: per model, tree and preset, the Native per-tree error over the "
    "Matched one, and each arm's family means; Native differs in resolution, input filtering and ground truth",
]


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
                raise ValueError(f"duplicate lighting_effect row for {model} {row['tree_id']} {match}")
            out[row["tree_id"]] = row["mask_mae_m"]
    return dict(sorted(out.items()))


def ratios(numerator, denominator, trees):
    return {
        tree: numerator[tree] / denominator[tree]
        for tree in sorted(trees)
        if tree in numerator and tree in denominator and denominator[tree]
    }


def view_passes(view):
    gate = view.get("gate") or {}
    return (
        gate.get("tolerance_m") == G0_TOLERANCE_M
        and (gate.get("mask_iou") or 0.0) >= G0_MIN_IOU
        and (gate.get("tree_depth_within_tolerance") or 0.0) >= G0_MIN_WITHIN
    )


def score_g0(pose_identity, families):
    """G0 per tree from every view's recorded grid-pass agreement; returns (prediction, passing trees, table)."""
    table = {}
    for tree in pose_identity["trees"]:
        views = tree.get("views") or []
        table[tree["tree_id"]] = {
            "family": families.get(tree["tree_id"]),
            "views_checked": len(views),
            "views_passing": count(views, view_passes),
            "recorded_ok": tree.get("ok"),
            "min_mask_iou": min(((v.get("gate") or {}).get("mask_iou", 0.0) for v in views), default=None),
            "min_within_1mm": min(
                ((v.get("gate") or {}).get("tree_depth_within_tolerance", 0.0) for v in views), default=None
            ),
            "cross_grid_views_would_pass": count(
                views, lambda v: (v.get("cross_grid_measurement") or {}).get("would_pass")
            ),
        }
    passing = sorted(t for t, r in table.items() if r["views_checked"] == G0_VIEWS and r["views_passing"] == G0_VIEWS)
    failing = sorted(set(table) - set(passing))
    recorded = sorted(pose_identity.get("matched_trees") or [])
    g0 = prediction(
        "G0",
        "Every tree's grid-pass depth and mask reproduce the published matrix in every view.",
        [
            clause(
                "all 8 trees pass the registered gate in all 6 views",
                {"trees": table, "passing": passing, "failing": failing},
                len(passing) == TREES and len(table) == TREES,
                refutes=bool(failing) or len(table) != TREES,
            ),
            clause(
                "the evaluation's recorded matched set equals the re-applied gate",
                {
                    "recorded_matched": recorded,
                    "excluded": pose_identity.get("excluded_trees"),
                    "not_checked": pose_identity.get("not_checked"),
                },
                recorded == passing and not pose_identity.get("not_checked"),
            ),
        ],
    )
    return g0, passing, table


def score_g1(matched, trees):
    rows = matched["lighting_effect"]
    c = per_tree(rows, MODEL["C"], lighting="source")
    clauses, observed = [], {"C": c}
    for arm in ("A", "B"):
        values = per_tree(rows, MODEL[arm], lighting="source")
        ratio = ratios(values, c, trees)
        observed[arm] = values
        observed[f"{arm}_over_C"] = ratio
        clauses.append(
            clause(
                f"{arm} >= 0.85 x C on Matched source in at least 6 of 8 trees",
                {"ratios": ratio, "at_least_0.85": count(ratio.values(), lambda r: r >= 0.85)},
                count(ratio.values(), lambda r: r >= 0.85) >= 6,
                refutes=count(ratio.values(), lambda r: r <= 0.70) >= 6,
            )
        )
    clauses.append(clause("values (measurement)", observed, True))
    return prediction(
        "G1",
        "On Matched, A's and B's source error is at least 0.85 x C's in at least 6 of 8 trees "
        "(the source gain is render-gap robustness).",
        clauses,
    )


def score_g2(matched, trees, families):
    rows = matched["lighting_effect"]
    envy = sorted(t for t in trees if families.get(t) == "envy")
    ufo = sorted(t for t in trees if families.get(t) == "ufo")
    b = per_tree(rows, MODEL["B"], lighting="evening")
    c = per_tree(rows, MODEL["C"], lighting="evening")
    a = per_tree(rows, MODEL["A"], lighting="evening")
    j = per_tree(rows, MODEL["J"], lighting="evening")
    b_over_c = ratios(b, c, envy)
    a_over_j = ratios(a, j, ufo)
    return prediction(
        "G2",
        "On Matched, B's evening error is at most 0.3 x C's in at least 3 of 4 Envy trees, and A's UFO evening "
        "error is at most 0.5 x J's in at least 3 of 4 UFO trees (the low-sun gain is lighting).",
        [
            clause(
                "B <= 0.3 x C on Matched evening in at least 3 of 4 Envy trees",
                {"B": {t: b.get(t) for t in envy}, "C": {t: c.get(t) for t in envy}, "B_over_C": b_over_c},
                count(b_over_c.values(), lambda r: r <= 0.3) >= 3,
                refutes=count(b_over_c.values(), lambda r: r >= 0.6) >= 3,
            ),
            clause(
                "A <= 0.5 x J on Matched evening in at least 3 of 4 UFO trees",
                {"A": {t: a.get(t) for t in ufo}, "J": {t: j.get(t) for t in ufo}, "A_over_J": a_over_j},
                count(a_over_j.values(), lambda r: r <= 0.5) >= 3,
            ),
        ],
    )


def score_g3(matched, published, trees):
    new = per_tree(matched["lighting_effect"], MODEL["F"], lighting="source")
    old = per_tree(published["lighting_effect"], MODEL["F"], lighting="source")
    ratio = ratios(new, old, trees)
    return prediction(
        "G3",
        "F's source error on Matched is at most 0.85 x its error on the published matrix in at least 6 of 8 trees.",
        [
            clause(
                "F Matched / F published <= 0.85 on source in at least 6 of 8 trees",
                {"F_matched": new, "F_published": old, "ratio": ratio},
                count(ratio.values(), lambda r: r <= 0.85) >= 6,
                refutes=count(ratio.values(), lambda r: r >= 1.0) >= 6,
            )
        ],
    )


def native_vs_matched(matched, native):
    """Measurement only: Native over Matched per model, tree and preset, and each arm's family means."""
    table = {}
    for label, model in MODEL.items():
        for lighting in sorted({row["lighting"] for row in matched["lighting_effect"]}):
            m = per_tree(matched["lighting_effect"], model, lighting=lighting)
            n = per_tree(native["lighting_effect"], model, lighting=lighting)
            r = ratios(n, m, m)
            table[f"{label} {lighting}"] = {
                "native_over_matched": r,
                "median_ratio": statistics.median(r.values()) if r else None,
            }
    families = {
        arm: [
            {key: row[key] for key in ("model", "lighting", "envy_mask_mae_m", "ufo_mask_mae_m")}
            for row in document["family_comparison"]
        ]
        for arm, document in (("matched", matched), ("native", native))
    }
    return {"per_tree": table, "family_means": families}


def checkpoint_hashes(document):
    return {row["model"]: row.get("checkpoint_sha256") for row in document["inputs"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    root = args.root
    documents = {name: json.loads((root / path).read_text()) for name, path in EVIDENCE.items()}
    pose_path = root / BATCH / "eval/pose_identity.json"
    status_path = root / BATCH / "eval/render_status.json"
    pose_identity = json.loads(pose_path.read_text())
    render_status = json.loads(status_path.read_text())
    matched, native, published = documents["matched"], documents["native"], documents["published"]
    families = {row["tree_id"]: row["family"] for row in published["lighting_effect"]}
    g0, trees, g0_table = score_g0(pose_identity, families)
    hashes = {name: checkpoint_hashes(documents[name]) for name in ("matched", "native", "published")}
    stalled = [t["tree_id"] for t in render_status["trees"] if not (t.get("render_ok") and t.get("present"))]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]))
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
        "inputs_sha256": {
            **{path: sha256(root / path) for path in EVIDENCE.values()},
            f"{BATCH}/eval/pose_identity.json": sha256(pose_path),
            f"{BATCH}/eval/render_status.json": sha256(status_path),
        },
        "checkpoints_sha256": hashes,
        "checkpoints_match_published": hashes["matched"] == hashes["published"] == hashes["native"],
        "render_status": {k: render_status.get(k) for k in ("registered", "native", "matched")},
        "stalled_or_missing_trees": stalled,
        "matched_trees": trees,
        "interpretations": INTERPRETATIONS,
        "g0_geometry": g0_table,
        "predictions": [
            g0,
            score_g1(matched, trees),
            score_g2(matched, trees, families),
            score_g3(matched, published, trees),
        ],
        "native_vs_matched": native_vs_matched(matched, native),
    }
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
