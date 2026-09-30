#!/usr/bin/env python3
"""Score the rendered-lighting training batch (protocol of September 27) against L1-L4 and R1-R7.

Model errors come from the four aggregate evidence files (matrix, anchoring, controls, Stage A) that the
committed aggregation tools wrote with all five models side by side: F frozen, J and C the published jitter
and control arms, A (rendered_jitter) and B (rendered_control). Render and training facts come from the
batches' own manifests and training records. Nothing is re-evaluated here.

Where the protocol leaves a detail open it is fixed here and recorded in ``interpretations``: per-tree values
are means over a tree's frames; R5's morning and noon test uses the family mean; R6's Stage A test uses the
source light, as the published arms did; R6 and R7 state no separate refutation, so they are refuted when
their stated condition fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md"
EVIDENCE = {
    "matrix": "docs/evidence/finetune_rendered_family_matrix_2026-09-30.json",
    "anchoring": "docs/evidence/finetune_rendered_anchoring_2026-09-30.json",
    "controls": "docs/evidence/finetune_rendered_controls_2026-09-30.json",
    "stage_a": "docs/evidence/finetune_rendered_stage_a_2026-09-30.json",
}
GENERALIZATION = "artifacts/generalization"
ARMS = {"A": "finetune-rendered-jitter-20260927", "B": "finetune-rendered-control-20260927"}
PILOT = "training-lighting-pilot-20260927/render/lpy_envy_00001/render_manifest.json"
FULL = "training-lighting-full-20260927"
MODEL = {
    "F": "da2_frozen",
    "J": "da2_jitter",
    "C": "da2_control",
    "A": "da2_rendered_jitter",
    "B": "da2_rendered_control",
}
REFERENCE_VAL_RMSE = 0.054
#: One evaluation render manifest; every matrix tree was rendered with the same preset suns.
EVALUATION_LIGHTING_MANIFEST = (
    "artifacts/generalization/phase-20260920/family_code_v2/pilot_lpy_envy_00000_job_21370027/render_manifest.json"
)
INTERPRETATIONS = [
    "per-tree values are means over the tree's frames in that cell",
    "R5 morning and noon compare family means of A and J on Envy",
    "R6 Stage A uses the source light (as the published arms did); morning and evening are reported beside it",
    "R6 and R7 give no separate refutation; each is refuted when its stated condition fails",
    "L4 counts the full render only (the pilot tree is L1-L3)",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count(values, test):
    return sum(1 for v in values if test(v))


def per_tree(rows, model, key="mask_mae_m", **match):
    """Mean of ``key`` per tree over rows of ``model`` matching every field in ``match``."""
    groups = defaultdict(list)
    for row in rows:
        if row["model"] == model and all(row.get(k) == v for k, v in match.items()) and row.get(key) is not None:
            groups[row["tree_id"]].append(row[key])
    return {tree: statistics.fmean(values) for tree, values in sorted(groups.items())}


def score_r1(controls):
    b = per_tree(controls["frames"], MODEL["B"], family="envy", condition="evening_x2.6")
    c = per_tree(controls["frames"], MODEL["C"], family="envy", condition="evening_x2.6")
    ratios = {tree: b[tree] / c[tree] for tree in c}
    return prediction(
        "R1",
        "B's evening_x2.6 error is at most 0.6 x C in at least 3 of 4 Envy trees.",
        [
            clause(
                "B <= 0.6 x C in at least 3 of 4 Envy trees",
                {"B": b, "C": c, "B_over_C": ratios, "B_mean": statistics.fmean(b.values())},
                count(ratios.values(), lambda r: r <= 0.6) >= 3,
                refutes=count(ratios.values(), lambda r: r > 0.8) >= 3,
            )
        ],
    )


def evening_ceiling(anchoring, model):
    return {
        row["family"]: row["affine_ceiling_mae_m"]
        for row in anchoring["cells"]
        if row["model"] == model and row["condition"] == "evening"
    }


def score_r2(anchoring):
    a, j = evening_ceiling(anchoring, MODEL["A"]), evening_ceiling(anchoring, MODEL["J"])
    ratios = {family: a[family] / j[family] for family in ("envy", "ufo")}
    return prediction(
        "R2",
        "A's evening affine ceiling is at most 0.8 x J in both families.",
        [
            clause(
                "A <= 0.8 x J in Envy and UFO",
                {"A": a, "J": j, "A_over_J": ratios},
                all(r <= 0.8 for r in ratios.values()),
                refutes=all(0.9 <= r <= 1.1 for r in ratios.values()),
            )
        ],
    )


def score_r3_r4(matrix):
    envy = per_tree(matrix["lighting_effect"], MODEL["A"], family="envy", lighting="evening")
    ufo = per_tree(matrix["lighting_effect"], MODEL["A"], family="ufo", lighting="evening")
    j_ufo = per_tree(matrix["lighting_effect"], MODEL["J"], family="ufo", lighting="evening")
    r3 = prediction(
        "R3",
        "A's Envy evening error is at most 0.20 m in at least 3 of 4 trees.",
        [
            clause(
                "A <= 0.20 m in at least 3 of 4 Envy trees",
                {"A": envy},
                count(envy.values(), lambda v: v <= 0.20) >= 3,
                refutes=count(envy.values(), lambda v: v >= 0.29) >= 3,
            )
        ],
    )
    r4 = prediction(
        "R4",
        "A's UFO evening error is at most 0.75 m in at least 3 of 4 trees.",
        [
            clause(
                "A <= 0.75 m in at least 3 of 4 UFO trees",
                {"A": ufo, "J": j_ufo},
                count(ufo.values(), lambda v: v <= 0.75) >= 3,
                refutes=count(ufo.values(), lambda v: v > 1.0) >= 3,
            )
        ],
    )
    return r3, r4


def family_mean(matrix, model, family, lighting):
    key = f"{family}_mask_mae_m"
    return next(r[key] for r in matrix["family_comparison"] if r["model"] == model and r["lighting"] == lighting)


def score_r5(matrix):
    rows = matrix["lighting_effect"]
    frozen = per_tree(rows, MODEL["F"], lighting="source")
    clauses = []
    for arm in ("A", "B"):
        own = per_tree(rows, MODEL[arm], lighting="source")
        rise = {tree: own[tree] - frozen[tree] for tree in frozen}
        clauses.append(
            clause(
                f"{arm}'s source error is at most F + 0.03 m in at least 6 of 8 trees",
                {"rise_over_F_m": rise},
                count(rise.values(), lambda d: d <= 0.03) >= 6,
                refutes=count(rise.values(), lambda d: d >= 0.05) > len(rise) / 2,
            )
        )
    daylight = {
        lighting: {arm: family_mean(matrix, MODEL[arm], "envy", lighting) for arm in ("A", "J")}
        for lighting in ("morning", "noon")
    }
    clauses.append(
        clause(
            "A's Envy morning and noon are no worse than J (family means)",
            daylight,
            all(v["A"] <= v["J"] for v in daylight.values()),
        )
    )
    return prediction(
        "R5",
        "Daylight is not paid for: source within F + 0.03 m, and A's Envy morning and noon no worse than J.",
        clauses,
    )


def score_r6(controls, stage_a):
    def close(model):
        return {
            r["family"]: r["target_mae_mean_m"]
            for r in controls["paired_effect"]
            if r["model"] == model and r["condition"] == "source/close/training_rig"
        }

    def stage(model):
        return {r["lighting"]: r["target_unmasked_mae_mean_m"] for r in stage_a["cells"] if r["model"] == model}

    clauses = []
    for arm, reference in (("A", "J"), ("B", "C")):
        own_close, ref_close = close(MODEL[arm]), close(MODEL[reference])
        own_stage, ref_stage = stage(MODEL[arm]), stage(MODEL[reference])
        ratios = {f"close_{f}": own_close[f] / ref_close[f] for f in ref_close}
        ratios["stage_a_source"] = own_stage["source"] / ref_stage["source"]
        within = all(0.8 <= r <= 1.2 for r in ratios.values())
        clauses.append(
            clause(
                f"{arm}'s close-range and Stage A target errors within 20% of {reference}",
                {"ratio": ratios, arm: {"close": own_close, "stage_a": own_stage}, reference: {"close": ref_close}}
                | {f"{reference}_stage_a": ref_stage},
                within,
                refutes=not within,
            )
        )
    return prediction(
        "R6", "Close-range and Stage A target errors stay within 20% of J (arm A) and C (arm B).", clauses
    )


def score_r7(root):
    clauses = []
    for arm, batch in ARMS.items():
        training = json.loads((root / GENERALIZATION / batch / "train/training.json").read_text())
        best = training["best"]["val_rmse"]
        near = abs(best - REFERENCE_VAL_RMSE) <= 0.01
        steps = sum(e["iterations"] for e in training["epochs"])
        clauses.append(
            clause(
                f"{arm}'s best validation RMSE within 0.01 of 0.054",
                {"best_val_rmse": best, "best_epoch": training["best"]["epoch"], "optimizer_steps": steps},
                near,
                refutes=not near,
            )
        )
    return prediction("R7", "Each arm's best validation RMSE is within 0.01 of 0.054.", clauses)


def score_render(root):
    pilot = json.loads((root / GENERALIZATION / PILOT).read_text())
    seconds = statistics.fmean(f["render_seconds"] for f in pilot["frames"])
    checks = [c["tree_mask_mean_abs_diff_0_255"] for c in pilot["source_checks"]]
    frames = []
    for manifest in sorted((root / GENERALIZATION / FULL / "render").glob("*/render_manifest.json")):
        frames += json.loads(manifest.read_text())["frames"]
    plan = json.loads((root / GENERALIZATION / FULL / "plan.json").read_text())
    registered = sum(len(e["frames"]) for e in plan["trees"].values() if e["status"] == "to_render")
    kept = [f for f in frames if f.get("ok")]
    dark = count((f["mean_luma"] for f in kept), lambda v: v <= 90) / len(kept)
    passed = len(kept) / registered
    return [
        prediction(
            "L1",
            "All 90 pilot frames pass the geometry check.",
            [
                clause(
                    "90 of 90 pass",
                    f"{pilot['frames_ok']} of {len(pilot['frames'])}",
                    pilot["frames_ok"] == 90,
                    refutes=pilot["frames_ok"] < len(pilot["frames"]),
                )
            ],
        ),
        prediction(
            "L2",
            "Pilot render time is at most 40 s per frame.",
            [clause("mean <= 40 s", round(seconds, 2), seconds <= 40, refutes=seconds > 40)],
        ),
        prediction(
            "L3",
            "The two source frames match the surviving RGB within 2/255 mean on the tree mask.",
            [clause("both <= 2/255", checks, all(c <= 2 for c in checks), refutes=any(c > 2 for c in checks))],
        ),
        prediction(
            "L4",
            "At least 99% of the full render passes the geometry check, and at least 30% of kept frames have mean "
            "luma at or below 90.",
            [
                clause(
                    ">= 99% pass",
                    {"passed": len(kept), "registered": registered, "fraction": round(passed, 4)},
                    passed >= 0.99,
                    refutes=passed < 0.99,
                ),
                clause(">= 30% dark", round(dark, 4), dark >= 0.30, refutes=dark < 0.30),
            ],
        ),
    ]


def holdout_check(root, evaluation_manifest):
    """Recompute the sun hold-out from the recorded draws, and read the evaluation's preset suns as rendered."""
    import training_lighting as tl

    frames = []
    for manifest in [
        *sorted((root / GENERALIZATION / FULL / "render").glob("*/render_manifest.json")),
        root / GENERALIZATION / PILOT,
    ]:
        frames += json.loads(manifest.read_text())["frames"]
    closest = {name: 180.0 for name in tl.PRESET_SUNS}
    for frame in frames:
        sun, yaw = frame["applied_sun"], frame["lighting"]["pose_yaw_offset_deg"]
        for name, distance in tl.holdout_distances(sun["azimuth_deg"], sun["elevation_deg"], yaw).items():
            closest[name] = min(closest[name], distance)
    rendered = json.loads((root / evaluation_manifest).read_text())["lighting"]
    evaluation = {name: [v["sun"]["azimuth_deg"], v["sun"]["elevation_deg"]] for name, v in rendered.items()}
    return {
        "training_draws": len(frames),
        "closest_training_sun_to_preset_deg": {k: round(v, 3) for k, v in closest.items()},
        "holdout_radius_deg": tl.HOLDOUT_RADIUS_DEG,
        "holds": all(v >= tl.HOLDOUT_RADIUS_DEG for v in closest.values()),
        "evaluation_suns_as_rendered_deg": evaluation,
        "evaluation_manifest": evaluation_manifest,
        "evaluation_suns_equal_presets": all(
            tuple(evaluation[name]) == tuple(tl.PRESET_SUNS[name]) for name in evaluation if name in tl.PRESET_SUNS
        ),
    }


def summary_table(matrix, controls, anchoring):
    """Family means per model and light (matrix), evening_x2.6 (controls) and the evening affine ceiling."""
    table = {}
    for key, model in MODEL.items():
        row = {}
        for lighting in ("source", "morning", "noon", "evening"):
            for family in ("envy", "ufo"):
                row[f"{family}_{lighting}"] = family_mean(matrix, model, family, lighting)
        for r in controls["cells"]:
            if r["model"] == model and r["condition"] == "evening_x2.6":
                row[f"{r['family']}_evening_x2.6"] = r["mask_mae_mean_m"]
        for family, value in evening_ceiling(anchoring, model).items():
            row[f"{family}_evening_affine_ceiling"] = value
        table[key] = row
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    root = args.root
    data = {name: json.loads((root / path).read_text()) for name, path in EVIDENCE.items()}
    r3, r4 = score_r3_r4(data["matrix"])
    predictions = [
        *score_render(root),
        score_r1(data["controls"]),
        score_r2(data["anchoring"]),
        r3,
        r4,
        score_r5(data["matrix"]),
        score_r6(data["controls"], data["stage_a"]),
        score_r7(root),
    ]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]))
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the rendered-lighting training batch (L1-L4, R1-R7). Synthetic training renders of the "
            "companion's Envy train split; evaluation on the registered Blender matrix, controls and the Isaac "
            "Stage A replay, identical bytes to the published arms. Simulator and renderer ground truth only."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "models": MODEL,
        "interpretations": INTERPRETATIONS,
        "inputs_sha256": {path: sha256(root / path) for path in EVIDENCE.values()},
        "sun_holdout": holdout_check(root, EVALUATION_LIGHTING_MANIFEST),
        "family_means": summary_table(data["matrix"], data["controls"], data["anchoring"]),
        "predictions": predictions,
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in predictions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
