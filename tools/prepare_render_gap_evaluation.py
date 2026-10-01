#!/usr/bin/env python3
"""Build the render-gap check's two evaluation plans from the training-renderer renders of the matrix trees.

Per registered tree, light and view of a frozen ``queue_render_gap.py`` batch:

* NATIVE: the new 1920x1080 RGB scored against its own new depth and tree mask. The new render manifest is
  passed to ``prepare_family_evaluation.py`` as it is.
* MATCHED: the new RGB downsampled to the matrix resolution (512x288) with ``cv2.INTER_AREA`` and scored
  against the PUBLISHED matrix depth and mask for the same tree, light and view. Its inputs are copies of the
  published matrix render manifests with only each frame's RGB replaced and the replacement recorded, so
  ``prepare_family_evaluation.py`` applies exactly the corrections it applied to the published plan (the
  half-pixel conversion of the two pilot manifests included). Both plans are written by that script, in its
  format, never by this one.

Pose identity is checked per view before a tree's matrix ground truth is used for MATCHED. The training
renderer renders each view's depth and tree mask once more on the matrix grid with the matrix renderer's
sampler, in the same Blender session as the 1920x1080 frames (``matrix_grid_geometry`` in its manifest).
``training_lighting.agreement``, the rendered-lighting geometry check, compares that pass with the published
matrix depth and mask: at least 99.5% of matrix tree-mask pixels within 1 mm and a mask IoU of at least 0.995.
A tree with any failing or missing view, or without the matrix-grid pass, is excluded from MATCHED and
recorded; it stays in NATIVE, which needs no matrix ground truth.

The 1920x1080 depth and mask, point-sampled at each matrix pixel centre (the native pixel that contains it),
are compared with the same thresholds and recorded as a measurement that never gates. Cycles takes depth and
object index at one fixed point per pixel, its centre, so a render on another pixel grid samples silhouettes
elsewhere: 1920/512 = 3.75, and every native sample lies 1/30 or 1/10 of a matrix pixel from the matrix one,
which on thin branches with steep depth gradients fails the thresholds at an identical pose. The seed plays no
part (on one grid, seed 0 and seed 1729 gave identical depth bytes in the render-gap review). Manifest-level
camera-pose, geometry-hash, focal-length and tree-transform comparisons, the agreement on interior tree pixels
and the median depth difference are recorded beside the gate and never gate.

Limits. A smoke render, a matrix-renderer manifest or another tree's manifest in the batch is refused before
anything is written. A missing or unfinished render is recorded and excluded from both arms. An arm with no
tree writes its ``status.json`` and no plan, and the evaluation job skips it; with no rendered tree at all the
job stops. Only ``--output`` is written; the published matrix files and the new renders are read.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_lighting as tl  # noqa: E402
from depth_generalization import sha256  # noqa: E402

FAMILY_SCRIPT = Path(__file__).resolve().parent / "prepare_family_evaluation.py"
ARMS = {
    "native": (
        "NATIVE arm: training-renderer RGB at 1920x1080 scored against the same render's own depth and tree mask."
    ),
    "matched": (
        "MATCHED arm: training-renderer RGB downsampled to 512x288 with cv2.INTER_AREA, scored against the "
        "published matrix depth and tree mask of the same tree, light and view."
    ),
}
GATE = {
    "tolerance_m": tl.AGREEMENT_TOLERANCE_M,
    "min_fraction_within_tolerance": tl.AGREEMENT_MIN_FRACTION,
    "min_mask_iou": tl.AGREEMENT_MIN_IOU,
    "function": "training_lighting.agreement(grid_depth, grid_mask, matrix_depth, matrix_mask)",
    "new_geometry": (
        "matrix_grid_geometry of the new render manifest: each view's depth and tree mask rendered on the matrix "
        "grid with the matrix renderer's sampler, in the same Blender session as the 1920x1080 frames"
    ),
}
CROSS_GRID = {
    "sampling": "point sample: the native pixel containing each matrix pixel centre, ((2i+1)*native)//(2*matrix)",
    "scope": "Measurement with the gate's thresholds; never gates.",
}
RGB_RESIZE = "cv2.resize(rgb, (matrix_width, matrix_height), interpolation=cv2.INTER_AREA)"
POSE_ATOL = 1e-6


def centre_indices(n_matrix, n_native):
    """Index of the native pixel that contains each matrix pixel centre, along one axis (exact integers)."""
    index = np.arange(n_matrix)
    return ((2 * index + 1) * n_native) // (2 * n_matrix)


def point_sample(array, shape):
    """``array`` on the native grid, sampled at the pixel centres of a ``shape`` grid over the same image."""
    native_h, native_w = array.shape[:2]
    matrix_h, matrix_w = shape
    if native_h * matrix_w != native_w * matrix_h:
        raise ValueError(f"Native {native_w}x{native_h} and matrix {matrix_w}x{matrix_h} grids differ in aspect")
    if native_h < matrix_h or native_w < matrix_w:
        raise ValueError("The native grid is coarser than the matrix grid")
    return array[np.ix_(centre_indices(matrix_h, native_h), centre_indices(matrix_w, native_w))]


def downsample_rgb(rgb, shape):
    """The MATCHED input image: area-averaged to the matrix grid."""
    return cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def read_mask(path):
    mask = np.asarray(Image.open(path)) > 0
    return mask.any(axis=2) if mask.ndim == 3 else mask


def load_geometry(depth_path, mask_path):
    """A depth array and its tree mask, which must share one grid."""
    depth, mask = np.load(depth_path, allow_pickle=False), read_mask(mask_path)
    if depth.shape != mask.shape:
        raise ValueError(f"Depth {depth_path} and mask {mask_path} differ in shape")
    return depth, mask


def erode(mask):
    """Pixels whose whole 3x3 neighbourhood is in ``mask``."""
    padded = np.pad(mask, 1)
    height, width = mask.shape
    interior = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return interior


def max_abs_difference(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def cross_grid_measurement(new_depth, new_mask, matrix_depth, matrix_mask):
    """The 1920x1080 geometry point-sampled at the matrix pixel centres against the matrix ground truth."""
    try:
        sampled_depth = point_sample(new_depth, matrix_depth.shape)
        sampled_mask = point_sample(new_mask, matrix_depth.shape)
    except ValueError as error:
        return {"skipped": str(error), **CROSS_GRID}
    measured = tl.agreement(sampled_depth, sampled_mask, matrix_depth, matrix_mask)
    on_tree = matrix_mask & np.isfinite(matrix_depth) & (matrix_depth < 1e6)
    interior = erode(on_tree)
    with np.errstate(invalid="ignore"):
        error = np.abs(sampled_depth.astype(float) - matrix_depth.astype(float))
    return {
        "mask_iou": measured["mask_iou"],
        "tree_depth_within_tolerance": measured["tree_depth_within_tolerance"],
        "would_pass": bool(measured["ok"]),
        "sampled_new_tree_pixels": int(sampled_mask.sum()),
        "median_abs_on_matrix_tree_m": float(np.median(error[on_tree])) if on_tree.any() else None,
        "interior_tree_pixels": int(interior.sum()),
        "interior_within_tolerance": (
            float((error[interior] <= tl.AGREEMENT_TOLERANCE_M).mean()) if interior.any() else None
        ),
        **CROSS_GRID,
    }


def view_identity(new_frame, grid_view, matrix_frame, grid_K=None):
    """One view: the gate (the matrix-grid pass against the matrix ground truth) and what is recorded beside it.

    Depth and mask are shared by every light of a view, so one frame per view is enough.
    """
    matrix_depth, matrix_mask = load_geometry(matrix_frame["depth"], matrix_frame["mask"])
    new_depth, new_mask = load_geometry(new_frame["depth"], new_frame["mask"])
    record = {
        "view_id": new_frame["view_id"],
        "native_shape": list(new_depth.shape),
        "matrix_shape": list(matrix_depth.shape),
    }
    if grid_view is None:
        gate = {"ok": False, "reason": "the render manifest has no matrix-grid pass for this view"}
    else:
        grid_depth, grid_mask = load_geometry(grid_view["depth"], grid_view["mask"])
        record["matrix_grid_shape"] = list(grid_depth.shape)
        if grid_depth.shape != matrix_depth.shape:
            gate = {"ok": False, "reason": f"matrix-grid pass {grid_depth.shape} vs matrix {matrix_depth.shape}"}
        else:
            gate = tl.agreement(grid_depth, grid_mask, matrix_depth, matrix_mask)
    on_tree = matrix_mask & np.isfinite(matrix_depth) & (matrix_depth < 1e6)
    matrix_K = matrix_frame["K"]
    record.update(
        gate=gate,
        ok=bool(gate["ok"]),
        cross_grid_measurement=cross_grid_measurement(new_depth, new_mask, matrix_depth, matrix_mask),
        diagnostics={
            "matrix_tree_pixels": int(on_tree.sum()),
            "camera_location_max_abs_diff_m": max_abs_difference(
                new_frame["camera_location"], matrix_frame["camera_location"]
            ),
            "camera_rotation_max_abs_diff_rad": max_abs_difference(
                new_frame["camera_rotation_euler"], matrix_frame["camera_rotation_euler"]
            ),
            "focal_ratio": float(new_frame["K"][0][0]) / float(matrix_K[0][0]),
            "expected_focal_ratio": new_depth.shape[1] / matrix_depth.shape[1],
            "matrix_grid_focal_max_abs_diff_px": (
                max_abs_difference([grid_K[0][0], grid_K[1][1]], [matrix_K[0][0], matrix_K[1][1]])
                if grid_K is not None
                else None
            ),
            "scope": "Recorded beside the gate; never gates.",
        },
    )
    return record


def frames_by_view(manifest):
    """One frame per view: depth and mask are rendered once per view and shared by its lights."""
    views = {}
    for frame in manifest["frames"]:
        views.setdefault(frame["view_id"], frame)
    return views


def tree_identity(new_manifest, matrix_manifest):
    """Per-view gate results; the tree passes only if it has the matrix-grid pass and every matrix view passes."""
    grid = new_manifest.get("matrix_grid_geometry") or {}
    grid_views = grid.get("views") or {}
    new_views, matrix_views = frames_by_view(new_manifest), frames_by_view(matrix_manifest)
    views = [
        view_identity(new_views[v], grid_views.get(v), matrix_views[v], grid.get("K"))
        for v in sorted(matrix_views)
        if v in new_views
    ]
    missing = sorted(set(matrix_views) - set(new_views))
    extra = sorted(set(new_views) - set(matrix_views))
    poses_equal = all(
        v["diagnostics"]["camera_location_max_abs_diff_m"] <= POSE_ATOL
        and v["diagnostics"]["camera_rotation_max_abs_diff_rad"] <= POSE_ATOL
        for v in views
    )
    return {
        "ok": bool(grid_views) and bool(views) and not missing and not extra and all(v["ok"] for v in views),
        "matrix_grid_pass": bool(grid_views),
        "matrix_grid_sampler": grid.get("sampler"),
        "views_passing": sum(1 for v in views if v["ok"]),
        "views_checked": len(views),
        "missing_views": missing,
        "extra_views": extra,
        "cross_grid_views_would_pass": sum(1 for v in views if v["cross_grid_measurement"].get("would_pass")),
        "views": views,
        "diagnostics": {
            "geometry_sha256_equal": new_manifest.get("geometry_sha256") == matrix_manifest.get("geometry_sha256"),
            "camera_poses_equal_within_1e-6": poses_equal,
            "tree_transform_max_abs_diff": (
                max_abs_difference(new_manifest["tree_transform"], matrix_manifest["tree_transform"])
                if new_manifest.get("tree_transform") and matrix_manifest.get("tree_transform")
                else None
            ),
            "scope": "Recorded beside the gate; never gates.",
        },
    }


def exclusion_reason(check):
    """Why a tree whose pose identity failed is kept out of MATCHED."""
    if not check["matrix_grid_pass"]:
        return "pose identity not checked: the render manifest has no matrix-grid geometry pass"
    reason = f"pose identity failed: {check['views_passing']} of {check['views_checked']} views pass"
    return reason + (f", missing {check['missing_views']}" if check["missing_views"] else "")


def check_new_manifest(data, tree_id):
    """Refuse a smoke render, a matrix-renderer manifest or another tree's; the caller records an unfinished one."""
    if data.get("smoke"):
        raise ValueError(f"{tree_id}: a smoke render is a code-path check, never evaluation data")
    if data.get("renderer_mode") != "training":
        raise ValueError(f"{tree_id}: not a training-renderer manifest (renderer_mode={data.get('renderer_mode')!r})")
    if data.get("tree_id") != tree_id:
        raise ValueError(f"{tree_id}: the render manifest is for {data.get('tree_id')!r}")


def provenance(data, entry, plan):
    """Source hashes of the new render against the plan; recorded, never gating."""
    external = plan.get("external_assets_sha256") or {}
    pairs = {
        "metadata": (data.get("source_metadata_sha256"), entry.get("metadata_sha256")),
        "generator": (data.get("source_generator_sha256"), external.get("Dataloader/generate_tree2.py")),
        "blend": (data.get("source_blend_sha256"), external.get("orchard_template.blend")),
    }
    return {name: (None if planned is None else found == planned) for name, (found, planned) in pairs.items()}


def matched_manifest(matrix, matrix_path, new, new_path, rgb_dir):
    """A copy of the published matrix manifest whose frames carry the downsampled new RGB."""
    data = copy.deepcopy(matrix)
    sources = {(f["lighting"], f["view_id"]): f for f in new["frames"]}
    for frame in data["frames"]:
        key = (frame["lighting"], frame["view_id"])
        if key not in sources:
            raise ValueError(f"{data['tree_id']}: the new render has no frame for {key}")
        source = sources[key]
        shape = np.load(frame["depth"], mmap_mode="r", allow_pickle=False).shape
        rgb = cv2.imread(source["rgb"], cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"RGB missing: {source['rgb']}")
        out = rgb_dir / frame["lighting"] / f"{frame['view_id']}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), downsample_rgb(rgb, shape)):
            raise OSError(f"Could not write {out}")
        frame.update(
            {
                "matrix_rgb": frame["rgb"],
                "rgb": str(out),
                "render_gap_arm": "matched",
                "render_gap_source_rgb": source["rgb"],
                "render_gap_source_rgb_sha256": sha256(source["rgb"]),
                "render_gap_rgb_resize": RGB_RESIZE,
            }
        )
    data["render_gap"] = {
        "arm": "matched",
        "scope": "RGB from the training-renderer render, downsampled; depth, mask, K and target from the matrix render",
        "matrix_render_manifest": str(matrix_path),
        "matrix_render_manifest_sha256": sha256(matrix_path),
        "new_render_manifest": str(new_path),
        "new_render_manifest_sha256": sha256(new_path),
    }
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_renders(batch, plan):
    """Every registered tree's render manifest that exists, each checked before anything is written."""
    renders = {}
    for entry in plan["trees"]:
        path = batch / "render" / entry["tree_id"] / "render_manifest.json"
        if path.is_file():
            data = json.loads(path.read_text())
            check_new_manifest(data, entry["tree_id"])
            renders[entry["tree_id"]] = (path, data)
    return renders


def build(batch, output, *, family_script=FAMILY_SCRIPT, python=sys.executable):
    batch, output = Path(batch), Path(output)
    plan = json.loads((batch / "plan.json").read_text())
    renders = read_renders(batch, plan)
    output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        (output / arm).mkdir()
    status, identity, members = [], [], {arm: [] for arm in ARMS}
    for entry in plan["trees"]:
        tree_id = entry["tree_id"]
        new_path = batch / "render" / tree_id / "render_manifest.json"
        record = {"tree_id": tree_id, "family": entry["family"], "render_manifest": str(new_path)}
        record.update(present=tree_id in renders, render_ok=False, native=False, matched=False)
        status.append(record)
        if not record["present"]:
            record["reason"] = "no render manifest"
            continue
        new = renders[tree_id][1]
        record["render_ok"] = bool(new.get("ok"))
        record["render_manifest_sha256"] = sha256(new_path)
        if not record["render_ok"]:
            record["reason"] = f"the render did not finish (ok false, {len(new.get('frames', []))} frames recorded)"
            continue
        record["provenance_matches_plan"] = provenance(new, entry, plan)
        members["native"].append(str(new_path))
        record["native"] = True
        matrix_path = Path(entry["matrix_render_manifest"])
        planned = entry.get("matrix_render_manifest_sha256")
        if planned is not None and sha256(matrix_path) != planned:
            raise ValueError(f"{tree_id}: the published matrix manifest changed since the plan: {matrix_path}")
        matrix = json.loads(matrix_path.read_text())
        check = tree_identity(new, matrix)
        identity.append({"tree_id": tree_id, "matrix_render_manifest": str(matrix_path), **check})
        if not check["ok"]:
            record["matched_excluded"] = exclusion_reason(check)
            continue
        manifest = matched_manifest(matrix, matrix_path, new, new_path, output / "matched/rgb" / tree_id)
        path = output / "matched/manifests" / tree_id / "render_manifest.json"
        write_json(path, manifest)
        members["matched"].append(str(path))
        record["matched"] = True
    write_json(
        output / "render_status.json",
        {
            "registered": len(plan["trees"]),
            "native": len(members["native"]),
            "matched": len(members["matched"]),
            "trees": status,
        },
    )
    write_json(
        output / "pose_identity.json",
        {
            "gate": GATE,
            "cross_grid_measurement": CROSS_GRID,
            "rule": "A tree enters MATCHED only if every view passes; a failing tree is excluded and recorded here.",
            "limit": (
                "Cycles takes depth and object index at each pixel's centre. 1920/512 is not an integer, so no "
                "1920x1080 pixel samples a matrix pixel centre, and the cross-grid measurement can fail at an "
                "identical pose; it never gates. The seed plays no part: on one grid, seed 0 and seed 1729 gave "
                "identical depth bytes."
            ),
            "matched_trees": [t["tree_id"] for t in identity if t["ok"]],
            "excluded_trees": [t["tree_id"] for t in identity if not t["ok"]],
            # Registered trees the gate never saw (no render, an unfinished render): G0 counts them as not passing.
            "not_checked": [
                {"tree_id": r["tree_id"], "reason": r.get("reason")}
                for r in status
                if r["tree_id"] not in {t["tree_id"] for t in identity}
            ],
            "trees": identity,
        },
    )
    for arm, scope in ARMS.items():
        arm_status = {"arm": arm, "registered": len(plan["trees"]), "trees": len(members[arm]), "plan": None}
        if members[arm]:
            inputs = output / arm / "inputs.json"
            write_json(inputs, {"scope": f"{plan['scope']} {scope}", "render_manifests": members[arm]})
            arm_plan = output / arm / "plan.json"
            subprocess.run([python, str(family_script), "--inputs", str(inputs), "--output", str(arm_plan)], check=True)
            arm_status.update(plan=str(arm_plan), plan_sha256=sha256(arm_plan), render_manifests=members[arm])
        else:
            arm_status["reason"] = "no registered tree qualified; see render_status.json and pose_identity.json"
        write_json(output / arm / "status.json", arm_status)
    summary = {arm: len(members[arm]) for arm in ARMS}
    print(json.dumps({"registered": len(plan["trees"]), **summary}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True, help="Frozen render-gap batch directory")
    parser.add_argument("--output", type=Path, required=True, help="Evaluation directory (the batch's eval/)")
    args = parser.parse_args()
    summary = build(args.batch, args.output)
    if not summary["native"]:
        raise SystemExit("no registered tree rendered; nothing to score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
