"""The render-gap check: the frozen plan and launcher, the two-arm evaluation builder and the renderer's options.

Everything runs without Blender, a GPU or the companion: the plan is built without hashing or from tiny stand-in
files, the builder works on synthetic 8x16 matrix and 30x60 native arrays (the 3.75 ratio of 512x288 to
1920x1080), and the renderer is imported for its argument rules, template check and matrix-grid settings only.

The synthetic renders test the plumbing: which geometry the pose-identity gate compares, what it records and how
a tree is admitted or excluded. Whether real renders pass is a property of Blender, not of this code; the one
synthetic that models it (a continuous scene sampled at both grids' pixel centres) records why the gate compares
a pass rendered on the matrix grid and keeps the cross-grid point sample as a measurement only.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
LIGHTS = ["source", "morning", "noon", "evening"]
VIEWS = [f"rig{rig}_{side}" for rig in range(3) for side in "lr"]
MATRIX_SHAPE, NATIVE_SHAPE = (8, 16), (30, 60)


def _load(monkeypatch, name):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- The frozen plan and the launcher -------------------------------------------------------------------------


def test_plan_registers_eight_trees_five_checkpoints_and_180_gpu_minutes(monkeypatch):
    q = _load(monkeypatch, "queue_render_gap")
    plan = q.render_gap_plan(hash_assets=False)
    trees = plan["trees"]
    assert [t["tree_id"] for t in trees] == list(q.TREES) and len(trees) == 8
    assert all(t["status"] == "to_render" for t in trees)
    assert sum(t["family"] == "envy" for t in trees) == sum(t["family"] == "ufo" for t in trees) == 4
    assert list(plan["models"]) == ["F", "J", "C", "A", "B"]
    assert plan["models"]["F"]["checkpoint_sha256"] == q.MODELS["da2"]["checkpoint_sha256"]
    for label, batch in (("J", "finetune-jitter-20260923"), ("B", "finetune-rendered-control-20260927")):
        assert plan["models"][label]["checkpoint"].endswith(f"{batch}/train/best.pth")
    assert plan["maximum_gpu_minutes"] == 180 == 8 * q.RENDER_MINUTES_RESERVED + q.EVAL_MINUTES_RESERVED
    assert plan["max_concurrent_gpus"] == 1 and plan["placement"] == {"partition": "gpu,ampere"}
    assert plan["renderer"] == "training" and plan["lights"] == LIGHTS and plan["views_per_light"] == 6
    assert plan["registered_frames"] == 192
    assert plan["template_sampler"]["samples"] == 4096 and plan["template_sampler"]["resolution"] == [1920, 1080, 100]
    # The ground truth MATCHED scores against: the pilots for the two 00000 trees, matrix_<i>_<tree> otherwise.
    gt = {t["tree_id"]: t["matrix_render_manifest"] for t in trees}
    assert gt["lpy_envy_00000"].endswith("pilot_lpy_envy_00000_job_21370027/render_manifest.json")
    assert gt["lpy_ufo_00000"].endswith("pilot_lpy_ufo_00000_job_21370028/render_manifest.json")
    assert gt["lpy_envy_00003"].endswith("family-matrix-20260923/matrix_0_lpy_envy_00003/render_manifest.json")
    assert gt["lpy_ufo_00003"].endswith("family-matrix-20260923/matrix_5_lpy_ufo_00003/render_manifest.json")
    gate = plan["pose_identity"]
    assert (gate["tolerance_m"], gate["min_fraction"], gate["min_iou"]) == (0.001, 0.995, 0.995)
    # The gate compares a pass rendered on the matrix grid with the matrix sampler; the cross-grid sample never gates.
    assert "512x288 matrix grid" in gate["new_geometry"] and "16 samples, seed 1729" in gate["new_geometry"]
    assert "never gates" in gate["cross_grid_measurement"]
    assert "60-minute limit" in plan["render_time_risk"] and "excluded from both arms" in plan["render_time_risk"]
    # NATIVE predictions dominate the storage estimate: 5 models x 192 frames x a 1920x1080 float32 array.
    assert plan["estimated_output_bytes"] > 5 * 192 * 1920 * 1080 * 4


def _fake_assets(tmp_path, q):
    companion = tmp_path / "companion"
    (companion / "trees/metadata").mkdir(parents=True)
    manifests, checkpoints = {}, {}
    for tree_id in q.TREES:
        (companion / "trees/metadata" / f"{tree_id}_metadata.json").write_text(json.dumps({"tree": tree_id}))
        manifest = tmp_path / "matrix" / tree_id / "render_manifest.json"
        manifest.parent.mkdir(parents=True)
        frames = [{"lighting": light, "view_id": view} for light in LIGHTS for view in VIEWS]
        manifest.write_text(json.dumps({"tree_id": tree_id, "ok": True, "frames": frames, "renderer": {}}))
        manifests[tree_id] = manifest
    (tmp_path / "ckpt").mkdir()
    for label in "FJCAB":
        path = tmp_path / "ckpt" / f"{label}.pth"
        path.write_bytes(label.encode())
        checkpoints[label] = {"checkpoint": str(path), "note": label}
    return companion, manifests, checkpoints


def test_plan_refuses_a_missing_or_changed_checkpoint_or_gt_manifest(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_render_gap")
    companion, manifests, checkpoints = _fake_assets(tmp_path, q)

    def plan(**overrides):
        kwargs = {
            "manifests": manifests,
            "checkpoints": checkpoints,
            "companion": companion,
            "matrix_evaluation_plan": None,
        }
        return q.render_gap_plan(**{**kwargs, **overrides})

    built = plan()
    assert len(built["models"]) == 5 and all(len(m["checkpoint_sha256"]) == 64 for m in built["models"].values())
    assert all(len(t["matrix_render_manifest_sha256"]) == 64 for t in built["trees"])
    assert built["maximum_gpu_minutes"] == 180

    a = Path(checkpoints["A"]["checkpoint"])
    a.unlink()
    with pytest.raises(FileNotFoundError, match="Checkpoint A"):
        plan()
    a.write_bytes(b"A")

    gt = manifests["lpy_ufo_00002"]
    text = gt.read_text()
    gt.unlink()
    with pytest.raises(FileNotFoundError, match="lpy_ufo_00002"):
        plan()
    gt.write_text(text)
    with pytest.raises(ValueError, match="No matrix GT"):
        plan(manifests={k: v for k, v in manifests.items() if k != "lpy_envy_00008"})

    # A checkpoint that changed since it was registered, or since its own evaluation recorded it, is refused.
    with pytest.raises(ValueError, match="registered hash"):
        plan(checkpoints={**checkpoints, "F": {**checkpoints["F"], "expected_sha256": "0" * 64}})
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"checkpoint_sha256": "1" * 64}))
    with pytest.raises(ValueError, match="own evaluation"):
        plan(checkpoints={**checkpoints, "J": {**checkpoints["J"], "recorded_status": str(status)}})

    # Only the manifests the published matrix evaluation scored are accepted as ground truth.
    published = tmp_path / "published_plan.json"
    sources = [{"path": str(path), "sha256": q.sha256(path)} for path in manifests.values()]
    published.write_text(json.dumps({"source_renders": sources}))
    assert plan(matrix_evaluation_plan=published)["matrix_evaluation_plan_sha256"] == q.sha256(published)
    sources[3]["sha256"] = "2" * 64
    published.write_text(json.dumps({"source_renders": sources}))
    with pytest.raises(ValueError, match="published matrix evaluation"):
        plan(matrix_evaluation_plan=published)


def test_repeated_or_single_family_registers_are_refused(monkeypatch):
    q = _load(monkeypatch, "queue_render_gap")
    with pytest.raises(ValueError, match="unique"):
        q.render_gap_plan(trees=("lpy_envy_00000", "lpy_envy_00000", "lpy_ufo_00000"), hash_assets=False)
    with pytest.raises(ValueError, match="both families"):
        q.render_gap_plan(trees=("lpy_envy_00000", "lpy_envy_00003"), hash_assets=False)


def test_array_index_resolves_the_tree_the_render_job_will_render(monkeypatch):
    q = _load(monkeypatch, "queue_render_gap")
    plan = q.render_gap_plan(hash_assets=False)
    # This mirrors the python snippet inside hpc/slurm/render_gap_render_frozen.sbatch.
    to_render = [t for t in plan["trees"] if t["status"] == "to_render"]
    assert to_render[0]["tree_id"] == "lpy_envy_00000" and to_render[7]["tree_id"] == "lpy_ufo_00003"
    with pytest.raises(IndexError):
        to_render[8]


def _stub_scheduler(monkeypatch, q, required=None):
    plan = q.render_gap_plan(hash_assets=False)
    monkeypatch.setattr(q, "render_gap_plan", lambda: json.loads(json.dumps(plan)))
    monkeypatch.setattr(q, "_git", lambda root, *args: b"" if args[0] == "diff" else b"abc123\n")

    def freeze(root, destination, revision):
        destination.mkdir()
        return {name: "0" * 64 for name in (q.REQUIRED_SOURCES if required is None else required)}

    monkeypatch.setattr(q, "freeze_source", freeze)
    monkeypatch.setattr(q.subprocess, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setenv("USER", "tester")
    calls = []

    def sbatch(command, env, batch, name):
        # The frozen plan and the passed preflight exist before every scheduler call.
        assert (batch / "plan.json").is_file()
        assert json.loads((batch / "storage_preflight.json").read_text())["ok"]
        calls.append((name, command))
        return str(100 + len(calls))

    monkeypatch.setattr(q, "_sbatch", sbatch)
    return calls


def test_submission_chains_one_render_array_and_one_evaluation(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_render_gap")
    calls = _stub_scheduler(monkeypatch, q)
    result = q.queue_batch(tmp_path, "render-gap-test", share_used_bytes=1_000_000_000_000)
    (render_name, render), (eval_name, evaluation) = calls
    assert (render_name, eval_name) == ("render", "eval")
    assert render[render.index("--array") + 1] == "0-7%1" and "--dependency" not in render
    assert render[render.index("--partition") + 1] == "gpu,ampere"
    assert render[render.index("--time") + 1] == "00:15:00"
    assert render[-2].endswith("code/hpc/slurm/render_gap_render_frozen.sbatch")
    assert evaluation[evaluation.index("--dependency") + 1] == "afterany:101"
    assert evaluation[evaluation.index("--partition") + 1] == "gpu,ampere"
    assert evaluation[evaluation.index("--time") + 1] == "01:00:00"
    assert evaluation[-2].endswith("code/hpc/slurm/render_gap_eval_frozen.sbatch")
    saved = json.loads((tmp_path / "artifacts/generalization/render-gap-test/plan.json").read_text())
    assert saved["code_revision"] == "abc123" and saved["maximum_gpu_minutes"] == 180
    assert set(q.REQUIRED_SOURCES) <= set(saved["source_sha256"])
    assert (result["render_array_job_id"], result["eval_job_id"]) == ("101", "102")


def test_queueing_behind_a_job_only_adds_an_afterany_dependency(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_render_gap")
    calls = _stub_scheduler(monkeypatch, q)
    q.queue_batch(tmp_path, "render-gap-after", share_used_bytes=1, after="21500000")
    render = calls[0][1]
    assert render[render.index("--dependency") + 1] == "afterany:21500000"
    with pytest.raises(ValueError, match="numeric"):
        q.queue_batch(tmp_path, "render-gap-bad-after", share_used_bytes=1, after="21500000;rm")


def test_nothing_is_submitted_before_the_protocol_is_committed(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_render_gap")
    calls = _stub_scheduler(monkeypatch, q, required=[n for n in q.REQUIRED_SOURCES if n != q.PROTOCOL])
    with pytest.raises(ValueError, match="not committed"):
        q.queue_batch(tmp_path, "render-gap-no-protocol", share_used_bytes=1)
    assert calls == []


def test_nothing_is_submitted_past_the_storage_line(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_render_gap")
    calls = _stub_scheduler(monkeypatch, q)
    with pytest.raises(RuntimeError, match="did not pass"):
        q.queue_batch(tmp_path, "render-gap-full", share_used_bytes=2_198_000_000_000)
    assert calls == []
    assert json.loads((tmp_path / "artifacts/generalization/render-gap-full/plan.json").read_text())["trees"]


def test_dirty_tree_and_bad_batch_id_are_refused_before_anything_is_frozen(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_render_gap")
    monkeypatch.setattr(q, "_git", lambda root, *args: b"tools/x.py\n" if args[0] == "diff" else b"abc\n")
    with pytest.raises(ValueError, match="Commit tracked changes"):
        q.queue_batch(tmp_path, "render-gap-test", share_used_bytes=1)
    with pytest.raises(ValueError, match="batch-id"):
        q.queue_batch(tmp_path, "../escape", share_used_bytes=1)
    assert not (tmp_path / "artifacts").exists()


# --- The two-arm evaluation builder ---------------------------------------------------------------------------


def test_native_grid_is_point_sampled_and_rgb_area_averaged_to_the_matrix_grid(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    # 1080/288 = 1920/512 = 3.75: the native pixel containing each matrix pixel centre, floor((i + 0.5) * 3.75).
    assert p.centre_indices(8, 30).tolist() == [int(np.floor((i + 0.5) * 3.75)) for i in range(8)]
    assert p.centre_indices(288, 1080).tolist() == [int(np.floor((i + 0.5) * 3.75)) for i in range(288)]
    assert p.centre_indices(512, 1920)[[0, 1, 511]].tolist() == [1, 5, 1918]
    native = np.arange(30 * 60, dtype=np.float32).reshape(NATIVE_SHAPE)
    sampled = p.point_sample(native, MATRIX_SHAPE)
    assert sampled.shape == MATRIX_SHAPE
    assert sampled[2, 3] == native[p.centre_indices(8, 30)[2], p.centre_indices(16, 60)[3]]
    with pytest.raises(ValueError, match="aspect"):
        p.point_sample(np.zeros((30, 50)), MATRIX_SHAPE)
    rgb = np.random.default_rng(0).integers(0, 256, size=(*NATIVE_SHAPE, 3), dtype=np.uint8)
    small = p.downsample_rgb(rgb, MATRIX_SHAPE)
    assert small.shape == (*MATRIX_SHAPE, 3)
    assert np.array_equal(small, cv2.resize(rgb, (16, 8), interpolation=cv2.INTER_AREA))


def _write_mask(path, mask):
    from PIL import Image

    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8)).save(path)


def _write_rgb(path, rgb):
    from PIL import Image

    Image.fromarray(rgb).save(path)


def _matrix_geometry():
    """A tree bar whose depth changes 1 cm per matrix column, so a shifted render cannot pass."""
    columns = np.arange(MATRIX_SHAPE[1], dtype=np.float32)
    depth = np.broadcast_to(2.0 + 0.01 * columns, MATRIX_SHAPE).astype(np.float32).copy()
    mask = np.zeros(MATRIX_SHAPE, dtype=bool)
    mask[1:7, 4:12] = True
    return depth, mask


def _native_geometry(p, depth, mask, shift=0, seed=0):
    """Native arrays whose pixel at each matrix centre holds the matrix value; every other pixel is noise."""
    rng = np.random.default_rng(seed)
    native_depth = (5.0 + rng.random(NATIVE_SHAPE)).astype(np.float32)
    native_mask = rng.random(NATIVE_SHAPE) > 0.5
    rows, cols = p.centre_indices(MATRIX_SHAPE[0], NATIVE_SHAPE[0]), p.centre_indices(MATRIX_SHAPE[1], NATIVE_SHAPE[1])
    native_depth[np.ix_(rows, cols)] = depth
    native_mask[np.ix_(rows, cols)] = mask
    return np.roll(native_depth, shift, axis=1), np.roll(native_mask, shift, axis=1)


def _K(width, height, focal):
    return [[focal, 0, width / 2 - 0.5], [0, focal, height / 2 - 0.5], [0, 0, 1]]


def _render(folder, tree_id, shape, depth, mask, *, seed, extra=None, grid=None):
    """A render manifest in render_family_lighting's format: 6 views x 4 lights, geometry shared per view.

    ``grid`` (depth, mask) adds the training renderer's matrix-grid pass, the same arrays for every view.
    """
    rng = np.random.default_rng(seed)
    (folder / "geometry").mkdir(parents=True)
    K = _K(shape[1], shape[0], 0.78 * shape[1])
    frames = []
    for index, view in enumerate(VIEWS):
        np.save(folder / "geometry" / f"{view}.npy", depth)
        _write_mask(folder / "geometry" / f"{view}_mask.png", mask)
        for light in LIGHTS:
            (folder / light).mkdir(exist_ok=True)
            rgb = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
            _write_rgb(folder / light / f"{view}.png", rgb)
            frames.append(
                {
                    "rgb": str(folder / light / f"{view}.png"),
                    "depth": str(folder / "geometry" / f"{view}.npy"),
                    "mask": str(folder / "geometry" / f"{view}_mask.png"),
                    "tree_id": tree_id,
                    "family": tree_id.split("_")[1],
                    "lighting": light,
                    "view_id": view,
                    "rig": index // 2,
                    "projected_target_pixel_xy": None,
                    "target_pixel_xy": None,
                    "target_visible": False,
                    "pixel_coordinate_convention": "zero-based pixel centers; Blender edge coordinates minus 0.5",
                    "camera_location": [-9.7 - 0.12 * (index % 2), -7.1, 0.85 + 0.15 * (index // 2)],
                    "camera_rotation_euler": [1.5395, 0.0, 3.1946],
                    "K": K,
                    "geometry_sha256": "g" * 64,
                }
            )
    manifest = {
        "schema_version": 1,
        "tree_id": tree_id,
        "family": tree_id.split("_")[1],
        "geometry_sha256": "g" * 64,
        "tree_transform": np.eye(4).tolist(),
        "target": None,
        "K": K,
        "frames": frames,
        "ok": True,
        **(extra or {}),
    }
    if grid is not None:
        (folder / "geometry_matrix_grid").mkdir()
        views = {}
        for view in VIEWS:
            views[view] = {
                "depth": str(folder / "geometry_matrix_grid" / f"{view}.npy"),
                "mask": str(folder / "geometry_matrix_grid" / f"{view}_mask.png"),
            }
            np.save(views[view]["depth"], grid[0])
            _write_mask(views[view]["mask"], grid[1])
        height, width = grid[0].shape
        manifest["matrix_grid_geometry"] = {
            "width": width,
            "height": height,
            "sampler": {"samples": 16, "seed": 1729, "use_animated_seed": False, "use_adaptive_sampling": False},
            "K": _K(width, height, 0.78 * width),
            "lighting": "source",
            "views": views,
        }
    (folder / "render_manifest.json").write_text(json.dumps(manifest))
    return folder / "render_manifest.json"


def _batch(p, tmp_path, trees, *, without_grid=()):
    """A frozen render-gap batch: ``trees`` maps tree_id to a column shift (None means the render never ran).

    A shift moves the whole render, its 1920x1080-style geometry and its matrix-grid pass alike; the trees in
    ``without_grid`` have no matrix-grid pass.
    """
    batch = tmp_path / "batch"
    batch.mkdir()
    depth, mask = _matrix_geometry()
    entries = []
    for seed, (tree_id, shift) in enumerate(trees.items()):
        matrix = _render(tmp_path / "matrix" / tree_id, tree_id, MATRIX_SHAPE, depth, mask, seed=100 + seed)
        entries.append(
            {
                "tree_id": tree_id,
                "family": tree_id.split("_")[1],
                "status": "to_render",
                "matrix_render_manifest": str(matrix),
            }
        )
        if shift is not None:
            native_depth, native_mask = _native_geometry(p, depth, mask, shift=shift, seed=seed)
            columns = shift * MATRIX_SHAPE[1] // NATIVE_SHAPE[1]
            grid = (np.roll(depth, columns, axis=1), np.roll(mask, columns, axis=1))
            _render(
                batch / "render" / tree_id,
                tree_id,
                NATIVE_SHAPE,
                native_depth,
                native_mask,
                seed=seed,
                extra={"renderer_mode": "training", "smoke": False},
                grid=None if tree_id in without_grid else grid,
            )
    (batch / "plan.json").write_text(json.dumps({"scope": "Synthetic render-gap batch.", "trees": entries}))
    return batch


def _identities(p, batch):
    plan = json.loads((batch / "plan.json").read_text())
    results = {}
    for entry in plan["trees"]:
        new = json.loads((batch / "render" / entry["tree_id"] / "render_manifest.json").read_text())
        matrix = json.loads(Path(entry["matrix_render_manifest"]).read_text())
        results[entry["tree_id"]] = p.tree_identity(new, matrix)
    return results


def test_pose_identity_gates_on_the_matrix_grid_pass_and_fails_a_shifted_render(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    results = _identities(p, _batch(p, tmp_path, {"lpy_envy_00000": 0, "lpy_ufo_00000": 8}))
    same, shifted = results["lpy_envy_00000"], results["lpy_ufo_00000"]
    assert same["ok"] and same["matrix_grid_pass"] and same["views_passing"] == 6
    assert all(v["gate"]["mask_iou"] == 1.0 and v["gate"]["tree_depth_within_tolerance"] == 1.0 for v in same["views"])
    assert all(v["matrix_grid_shape"] == list(MATRIX_SHAPE) for v in same["views"])
    view = same["views"][0]
    assert view["diagnostics"]["focal_ratio"] == pytest.approx(3.75)
    assert view["diagnostics"]["matrix_grid_focal_max_abs_diff_px"] == 0.0
    assert same["diagnostics"]["camera_poses_equal_within_1e-6"] and same["diagnostics"]["geometry_sha256_equal"]
    # The cross-grid point sample is recorded beside the gate; these native centre pixels hold the matrix values.
    assert view["cross_grid_measurement"]["would_pass"] and same["cross_grid_views_would_pass"] == 6
    # Eight native pixels are two matrix columns: the mask and the 1 cm per column depth both move.
    assert not shifted["ok"] and shifted["views_passing"] == 0
    gate = shifted["views"][0]["gate"]
    assert gate["mask_iou"] < 0.995 and gate["tree_depth_within_tolerance"] < 0.995


def _scene(x, y):
    """A continuous image in matrix-pixel units: slanted bars 1.2 px wide whose depth rises 5 cm per pixel."""
    on_bar = np.abs(np.mod(x - 0.3 * y, 4.0) - 2.0) < 0.6
    return np.where(on_bar, 2.0 + 0.05 * x + 0.015 * y, 1e10).astype(np.float32), on_bar


def _sample_scene(shape, pixels_per_matrix_pixel):
    """The scene sampled once at each pixel centre of a ``shape`` grid, as Cycles samples depth and object index."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    return _scene((cols + 0.5) / pixels_per_matrix_pixel, (rows + 0.5) / pixels_per_matrix_pixel)


def test_cross_grid_sampling_fails_at_an_identical_pose_and_never_gates(monkeypatch, tmp_path):
    """Why the gate compares a matrix-grid pass: no 1920x1080 pixel centre is a 512x288 pixel centre."""
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    offsets = (p.centre_indices(512, 1920) + 0.5) / 3.75 - (np.arange(512) + 0.5)
    assert sorted({round(abs(float(o)), 9) for o in offsets}) == [round(1 / 30, 9), round(1 / 10, 9)]
    matrix_depth, matrix_mask = _sample_scene(MATRIX_SHAPE, 1.0)
    native_depth, native_mask = _sample_scene(NATIVE_SHAPE, NATIVE_SHAPE[1] / MATRIX_SHAPE[1])
    matrix = _render(tmp_path / "matrix", "lpy_envy_00000", MATRIX_SHAPE, matrix_depth, matrix_mask, seed=1)
    new = _render(
        tmp_path / "new",
        "lpy_envy_00000",
        NATIVE_SHAPE,
        native_depth,
        native_mask,
        seed=2,
        extra={"renderer_mode": "training", "smoke": False},
        grid=_sample_scene(MATRIX_SHAPE, 1.0),
    )
    result = p.tree_identity(json.loads(new.read_text()), json.loads(matrix.read_text()))
    # One scene, one camera: the pass rendered on the matrix grid reproduces the ground truth and admits the tree,
    assert result["ok"] and result["views_passing"] == 6
    # while the native geometry point-sampled at the matrix centres fails at the identical pose.
    measured = result["views"][0]["cross_grid_measurement"]
    assert not measured["would_pass"] and result["cross_grid_views_would_pass"] == 0
    assert measured["tree_depth_within_tolerance"] < 0.995


def test_build_writes_both_arms_and_records_every_exclusion(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    batch = _batch(p, tmp_path, {"lpy_envy_00000": 0, "lpy_ufo_00000": 8, "lpy_ufo_00001": None})
    out = batch / "eval"
    out.mkdir()
    assert p.build(batch, out) == {"native": 2, "matched": 1}

    status = {t["tree_id"]: t for t in json.loads((out / "render_status.json").read_text())["trees"]}
    assert status["lpy_ufo_00001"]["present"] is False and not status["lpy_ufo_00001"]["native"]
    assert status["lpy_ufo_00000"]["native"] and not status["lpy_ufo_00000"]["matched"]
    assert status["lpy_ufo_00000"]["matched_excluded"].startswith("pose identity failed: 0 of 6")
    identity = json.loads((out / "pose_identity.json").read_text())
    assert identity["matched_trees"] == ["lpy_envy_00000"] and identity["excluded_trees"] == ["lpy_ufo_00000"]
    assert identity["gate"]["min_mask_iou"] == 0.995 and len(identity["trees"][1]["views"]) == 6
    assert "matrix_grid_geometry" in identity["gate"]["new_geometry"] and "or seed" not in identity["limit"]

    native = json.loads((out / "native/plan.json").read_text())
    matched = json.loads((out / "matched/plan.json").read_text())
    # Both plans are prepare_family_evaluation.py's own output.
    keys = {"schema_version", "scope", "source_renders", "frames", "depth_convention", "target_scope", "temporal_scope"}
    assert set(native) == set(matched) == keys
    assert len(native["frames"]) == 48 and len(matched["frames"]) == 24
    assert native["scope"].endswith(p.ARMS["native"]) and matched["scope"].endswith(p.ARMS["matched"])
    assert all(f["rgb"].startswith(str(batch / "render")) for f in native["frames"])
    matrix = json.loads((tmp_path / "matrix/lpy_envy_00000/render_manifest.json").read_text())
    published = {(f["lighting"], f["view_id"]): f for f in matrix["frames"]}
    for frame in matched["frames"]:
        source = published[(frame["lighting"], frame["view_id"])]
        # Published depth, mask and K; only the RGB is the new render, area-averaged to the matrix grid.
        assert (frame["depth"], frame["mask"], frame["K"]) == (source["depth"], source["mask"], source["K"])
        assert frame["matrix_rgb"] == source["rgb"] and frame["rgb"].startswith(str(out / "matched/rgb"))
        expected = cv2.resize(cv2.imread(frame["render_gap_source_rgb"]), (16, 8), interpolation=cv2.INTER_AREA)
        assert np.array_equal(cv2.imread(frame["rgb"]), expected)
    assert json.loads((out / "matched/status.json").read_text())["trees"] == 1


def test_a_render_without_the_matrix_grid_pass_stays_native_and_never_enters_matched(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    batch = _batch(p, tmp_path, {"lpy_envy_00000": 0, "lpy_ufo_00000": 0}, without_grid={"lpy_ufo_00000"})
    out = batch / "eval"
    assert p.build(batch, out) == {"native": 2, "matched": 1}
    status = {t["tree_id"]: t for t in json.loads((out / "render_status.json").read_text())["trees"]}
    assert status["lpy_ufo_00000"]["native"] and not status["lpy_ufo_00000"]["matched"]
    assert "no matrix-grid geometry pass" in status["lpy_ufo_00000"]["matched_excluded"]
    trees = {t["tree_id"]: t for t in json.loads((out / "pose_identity.json").read_text())["trees"]}
    excluded = trees["lpy_ufo_00000"]
    assert not excluded["ok"] and not excluded["matrix_grid_pass"] and excluded["views_passing"] == 0
    assert all("no matrix-grid pass" in v["gate"]["reason"] for v in excluded["views"])
    # Its cross-grid measurement would have passed on these arrays; a measurement never admits a tree.
    assert excluded["cross_grid_views_would_pass"] == 6


def test_an_arm_with_no_qualifying_tree_writes_its_status_and_no_plan(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    batch = _batch(p, tmp_path, {"lpy_envy_00000": 8, "lpy_ufo_00000": 8})
    out = batch / "eval"
    assert p.build(batch, out) == {"native": 2, "matched": 0}
    status = json.loads((out / "matched/status.json").read_text())
    assert status["plan"] is None and status["trees"] == 0 and "pose_identity.json" in status["reason"]
    assert not (out / "matched/plan.json").exists() and (out / "native/plan.json").is_file()


def test_smoke_and_matrix_renderer_manifests_are_refused_before_anything_is_written(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    p = _load(monkeypatch, "prepare_render_gap_evaluation")
    with pytest.raises(ValueError, match="smoke"):
        p.check_new_manifest(
            {"tree_id": "lpy_envy_00000", "renderer_mode": "training", "smoke": True}, "lpy_envy_00000"
        )
    with pytest.raises(ValueError, match="training-renderer"):
        p.check_new_manifest({"tree_id": "lpy_envy_00000"}, "lpy_envy_00000")
    with pytest.raises(ValueError, match="lpy_envy_00003"):
        p.check_new_manifest({"tree_id": "lpy_envy_00003", "renderer_mode": "training"}, "lpy_envy_00000")
    batch = _batch(p, tmp_path, {"lpy_envy_00000": 0, "lpy_ufo_00000": 0})
    manifest = batch / "render/lpy_ufo_00000/render_manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "smoke": True}))
    with pytest.raises(ValueError, match="smoke"):
        p.build(batch, batch / "eval")
    # Every manifest is checked before the first write, so a refusal leaves no half-built evaluation behind.
    assert not (batch / "eval").exists()


# --- The renderer's options -----------------------------------------------------------------------------------

BASE = ["--companion", "/companion", "--tree-id", "lpy_envy_00003", "--output", "/out"]


def test_matrix_renderer_stays_the_default(monkeypatch):
    rfl = _load(monkeypatch, "render_family_lighting")
    args = rfl.parse_args(BASE)
    assert (args.renderer, args.width, args.samples, args.lights) == ("matrix", 512, 16, LIGHTS)
    assert not args.one_view and not args.smoke
    # hpc/slurm/family_matrix_frozen.sbatch passes width, samples and lights explicitly and no --renderer.
    assert vars(rfl.parse_args([*BASE, "--width", "512", "--samples", "16", "--lights", *LIGHTS])) == vars(args)
    assert rfl.parse_args([*BASE, "--width", "256", "--one-view"]).width == 256
    assert "--renderer" not in (ROOT / "hpc/slurm/family_matrix_frozen.sbatch").read_text()


def test_training_renderer_refuses_matrix_options_and_smoke_is_training_only(monkeypatch):
    rfl = _load(monkeypatch, "render_family_lighting")
    args = rfl.parse_args([*BASE, "--renderer", "training"])
    assert args.renderer == "training" and args.width is None and args.samples is None and not args.smoke
    assert rfl.parse_args([*BASE, "--renderer", "training", "--smoke"]).smoke
    for extra in (
        ["--smoke"],
        ["--renderer", "training", "--width", "512"],
        ["--renderer", "training", "--samples", "16"],
    ):
        with pytest.raises(SystemExit):
            rfl.parse_args([*BASE, *extra])
    command = (ROOT / "hpc/slurm/render_gap_render_frozen.sbatch").read_text().split("exec ", 1)[1]
    assert "--renderer training" in command
    assert all(option not in command for option in ("--smoke", "--width", "--samples"))


def test_template_check_is_the_training_renderers(monkeypatch):
    rfl = _load(monkeypatch, "render_family_lighting")
    module = ast.parse((TOOLS / "render_training_lighting.py").read_text())
    literal = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TEMPLATE_SAMPLER" for t in node.targets)
    )
    assert ast.literal_eval(literal) == rfl.TEMPLATE_SAMPLER
    template = {**rfl.TEMPLATE_SAMPLER, "view_transform": "Filmic", "resolution": [1920, 1080, 100], "device": "GPU"}
    assert rfl.template_differences(template, 1920, 1080) == {}
    matrix_like = {
        **template,
        "samples": 16,
        "seed": 1729,
        "use_adaptive_sampling": False,
        "resolution": [512, 288, 100],
    }
    assert set(rfl.template_differences(matrix_like, 1920, 1080)) == {
        "samples",
        "seed",
        "use_adaptive_sampling",
        "resolution",
    }
    changed = {**template, "view_transform": "Standard", "denoiser": "OPTIX", "filter_width": 1.0}
    assert set(rfl.template_differences(changed, 1920, 1080)) == {"view_transform", "denoiser", "filter_width"}


def _template_scene():
    """A stand-in for the scene after generate_tree2.render_tree: the template's training sampler at 1920x1080."""
    cycles = SimpleNamespace(
        device="GPU",
        samples=4096,
        use_adaptive_sampling=True,
        adaptive_threshold=0.01,
        adaptive_min_samples=0,
        use_denoising=True,
        denoiser="OPENIMAGEDENOISE",
        seed=0,
        use_animated_seed=False,
        filter_width=1.5,
        max_bounces=12,
        sampling_pattern="AUTOMATIC",
    )
    render = SimpleNamespace(resolution_x=1920, resolution_y=1080, resolution_percentage=100)
    view = SimpleNamespace(view_transform="Filmic", look="None", exposure=0.0, gamma=1.0)
    return SimpleNamespace(cycles=cycles, render=render, view_settings=view)


def test_smoke_settings_and_the_recorded_sampler(monkeypatch):
    rfl = _load(monkeypatch, "render_family_lighting")
    scene = _template_scene()
    settings = rfl.sampler_settings(scene)
    assert rfl.template_differences(settings, 1920, 1080) == {}
    assert (
        settings["max_bounces"] == 12 and settings["sampling_pattern"] == "AUTOMATIC" and "time_limit" not in settings
    )
    rfl.apply_smoke(scene)
    smoke = rfl.sampler_settings(scene)
    assert (smoke["device"], smoke["samples"], smoke["use_adaptive_sampling"], smoke["resolution"]) == (
        "CPU",
        1,
        False,
        [64, 36, 100],
    )
    assert set(rfl.template_differences(smoke, 1920, 1080)) == {"samples", "use_adaptive_sampling", "resolution"}


def test_matrix_grid_pass_uses_the_matrix_sampler_and_restores_the_training_settings(monkeypatch, tmp_path):
    rfl = _load(monkeypatch, "render_family_lighting")
    matrix_plan = _load(monkeypatch, "queue_family_matrix").matrix_plan(hash_assets=False)
    state = rfl.matrix_grid_state()
    # The published matrix renderer's grid and sampler: width 512 at 16:9, 16 fixed samples, seed 1729.
    assert (state["resolution_x"], state["samples"], state["seed"]) == (
        matrix_plan["width"],
        matrix_plan["samples"],
        matrix_plan["seed"],
    )
    assert (state["resolution_y"], state["resolution_percentage"]) == (288, 100)
    assert state["use_adaptive_sampling"] is False and state["use_animated_seed"] is False
    smoke = rfl.matrix_grid_state(smoke=True)
    assert smoke == {**state, "samples": 1}
    scene = _template_scene()
    training = rfl.sampler_settings(scene)
    saved = rfl.scene_state(scene)
    rfl.set_scene_state(scene, state)
    grid = rfl.sampler_settings(scene)
    assert (grid["resolution"], grid["samples"], grid["seed"], grid["use_adaptive_sampling"]) == (
        [512, 288, 100],
        16,
        1729,
        False,
    )
    # Only the grid and the sampler's count, seed and adaptivity change; the denoiser and filter stay the template's.
    assert {k for k in training if training[k] != grid[k]} == {"resolution", "samples", "seed", "use_adaptive_sampling"}
    rfl.set_scene_state(scene, saved)
    assert rfl.sampler_settings(scene) == training and rfl.template_differences(training, 1920, 1080) == {}
    rfl.write_manifest(tmp_path, {"ok": False, "frames": [{"view_id": "rig0_l"}]})
    assert json.loads((tmp_path / "render_manifest.json").read_text())["frames"] == [{"view_id": "rig0_l"}]
    assert not (tmp_path / "render_manifest.json.tmp").exists()
