"""The planner keeps every registered frame in the denominator and derives the photometric cells itself."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    out = {}
    for name in ("generalization_controls", "queue_generalization_controls", "prepare_controls_evaluation"):
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out[name] = module
    return out


def _render(folder, gc, tree_id, *, drop=()):
    """A synthetic render manifest with the real layout and tiny images."""
    rng = np.random.default_rng(hash(tree_id) % 1000)
    folder.mkdir(parents=True)
    frames, skipped = [], []
    for cond in gc.CONDITIONS:
        if cond["id"] in drop:
            skipped.append({"condition": cond["id"], "reason": "no_geometric_target"})
            continue
        spec = gc.CAMERA_MODELS[cond["camera_model"]]
        w, h = spec["width"] // 16, spec["height"] // 16
        key = gc.geometry_key(cond["camera_model"], cond["pose_set"])
        geometry = folder / "geometry" / key
        geometry.mkdir(parents=True, exist_ok=True)
        cond_dir = folder / "conditions" / cond["id"].replace("/", "__")
        cond_dir.mkdir(parents=True)
        for view in gc.pose_ids(gc.POSE_SETS[cond["pose_set"]]):
            depth, mask = geometry / f"{view}.npy", geometry / f"{view}_mask.png"
            if not depth.exists():
                np.save(depth, np.full((h, w), 2.0, dtype=np.float32))
                m = np.zeros((h, w), dtype=np.uint8)
                m[:, w // 2] = 255
                cv2.imwrite(str(mask), m)
            level = 55 if cond["lighting"].startswith("evening") else 140
            image = np.clip(rng.normal(level, 20, size=(h, w, 3)), 0, 255).astype(np.uint8)
            rgb = cond_dir / f"{view}.png"
            cv2.imwrite(str(rgb), image)
            frames.append(
                {
                    "rgb": str(rgb),
                    "depth": str(depth),
                    "mask": str(mask),
                    "tree_id": tree_id,
                    "family": gc.condition(cond["id"]) and tree_id.split("_")[1],
                    "condition": cond["id"],
                    "condition_group": cond["group"],
                    "lighting": cond["lighting"],
                    "camera_model": cond["camera_model"],
                    "pose_set": cond["pose_set"],
                    "geometry_key": key,
                    "dino_rig_compatible": gc.dino_rig_compatible(cond["camera_model"], cond["pose_set"]),
                    "view_id": view,
                    "width": w,
                    "height": h,
                    "K": [
                        [spec["lens_mm"] / spec["sensor_width_mm"] * w, 0, w / 2 - 0.5],
                        [0, 1, h / 2 - 0.5],
                        [0, 0, 1],
                    ],
                    "target_visible": True,
                    "target_pixel_xy": [w / 2, h / 2],
                }
            )
    manifest = {
        "ok": True,
        "tree_id": tree_id,
        "geometry_sha256": "g",
        "frames": frames,
        "skipped": skipped,
        "lighting": {},
    }
    (folder / "render_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_missing_trees_and_conditions_stay_in_the_denominator(tmp_path, modules):
    gc, launcher, prepare = (
        modules[k] for k in ("generalization_controls", "queue_generalization_controls", "prepare_controls_evaluation")
    )
    plan = launcher.controls_plan(trees=("lpy_envy_00003", "lpy_ufo_00001", "lpy_ufo_00002"), hash_assets=False)
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "plan.json").write_text(json.dumps(plan))
    _render(batch / "controls_0_lpy_envy_00003", gc, "lpy_envy_00003")
    _render(
        batch / "controls_1_lpy_ufo_00001",
        gc,
        "lpy_ufo_00001",
        drop=("source/close/isaac_wrist", "source/sweep/training_rig"),
    )
    # lpy_ufo_00002 never rendered.
    out = tmp_path / "eval"
    out.mkdir()
    final = prepare.build(batch, out)
    assert final["registered_frames"] == 3 * (62 + 24)
    assert final["scored_frames"] == (62 + 24) + (62 - 6 - 8 + 24)
    reasons = {(m["tree_id"], m["condition"]): m["reason"] for m in final["missing"]}
    assert reasons[("lpy_ufo_00002", "*")] == "manifest_absent"
    assert reasons[("lpy_ufo_00001", "source/close/isaac_wrist")] == "no_geometric_target"
    assert reasons[("lpy_ufo_00001", "source/sweep/training_rig")] == "no_geometric_target"
    status = json.loads((out / "render_status.json").read_text())
    assert status["registered"] == 3 and status["with_manifest"] == 2
    photometric = {f["condition"] for f in final["frames"] if f.get("condition_group") == "photometric"}
    assert photometric == {f"evening/{v}" for v in gc.PHOTOMETRIC_VARIANTS}
    gamma = [f for f in final["frames"] if f["condition"] == "evening/gamma"]
    assert len(gamma) == 12 and all(f["dino_rig_compatible"] and f["lighting"] == "evening" for f in gamma)
    assert final["luma_by_condition"]["evening/gamma"]["mean"] > final["luma_by_condition"]["evening"]["mean"]
    assert final["luma_by_condition"]["source"]["n"] == 12
    # Camera cells under the source light were not used as the luma reference.
    manifest = json.loads(Path(final["photometric_manifest"]).read_text())
    assert manifest["target_luma"] == pytest.approx(final["luma_by_condition"]["source"]["mean"], abs=0.01)
    with pytest.raises(FileExistsError):
        prepare.build(batch, out)


def test_lighting_cells_must_share_geometry_and_differ_in_pixels(tmp_path, modules):
    gc, launcher, prepare = (
        modules[k] for k in ("generalization_controls", "queue_generalization_controls", "prepare_controls_evaluation")
    )
    plan = launcher.controls_plan(trees=("lpy_envy_00003", "lpy_ufo_00001"), hash_assets=False)
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "plan.json").write_text(json.dumps(plan))
    manifest = _render(batch / "controls_0_lpy_envy_00003", gc, "lpy_envy_00003")
    # Copy the source bytes over the overcast cell: identical pixels must be refused.
    source = {f["view_id"]: f["rgb"] for f in manifest["frames"] if f["condition"] == "source"}
    for f in manifest["frames"]:
        if f["condition"] == "overcast":
            Path(f["rgb"]).write_bytes(Path(source[f["view_id"]]).read_bytes())
    with pytest.raises(ValueError, match="identical bytes"):
        prepare.build(batch, tmp_path / "eval")
