import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

spec = importlib.util.spec_from_file_location(
    "photometric_normalization", Path(__file__).parents[1] / "tools/photometric_normalization.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dark_frame(seed=0, shape=(24, 32)):
    rng = np.random.default_rng(seed)
    # A dim frame with real structure: mean luma near 55, like the evening preset.
    return np.clip(rng.normal(55, 20, size=(*shape, 3)), 0, 255).astype(np.uint8)


def test_exposure_and_gamma_hit_the_target_luma_without_touching_geometry():
    image = dark_frame()
    for variant in ("exposure", "gamma"):
        out, params = m.normalize(image, variant, target_luma=142.0)
        assert out.shape == image.shape and out.dtype == np.uint8
        # Within a few luma levels: clipping at 255 and 8-bit rounding bound the residual.
        assert m.luma(out) == pytest.approx(142.0, abs=3.0)
        assert params, "the applied parameter must be recorded"
    _, exposure = m.normalize(image, "exposure", target_luma=142.0)
    assert exposure["gain"] > 1.0
    _, gamma = m.normalize(image, "gamma", target_luma=142.0)
    assert 0.0 < gamma["gamma"] < 1.0  # brightening exponent


def test_equalization_variants_use_fixed_parameters():
    image = dark_frame(1)
    out, params = m.normalize(image, "clahe")
    assert params == {"clip_limit": 2.0, "tile": 8}
    assert out.shape == image.shape
    out, params = m.normalize(image, "hist_eq")
    assert params == {}
    assert m.luma(out) > m.luma(image)


def test_rejects_wrong_dtype_and_unknown_variant():
    with pytest.raises(ValueError):
        m.normalize(np.zeros((4, 4, 3), dtype=np.float32), "exposure")
    with pytest.raises(ValueError):
        m.normalize(dark_frame(), "auto_contrast")


def test_export_writes_scoreable_plans_and_refuses_overwrite(tmp_path):
    frames = []
    for lighting, seed in (("source", 0), ("source", 1), ("evening", 2), ("evening", 3)):
        rgb = tmp_path / f"{lighting}_{seed}.png"
        image = dark_frame(seed) if lighting == "evening" else np.full((24, 32, 3), 140, dtype=np.uint8)
        cv2.imwrite(str(rgb), image)
        frames.append(
            {"rgb": str(rgb), "depth": f"/gt/{seed}.npy", "mask": f"/gt/{seed}.png", "lighting": lighting, "tree": "t"}
        )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": 1, "depth_convention": "z", "frames": frames}))
    out = tmp_path / "normalized"
    manifest = m.export(plan_path, out, lighting="evening", variants=("exposure", "clahe"))
    assert manifest["frames"] == 2
    assert manifest["reference_luma_measured_on_source_frames"] == pytest.approx(140.0, abs=1.0)
    assert manifest["target_luma"] == manifest["reference_luma_measured_on_source_frames"]
    assert manifest["source_plan_sha256"] == m.sha256(plan_path)
    for variant in ("exposure", "clahe"):
        variant_plan = json.loads(Path(manifest["variants"][variant]["plan"]).read_text())
        assert variant_plan["photometric_normalization"] == variant
        assert variant_plan["depth_convention"] == "z"
        assert len(variant_plan["frames"]) == 2
        for frame in variant_plan["frames"]:
            assert frame["lighting"] == "evening"
            # Depth, mask and metadata pass through; only the RGB path changes.
            assert frame["depth"].startswith("/gt/") and frame["mask"].startswith("/gt/")
            assert Path(frame["rgb"]).exists() and frame["rgb"] != frame["original_rgb"]
            assert cv2.imread(frame["rgb"]).shape == (24, 32, 3)
    exposure = manifest["variants"]["exposure"]
    assert exposure["mean_luma_after"] == pytest.approx(140.0, abs=3.0)
    assert all("luma_before" in p and "gain" in p for p in exposure["per_frame"])
    with pytest.raises(FileExistsError):
        m.export(plan_path, out, lighting="evening")
    with pytest.raises(ValueError):
        m.export(plan_path, tmp_path / "other", lighting="noon")
