from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from isaaclab_pruning.sensors.tof_raycaster import (
    MOCK_PRUNER_BASE_PRIM_EXPR,
    TOF_SITE_OFFSETS_M,
    TOF_SITE_PRIM_EXPRS,
    VL53L8CX_INTRINSICS,
    make_vl53l8cx_raycaster_cfg,
    pinhole_intrinsics_from_diagonal_fov,
)
from isaaclab_pruning.sim.prim_paths import ROBOT_PRIM_EXPR, TREE_PRIM_EXPR


def test_8x8_intrinsics_preserve_the_65_degree_diagonal_fov() -> None:
    intrinsics = VL53L8CX_INTRINSICS

    assert (intrinsics.height, intrinsics.width) == (8, 8)
    assert intrinsics.diagonal_fov_deg == 65.0
    assert intrinsics.horizontal_fov_deg == pytest.approx(intrinsics.vertical_fov_deg)
    reconstructed = math.degrees(
        2.0 * math.atan(math.hypot(intrinsics.width, intrinsics.height) / (2.0 * intrinsics.focal_length_px))
    )
    assert reconstructed == pytest.approx(65.0)
    assert intrinsics.matrix_row_major == pytest.approx(
        (
            intrinsics.focal_length_px,
            0.0,
            4.0,
            0.0,
            intrinsics.focal_length_px,
            4.0,
            0.0,
            0.0,
            1.0,
        )
    )


@pytest.mark.parametrize(
    ("width", "height", "diagonal_fov_deg"),
    [(0, 8, 65.0), (8, -1, 65.0), (8, 8, 0.0), (8, 8, 180.0), (8, 8, float("nan"))],
)
def test_pinhole_conversion_rejects_invalid_models(width: int, height: int, diagonal_fov_deg: float) -> None:
    with pytest.raises(ValueError):
        pinhole_intrinsics_from_diagonal_fov(
            width=width,
            height=height,
            diagonal_fov_deg=diagonal_fov_deg,
        )


def test_site_paths_use_the_reviewed_generated_urdf_frames() -> None:
    assert ROBOT_PRIM_EXPR.startswith("/World/")
    assert "{ENV_REGEX_NS}" not in MOCK_PRUNER_BASE_PRIM_EXPR
    assert TOF_SITE_PRIM_EXPRS["tof0"] == f"{MOCK_PRUNER_BASE_PRIM_EXPR}/mock_pruner__tof0"
    assert TOF_SITE_PRIM_EXPRS["tof1"] == f"{MOCK_PRUNER_BASE_PRIM_EXPR}/mock_pruner__tof1"
    assert TOF_SITE_OFFSETS_M == {
        "tof0": (0.04685226669, 0.0, 0.14444246761),
        "tof1": (-0.04685226669, 0.0, 0.14444246761),
    }


def test_factory_emits_the_v60_multi_mesh_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePatternCfg:
        @classmethod
        def from_intrinsic_matrix(cls, **kwargs):
            return SimpleNamespace(**kwargs)

    class FakeOffsetCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeMultiMeshRayCasterCameraCfg:
        OffsetCfg = FakeOffsetCfg

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sensors_module = ModuleType("isaaclab.sensors")
    sensors_module.MultiMeshRayCasterCameraCfg = FakeMultiMeshRayCasterCameraCfg
    ray_caster_module = ModuleType("isaaclab.sensors.ray_caster")
    ray_caster_module.patterns = SimpleNamespace(PinholeCameraPatternCfg=FakePatternCfg)
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors.ray_caster", ray_caster_module)

    cfg = make_vl53l8cx_raycaster_cfg("tof0")

    # v60 beta drops a non-rigid site's resolved offset. The runtime tracks the
    # rigid base and uses the importer-verified site translation explicitly.
    assert cfg.prim_path == MOCK_PRUNER_BASE_PRIM_EXPR
    assert cfg.mesh_prim_paths == [TREE_PRIM_EXPR]
    assert cfg.spawn is None
    assert cfg.update_period == pytest.approx(1.0 / 15.0)
    assert cfg.offset.pos == TOF_SITE_OFFSETS_M["tof0"]
    assert cfg.offset.rot == (0.0, 0.0, 0.0, 1.0)  # Isaac ray camera API is xyzw.
    assert cfg.offset.convention == "ros"  # +Z optical axis.
    assert cfg.pattern_cfg.width == 8
    assert cfg.pattern_cfg.height == 8
    assert cfg.pattern_cfg.intrinsic_matrix == pytest.approx(list(VL53L8CX_INTRINSICS.matrix_row_major))
    assert cfg.max_distance == 3.4
    assert cfg.data_types == ["distance_to_camera"]
    assert cfg.depth_clipping_behavior == "none"


def test_factory_rejects_unknown_sensor_or_empty_targets() -> None:
    with pytest.raises(ValueError, match="Unknown ToF sensor"):
        make_vl53l8cx_raycaster_cfg("tof2")
    with pytest.raises(ValueError, match="mesh_prim_paths"):
        make_vl53l8cx_raycaster_cfg("tof0", mesh_prim_paths=())
