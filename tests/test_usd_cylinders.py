from __future__ import annotations

import numpy as np
import pytest

from isaaclab_pruning.geometry import Cylinder
from isaaclab_pruning.usd import write_cylinder_tree_usd


def _sample_cylinders() -> list[Cylinder]:
    return [
        Cylinder(
            record_id="(0, 0, 0)",
            part_name="trunk_1",
            centroid=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 1.0]),
            radius=0.05,
            length=1.0,
        ),
        Cylinder(
            record_id="(0, 0, 1)",
            part_name="spur_1",
            centroid=np.array([0.2, 0.0, 0.75]),
            orientation=np.array([1.0, 0.0, 0.0]),
            radius=0.006,
            length=0.1,
        ),
    ]


@pytest.mark.isaacsim_ci
def test_usd_writer_preserves_geometry_semantics_and_collision_lod(tmp_path) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    output = tmp_path / "tree.usda"
    summary = write_cylinder_tree_usd(_sample_cylinders(), output, tree_id="lpy_envy_00000")
    stage = Usd.Stage.Open(str(output))
    trunk = UsdGeom.Cylinder.Get(
        stage,
        "/lpy_envy_00000/trunk/trunk_1/cylinder_00000",
    )
    spur = UsdGeom.Cylinder.Get(
        stage,
        "/lpy_envy_00000/spur/spur_1/cylinder_00001",
    )

    assert trunk.GetHeightAttr().Get() == pytest.approx(1.0)
    assert trunk.GetRadiusAttr().Get() == pytest.approx(0.05)
    assert trunk.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert not spur.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert summary["organ_counts"] == {"spur": 1, "trunk": 1}
    assert summary["collision_cylinders"] == 1


def test_usd_writer_requires_pxr(monkeypatch, tmp_path) -> None:
    import isaaclab_pruning.usd.cylinders as module

    def _missing_pxr() -> tuple:
        raise RuntimeError("USD authoring requires the `pxr` modules bundled with Isaac Sim.")

    monkeypatch.setattr(module, "_require_pxr", _missing_pxr)
    with pytest.raises(RuntimeError, match="pxr"):
        write_cylinder_tree_usd(
            _sample_cylinders(),
            tmp_path / "missing.usda",
            tree_id="lpy_envy_00000",
        )
