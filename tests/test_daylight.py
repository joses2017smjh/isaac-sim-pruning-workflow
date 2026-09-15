"""Daylight presets preserve source mode and use world-space sunlight."""

import math

import numpy as np
import pytest

from isaaclab_pruning.sim.daylight import PRESETS, apply_daylight, daylight_settings


@pytest.mark.parametrize("name", PRESETS)
def test_presets_are_deterministic_upward_unit_vectors(name):
    settings = daylight_settings(name)
    assert settings == daylight_settings(name)
    direction = settings["world_direction_to_sun"]
    assert np.linalg.norm(direction) == pytest.approx(1)
    assert direction[2] == pytest.approx(math.sin(math.radians(settings["elevation_deg"])))
    assert direction[2] > 0
    assert settings["astronomically_calibrated"] is False


def test_source_mode_never_mutates_lights():
    assert apply_daylight(None, None, "source")["direction"] == "unchanged exported Sun"


@pytest.mark.parametrize("name", ["night", "", "Morning"])
def test_unknown_preset_fails_before_touching_stage(name):
    with pytest.raises(ValueError, match="Unknown daylight preset"):
        apply_daylight(None, None, name)


@pytest.mark.parametrize("name", PRESETS)
def test_usd_direction_ignores_rotated_parent(name):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Orchard")
    parent.AddRotateZOp().Set(150.0)
    sun = UsdLux.DistantLight.Define(stage, "/Orchard/Sun")
    dome = UsdLux.DomeLight.Define(stage, "/Dome")
    settings = apply_daylight(sun, dome, name)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(sun.GetPrim())
    actual = np.asarray(world.TransformDir(Gf.Vec3d(0, 0, -1)))
    np.testing.assert_allclose(actual, -np.asarray(settings["world_direction_to_sun"]), atol=1e-12)
    assert sun.GetIntensityAttr().Get() == pytest.approx(settings["sun_intensity"])
    assert dome.GetIntensityAttr().Get() == pytest.approx(450.0)
