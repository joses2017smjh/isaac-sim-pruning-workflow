"""Deterministic artistic daylight presets; not astronomical or radiometric calibration."""

from __future__ import annotations

import math

PRESETS = {
    "morning": (60.0, 18.0, 1600.0, (1.0, 0.78, 0.55)),
    "noon": (160.0, 65.0, 2400.0, (1.0, 0.98, 0.93)),
    "evening": (270.0, 12.0, 1300.0, (1.0, 0.56, 0.32)),
}


def daylight_settings(name: str) -> dict:
    """Azimuth is counterclockwise from world +X; elevation is above XY."""
    if name == "source":
        return {"preset": name, "direction": "unchanged exported Sun", "astronomically_calibrated": False}
    if name not in PRESETS:
        raise ValueError(f"Unknown daylight preset {name!r}; choose source, morning, noon, evening")
    azimuth, elevation, intensity, color = PRESETS[name]
    az, el = math.radians(azimuth), math.radians(elevation)
    return {
        "preset": name,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "world_direction_to_sun": [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        "sun_intensity": intensity,
        "sun_color_rgb": list(color),
        "dome_intensity": 450.0,
        "dome_color_rgb": [0.82, 0.90, 1.0],
        "astronomically_calibrated": False,
        "scope": "Artistic fixed sunlight; no time/location, weather, sensor or exposure calibration",
    }


def apply_daylight(sun, dome, name: str) -> dict:
    """Override the composed light, never the source USD; preserve the default run."""
    settings = daylight_settings(name)
    if name == "source":
        return settings
    from pxr import Gf, UsdGeom

    direction = Gf.Vec3d(*(-value for value in settings["world_direction_to_sun"]))
    transform = UsdGeom.Xformable(sun.GetPrim())
    # DistantLight shines down local -Z. Reset inherited orchard rotation so
    # requested azimuth/elevation refer to world coordinates, not the tree frame.
    transform.MakeMatrixXform().Set(Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), direction)))
    transform.SetResetXformStack(True)
    sun.GetIntensityAttr().Set(settings["sun_intensity"])
    sun.CreateColorAttr(Gf.Vec3f(*settings["sun_color_rgb"]))
    dome.GetIntensityAttr().Set(settings["dome_intensity"])
    dome.GetColorAttr().Set(Gf.Vec3f(*settings["dome_color_rgb"]))
    settings["authored_on_composed_stage_only"] = True
    return settings
