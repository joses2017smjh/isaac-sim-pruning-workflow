"""Fixed definitions for the generalization-controls experiment; no Blender here.

Everything the renderer, the launcher, the evaluation planner and the tests need
to agree on lives in this module: the camera models, the pose sets, the
repo-local lighting controls and the list of conditions rendered per tree. The
renderer imports it inside Blender; the tests import it on the submit node.

Two axes, each a controlled change from the registered matrix cell
(``source`` light, training-rig camera, far rig):

* lighting, at the far rig with the training camera: the companion's
  ``overcast`` preset and two repo-local controls that scale a companion preset's
  sun energy and world strength by a factor fixed before rendering;
* camera, under the ``source`` light: the Isaac wrist camera model, close-range
  rigs at the Isaac working distances, the Isaac upward pitch, and a distance
  sweep on the training camera.

The repo-local controls are injected into the companion's preset table for the
duration of one ``apply_preset`` call. No file in the companion tree is written.
"""

from __future__ import annotations

import math

SCHEMA_VERSION = 1

#: Ratio of mean luma between the matrix's source and evening frames (142.3 / 54.9
#: on the registered 48-frame cells). Fixed here before any control is rendered.
LUMA_FACTOR = 2.6

#: Isaac upward pitch of the wrist optical axis relative to the level training
#: pose, from the camera audit of the Stage A run_00 frames.
ISAAC_PITCH_DEG = 39.7

CAMERA_MODELS = {
    "training_rig": {
        "lens_mm": 28.0,
        "sensor_width_mm": 36.0,
        "sensor_fit": "HORIZONTAL",
        "width": 512,
        "height": 288,
        "note": "generate_tree2 camera (28 mm on the Blender default 36 mm sensor) at the matrix resolution",
    },
    "isaac_wrist": {
        "lens_mm": 20.0,
        "sensor_width_mm": 30.0,
        "sensor_fit": "HORIZONTAL",
        "width": 480,
        "height": 320,
        "note": "Isaac wrist camera from the Stage A evidence: focal 20 mm, aperture 30 mm, fx = fy = 320 px",
    },
}

POSE_SETS = {
    "far": {
        "kind": "heights",
        "heights_m": [0.85, 1.0, 1.15],
        "baseline_m": 0.12,
        "note": "The matrix rig: fixed X/Y reference, three heights, 0.12 m stereo baseline, level view",
    },
    "close": {
        "kind": "distances",
        "distances_m": [0.16, 0.25, 0.39],
        "baseline_m": 0.02,
        "pitch_deg": 0.0,
        "note": "Camera on the level viewing direction at the Isaac working distances from the spur centre",
    },
    "close_up40": {
        "kind": "distances",
        "distances_m": [0.16, 0.25, 0.39],
        "baseline_m": 0.02,
        "pitch_deg": ISAAC_PITCH_DEG,
        "note": "As close, with the optical axis pitched up by the Isaac wrist angle (camera below the spur)",
    },
    "sweep": {
        "kind": "sweep",
        "distances_m": [0.16, 0.25, 0.39, 0.6, 0.9, 1.3, 1.8, 2.3],
        "note": "Single centred pose per distance on the level viewing direction, from Isaac range to matrix range",
    },
}

#: Repo-local lighting controls: (companion preset, multiplicative factor on sun
#: energy and world strength). Filmic tone mapping is not linear, so the achieved
#: luma is recorded by the evaluation planner rather than assumed.
LIGHTING_CONTROLS = {
    "evening_x2.6": {"base": "evening", "factor": LUMA_FACTOR, "corner": "bright, low sun"},
    "overcast_div2.6": {"base": "overcast", "factor": 1.0 / LUMA_FACTOR, "corner": "dark, diffuse"},
}

CONDITIONS = [
    {"id": "source", "group": "lighting", "lighting": "source", "camera_model": "training_rig", "pose_set": "far"},
    {"id": "evening", "group": "lighting", "lighting": "evening", "camera_model": "training_rig", "pose_set": "far"},
    {
        "id": "evening_x2.6",
        "group": "lighting",
        "lighting": "evening_x2.6",
        "camera_model": "training_rig",
        "pose_set": "far",
    },
    {"id": "overcast", "group": "lighting", "lighting": "overcast", "camera_model": "training_rig", "pose_set": "far"},
    {
        "id": "overcast_div2.6",
        "group": "lighting",
        "lighting": "overcast_div2.6",
        "camera_model": "training_rig",
        "pose_set": "far",
    },
    {
        "id": "source/far/isaac_wrist",
        "group": "camera",
        "lighting": "source",
        "camera_model": "isaac_wrist",
        "pose_set": "far",
    },
    {
        "id": "source/close/training_rig",
        "group": "camera",
        "lighting": "source",
        "camera_model": "training_rig",
        "pose_set": "close",
    },
    {
        "id": "source/close/isaac_wrist",
        "group": "camera",
        "lighting": "source",
        "camera_model": "isaac_wrist",
        "pose_set": "close",
    },
    {
        "id": "source/close_up40/isaac_wrist",
        "group": "camera",
        "lighting": "source",
        "camera_model": "isaac_wrist",
        "pose_set": "close_up40",
    },
    {
        "id": "source/sweep/training_rig",
        "group": "distance_sweep",
        "lighting": "source",
        "camera_model": "training_rig",
        "pose_set": "sweep",
    },
]

#: Test-time photometric variants scored on the plain ``evening`` renders.
PHOTOMETRIC_VARIANTS = ("exposure", "gamma", "hist_eq", "clahe")

#: The cell every control is compared with, and the paired baseline of each.
BASELINE_OF = {
    "evening": "source",
    "evening_x2.6": "evening",
    "overcast": "source",
    "overcast_div2.6": "overcast",
    "source/far/isaac_wrist": "source",
    "source/close/training_rig": "source",
    "source/close/isaac_wrist": "source/close/training_rig",
    "source/close_up40/isaac_wrist": "source/close/isaac_wrist",
    "source/sweep/training_rig": "source",
    **{f"evening/{variant}": "evening" for variant in PHOTOMETRIC_VARIANTS},
}


def condition(condition_id):
    for entry in CONDITIONS:
        if entry["id"] == condition_id:
            return entry
    raise KeyError(f"Unknown condition: {condition_id}")


def dino_rig_compatible(camera_model, pose_set) -> bool:
    """The DINO checkpoint expects six views of the matrix rig at the training camera."""
    return camera_model == "training_rig" and pose_set == "far"


def geometry_key(camera_model, pose_set) -> str:
    """Depth and mask are shared by every condition with the same camera and poses."""
    return f"{pose_set}__{camera_model}"


def frames_per_tree(conditions=CONDITIONS, pose_sets=POSE_SETS) -> int:
    return sum(len(pose_ids(pose_sets[c["pose_set"]])) for c in conditions)


def pose_ids(spec):
    if spec["kind"] == "heights":
        return [f"rig{i}_{side}" for i in range(len(spec["heights_m"])) for side in ("l", "r")]
    if spec["kind"] == "distances":
        return [f"d{i}_{side}" for i in range(len(spec["distances_m"])) for side in ("l", "r")]
    if spec["kind"] == "sweep":
        return [f"sweep{i}" for i in range(len(spec["distances_m"]))]
    raise ValueError(f"Unknown pose set kind: {spec['kind']}")


def _add(a, b, scale=1.0):
    return [float(x + scale * y) for x, y in zip(a, b)]


def pitched_forward(forward, up, pitch_deg):
    """Rotate the viewing direction towards ``up`` by ``pitch_deg`` (positive looks up)."""
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    return [float(c * f + s * u) for f, u in zip(forward, up)]


def poses(spec, *, reference_xy, rotation_ref, forward, right, up, target_centroid=None, pitched_rotation=None):
    """Camera poses of one pose set as ``{id, location, rotation_euler}`` dictionaries.

    ``forward``, ``right`` and ``up`` are the camera axes at ``rotation_ref``.
    Distance-based sets need the target centroid; ``pitched_rotation`` is the
    Euler rotation whose forward axis is ``pitched_forward(forward, up, pitch)``,
    computed by the caller in Blender and checked here against the pitch.
    """
    ids = pose_ids(spec)
    if spec["kind"] == "heights":
        out = []
        for i, z in enumerate(spec["heights_m"]):
            base = [reference_xy[0], reference_xy[1], z]
            for side, sign in (("l", -1.0), ("r", 1.0)):
                out.append(
                    {
                        "id": f"rig{i}_{side}",
                        "location": _add(base, right, sign * spec["baseline_m"]),
                        "rotation_euler": list(rotation_ref),
                    }
                )
        return out
    if target_centroid is None:
        raise ValueError("Distance-based pose sets need the target centroid")
    pitch = spec.get("pitch_deg", 0.0)
    if pitch and pitched_rotation is None:
        raise ValueError("A pitched pose set needs the pitched camera rotation")
    view = pitched_forward(forward, up, pitch) if pitch else list(forward)
    rotation = list(pitched_rotation) if pitch else list(rotation_ref)
    out = []
    if spec["kind"] == "sweep":
        for pose_id, d in zip(ids, spec["distances_m"]):
            out.append({"id": pose_id, "location": _add(target_centroid, view, -d), "rotation_euler": rotation})
        return out
    for i, d in enumerate(spec["distances_m"]):
        centre = _add(target_centroid, view, -d)
        for side, sign in (("l", -1.0), ("r", 1.0)):
            out.append(
                {
                    "id": f"d{i}_{side}",
                    "location": _add(centre, right, sign * spec["baseline_m"]),
                    "rotation_euler": rotation,
                }
            )
    return out


def lighting_config(name):
    """The lighting configuration for ``name`` and where it comes from."""
    from daylight_presets import PRESETS, preset

    if name in PRESETS:
        return preset(name), {"origin": "companion daylight_presets.PRESETS", "name": name}
    if name not in LIGHTING_CONTROLS:
        raise ValueError(f"Unknown lighting: {name}")
    control = LIGHTING_CONTROLS[name]
    config = preset(control["base"])
    config["world_strength"] = float(config["world_strength"] * control["factor"])
    if config["sun"]:
        config["sun"]["energy"] = float(config["sun"]["energy"] * control["factor"])
    origin = {
        "origin": "repo-local control (tools/generalization_controls.py)",
        "name": name,
        "base_preset": control["base"],
        "factor": control["factor"],
        "factor_note": "Source-to-evening mean-luma ratio on the registered matrix, fixed before rendering",
        "corner": control["corner"],
    }
    return config, origin


def apply_lighting(scene, name, seed):
    """Apply a companion preset or a repo-local control through the companion's own code path."""
    import daylight_presets

    config, origin = lighting_config(name)
    if name in daylight_presets.PRESETS:
        return dict(daylight_presets.apply_preset(scene, name, seed), control=origin)
    daylight_presets.PRESETS[name] = config
    try:
        applied = daylight_presets.apply_preset(scene, name, seed)
    finally:
        daylight_presets.PRESETS.pop(name, None)
    return dict(applied, control=origin)


def intrinsics(camera_model):
    """Pinhole intrinsics implied by the camera model, zero-based pixel centres."""
    spec = CAMERA_MODELS[camera_model]
    f = spec["lens_mm"] / spec["sensor_width_mm"] * spec["width"]
    return [[f, 0.0, spec["width"] / 2 - 0.5], [0.0, f, spec["height"] / 2 - 0.5], [0.0, 0.0, 1.0]]


def horizontal_fov_deg(camera_model):
    spec = CAMERA_MODELS[camera_model]
    return math.degrees(2 * math.atan(spec["sensor_width_mm"] / (2 * spec["lens_mm"])))
