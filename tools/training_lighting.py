"""Fixed definitions for the rendered-lighting training set; no Blender here.

The surviving training frames (6,270 on disk, 4,980 in the train split) were
rendered under one light, the companion's ``source`` preset. This module
defines a second rendering of the train-split frames at their exact recorded
poses with one seeded lighting draw per frame, so a fine-tune can see low sun,
warm colour and dark sky without any evaluation preset leaking in.

The sampler (``rendered-lighting-v1``) draws per frame, seeded from the frame's
identity: a low-sun regime with probability 0.6 and a high-sun regime
otherwise, sun azimuth uniform over the world, sun energy log-uniform, sun
colour from a colour temperature through a declared Planckian formula, and a
world colour mixed between the daylight and dusk worlds. A draw whose sun
direction falls within 15 degrees of any registered preset sun (source,
morning, noon, evening and its x2.6 variant), in the world frame or relative to
the pose's yaw, is rejected and redrawn; the sunless overcast presets are
outside the support because every draw has a sun of energy at least 1. The
hold-out tests interpolation from at least 15 degrees away, not extrapolation.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

SAMPLER_VERSION = "rendered-lighting-v1"
#: generate_tree2.ROTATION_REF[2]: the reference camera yaw every pose's yaw is measured from.
ROTATION_REF_RZ = 3.194649314880371
DAY_WORLD = (0.70, 0.75, 0.82)
DUSK_WORLD = (0.10, 0.13, 0.20)
REGIMES = {
    "low": {
        "p": 0.6,
        "elevation_deg": (4.0, 25.0),
        "cct_k": (2500.0, 4500.0),
        "world_mix": (0.5, 1.0),
        "world_strength": (0.07, 0.35),
    },
    "high": {
        "p": 0.4,
        "elevation_deg": (25.0, 75.0),
        "cct_k": (4500.0, 6500.0),
        "world_mix": (0.0, 0.5),
        "world_strength": (0.25, 0.80),
    },
}
AZIMUTH_DEG = (0.0, 360.0)
SUN_ENERGY = (1.0, 6.0)
SUN_ANGLE_DEG = 8.0
HOLDOUT_RADIUS_DEG = 15.0
#: Direction TO the sun of every registered evaluation preset (daylight_presets convention).
PRESET_SUNS = {"source": (0.0, 90.0), "morning": (60.0, 18.0), "noon": (160.0, 65.0), "evening": (315.0, 12.0)}
MAX_ATTEMPTS = 10000
M_XYZ_TO_709 = np.array(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]
)
#: Agreement required between a re-rendered frame's depth and mask and the surviving files it reuses.
AGREEMENT_TOLERANCE_M = 0.001
AGREEMENT_MIN_FRACTION = 0.995
AGREEMENT_MIN_IOU = 0.995


def planck_xy(temperature_k):
    """Kang et al. (2002) cubic fit of the Planckian locus, valid 1667-25000 K."""
    t = float(temperature_k)
    if not 1667.0 <= t <= 25000.0:
        raise ValueError("colour temperature outside the fitted range")
    if t < 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t < 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t < 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def cct_to_linear_rgb(temperature_k):
    """Planckian chromaticity to linear Rec.709 (Blender scene linear), brightest channel 1."""
    x, y = planck_xy(temperature_k)
    rgb = np.clip(M_XYZ_TO_709 @ np.array([x / y, 1.0, (1 - x - y) / y]), 0, None)
    return (rgb / rgb.max()).tolist()


def unit(azimuth_deg, elevation_deg):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def yaw_offset_deg(rotation_euler):
    """Pose yaw relative to the reference camera, wrapped to (-180, 180]."""
    delta = (float(rotation_euler[2]) - ROTATION_REF_RZ + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(delta)


def holdout_distances(azimuth_deg, elevation_deg, yaw_deg):
    """Smallest angular distance to each preset sun, world-frame or rotated with the pose."""
    sun = unit(azimuth_deg, elevation_deg)
    return {
        name: min(angle_deg(sun, unit(paz, pel)), angle_deg(sun, unit(paz + yaw_deg, pel)))
        for name, (paz, pel) in PRESET_SUNS.items()
    }


def frame_seed(tree_id, set_id, shot, view):
    key = f"{SAMPLER_VERSION}|{tree_id}|{set_id}|{shot}|{view}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _loguniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample(tree_id, set_id, shot, view, rotation_euler):
    """One lighting draw for one frame, identical every time it is asked for."""
    seed = frame_seed(tree_id, set_id, shot, view)
    rng = np.random.default_rng(seed)
    yaw = yaw_offset_deg(rotation_euler)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        regime = "low" if rng.random() < REGIMES["low"]["p"] else "high"
        spec = REGIMES[regime]
        elevation = float(rng.uniform(*spec["elevation_deg"]))
        azimuth = float(rng.uniform(*AZIMUTH_DEG))
        cct = float(rng.uniform(*spec["cct_k"]))
        energy = _loguniform(rng, *SUN_ENERGY)
        mix = float(rng.uniform(*spec["world_mix"]))
        strength = _loguniform(rng, *spec["world_strength"])
        distances = holdout_distances(azimuth, elevation, yaw)
        if min(distances.values()) > HOLDOUT_RADIUS_DEG:
            world = [float((1 - mix) * a + mix * b) for a, b in zip(DAY_WORLD, DUSK_WORLD)]
            return {
                "sampler": SAMPLER_VERSION,
                "seed": seed,
                "attempts": attempt,
                "regime": regime,
                "pose_yaw_offset_deg": yaw,
                "sun_cct_k": cct,
                "world_mix_to_dusk": mix,
                "holdout_min_distance_deg": distances,
                "config": {
                    "world_color": world,
                    "world_strength": strength,
                    "sun": {
                        "azimuth_deg": azimuth,
                        "elevation_deg": elevation,
                        "energy": energy,
                        "color": cct_to_linear_rgb(cct),
                        "angle_deg": SUN_ANGLE_DEG,
                    },
                },
            }
    raise RuntimeError("hold-out rejection did not terminate")


def lighting_label(draw):
    return f"rl1_{draw['seed']:016x}"


def annotation_path(companion, row):
    """The generator's annotation JSON for a manifest row: its exact recorded camera pose."""
    suffix = "" if row["view"] == "c" else "_" + row["view"]
    return (
        Path(companion)
        / "Data/full_spur/ann"
        / row["bark"]
        / row["tree"]
        / row["set_id"]
        / f"{row['tree']}_{row['shot']}{suffix}.json"
    )


def surviving_rows(manifest_csv):
    """Train-split rows whose RGB, depth and mask all exist, in manifest order."""
    with Path(manifest_csv).open(newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if all(Path(row[key]).is_file() for key in ("rgb_path", "depth_path", "mask_path"))
        ]


def frames_by_tree(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["tree"], []).append(
            {key: row[key] for key in ("bark", "tree", "set_id", "shot", "view", "rgb_path", "depth_path", "mask_path")}
        )
    return grouped


def agreement(new_depth, new_mask, old_depth, old_mask, tolerance_m=AGREEMENT_TOLERANCE_M):
    """How well a re-render's geometry reproduces the surviving depth and mask it will reuse."""
    new_depth, old_depth = np.asarray(new_depth, dtype=float), np.asarray(old_depth, dtype=float)
    new_mask, old_mask = np.asarray(new_mask, dtype=bool), np.asarray(old_mask, dtype=bool)
    if new_depth.shape != old_depth.shape or new_mask.shape != old_mask.shape or new_depth.shape != new_mask.shape:
        raise ValueError("re-render and surviving frame differ in shape")
    union = new_mask | old_mask
    iou = float((new_mask & old_mask).sum() / union.sum()) if union.any() else 0.0
    on_tree = old_mask & np.isfinite(old_depth) & (old_depth < 1e6)
    close = on_tree & np.isfinite(new_depth) & (np.abs(new_depth - old_depth) <= tolerance_m)
    fraction = float(close.sum() / on_tree.sum()) if on_tree.any() else 0.0
    return {
        "mask_iou": iou,
        "tree_depth_within_tolerance": fraction,
        "tolerance_m": tolerance_m,
        "ok": iou >= AGREEMENT_MIN_IOU and fraction >= AGREEMENT_MIN_FRACTION,
    }
