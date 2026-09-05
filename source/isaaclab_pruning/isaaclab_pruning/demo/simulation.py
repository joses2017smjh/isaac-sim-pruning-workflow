"""Deterministic CPU pruning episodes using the repository's real primitives.

This is an analytic integration demo, not robot dynamics or a live Isaac run.
Tool poses follow commands exactly. A scripted insertion follows the ToF
controller's 80 mm standoff; geometric checks use the versioned legacy cutter
proxies and known scene geometry. No branch is physically severed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from isaaclab_pruning.baselines.tof_servo import (
    ToFServoGains,
    deproject_tof,
    fit_branch_axis,
    scripted_absolute_pose,
    scripted_tof_action,
)
from isaaclab_pruning.geometry import (
    Cylinder,
    cutter_boxes_from_spec,
    nearby_wood_in_failure_zone,
    oracle_cut_point,
)
from isaaclab_pruning.robot import load_ur5e_pruner_spec
from isaaclab_pruning.sensors import ToFNoiseConfig, apply_tof_noise, fuse_depths, ray_finite_cylinder_t
from isaaclab_pruning.sensors.tof_raycaster import (
    TOF_SITE_OFFSETS_M,
    VL53L8CX_INTRINSICS,
    VL53L8CX_MAX_RANGE_M,
    VL53L8CX_UPDATE_RATE_HZ,
)
from isaaclab_pruning.task import episode_start_target
from isaaclab_pruning.task.success import evaluate_cut_success

SCENARIOS = ("nominal", "sensor_dropout", "cluttered")
TITLES = {
    "nominal": "01 / Clear approach",
    "sensor_dropout": "02 / Sensor blackout",
    "cluttered": "03 / Nearby wood",
}


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 7
    max_steps: int = 100
    dropout_after_step: int = 16
    lost_lock_limit: int = 4
    insertion_step_m: float = 0.003

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.lost_lock_limit < 1 or self.dropout_after_step < 0:
            raise ValueError("Step counts must be positive; dropout step may also be zero.")
        if not math.isfinite(self.insertion_step_m) or not 0 < self.insertion_step_m <= 0.005:
            raise ValueError("insertion_step_m must be finite and in (0, 0.005].")


def rotation_matrix(pose: np.ndarray) -> np.ndarray:
    """Rotation for a normalized xyz + wxyz tool pose."""
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_scene(scenario: str) -> list[Cylinder]:
    """Original procedural wood, in metres, with an exposed target branch."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
    cylinders = [
        Cylinder("target", "spur_demo", (0.0, 0.0, 0.34), (0.10, 1.0, 0.0), 0.014, 0.29),
        Cylinder("limb", "branch_demo", (-0.25, -0.28, 0.35), (1.0, 0.45, -0.12), 0.025, 0.24),
        Cylinder("trunk", "trunk_demo", (-0.36, -0.10, 0.38), (0.0, 1.0, 0.0), 0.037, 0.51),
        Cylinder("twig", "branch_context", (-0.23, 0.29, 0.385), (1.0, 0.4, 0.0), 0.011, 0.23),
    ]
    if scenario == "cluttered":
        cylinders.append(Cylinder("obstacle", "branch_obstacle", (0.016, 0.065, 0.312), (1.0, 0.0, 0.0), 0.008, 0.065))
    return cylinders


def sensor_rays() -> np.ndarray:
    """The same unit pinhole rays used by the 8x8 hardware contract."""
    intrinsics = VL53L8CX_INTRINSICS
    v, u = np.mgrid[: intrinsics.height, : intrinsics.width]
    rays = np.stack(
        (
            (u + 0.5 - intrinsics.width / 2) / intrinsics.focal_length_px,
            (v + 0.5 - intrinsics.height / 2) / intrinsics.focal_length_px,
            np.ones_like(u),
        ),
        axis=-1,
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def cast_tof(pose: np.ndarray, cylinders: list[Cylinder]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast both offset ToF grids against finite cylinder surfaces and caps.

    Returns ranges, hit radii, and world hit points, shaped (2, 8, 8[, 3]).
    Misses remain infinity until the repository noise/status stage handles them.
    """
    spec = load_ur5e_pruner_spec()
    rotation = rotation_matrix(pose)
    directions = sensor_rays() @ rotation.T
    ranges = np.full((2, 8, 8), np.inf)
    radii = np.full_like(ranges, np.nan)
    points = np.full((2, 8, 8, 3), np.nan)
    for sensor_index, offset in enumerate(TOF_SITE_OFFSETS_M.values()):
        relative = np.asarray(offset) - np.asarray(spec.control_tool_translation_in_physics_body_m)
        origin = pose[:3] + rotation @ relative
        for row in range(8):
            for column in range(8):
                direction = directions[row, column]
                for cylinder in cylinders:
                    distance = ray_finite_cylinder_t(origin, direction, cylinder, t_max=VL53L8CX_MAX_RANGE_M)
                    if distance < ranges[sensor_index, row, column]:
                        ranges[sensor_index, row, column] = distance
                        radii[sensor_index, row, column] = cylinder.radius
                        points[sensor_index, row, column] = origin + distance * direction
    return ranges, radii, points


def _sensor_frame(pose, cylinders, generator, blackout):
    raw, radii, hit_points = cast_tof(pose, cylinders)
    ranges = torch.from_numpy(raw)
    config = ToFNoiseConfig(dropout_probability=1.0 if blackout else 0.05)
    tof = apply_tof_noise(ranges, hit_radii_m=torch.from_numpy(radii), config=config, generator=generator)
    # An explicitly synthetic metric estimate on the SAME rays. There is no
    # cross-camera reprojection or learned depth inference in this CPU demo.
    metric_config = ToFNoiseConfig(
        min_sigma_m=0.012, range_sigma_fraction=0.06, dropout_probability=0.0, thin_dropout_probability=0.0
    )
    metric = apply_tof_noise(ranges, config=metric_config, generator=generator)
    fused, variance = fuse_depths(
        torch.stack((tof.range_m, metric.range_m)),
        torch.stack((tof.variance_m2, metric.variance_m2)),
        valid=torch.stack((tof.valid, metric.valid)),
    )
    return raw, hit_points, tof, metric, fused, variance


def _cut_checks(pose, cylinders, target, spec):
    pose_tensor = torch.from_numpy(pose).unsqueeze(0)
    mouth, failure = cutter_boxes_from_spec(
        eef_pose_w=pose_tensor,
        mouth_half_extents=spec.mouth_half_extents_m,
        mouth_offset_eef=spec.mouth_offset_m,
        failure_half_extents=spec.failure_half_extents_m,
        failure_offset_eef=spec.failure_offset_m,
    )
    others = nearby_wood_in_failure_zone(
        torch.tensor(np.array([c.centroid for c in cylinders])).unsqueeze(0),
        torch.tensor(np.array([c.orientation for c in cylinders])).unsqueeze(0),
        torch.tensor([[c.length for c in cylinders]]),
        failure,
        exclude_mask=torch.tensor([[c.record_id == "target" for c in cylinders]]),
    )
    closing_axis = torch.from_numpy(rotation_matrix(pose)[:, 0]).unsqueeze(0)
    result = evaluate_cut_success(
        branch_centroid_w=target.position_w,
        branch_axis_w=target.axis_w,
        branch_length_m=target.length_m,
        cutter_closing_axis_w=closing_axis,
        mouth_box=mouth,
        failure_box=failure,
        other_wood_in_failure_zone=others,
        perpendicularity_tolerance_deg=spec.perpendicularity_tolerance_deg,
    )
    checks = {key: bool(getattr(result, key)[0]) for key in ("success", "mouth_hit", "failure_clear", "perpendicular")}
    checks["perpendicularity_error_deg"] = float(result.perpendicularity_error_deg[0])
    checks["other_wood_in_failure_zone"] = bool(others[0])
    return checks


def _json_values(value: Any) -> Any:
    """Portable strict JSON: sensor misses use null, never NaN/Infinity."""
    if isinstance(value, torch.Tensor):
        return _json_values(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, (float, np.floating)):
        return round(float(value), 7) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


def run_episode(scenario: str, config: DemoConfig | None = None) -> dict[str, Any]:
    """Execute one sensed approach and return all measured frames and metrics."""
    cfg = config or DemoConfig()
    cylinders = make_scene(scenario)
    spec = load_ur5e_pruner_spec()
    target = episode_start_target(
        oracle_cut_point(cylinders),
        batch=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
        source="procedural_cylinder_oracle",
    )
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    pose = np.array([0.012, -0.012, 0.0, 1.0, 0.0, 0.0, 0.0])
    gains = ToFServoGains(pan=0.35, pitch=0.20, roll=0.35, approach=0.8, max_delta_m=0.012)
    phase = "APPROACH"
    outcome = "timeout"
    lost_lock = 0
    insert_remaining = 0.0
    frames = []
    squared_errors = {key: [] for key in ("tof", "metric", "fused")}
    paired_squared_errors = {key: [] for key in ("tof", "metric", "fused")}
    for step in range(cfg.max_steps):
        blackout = scenario == "sensor_dropout" and step >= cfg.dropout_after_step
        raw, hit_points, tof, metric, fused, fused_variance = _sensor_frame(pose, cylinders, generator, blackout)
        checks = _cut_checks(pose, cylinders, target, spec)
        count = int(tof.valid.sum())
        lost_lock = lost_lock + 1 if count < 3 else 0
        if not checks["failure_clear"]:
            phase, outcome = "STOP / WOOD", "failure_zone_blocked"
        elif phase == "INSERT" and checks["success"]:
            phase, outcome = "CUT / ACCEPTED", "success"
        elif phase == "APPROACH" and lost_lock >= cfg.lost_lock_limit:
            phase, outcome = "STOP / NO LOCK", "sensor_lock_lost"

        paired = torch.from_numpy(np.isfinite(raw)) & tof.valid & metric.valid & torch.isfinite(fused)
        for key, measured in (("tof", tof.range_m), ("metric", metric.range_m), ("fused", fused)):
            delta = measured - torch.from_numpy(raw)
            finite = torch.isfinite(delta)
            squared_errors[key].extend(delta[finite].square().tolist())
            paired_squared_errors[key].extend(delta[paired].square().tolist())
        visible_hits = hit_points[tof.valid.numpy()]
        frames.append(
            _json_values(
                {
                    "step": step,
                    "time_s": step / VL53L8CX_UPDATE_RATE_HZ,
                    "phase": phase,
                    "pose_w": pose.copy(),
                    "target_distance_m": float(np.linalg.norm(target.position_w.numpy()[0] - pose[:3])),
                    "tof_m": tof.range_m,
                    "tof_valid": tof.valid,
                    "tof_status": tof.status,
                    "metric_m": metric.range_m,
                    "fused_m": fused,
                    "fused_variance_m2": fused_variance,
                    "valid_returns": count,
                    "hit_points_w": visible_hits,
                    "checks": checks,
                }
            )
        )
        if outcome != "timeout":
            break
        if phase == "INSERT":
            delta = np.zeros(7)
            delta[3] = 1.0
            delta[2] = min(cfg.insertion_step_m, insert_remaining)
            insert_remaining -= delta[2]
            if delta[2] < 1e-8:
                phase, outcome = "STOP / NO CUT", "insertion_exhausted"
                frames[-1]["phase"] = phase
                break
            pose = scripted_absolute_pose(torch.from_numpy(pose[None]), torch.from_numpy(delta[None])).numpy()[0]
            continue

        delta = scripted_tof_action(tof.range_m[0:1], tof.range_m[1:2], tof.valid[0:1], tof.valid[1:2], gains=gains)
        if count >= 3:
            points = torch.cat(
                [
                    deproject_tof(tof.range_m[i : i + 1], tof.valid[i : i + 1], offset)
                    for i, offset in enumerate(TOF_SITE_OFFSETS_M.values())
                ],
                dim=1,
            )
            points -= torch.tensor(spec.control_tool_translation_in_physics_body_m)
            centroid, _, _ = fit_branch_axis(points)
            if float(centroid[0, 2]) <= 0.089 and abs(float(centroid[0, 0])) < 0.008:
                phase = "INSERT"
                # Bounded final stroke uses ToF surface distance and the oracle
                # target radius; all close approach steps use the OBB stop gate.
                insert_remaining = max(0.0, float(centroid[0, 2]) + float(target.radius_m[0]) - 0.038)
                continue
        pose = scripted_absolute_pose(torch.from_numpy(pose[None]), delta).numpy()[0]

    if outcome == "timeout":
        frames[-1]["phase"] = "STOP / TIMEOUT"
    positions = np.array([frame["pose_w"][:3] for frame in frames])
    metrics = {
        "outcome": outcome,
        "success": outcome == "success",
        "steps": len(frames),
        "simulated_duration_s": frames[-1]["time_s"],
        "tool_path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "initial_target_distance_m": frames[0]["target_distance_m"],
        "final_target_distance_m": frames[-1]["target_distance_m"],
        "mean_valid_return_fraction": float(np.mean([frame["valid_returns"] / 128 for frame in frames])),
        "final_checks": frames[-1]["checks"],
        "range_rmse_mm": {
            key: 1000 * math.sqrt(float(np.mean(values))) if values else None for key, values in squared_errors.items()
        },
        "paired_range_rmse_mm": {
            key: 1000 * math.sqrt(float(np.mean(values))) if values else None
            for key, values in paired_squared_errors.items()
        },
        "paired_range_sample_count": len(paired_squared_errors["tof"]),
        "arm_collision_evaluated": False,
    }
    return _json_values(
        {
            "scenario": scenario,
            "title": TITLES[scenario],
            "seed": cfg.seed,
            "scene": [asdict(cylinder) for cylinder in cylinders],
            "metrics": metrics,
            "frames": frames,
        }
    )


def run_demo(config: DemoConfig | None = None, scenarios: tuple[str, ...] = SCENARIOS) -> dict[str, Any]:
    """Bundle replay data and explicit scope into a deterministic strict-JSON report."""
    cfg = config or DemoConfig()
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Select at least one scenario, with no duplicates.")
    spec = load_ur5e_pruner_spec()
    return {
        "schema_version": 1,
        "demo_kind": "analytic_cpu_pruning_integration",
        "title": "Pruning / perception to motion",
        "config": asdict(cfg),
        "scope": {
            "geometry": "Original procedural cylinders; exact finite-cylinder ray intersections.",
            "controller": "Repository scripted ToF servo, then bounded scripted insertion with geometric stop checks.",
            "motion": "Ideal tool pose integration; no arm, IK, contacts, or robot dynamics.",
            "sensor": "Two simulated 8x8 ToF arrays; calibrated offsets; configured noise and dropout.",
            "fusion": "Inverse-variance fusion with a synthetic noisy metric estimate on identical rays.",
            "oracle": "Episode-start target metadata and known scene geometry for cutter checks.",
            "success": "Target centerline in legacy mouth proxy, clear failure zone, and perpendicular closing axis.",
            "limits": "No trained policy, live Isaac capture, learned depth, hardware, or physical branch severing.",
        },
        "cutter": {
            "source": spec.cutter_source,
            "mouth_half_extents_m": spec.mouth_half_extents_m,
            "mouth_offset_m": spec.mouth_offset_m,
            "failure_half_extents_m": spec.failure_half_extents_m,
            "failure_offset_m": spec.failure_offset_m,
        },
        "sensor": {
            "shape": [2, 8, 8],
            "rate_hz": VL53L8CX_UPDATE_RATE_HZ,
            "diagonal_fov_deg": VL53L8CX_INTRINSICS.diagonal_fov_deg,
            "offsets_in_tool_m": [
                list(np.asarray(offset) - np.asarray(spec.control_tool_translation_in_physics_body_m))
                for offset in TOF_SITE_OFFSETS_M.values()
            ],
        },
        "episodes": [run_episode(scenario, cfg) for scenario in scenarios],
    }
