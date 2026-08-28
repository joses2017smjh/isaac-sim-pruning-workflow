"""Observation variants A (flow), B (ToF), C (metric student), D (fused)."""

from __future__ import annotations

from enum import Enum

import torch

from isaaclab_pruning.sensors.fusion import fuse_depths


class ObservationVariant(str, Enum):
    FLOW = "A_flow"
    TOF = "B_tof"
    METRIC = "C_metric"
    FUSED = "D_fused"


# VL53L8CX on the mock pruner. B/C/D share this grid so a ToF-vs-metric
# comparison is content, not width. A is 2*H*W on the same grid and stays a
# different last-dim; say so rather than padding it. Native-resolution C is a
# second table (NATIVE_METRIC_HW), not the default.
TOF_HW = (8, 8)
WIDTH_MATCHED_HW = TOF_HW
NATIVE_METRIC_HW = (256, 256)
PROPRIO_EEF_DIM = 7
ARM_JOINT_COUNT = 6


def observation_width(
    variant: ObservationVariant,
    *,
    n_joints: int = ARM_JOINT_COUNT,
    flow_hw: tuple[int, int] = WIDTH_MATCHED_HW,
    tof_hw: tuple[int, int] = WIDTH_MATCHED_HW,
    metric_hw: tuple[int, int] = WIDTH_MATCHED_HW,
) -> int:
    """Last-dim size of ``build_observation``.

    A, B, and C must not share a width (the BHL 194/194/194 trap). C and D
    match on WIDTH_MATCHED_HW; contents must still differ when ToF ≠ metric.
    """
    if n_joints < 1:
        raise ValueError("n_joints must be positive.")
    base = 3 + n_joints + n_joints + PROPRIO_EEF_DIM
    if variant is ObservationVariant.FLOW:
        return base + 2 * flow_hw[0] * flow_hw[1]
    if variant is ObservationVariant.TOF:
        pixels = tof_hw[0] * tof_hw[1]
        return base + 4 * pixels
    if variant is ObservationVariant.METRIC:
        return base + metric_hw[0] * metric_hw[1]
    if variant is ObservationVariant.FUSED:
        return base + metric_hw[0] * metric_hw[1]
    raise ValueError(f"Unknown observation variant: {variant}.")


def proprioception(joint_pos: torch.Tensor, joint_vel: torch.Tensor, eef_pose_w: torch.Tensor) -> torch.Tensor:
    if joint_pos.shape[:-1] != joint_vel.shape[:-1] or eef_pose_w.shape[-1] != 7:
        raise ValueError("Proprioception tensors have incompatible batch shapes.")
    return torch.cat((joint_pos, joint_vel, eef_pose_w), dim=-1)


def flatten_flow(flow_hw2: torch.Tensor) -> torch.Tensor:
    """Isaac ``motion_vectors`` ``(..., H, W, 2)`` -> ``(..., 2*H*W)``."""
    if flow_hw2.shape[-1] != 2:
        raise ValueError("flow must have last dimension 2.")
    return flow_hw2.reshape(*flow_hw2.shape[:-3], -1)


def flatten_tof(range0: torch.Tensor, range1: torch.Tensor, valid0: torch.Tensor, valid1: torch.Tensor) -> torch.Tensor:
    filled0 = torch.nan_to_num(range0, nan=0.0)
    filled1 = torch.nan_to_num(range1, nan=0.0)
    return torch.cat(
        (
            filled0.reshape(*filled0.shape[:-2], -1),
            filled1.reshape(*filled1.shape[:-2], -1),
            valid0.to(dtype=filled0.dtype).reshape(*valid0.shape[:-2], -1),
            valid1.to(dtype=filled1.dtype).reshape(*valid1.shape[:-2], -1),
        ),
        dim=-1,
    )


def flatten_metric(depth: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(depth, nan=0.0).reshape(*depth.shape[:-2], -1)


def resample_hw(image: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear-resample ``(..., H, W)`` onto ``hw``. Identity when already matched."""
    if tuple(image.shape[-2:]) == hw:
        return image
    leading = image.shape[:-2]
    flat = image.reshape(-1, 1, image.shape[-2], image.shape[-1]).to(dtype=torch.float32)
    out = torch.nn.functional.interpolate(flat, size=hw, mode="bilinear", align_corners=False)
    return out.reshape(*leading, hw[0], hw[1]).to(dtype=image.dtype)


def fuse_tof_and_metric(
    tof0: torch.Tensor,
    tof1: torch.Tensor,
    metric_depth: torch.Tensor,
    tof0_var: torch.Tensor,
    tof1_var: torch.Tensor,
    metric_var: torch.Tensor,
    tof0_valid: torch.Tensor | None,
    tof1_valid: torch.Tensor | None,
) -> torch.Tensor:
    """Inverse-variance fuse of both VL53L8CX channels plus the metric student.

    Metric is resampled onto the ToF grid. Using only tof0 would confound
    fusion with dropping half the hardware.
    """
    hw = (int(tof0.shape[-2]), int(tof0.shape[-1]))
    if tuple(tof1.shape[-2:]) != hw:
        raise ValueError(f"tof1 grid {tuple(tof1.shape[-2:])} != tof0 grid {hw}.")
    metric_r = resample_hw(metric_depth, hw)
    metric_var_r = resample_hw(metric_var, hw)
    if tuple(metric_r.shape[-2:]) != hw:
        raise AssertionError(f"metric resampled to {tuple(metric_r.shape[-2:])}, expected {hw}.")
    depths = torch.stack((tof0, tof1, metric_r), dim=0)
    variances = torch.stack((tof0_var, tof1_var, metric_var_r), dim=0)
    if tof0_valid is None or tof1_valid is None:
        valid = None
    else:
        valid = torch.stack((tof0_valid, tof1_valid, torch.isfinite(metric_r)), dim=0)
    fused, _ = fuse_depths(depths, variances, valid=valid)
    return fused


def build_observation(
    variant: ObservationVariant,
    *,
    goal_w: torch.Tensor,
    proprio: torch.Tensor,
    flow_hw2: torch.Tensor | None = None,
    tof0: torch.Tensor | None = None,
    tof1: torch.Tensor | None = None,
    tof0_valid: torch.Tensor | None = None,
    tof1_valid: torch.Tensor | None = None,
    tof0_var: torch.Tensor | None = None,
    tof1_var: torch.Tensor | None = None,
    metric_depth: torch.Tensor | None = None,
    metric_var: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble a policy observation. DA2-ft is never an input here."""
    parts = [goal_w, proprio]
    if variant is ObservationVariant.FLOW:
        if flow_hw2 is None:
            raise ValueError("Variant A requires flow_hw2.")
        parts.append(flatten_flow(flow_hw2))
    elif variant is ObservationVariant.TOF:
        if tof0 is None or tof1 is None or tof0_valid is None or tof1_valid is None:
            raise ValueError("Variant B requires both ToF images and validity masks.")
        parts.append(flatten_tof(tof0, tof1, tof0_valid, tof1_valid))
    elif variant is ObservationVariant.METRIC:
        if metric_depth is None:
            raise ValueError("Variant C requires a distilled metric-depth student tensor.")
        parts.append(flatten_metric(metric_depth))
    elif variant is ObservationVariant.FUSED:
        if (
            tof0 is None
            or tof1 is None
            or metric_depth is None
            or tof0_var is None
            or tof1_var is None
            or metric_var is None
        ):
            raise ValueError("Variant D requires both ToF channels, metric depth, and all three variances.")
        fused = fuse_tof_and_metric(
            tof0,
            tof1,
            metric_depth,
            tof0_var,
            tof1_var,
            metric_var,
            tof0_valid,
            tof1_valid,
        )
        parts.append(flatten_metric(fused))
    else:
        raise ValueError(f"Unknown observation variant: {variant}.")
    observation = torch.cat(parts, dim=-1)
    n_joints = (proprio.shape[-1] - PROPRIO_EEF_DIM) // 2
    width_kwargs: dict = {"n_joints": n_joints}
    if variant is ObservationVariant.FLOW:
        width_kwargs["flow_hw"] = tuple(flow_hw2.shape[-3:-1])
    elif variant is ObservationVariant.TOF:
        width_kwargs["tof_hw"] = tuple(tof0.shape[-2:])
    elif variant is ObservationVariant.FUSED:
        width_kwargs["metric_hw"] = tuple(tof0.shape[-2:])
    else:
        width_kwargs["metric_hw"] = tuple(metric_depth.shape[-2:])
    expected = observation_width(variant, **width_kwargs)
    if observation.shape[-1] != expected:
        raise ValueError(f"Observation width {observation.shape[-1]} != contract {expected} for {variant}.")
    return observation
