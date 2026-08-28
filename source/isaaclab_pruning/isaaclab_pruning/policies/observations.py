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
        if tof0 is None or metric_depth is None or tof0_var is None or metric_var is None:
            raise ValueError("Variant D requires ToF0, metric depth, and both variances.")
        fused, _ = fuse_depths(
            torch.stack((tof0, metric_depth), dim=0),
            torch.stack((tof0_var, metric_var), dim=0),
            valid=None if tof0_valid is None else torch.stack((tof0_valid, torch.isfinite(metric_depth)), dim=0),
        )
        parts.append(flatten_metric(fused))
    else:
        raise ValueError(f"Unknown observation variant: {variant}.")
    return torch.cat(parts, dim=-1)
