#!/usr/bin/env python3
"""Render component demos from measured contracts. No Isaac Sim required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "isaaclab_pruning"))
OUT = ROOT / "docs" / "demo"

from isaaclab_pruning.geometry import Cylinder, cylinder_endpoints  # noqa: E402
from isaaclab_pruning.policies.observations import (  # noqa: E402
    ObservationVariant,
    fuse_tof_and_metric,
    observation_width,
)
from isaaclab_pruning.robot import load_ur5e_pruner_spec  # noqa: E402


def _sample_tree() -> list[Cylinder]:
    return [
        Cylinder("t", "trunk_1", np.array([0.0, 0.0, 0.6]), np.array([0.0, 0.0, 1.0]), 0.04, 1.2),
        Cylinder("b", "branch_1", np.array([0.15, 0.0, 1.0]), np.array([1.0, 0.0, 0.2]), 0.012, 0.4),
        Cylinder("s", "spur_1", np.array([0.32, 0.02, 1.05]), np.array([0.2, 1.0, 0.0]), 0.005, 0.12),
    ]


def _cylinder_lines(cylinders: list[Cylinder]) -> Line3DCollection:
    segs, colors, widths = [], [], []
    palette = {"trunk": "#8d6e63", "branch": "#43a047", "spur": "#fb8c00"}
    for cyl in cylinders:
        a, b = cylinder_endpoints(cyl)
        segs.append([a, b])
        colors.append(palette.get(cyl.organ_class, "#90a4ae"))
        widths.append(max(1.5, cyl.radius * 80))
    return Line3DCollection(segs, colors=colors, linewidths=widths)


def write_tree() -> Path:
    fig = plt.figure(figsize=(6, 6), facecolor="#0e1116")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0e1116")
    ax.add_collection3d(_cylinder_lines(_sample_tree()))
    ax.set_xlim(-0.2, 0.5)
    ax.set_ylim(-0.2, 0.4)
    ax.set_zlim(0.0, 1.3)
    ax.set_title("UsdGeom.Cylinder tree  ·  not capsules", color="white", fontsize=11)
    ax.tick_params(colors="#90a4ae")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    path = OUT / "tree_cylinders.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_widths() -> Path:
    labels = ["A flow", "B ToF×2", "C metric", "D fused"]
    values = [
        observation_width(ObservationVariant.FLOW),
        observation_width(ObservationVariant.TOF),
        observation_width(ObservationVariant.METRIC),
        observation_width(ObservationVariant.FUSED),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    bars = ax.bar(labels, values, color=["#7e57c2", "#29b6f6", "#66bb6a", "#ffa726"])
    ax.axhline(128, color="#ef5350", ls="--", lw=1, label="BHL trap (128)")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 6, str(value), ha="center", color="white", fontsize=10)
    ax.set_ylabel("observation last-dim", color="white")
    ax.set_title("A≠B≠C  ·  C=D width  ·  never 128", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0e1116", labelcolor="white", frameon=False)
    for spine in ax.spines.values():
        spine.set_color("#455a64")
    path = OUT / "obs_widths.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_fusion_gif() -> Path:
    rng = np.random.default_rng(0)
    tof0 = torch.tensor(0.40 + 0.04 * rng.standard_normal((8, 8)), dtype=torch.float32)
    tof1 = torch.tensor(0.42 + 0.04 * rng.standard_normal((8, 8)), dtype=torch.float32)
    metric = torch.tensor(1.20 + 0.08 * rng.standard_normal((16, 16)), dtype=torch.float32)
    frames = []
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), facecolor="#0e1116")
    titles = ["ToF0 8×8", "metric 16×16 → 8×8", "D = fuse(ToF0,ToF1,metric)"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_facecolor("#0e1116")
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ims = [
        axes[0].imshow(tof0.numpy(), vmin=0.2, vmax=1.4, cmap="magma"),
        axes[1].imshow(metric.numpy(), vmin=0.2, vmax=1.4, cmap="magma"),
        axes[2].imshow(np.zeros((8, 8)), vmin=0.2, vmax=1.4, cmap="magma"),
    ]

    def update(frame: int):
        noise = 0.02 * np.sin(frame / 3.0)
        t0 = tof0 + noise
        t1 = tof1 - 0.5 * noise
        fused = fuse_tof_and_metric(
            t0.unsqueeze(0),
            t1.unsqueeze(0),
            metric.unsqueeze(0),
            torch.full_like(t0, 1e-4).unsqueeze(0),
            torch.full_like(t1, 1e-4).unsqueeze(0),
            torch.full_like(metric, 1e-2).unsqueeze(0),
            torch.ones_like(t0, dtype=torch.bool).unsqueeze(0),
            torch.ones_like(t1, dtype=torch.bool).unsqueeze(0),
        )[0]
        ims[0].set_data(t0.numpy())
        ims[2].set_data(fused.numpy())
        return ims

    anim = animation.FuncAnimation(fig, update, frames=24, interval=80, blit=True)
    path = OUT / "fusion_d.gif"
    anim.save(path, writer="pillow", dpi=110)
    plt.close(fig)
    frames.append(path)
    return path


def write_gate0() -> Path:
    evidence = json.loads((ROOT / "docs" / "evidence" / "isaac_smoke_21077170.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    ax.axis("off")
    ax.set_title("Gate 0  ·  Isaac Sim 6.0 RTX  ·  job 21077170", color="white", pad=12)
    lines = [
        f"node  {evidence['node']}   A40",
        f"cube planar z   {evidence['cube_depth_m']:.4f} m   expect 1.5000",
        f"plane planar z  {evidence['plane_depth_m']:.4f} m   expect 2.0000",
        f"RGB std         {evidence['cube_rgb_std']:.1f}     not a black frame",
        f"trees           {evidence['trees']['count']} Envy debug USDA",
    ]
    for index, line in enumerate(lines):
        ax.text(
            0.04,
            0.78 - 0.16 * index,
            line,
            color="#eceff1",
            fontsize=11,
            family="monospace",
            transform=ax.transAxes,
        )
    path = OUT / "gate0_rtx.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_cutter() -> Path:
    spec = load_ur5e_pruner_spec()
    fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")

    def rect(half, offset, color, label):
        x, y = offset[1] - half[1], offset[2] - half[2]
        ax.add_patch(plt.Rectangle((x, y), 2 * half[1], 2 * half[2], fill=False, ec=color, lw=2, label=label))

    rect(spec.mouth_half_extents_m, spec.mouth_offset_m, "#ff7043", "mouth AABB")
    rect(spec.failure_half_extents_m, spec.failure_offset_m, "#ef5350", "failure AABB")
    ax.set_aspect("equal")
    ax.set_xlabel("EEF Y (m)", color="white")
    ax.set_ylabel("EEF Z (m)", color="white")
    ax.set_title("Cutter boxes from pybullet-tree-sim STL", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0e1116", labelcolor="white", frameon=False)
    path = OUT / "cutter_boxes.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = [write_tree(), write_widths(), write_fusion_gif(), write_gate0(), write_cutter()]
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
