"""V-trellis orchard furniture as metric geometry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_CONFIG_PACKAGE = "isaaclab_pruning.config.orchard"


@dataclass(frozen=True)
class Post:
    name: str
    centroid: np.ndarray
    height_m: float
    radius_m: float


@dataclass(frozen=True)
class Wire:
    name: str
    centroid: np.ndarray
    length_m: float
    radius_m: float
    axis: np.ndarray


@dataclass(frozen=True)
class OrchardLayout:
    tree_translations: tuple[tuple[float, float, float], ...]
    tree_tilt_x_deg: float
    posts: tuple[Post, ...]
    wires: tuple[Wire, ...]
    ground_size_m: tuple[float, float, float]
    ground_color: tuple[float, float, float]
    baseline_bark: str
    lighting: dict[str, Any]


def load_orchard_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        source = resources.files(_CONFIG_PACKAGE).joinpath("v_trellis.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(source)
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_v_trellis_layout(config: dict[str, Any] | None = None) -> OrchardLayout:
    """Place three tilted Envy trees, posts, and trellis wires in one env."""
    cfg = config or load_orchard_config()
    count = int(cfg["trees_per_env"])
    spacing = float(cfg["in_row_spacing_m"])
    row_y = float(cfg["row_y_m"])
    origin_x = -0.5 * (count - 1) * spacing
    translations = tuple((origin_x + index * spacing, row_y, 0.0) for index in range(count))

    post_cfg = cfg["posts"]
    posts = tuple(
        Post(
            name=f"post_{index}",
            centroid=np.array(
                [translation[0], translation[1] + float(post_cfg["y_offset_m"]), 0.5 * float(post_cfg["height_m"])],
                dtype=np.float64,
            ),
            height_m=float(post_cfg["height_m"]),
            radius_m=float(post_cfg["radius_m"]),
        )
        for index, translation in enumerate(translations)
    )

    wire_cfg = cfg["wires"]
    mid_x = 0.5 * (translations[0][0] + translations[-1][0])
    wires = tuple(
        Wire(
            name=f"wire_{index}",
            centroid=np.array([mid_x, row_y, float(height)], dtype=np.float64),
            length_m=float(wire_cfg["length_m"]),
            radius_m=float(wire_cfg["radius_m"]),
            axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        for index, height in enumerate(wire_cfg["heights_m"])
    )
    return OrchardLayout(
        tree_translations=translations,
        tree_tilt_x_deg=float(cfg["tree_tilt_x_deg"]),
        posts=posts,
        wires=wires,
        ground_size_m=tuple(float(value) for value in cfg["ground"]["size_m"]),
        ground_color=tuple(float(value) for value in cfg["ground"]["color"]),
        baseline_bark=str(cfg["baseline_bark"]),
        lighting=dict(cfg["lighting"]),
    )
