"""Train / eval tree identity splits."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

_CONFIG_PACKAGE = "isaaclab_pruning.config.eval"


@dataclass(frozen=True)
class TreeSplits:
    held_out_envy: tuple[str, ...]
    debug_envy: tuple[str, ...]
    camera_rect_depth_m: float
    seeds: tuple[int, ...]

    def is_held_out(self, tree_id: str) -> bool:
        return tree_id in self.held_out_envy

    def train_envy(self, available: list[str]) -> list[str]:
        held = set(self.held_out_envy)
        return [tree_id for tree_id in available if tree_id not in held]


def load_tree_splits(path: str | Path | None = None) -> TreeSplits:
    if path is None:
        source = resources.files(_CONFIG_PACKAGE).joinpath("splits.yaml").read_text(encoding="utf-8")
        payload = yaml.safe_load(source)
    else:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TreeSplits(
        held_out_envy=tuple(payload["held_out_envy"]),
        debug_envy=tuple(payload["debug_envy"]),
        camera_rect_depth_m=float(payload["camera_rect_depth_m"]),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
    )
