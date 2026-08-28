"""Five-seed training protocol. No single-seed numbers are evidence."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from isaaclab_pruning.policies.observations import ObservationVariant

_CONFIG_PACKAGE = "isaaclab_pruning.config.agents"


@dataclass(frozen=True)
class TrainingProtocol:
    seeds: tuple[int, ...]
    variants: tuple[ObservationVariant, ...]
    num_envs_raycast: int
    num_envs_tiled_rgb: int
    da2_in_ppo_loop: bool
    slider_held_fixed: bool
    baselines: tuple[str, ...]

    def run_ids(self) -> tuple[str, ...]:
        return tuple(f"{variant.value}-seed{seed}" for variant in self.variants for seed in self.seeds)


def load_training_protocol(path: str | Path | None = None) -> TrainingProtocol:
    if path is None:
        source = resources.files(_CONFIG_PACKAGE).joinpath("protocol.yaml").read_text(encoding="utf-8")
        payload = yaml.safe_load(source)
    else:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    variants = tuple(ObservationVariant(name) for name in payload["variants"])
    return TrainingProtocol(
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        variants=variants,
        num_envs_raycast=int(payload["num_envs_raycast"]),
        num_envs_tiled_rgb=int(payload["num_envs_tiled_rgb"]),
        da2_in_ppo_loop=bool(payload["da2_in_ppo_loop"]),
        slider_held_fixed=bool(payload["slider_held_fixed"]),
        baselines=tuple(payload["baselines"]),
    )


def assert_ready_for_policy_claim(baselines_complete: dict[str, bool]) -> None:
    protocol = load_training_protocol()
    missing = [name for name in protocol.baselines if not baselines_complete.get(name)]
    if missing:
        raise RuntimeError(f"Cannot report a learned policy; missing baselines: {', '.join(missing)}.")
    if protocol.da2_in_ppo_loop:
        raise RuntimeError("DA2-ft must not run inside the PPO loop.")
