from __future__ import annotations

from pathlib import Path

import torch
import yaml

from isaaclab_pruning.sensors.tof_noise import (
    ToFNoiseConfig,
    ToFStatus,
    apply_tof_noise,
)


def test_range_limits_are_reported_as_status_not_fake_depth() -> None:
    config = ToFNoiseConfig(
        min_sigma_m=0.0,
        range_sigma_fraction=0.0,
        dropout_probability=0.0,
        thin_dropout_probability=0.0,
    )
    ranges = torch.tensor([0.01, 0.03, 1.0, 3.4, 4.0, float("inf")])
    observation = apply_tof_noise(ranges, config=config)

    assert observation.valid.tolist() == [False, True, True, True, False, False]
    assert observation.status.tolist() == [1, 0, 0, 0, 1, 1]
    assert torch.isnan(observation.range_m[[0, 4, 5]]).all()
    torch.testing.assert_close(observation.range_m[1:4], ranges[1:4])


def test_random_dropout_takes_precedence_over_thin_target_dropout() -> None:
    config = ToFNoiseConfig(
        min_sigma_m=0.0,
        range_sigma_fraction=0.0,
        dropout_probability=1.0,
        thin_dropout_probability=1.0,
    )
    observation = apply_tof_noise(
        torch.ones(4),
        hit_radii_m=torch.full((4,), 0.001),
        config=config,
    )
    assert observation.status.tolist() == [int(ToFStatus.RANDOM_DROPOUT)] * 4


def test_thin_target_dropout_uses_hit_radius() -> None:
    config = ToFNoiseConfig(
        min_sigma_m=0.0,
        range_sigma_fraction=0.0,
        dropout_probability=0.0,
        thin_dropout_probability=1.0,
    )
    observation = apply_tof_noise(
        torch.ones(2),
        hit_radii_m=torch.tensor([0.007, 0.009]),
        config=config,
    )
    assert observation.status.tolist() == [
        int(ToFStatus.THIN_TARGET_DROPOUT),
        int(ToFStatus.VALID),
    ]


def test_variance_matches_the_declared_range_model() -> None:
    config = ToFNoiseConfig(
        min_sigma_m=0.003,
        range_sigma_fraction=0.03,
        dropout_probability=0.0,
        thin_dropout_probability=0.0,
    )
    generator = torch.Generator().manual_seed(7)
    observation = apply_tof_noise(torch.tensor([0.05, 1.0]), config=config, generator=generator)
    torch.testing.assert_close(observation.variance_m2, torch.tensor([0.003**2, 0.03**2]))


def test_mock_pruner_rig_preserves_hardware_offsets_and_no_camera_guess() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "isaaclab_pruning"
        / "isaaclab_pruning"
        / "config"
        / "rigs"
        / "mock_pruner_vl53l8cx.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sensors = {sensor["name"]: sensor for sensor in config["sensors"]}
    assert sensors["tof0"]["mount_offset_m"] == [0.04685226669, 0.0, 0.14444246761]
    assert sensors["tof1"]["mount_offset_m"] == [-0.04685226669, 0.0, 0.14444246761]
    assert config["wrist_camera"]["enabled"] is False
    assert config["wrist_camera"]["offset"] == [0.0, -0.06, 0.10]
    assert config["wrist_camera"]["selected"] == "close_lateral"
