from __future__ import annotations

import numpy as np
import pytest
import torch

from isaaclab_pruning.baselines import scripted_absolute_pose, scripted_tof_action, ur5e_pruner_oracle_status
from isaaclab_pruning.eval import CAMERA_RECT_DEPTH_M, ranking_inversion, success_vs_cut_error
from isaaclab_pruning.eval.box_check import assert_box_agrees, camera_rect_extent
from isaaclab_pruning.geometry import Cylinder
from isaaclab_pruning.ladder import inject_cut_point_error, sample_ladder
from isaaclab_pruning.policies import (
    ObservationVariant,
    assert_ready_for_policy_claim,
    build_observation,
    load_training_protocol,
    observation_width,
    proprioception,
)
from isaaclab_pruning.sensors import CANDIDATES, fuse_depths, score_candidate
from isaaclab_pruning.task import (
    RewardWeights,
    curriculum_stage,
    dense_pruning_reward,
    episode_start_target,
    select_curriculum_cut,
)
from isaaclab_pruning.task.success import CutSuccess


def _identity_success(batch: int, success: bool = False) -> CutSuccess:
    flag = torch.tensor([success] * batch)
    zeros = torch.zeros(batch)
    return CutSuccess(
        success=flag,
        mouth_hit=flag,
        failure_clear=torch.ones(batch, dtype=torch.bool),
        perpendicular=torch.ones(batch, dtype=torch.bool),
        collision_free=torch.ones(batch, dtype=torch.bool),
        perpendicularity_error_deg=zeros,
    )


def test_inverse_variance_fusion_prefers_the_low_variance_sensor() -> None:
    depths = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    variances = torch.tensor([[1.0, 1e6], [1e6, 1.0]])
    fused, fused_var = fuse_depths(depths, variances)
    torch.testing.assert_close(fused, torch.tensor([1.0, 4.0]), atol=1e-4, rtol=0)
    assert torch.all(fused_var < 1.0)


def test_wrist_camera_candidates_are_not_selected() -> None:
    intrinsic = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    scores = [score_candidate(candidate, (0.0, 0.0, 0.4), intrinsic, 640, 480) for candidate in CANDIDATES]
    assert all(score["selected"] is False for score in scores)
    assert any(score["visible"] for score in scores)


def test_reward_grows_when_the_cut_succeeds() -> None:
    eef = torch.zeros(2, 3)
    cut = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    actions = torch.zeros(2, 7)
    fail = dense_pruning_reward(eef, cut, _identity_success(2, False), actions, weights=RewardWeights(time=0.0))
    win = dense_pruning_reward(eef, cut, _identity_success(2, True), actions, weights=RewardWeights(time=0.0))
    assert torch.all(win > fail)


def test_curriculum_starts_on_thick_branches() -> None:
    classes, min_radius = curriculum_stage(0.0)
    assert classes == ("branch",)
    assert min_radius == 0.015
    cylinders = [
        Cylinder("0", "branch_1", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, 0.3),
        Cylinder("1", "spur_1", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.003, 0.1),
    ]
    chosen = select_curriculum_cut(cylinders, 0.0)
    assert chosen.part_name == "branch_1"


def test_scripted_tof_moves_toward_a_centered_return() -> None:
    ranges = torch.full((1, 8, 8), 0.4)
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    action = scripted_tof_action(ranges, ranges, valid, valid)
    assert action.shape == (1, 7)
    assert action[0, 2] > 0
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    posed = scripted_absolute_pose(identity, action)
    assert posed[0, 2] > 0
    assert not ur5e_pruner_oracle_status(urdf_usd_path=None).configured


def test_observation_variants_and_five_seed_protocol() -> None:
    protocol = load_training_protocol()
    assert protocol.seeds == (0, 1, 2, 3, 4)
    assert not protocol.da2_in_ppo_loop
    assert len(protocol.run_ids()) == 20
    with pytest.raises(RuntimeError, match="missing baselines"):
        assert_ready_for_policy_claim({"scripted_tof": False, "curobo_oracle": False})

    proprio = proprioception(torch.zeros(1, 6), torch.zeros(1, 6), torch.zeros(1, 7))
    goal = torch.ones(1, 3)
    tof = torch.ones(1, 8, 8)
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    obs_b = build_observation(
        ObservationVariant.TOF,
        goal_w=goal,
        proprio=proprio,
        tof0=tof,
        tof1=tof,
        tof0_valid=valid,
        tof1_valid=valid,
    )
    flow = torch.zeros(1, 4, 4, 2)
    obs_a = build_observation(ObservationVariant.FLOW, goal_w=goal, proprio=proprio, flow_hw2=flow)
    assert obs_b.shape[-1] > obs_a.shape[-1]
    widths = {
        observation_width(ObservationVariant.FLOW, n_joints=6, flow_hw=(4, 4)),
        observation_width(ObservationVariant.TOF, n_joints=6, tof_hw=(8, 8)),
        observation_width(ObservationVariant.METRIC, n_joints=6, metric_hw=(8, 8)),
    }
    assert len(widths) == 3
    matched = {
        observation_width(ObservationVariant.TOF),
        observation_width(ObservationVariant.METRIC),
        observation_width(ObservationVariant.FUSED),
        observation_width(ObservationVariant.FLOW),
    }
    assert 128 not in matched
    assert observation_width(ObservationVariant.METRIC) == observation_width(ObservationVariant.FUSED)
    assert len(
        {
            observation_width(ObservationVariant.FLOW),
            observation_width(ObservationVariant.TOF),
            observation_width(ObservationVariant.METRIC),
        }
    ) == 3
    native = observation_width(ObservationVariant.METRIC, metric_hw=(256, 256))
    assert native != observation_width(ObservationVariant.METRIC)
    metric = torch.full((1, 8, 8), 2.0)
    obs_c = build_observation(ObservationVariant.METRIC, goal_w=goal, proprio=proprio, metric_depth=metric)
    obs_d = build_observation(
        ObservationVariant.FUSED,
        goal_w=goal,
        proprio=proprio,
        tof0=tof,
        tof1=tof,
        tof0_valid=valid,
        tof1_valid=valid,
        metric_depth=metric,
        tof0_var=torch.full_like(tof, 1e-8),
        tof1_var=torch.full_like(tof, 1e-8),
        metric_var=torch.full_like(metric, 1e6),
    )
    assert obs_c.shape == obs_d.shape
    assert not torch.allclose(obs_c, obs_d)
    native_metric = torch.full((1, 16, 16), 2.0)
    obs_d_native = build_observation(
        ObservationVariant.FUSED,
        goal_w=goal,
        proprio=proprio,
        tof0=tof,
        tof1=tof,
        tof0_valid=valid,
        tof1_valid=valid,
        metric_depth=native_metric,
        tof0_var=torch.full_like(tof, 1e-8),
        tof1_var=torch.full_like(tof, 1e-8),
        metric_var=torch.full_like(native_metric, 1e6),
    )
    assert obs_d_native.shape == obs_d.shape


def test_ladder_is_nominal_at_zero_and_injects_cut_error() -> None:
    progress = torch.zeros(4)
    sample = sample_ladder(progress)
    torch.testing.assert_close(sample.tof_sigma_fraction, torch.zeros(4))
    torch.testing.assert_close(sample.cut_point_error_m, torch.zeros(4))
    hard = sample_ladder(torch.ones(4))
    assert torch.all(hard.cut_point_error_m == 0.10)
    shifted = inject_cut_point_error(torch.zeros(4, 3), torch.full((4,), 0.05), torch.Generator().manual_seed(0))
    torch.testing.assert_close(torch.linalg.vector_norm(shifted, dim=-1), torch.full((4,), 0.05), atol=1e-5, rtol=0)


def test_eval_box_and_ranking_inversion() -> None:
    assert CAMERA_RECT_DEPTH_M == 0.30
    points = np.array([[-0.1, -0.1, 0.30], [0.1, 0.1, 0.30], [0.0, 0.0, 0.30]])
    extent = camera_rect_extent(points)
    assert extent["width_m"] == pytest.approx(0.2)
    assert_box_agrees(0.30)
    result = ranking_inversion({"A": 0.9, "B": 0.4}, {"A": 0.2, "B": 0.8})
    assert result.inverted
    centers, rates = success_vs_cut_error(
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.01, 0.02, 0.08, 0.09]),
        torch.tensor([0.0, 0.05, 0.10]),
    )
    assert rates[0] == 1.0
    assert rates[1] == 0.0
    _ = centers


def test_episode_start_is_the_only_perception_call() -> None:
    cut = select_curriculum_cut(
        [Cylinder("0", "spur_1", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.006, 0.1)],
        1.0,
    )
    target = episode_start_target(cut, batch=2, device=torch.device("cpu"), source="oracle")
    assert target.source == "oracle"
    assert target.position_w.shape == (2, 3)
