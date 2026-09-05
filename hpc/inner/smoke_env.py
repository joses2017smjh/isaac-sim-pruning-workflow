"""Batched v60 env smoke: trainer import, A–D obs contract, one step, PhysX contact.

One job. Asserts, not log archaeology. Writes docs/evidence/smoke_<jobid>.json.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import sys
import traceback
from pathlib import Path


def _probe_trainer(name: str) -> str | None:
    """Import an RL library without launching Kit."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "imported"))


report: dict = {
    "ok": False,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    "bhl_stack": os.environ.get("BHL_STACK"),
    "skrl": _probe_trainer("skrl"),
    "rsl_rl": _probe_trainer("rsl_rl"),
    "asset_id": os.environ.get("PRUNING_ASSET_ID"),
    "usd": os.environ.get("PRUNING_USD"),
    "usd_evidence": os.environ.get("PRUNING_USD_EVIDENCE"),
    "phase": "import",
}


def _bench_path() -> Path:
    return Path(os.environ.get("BENCH_OUT", "/tmp/pruning_env_smoke.json"))


def _flush_report() -> None:
    out = _bench_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write(code: int) -> None:
    print(json.dumps(report, indent=2, default=str, allow_nan=False), flush=True)
    _flush_report()
    raise SystemExit(code)


def _range_frame_json(values) -> list:
    """Serialize a batched range tensor with standards-compliant null misses."""
    nested = values.detach().cpu().tolist()
    return [
        [[float(value) if math.isfinite(float(value)) else None for value in row] for row in frame] for frame in nested
    ]


_flush_report()


if report["skrl"] is None and report["rsl_rl"] is None:
    report["error"] = "No RL library on this interpreter (skrl and rsl_rl both missing)."
    _write(1)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    pruning_root = Path(os.environ["PRUNING_ROOT"])
    sys.path.insert(0, str(pruning_root / "source" / "isaaclab_pruning"))

    import torch

    from isaaclab.utils.math import quat_apply, subtract_frame_transforms

    from isaaclab_pruning.policies.observations import ObservationVariant, observation_width
    from isaaclab_pruning.sim.lab3_compat import apply as apply_lab3
    from isaaclab_pruning.sim.lab3_compat import as_torch
    from isaaclab_pruning.sim.pose_conventions import pose_xyzw_to_wxyz, quaternion_wxyz_to_xyzw
    from isaaclab_pruning.sim.pruning_env import make_pruning_env_cls
    from isaaclab_pruning.sim.pruning_env_cfg import PruningEnvCfg
    from isaaclab_pruning.sim.tof_smoke_geometry import TOF_SMOKE_TARGET, enable_tof_smoke_target

    report["lab3_compat"] = apply_lab3()

    cfgs = {}
    spaces = {}
    for variant in ObservationVariant:
        cfg = PruningEnvCfg(observation_variant=variant.value)
        cfg.tof_noise_enabled = False
        expected = observation_width(
            variant,
            n_joints=cfg.n_joints,
            flow_hw=cfg.flow_hw,
            tof_hw=cfg.tof_hw,
            metric_hw=cfg.metric_hw,
        )
        assert cfg.observation_space == expected, (variant, cfg.observation_space, expected)
        assert cfg.observation_space != 128, variant
        cfgs[variant.value] = cfg
        spaces[variant.value] = int(cfg.observation_space)
    enable_tof_smoke_target(cfgs["B_tof"])
    report["observation_space"] = spaces
    assert spaces["A_flow"] != spaces["B_tof"] != spaces["C_metric"]
    assert spaces["A_flow"] != spaces["C_metric"]
    assert spaces["C_metric"] == spaces["D_fused"]

    Env = make_pruning_env_cls()
    report["phase"] = "construct"
    _flush_report()
    env = Env(cfg=cfgs["B_tof"])
    smoke_target = env.tof_smoke_target_state()
    assert smoke_target["enabled"] is True
    assert smoke_target["stage_prim_valid"] is True
    assert smoke_target["stage_prim_type"] == "Xform"
    assert smoke_target["geometry_prim_type"] == "Cube"
    assert smoke_target["collision_api_applied"] is False
    assert smoke_target["rigid_body_api_applied"] is False
    assert smoke_target["prim_path"] == TOF_SMOKE_TARGET.prim_path
    assert (
        max(
            abs(actual - expected)
            for actual, expected in zip(smoke_target["actual_position_w_m"], TOF_SMOKE_TARGET.position_w_m, strict=True)
        )
        < 1.0e-6
    )
    assert (
        max(
            abs(actual - expected)
            for actual, expected in zip(smoke_target["actual_size_m"], TOF_SMOKE_TARGET.size_m, strict=True)
        )
        < 1.0e-6
    )
    assert smoke_target["sensor_mesh_prim_paths"] == {
        "tof0": [TOF_SMOKE_TARGET.prim_expr],
        "tof1": [TOF_SMOKE_TARGET.prim_expr],
    }
    report["tof_smoke_target"] = smoke_target
    report["phase"] = "reset"
    _flush_report()
    obs, _ = env.reset(seed=0)
    policy = obs["policy"]
    assert policy.shape[-1] == spaces["B_tof"], (policy.shape, spaces["B_tof"])

    last_dims = {}
    tensors = {}
    for variant in ObservationVariant:
        tensor = env.observation_for_variant(variant)
        last_dims[variant.value] = int(tensor.shape[-1])
        tensors[variant] = tensor
        assert tensor.shape[-1] == spaces[variant.value], (variant, tensor.shape[-1], spaces[variant.value])
        assert cfgs[variant.value].observation_space == tensor.shape[-1]

    report["obs_last_dim"] = last_dims
    assert last_dims["A_flow"] != last_dims["B_tof"]
    assert last_dims["B_tof"] != last_dims["C_metric"]
    assert last_dims["A_flow"] != last_dims["C_metric"]
    assert last_dims["C_metric"] == last_dims["D_fused"]
    assert not __import__("torch").allclose(tensors[ObservationVariant.METRIC], tensors[ObservationVariant.FUSED])

    assert env.spec.physics_eef_body == "mock_pruner__base"
    assert env.spec.control_tool_frame == "mock_pruner__tool0"
    assert env.spec.eef_body == env.spec.physics_eef_body
    body_w = as_torch(env.robot.data.body_pose_w)[:, env.robot_entity_cfg.body_ids[0], 0:7]
    tool_w_initial = env._control_tool_pose_w().clone()
    tool_separation_m = torch.linalg.vector_norm(tool_w_initial[:, :3] - body_w[:, :3], dim=-1)
    torch.testing.assert_close(
        tool_separation_m,
        torch.full_like(tool_separation_m, 0.1601525),
        atol=1.0e-5,
        rtol=0.0,
    )
    root_w = as_torch(env.robot.data.root_pose_w)
    tool_pos_b, tool_quat_b = subtract_frame_transforms(
        root_w[:, :3], root_w[:, 3:7], tool_w_initial[:, :3], quaternion_wxyz_to_xyzw(tool_w_initial[:, 3:7])
    )
    body_pos_b, _ = subtract_frame_transforms(root_w[:, :3], root_w[:, 3:7], body_w[:, :3], body_w[:, 3:7])
    hold = pose_xyzw_to_wxyz(torch.cat((tool_pos_b, tool_quat_b), dim=-1)).contiguous()
    old_body_command_error_m = torch.linalg.vector_norm(hold[:, :3] - body_pos_b, dim=-1)
    assert bool((old_body_command_error_m > 0.15).all().item())

    # Let at least two 15 Hz frames arrive while holding the true control tool.
    report["phase"] = "hold_and_live_tof_warmup"
    _flush_report()
    warmup_steps = max(6, int(env.cfg.tof0_cfg.update_period / env.step_dt) + 2)
    report["hold_trace"] = []
    for _ in range(warmup_steps):
        next_obs, rewards, terminated, truncated, extras = env.step(hold)
        report["hold_trace"].append(
            {
                "tool_pose_wxyz": env._control_tool_pose_w().detach().cpu().tolist(),
                "joint_pos": as_torch(env.robot.data.joint_pos).detach().cpu().tolist(),
                "terminated": terminated.detach().cpu().tolist(),
                "truncated": truncated.detach().cpu().tolist(),
            }
        )
    tool_w_held = env._control_tool_pose_w().clone()
    hold_translation_drift_m = torch.linalg.vector_norm(tool_w_held[:, :3] - tool_w_initial[:, :3], dim=-1)
    quat_dot = torch.sum(tool_w_held[:, 3:7] * tool_w_initial[:, 3:7], dim=-1).abs().clamp(max=1.0)
    hold_rotation_drift_rad = 2.0 * torch.acos(quat_dot)
    report["hold_diagnostic"] = {
        "initial_tool_pose_wxyz": tool_w_initial.detach().cpu().tolist(),
        "command_pose_b_wxyz": hold.detach().cpu().tolist(),
        "translation_drift_m": hold_translation_drift_m.detach().cpu().tolist(),
        "rotation_drift_rad": hold_rotation_drift_rad.detach().cpu().tolist(),
        "tof_state": env.tof_state(),
        "contact": env.contact_state(),
    }
    _flush_report()
    assert bool((hold_translation_drift_m < 5.0e-3).all().item())
    assert bool((hold_rotation_drift_rad < 2.0e-2).all().item())

    tof_before = env.tof_state()
    raw_before = {name: getattr(env, f"{name}_raw").clone() for name in env.tof_sensors}
    assert tof_before["source"] == "live_multi_mesh_ray_caster"
    assert tof_before["noise_enabled"] is False
    for name, state in tof_before["sensors"].items():
        assert state["class"] == "MultiMeshRayCasterCamera"
        assert state["observation"]["shape"] == [env.num_envs, 8, 8]
        assert state["observation"]["valid_fraction"] == 1.0, (name, state)
        assert 0.45 < state["raw"]["min_m"] < state["raw"]["max_m"] < 0.70, (name, state)
        assert min(state["frame"]) >= 1

    # Move the tool 5 mm along its local +Z (the ToF optical direction), then
    # require both physical motion and a changed ray-cast range table.
    report["phase"] = "tool_motion_and_tof_response"
    _flush_report()
    local_step = torch.zeros((env.num_envs, 3), device=env.device)
    local_step[:, 2] = 0.005
    move = hold.clone()
    move[:, :3] += quat_apply(tool_quat_b, local_step)
    initial_move_error = torch.linalg.vector_norm(move[:, :3] - hold[:, :3], dim=-1)
    motion_steps = max(12, 2 * warmup_steps)
    for _ in range(motion_steps):
        next_obs, rewards, terminated, truncated, extras = env.step(move)
    tool_w_moved = env._control_tool_pose_w().clone()
    root_w_after = as_torch(env.robot.data.root_pose_w)
    moved_pos_b, _ = subtract_frame_transforms(
        root_w_after[:, :3],
        root_w_after[:, 3:7],
        tool_w_moved[:, :3],
        quaternion_wxyz_to_xyzw(tool_w_moved[:, 3:7]),
    )
    final_move_error = torch.linalg.vector_norm(move[:, :3] - moved_pos_b, dim=-1)
    assert bool((final_move_error < initial_move_error).all().item())
    assert bool((torch.linalg.vector_norm(moved_pos_b - hold[:, :3], dim=-1) > 5.0e-4).all().item())

    tof_after = env.tof_state()
    range_change = {}
    for name in env.tof_sensors:
        before = raw_before[name]
        after = getattr(env, f"{name}_raw")
        shared = torch.isfinite(before) & torch.isfinite(after)
        assert bool(shared.any().item()), name
        delta = (after - before).abs()[shared]
        toward_delta = (before - after)[shared]
        range_change[name] = {
            "shared_finite_pixels": int(shared.sum().item()),
            "median_abs_delta_m": float(delta.median().item()),
            "max_abs_delta_m": float(delta.max().item()),
            "median_toward_delta_m": float(toward_delta.median().item()),
        }
        assert range_change[name]["shared_finite_pixels"] == before.numel(), (name, range_change[name])
        assert range_change[name]["max_abs_delta_m"] > 1.0e-4, (name, range_change[name])
        assert range_change[name]["median_toward_delta_m"] > 1.0e-4, (name, range_change[name])
        assert min(tof_after["sensors"][name]["frame"]) > min(tof_before["sensors"][name]["frame"])

    contact = env.contact_state()
    assert contact["finite"]
    report["step"] = {
        "reward": float(rewards.mean().item()),
        "terminated": bool(terminated.any().item()),
        "truncated": bool(truncated.any().item()),
        "obs_last_dim": int(next_obs["policy"].shape[-1]),
    }
    report["tool_frame"] = {
        "physics_body": env.spec.physics_eef_body,
        "control_tool": env.spec.control_tool_frame,
        "lab_quaternion_order": "xyzw",
        "core_and_action_quaternion_order": "wxyz",
        "body_to_tool_distance_m": tool_separation_m.detach().cpu().tolist(),
        "old_body_command_error_m": old_body_command_error_m.detach().cpu().tolist(),
        "hold_translation_drift_m": hold_translation_drift_m.detach().cpu().tolist(),
        "hold_rotation_drift_rad": hold_rotation_drift_rad.detach().cpu().tolist(),
        "move_initial_error_m": initial_move_error.detach().cpu().tolist(),
        "move_final_error_m": final_move_error.detach().cpu().tolist(),
    }
    report["tof_before"] = tof_before
    report["tof_after"] = tof_after
    report["tof_range_change"] = range_change
    report["tof_raw_frames"] = {
        "before": {name: _range_frame_json(values) for name, values in raw_before.items()},
        "after": {name: _range_frame_json(getattr(env, f"{name}_raw")) for name in env.tof_sensors},
    }
    report["contact"] = contact
    names = list(getattr(env.robot.data, "joint_names", []) or getattr(env.robot, "joint_names", []))
    report["joint_names"] = [str(name) for name in names]
    report["n_arm_joints"] = len(env.robot_entity_cfg.joint_ids)
    report["slider_in_joint_names"] = any("linear_slider" in str(name) for name in names)
    assert report["n_arm_joints"] == 6
    assert not report["slider_in_joint_names"]
    report["ok"] = True
    # Kit's close path may terminate the interpreter.  Persist the complete
    # application result before either environment or app cleanup so the outer
    # Slurm wrapper never has to infer success from a process exit code.
    _flush_report()
    print(json.dumps(report, indent=2, default=str, allow_nan=False), flush=True)
    env.close()
except Exception as error:  # noqa: BLE001 — always emit JSON
    report["ok"] = False
    report["error"] = repr(error)
    report["traceback"] = traceback.format_exc()
    _flush_report()
    print(json.dumps(report, indent=2, default=str, allow_nan=False), flush=True)
    with contextlib.suppress(Exception):
        env.close()
    with contextlib.suppress(Exception):
        simulation_app.close()
    raise SystemExit(1)

simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
