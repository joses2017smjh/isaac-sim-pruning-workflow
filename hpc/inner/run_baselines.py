"""Run scripted ToF in the pruning env and record CuRobo readiness status.

This is the Isaac baseline job. It does not start PPO. CuRobo motion planning
is not executed here, and no task success rate is inferred from configuration.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

report: dict = {
    "ok": False,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    "bhl_stack": os.environ.get("BHL_STACK"),
    "scripted_tof": False,
    "curobo_oracle": False,
    "result_scope": "runtime sensor-and-motion gate only; no task success rate is measured",
}


def _bench_path() -> Path:
    return Path(os.environ.get("BENCH_OUT", "/tmp/pruning_baselines.json"))


def _assert_json_finite(value: Any, path: str = "report") -> None:
    """Reject non-finite JSON numbers with their evidence path."""
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite value at {path}: {value!r}.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")


def _replace_nonfinite(value: Any, path: str = "report") -> tuple[Any, list[str]]:
    """Replace invalid floats with null while preserving explicit failure paths."""
    paths: list[str] = []

    def visit(item: Any, item_path: str) -> Any:
        if isinstance(item, Real) and not isinstance(item, bool) and not math.isfinite(float(item)):
            paths.append(item_path)
            return None
        if isinstance(item, Mapping):
            return {key: visit(child, f"{item_path}.{key}") for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [visit(child, f"{item_path}[{index}]") for index, child in enumerate(item)]
        return item

    return visit(value, path), paths


def _json_payload(value: Any) -> str:
    _assert_json_finite(value)
    return json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"


def _flush() -> None:
    out = _bench_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json_payload(report), encoding="utf-8")


def _write(code: int) -> None:
    try:
        payload = _json_payload(report)
    except (FloatingPointError, ValueError) as error:
        sanitized, paths = _replace_nonfinite(report)
        report.clear()
        report.update(sanitized)
        report["ok"] = False
        report["serialization_error"] = str(error)
        report["nonfinite_paths"] = paths
        code = 1
        payload = json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
    print(payload, end="", flush=True)
    out = _bench_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    raise SystemExit(code)


_flush()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    pruning_root = Path(os.environ["PRUNING_ROOT"])
    sys.path.insert(0, str(pruning_root / "source" / "isaaclab_pruning"))

    import torch

    from isaaclab.utils.math import subtract_frame_transforms

    from isaaclab_pruning.baselines.curobo import ur5e_pruner_oracle_status
    from isaaclab_pruning.baselines.runtime_gate import validate_scripted_tof_runtime
    from isaaclab_pruning.baselines.tof_servo import scripted_absolute_pose, scripted_tof_action
    from isaaclab_pruning.robot import imported_usd_path
    from isaaclab_pruning.sim.lab3_compat import apply as apply_lab3
    from isaaclab_pruning.sim.lab3_compat import as_torch
    from isaaclab_pruning.sim.pose_conventions import pose_xyzw_to_wxyz, quaternion_wxyz_to_xyzw
    from isaaclab_pruning.sim.pruning_env import make_pruning_env_cls
    from isaaclab_pruning.sim.pruning_env_cfg import PruningEnvCfg

    report["lab3_compat"] = apply_lab3()
    usd = imported_usd_path()
    status = ur5e_pruner_oracle_status(urdf_usd_path=str(usd) if usd.is_file() else None)
    try:
        import curobo  # noqa: F401
    except ImportError:
        report["curobo_import"] = None
    else:
        report["curobo_import"] = getattr(curobo, "__version__", "imported")
    # This job does not construct a CuRobo world or execute a motion plan.  A
    # present package plus configuration is readiness evidence, not an oracle
    # result and never a task success rate.
    report["curobo_status"] = {
        "configured": bool(status.configured),
        "package_imported": report["curobo_import"] is not None,
        "ready_to_run": bool(status.configured and report["curobo_import"] is not None),
        "runtime_exercised": False,
        "oracle_complete": False,
        "reason": status.reason,
    }
    report["curobo_oracle"] = False

    Env = make_pruning_env_cls()
    env = Env(cfg=PruningEnvCfg(observation_variant="B_tof"))
    obs, _ = env.reset(seed=0)

    def tool_pose_b():
        tool_w = env._control_tool_pose_w()
        root_w = as_torch(env.robot.data.root_pose_w)
        tool_pos_b, tool_quat_b = subtract_frame_transforms(
            root_w[:, :3], root_w[:, 3:7], tool_w[:, :3], quaternion_wxyz_to_xyzw(tool_w[:, 3:7])
        )
        return pose_xyzw_to_wxyz(torch.cat((tool_pos_b, tool_quat_b), dim=-1))

    # Hold the actual control tool long enough to consume at least one complete
    # 15 Hz sensor interval.  Reset buffers start invalid on purpose.
    hold = tool_pose_b().clone()
    warmup_steps = max(6, int(env.cfg.tof0_cfg.update_period / env.step_dt) + 2)
    for _ in range(warmup_steps):
        obs, reward, terminated, truncated, extras = env.step(hold)
        warmup_reward = float(reward.mean().item())
        if not math.isfinite(warmup_reward):
            raise FloatingPointError(f"Warm-up reward is non-finite: {warmup_reward!r}.")
    tof_before = env.tof_state()
    _assert_json_finite(tof_before, "tof_before")
    baseline_start_pose = tool_pose_b().clone()
    if not bool(torch.isfinite(baseline_start_pose).all().item()):
        raise FloatingPointError("Control-tool pose is non-finite after ToF warm-up.")

    requested_steps = int(os.environ.get("BASELINE_STEPS", "32"))
    if requested_steps < 1:
        raise ValueError(f"BASELINE_STEPS must be positive, got {requested_steps}.")
    # A short caller override must not weaken the fresh-frame gate.
    scripted_steps = max(requested_steps, warmup_steps)
    rewards = []
    max_command_translation = torch.zeros(env.num_envs, device=env.device)
    max_tool_translation = torch.zeros(env.num_envs, device=env.device)
    terminated_any = False
    truncated_any = False
    for _ in range(scripted_steps):
        pose = tool_pose_b()
        if not bool(torch.isfinite(pose).all().item()):
            raise FloatingPointError("Control-tool pose became non-finite.")
        delta = scripted_tof_action(env.tof0, env.tof1, env.tof0_valid, env.tof1_valid)
        if not bool(torch.isfinite(delta).all().item()):
            raise FloatingPointError("Scripted ToF command delta is non-finite.")
        max_command_translation = torch.maximum(max_command_translation, torch.linalg.vector_norm(delta[:, :3], dim=-1))
        command = scripted_absolute_pose(pose, delta)
        if not bool(torch.isfinite(command).all().item()):
            raise FloatingPointError("Scripted absolute tool command is non-finite.")
        next_obs, reward, terminated, truncated, extras = env.step(command)
        reward_value = float(reward.mean().item())
        if not math.isfinite(reward_value):
            raise FloatingPointError(f"Scripted reward is non-finite: {reward_value!r}.")
        rewards.append(reward_value)
        obs = next_obs
        current_pose = tool_pose_b()
        if not bool(torch.isfinite(current_pose).all().item()):
            raise FloatingPointError("Control-tool response became non-finite.")
        max_tool_translation = torch.maximum(
            max_tool_translation,
            torch.linalg.vector_norm(current_pose[:, :3] - baseline_start_pose[:, :3], dim=-1),
        )

        terminated_any |= bool(terminated.any().item())
        truncated_any |= bool(truncated.any().item())

    tof_after = env.tof_state()
    _assert_json_finite(tof_after, "tof_after")
    if not bool(torch.isfinite(max_command_translation).all().item()):
        raise FloatingPointError("Maximum scripted command translation is non-finite.")
    if not bool(torch.isfinite(max_tool_translation).all().item()):
        raise FloatingPointError("Maximum measured tool translation is non-finite.")
    reward_mean = sum(rewards) / len(rewards)
    if not math.isfinite(reward_mean):
        raise FloatingPointError(f"Mean scripted reward is non-finite: {reward_mean!r}.")
    contact = env.contact_state()
    _assert_json_finite(contact, "contact")
    report["tof_before"] = tof_before
    report["tof_after"] = tof_after
    report["scripted_steps_requested"] = requested_steps
    report["scripted_steps"] = scripted_steps
    report["warmup_steps"] = warmup_steps
    report["scripted_reward_mean"] = reward_mean
    report["terminated_any"] = terminated_any
    report["truncated_any"] = truncated_any
    report["scripted_motion"] = {
        "max_command_translation_m": max_command_translation.detach().cpu().tolist(),
        "max_tool_translation_m": max_tool_translation.detach().cpu().tolist(),
    }
    report["contact"] = contact
    _flush()

    runtime_gate = validate_scripted_tof_runtime(
        tof_before,
        tof_after,
        expected_num_envs=env.num_envs,
        expected_hw=tuple(env.cfg.tof_hw),
        max_command_translation_m=report["scripted_motion"]["max_command_translation_m"],
        max_tool_translation_m=report["scripted_motion"]["max_tool_translation_m"],
    )
    report["scripted_runtime_gate"] = runtime_gate
    report["scripted_tof"] = True
    report["ok"] = bool(report["scripted_tof"] and report["contact"]["finite"])
    # Persist the application gate before cleanup.  Kit may terminate the
    # interpreter from close(), so writing afterward can lose both successes
    # and caught tracebacks while leaving a misleading process exit code.
    _flush()
    print(json.dumps(report, indent=2, default=str, allow_nan=False), flush=True)
    env.close()
except Exception as error:  # noqa: BLE001
    report["ok"] = False
    report["error"] = repr(error)
    report["traceback"] = traceback.format_exc()
    _flush()
    print(json.dumps(report, indent=2, default=str, allow_nan=False), flush=True)
    with contextlib.suppress(Exception):
        env.close()
    with contextlib.suppress(Exception):
        simulation_app.close()
    raise SystemExit(1)

simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
