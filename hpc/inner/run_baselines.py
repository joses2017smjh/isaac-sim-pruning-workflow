"""Run scripted ToF in the pruning env and record CuRobo configuration status.

This is the Isaac baseline job. It does not start PPO. CuRobo motion planning
is recorded as configured-or-not; a missing CuRobo wheel is not a fake ~60%.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

report: dict = {
    "ok": False,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    "bhl_stack": os.environ.get("BHL_STACK"),
    "scripted_tof": False,
    "curobo_oracle": False,
}


def _bench_path() -> Path:
    return Path(os.environ.get("BENCH_OUT", "/tmp/pruning_baselines.json"))


def _flush() -> None:
    out = _bench_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def _write(code: int) -> None:
    print(json.dumps(report, indent=2, default=str), flush=True)
    _flush()
    raise SystemExit(code)


_flush()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    pruning_root = Path(os.environ["PRUNING_ROOT"])
    sys.path.insert(0, str(pruning_root / "source" / "isaaclab_pruning"))

    from isaaclab_pruning.baselines.curobo import ur5e_pruner_oracle_status
    from isaaclab_pruning.baselines.tof_servo import scripted_absolute_pose, scripted_tof_action
    from isaaclab_pruning.robot import imported_usd_path
    from isaaclab_pruning.sim.lab3_compat import apply as apply_lab3
    from isaaclab_pruning.sim.lab3_compat import as_torch
    from isaaclab_pruning.sim.pruning_env import make_pruning_env_cls
    from isaaclab_pruning.sim.pruning_env_cfg import PruningEnvCfg

    report["lab3_compat"] = apply_lab3()
    usd = imported_usd_path()
    status = ur5e_pruner_oracle_status(urdf_usd_path=str(usd) if usd.is_file() else None)
    report["curobo_status"] = {"configured": status.configured, "reason": status.reason}
    try:
        import curobo  # noqa: F401
    except ImportError:
        report["curobo_import"] = None
    else:
        report["curobo_import"] = getattr(curobo, "__version__", "imported")

    Env = make_pruning_env_cls()
    env = Env(cfg=PruningEnvCfg(observation_variant="B_tof"))
    obs, _ = env.reset(seed=0)
    rewards = []
    for _ in range(int(os.environ.get("BASELINE_STEPS", "32"))):
        pose = as_torch(env.robot.data.body_pose_w)[:, env.robot_entity_cfg.body_ids[0], 0:7]
        delta = scripted_tof_action(env.tof0, env.tof1, env.tof0_valid, env.tof1_valid)
        command = scripted_absolute_pose(pose, delta)
        next_obs, reward, terminated, truncated, extras = env.step(command)
        rewards.append(float(reward.mean().item()))
        obs = next_obs
    report["scripted_tof"] = True
    report["scripted_steps"] = len(rewards)
    report["scripted_reward_mean"] = sum(rewards) / max(len(rewards), 1)
    report["contact"] = env.contact_state()
    report["curobo_oracle"] = bool(status.configured and report.get("curobo_import"))
    report["ok"] = bool(report["scripted_tof"] and report["contact"]["finite"])
    env.close()
except Exception as error:  # noqa: BLE001
    report["error"] = repr(error)
    report["traceback"] = traceback.format_exc()
    _write(1)

simulation_app.close()
_write(0 if report.get("ok") else 1)
