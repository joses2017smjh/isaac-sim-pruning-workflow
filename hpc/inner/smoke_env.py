"""Batched v60 env smoke: trainer import, A–D obs contract, one step, PhysX contact.

One job. Asserts, not log archaeology. Writes docs/evidence/smoke_<jobid>.json.
"""
from __future__ import annotations

import contextlib
import importlib
import json
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
}


def _write(code: int) -> None:
    out = Path(os.environ.get("BENCH_OUT", "/tmp/pruning_env_smoke.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=str)
    print(text, flush=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(code)


if report["skrl"] is None and report["rsl_rl"] is None:
    report["error"] = "No RL library on this interpreter (skrl and rsl_rl both missing)."
    _write(1)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    pruning_root = Path(os.environ["PRUNING_ROOT"])
    sys.path.insert(0, str(pruning_root / "source" / "isaaclab_pruning"))

    from isaaclab_pruning.policies.observations import ObservationVariant, observation_width
    from isaaclab_pruning.sim.lab3_compat import apply as apply_lab3
    from isaaclab_pruning.sim.pruning_env import make_pruning_env_cls
    from isaaclab_pruning.sim.pruning_env_cfg import PruningEnvCfg

    report["lab3_compat"] = apply_lab3()

    cfgs = {}
    spaces = {}
    for variant in ObservationVariant:
        cfg = PruningEnvCfg(observation_variant=variant.value)
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
    report["observation_space"] = spaces
    assert spaces["A_flow"] != spaces["B_tof"] != spaces["C_metric"]
    assert spaces["A_flow"] != spaces["C_metric"]
    assert spaces["C_metric"] == spaces["D_fused"]

    Env = make_pruning_env_cls()
    env = Env(cfg=cfgs["B_tof"])
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

    pose = env.robot.data.body_pose_w
    pose = pose.torch if type(pose).__name__ == "ProxyArray" else pose
    hold = pose[:, env.robot_entity_cfg.body_ids[0], 0:7].contiguous()
    next_obs, rewards, terminated, truncated, extras = env.step(hold)
    contact = env.contact_state()
    assert contact["finite"]
    report["step"] = {
        "reward": float(rewards.mean().item()),
        "terminated": bool(terminated.any().item()),
        "truncated": bool(truncated.any().item()),
        "obs_last_dim": int(next_obs["policy"].shape[-1]),
    }
    report["contact"] = contact
    names = list(getattr(env.robot.data, "joint_names", []) or getattr(env.robot, "joint_names", []))
    report["joint_names"] = [str(name) for name in names]
    report["n_arm_joints"] = len(env.robot_entity_cfg.joint_ids)
    report["slider_in_joint_names"] = any("linear_slider" in str(name) for name in names)
    assert report["n_arm_joints"] == 6
    assert not report["slider_in_joint_names"]
    report["ok"] = True
    env.close()
except Exception as error:  # noqa: BLE001 — always emit JSON
    report["error"] = repr(error)
    report["traceback"] = traceback.format_exc()
    with contextlib.suppress(Exception):
        simulation_app.close()
    _write(1)

simulation_app.close()
_write(0 if report.get("ok") else 1)
