"""Isaac Lab 3.x shims copied from the BHL v60 port (Jose, 2026-08).

5.1 RTX segfaults here. The working renderer is Isaac Sim 6.0 + Lab 3.0.0b2,
which is warp-first. Lab's own torch.jit math rejects ``ProxyArray``. Call
``apply()`` before constructing a DirectRLEnv on v60. No-op on 2.x.
See ``docs/ISAAC_STACK.md``.

Timebox: three surface patches (noise alias, physx→physics, ProxyArray unwrap).
A fourth is the signal to subclass Lab 3's DirectRLEnv instead of accumulating
compat debt. 3.0.0b2 will move under this tree.
"""

from __future__ import annotations

from typing import Any


def apply() -> list[str]:
    """Install Lab 3 shims. Returns the names of the ones that were needed."""
    applied: list[str] = []
    try:
        import isaaclab.utils.noise as noise
    except ImportError:
        return applied

    if not hasattr(noise, "AdditiveUniformNoiseCfg"):
        noise.AdditiveUniformNoiseCfg = noise.UniformNoiseCfg
        applied.append("AdditiveUniformNoiseCfg -> UniformNoiseCfg")
    if not hasattr(noise, "AdditiveGaussianNoiseCfg") and hasattr(noise, "GaussianNoiseCfg"):
        noise.AdditiveGaussianNoiseCfg = noise.GaussianNoiseCfg
        applied.append("AdditiveGaussianNoiseCfg -> GaussianNoiseCfg")

    try:
        from isaaclab.sim.simulation_cfg import SimulationCfg
    except Exception:  # pragma: no cover
        SimulationCfg = None

    if SimulationCfg is not None and not hasattr(SimulationCfg, "physx"):
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        def _get_physx(self):
            current = getattr(self, "physics", None)
            if not isinstance(current, PhysxCfg):
                current = PhysxCfg()
                self.physics = current
            return current

        def _set_physx(self, value):
            self.physics = value

        SimulationCfg.physx = property(_get_physx, _set_physx)
        applied.append("SimulationCfg.physx -> .physics (PhysxCfg)")

    try:
        from isaaclab.utils.warp.proxy_array import ProxyArray
    except Exception:  # pragma: no cover
        ProxyArray = None

    if ProxyArray is not None:
        import functools

        import isaaclab.utils.math as math_utils

        def _unwrap(value: Any) -> Any:
            return value.torch if isinstance(value, ProxyArray) else value

        def _wrap(fn):
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                return fn(*[_unwrap(arg) for arg in args], **{key: _unwrap(val) for key, val in kwargs.items()})

            return inner

        wrapped = 0
        for name in dir(math_utils):
            fn = getattr(math_utils, name, None)
            if name.startswith("_") or not callable(fn):
                continue
            if type(fn).__name__ != "ScriptFunction":
                continue
            setattr(math_utils, name, _wrap(fn))
            wrapped += 1
        if wrapped:
            applied.append(f"isaaclab.utils.math: unwrap ProxyArray on {wrapped} scripted fns")
    return applied


def as_torch(value: Any) -> Any:
    """Zero-copy ``ProxyArray`` → tensor. Identity for everything else."""
    tensor = getattr(value, "torch", None)
    if tensor is not None and type(value).__name__ == "ProxyArray":
        return tensor
    return value
