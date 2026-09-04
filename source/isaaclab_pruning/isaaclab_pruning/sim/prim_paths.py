"""Globally rooted prim expressions for the manually built DirectRLEnv scene.

``{ENV_REGEX_NS}`` is expanded by ``InteractiveScene`` when assets are declared
on its configuration.  The pruning ``DirectRLEnv`` constructs assets directly
inside ``_setup_scene``, so Isaac Lab 3 requires the equivalent absolute regex.
"""

DIRECT_ENV_NAMESPACE_EXPR = "/World/envs/env_.*"
ROBOT_PRIM_EXPR = f"{DIRECT_ENV_NAMESPACE_EXPR}/Robot"
TREE_PRIM_EXPR = f"{DIRECT_ENV_NAMESPACE_EXPR}/Tree"
TOF_SMOKE_TARGET_PRIM_EXPR = f"{DIRECT_ENV_NAMESPACE_EXPR}/ToFSmokeTarget"


__all__ = [
    "DIRECT_ENV_NAMESPACE_EXPR",
    "ROBOT_PRIM_EXPR",
    "TOF_SMOKE_TARGET_PRIM_EXPR",
    "TREE_PRIM_EXPR",
]
