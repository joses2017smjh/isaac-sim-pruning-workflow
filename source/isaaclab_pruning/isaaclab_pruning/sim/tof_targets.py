"""Which meshes the simulated ToF sensors cast against in the orchard scene.

The sensors cast against a list of mesh prims, not against the rendered scene. Until November 25, 2026 that list
was the orchard's collision meshes (trees, posts and wires) and the selected spur's body. The orchard's ground
relief was never in it: the relief is drawn and nothing collides with it, so the sensors could not see a ground
surface that the camera could. An audit found recorded runs in which the robot was beneath that surface
(docs/GROUND_AUDIT_2026-11-25.md).

``PRUNING_TOF_GROUND=1`` adds the ground relief to the list. It is off by default, so a run without it casts
against exactly what every earlier recording did. As of November 25, 2026 the switch has unit tests only: no
simulator run has used it, and what it would change in the recordings was measured offline
(tools/replay_tof_ground.py).
"""

from __future__ import annotations

TOF_GROUND_ENV = "PRUNING_TOF_GROUND"


def tof_ground_requested(environ):
    """True when the environment asks for the ground relief in the ToF mesh set."""
    value = environ.get(TOF_GROUND_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{TOF_GROUND_ENV} must be 0 or 1, not {value!r}")
    return value == "1"


def tof_ray_targets(collision_mesh_paths, selected_mesh_path, selected_body_target, ground_mesh_paths=()):
    """The ray-cast targets of both ToF sensors, in a fixed order.

    The static collision meshes without the selected spur's own mesh, then the ground relief's meshes if any, then
    the selected spur as a tracked body (it moves when it is cut). With no ground meshes this is the list every
    recording before November 25, 2026 used."""
    ground = [str(path) for path in ground_mesh_paths]
    if len(set(ground)) != len(ground):
        raise ValueError("ground mesh paths must be distinct")
    static = [path for path in collision_mesh_paths if path != selected_mesh_path]
    if set(ground) & set(static):
        raise ValueError("a ground mesh is already among the collision meshes")
    return [*static, *ground, selected_body_target]
