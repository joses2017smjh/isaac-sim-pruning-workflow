#!/usr/bin/env python3
"""The orchard's ground relief for the offline tools: where it is, and what is beneath it. No simulator import.

The renderer draws the exported orchard's ground relief and nothing collides with it. The simulated ToF sensors
cast against the trees only, and so does ``tools/known_map_planner.py``, whose export helper skips the ground on
purpose: its output hashes are part of registered evidence, so the ground is streamed here separately instead.

The relief is an open sheet, not a closed solid, so a clearance or signed-distance test cannot tell that a robot
which starts under it is under it. ``beneath`` uses a vertical ray instead: a point is beneath the surface when
the ground is above it and there is no ground below it.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np

GROUND = "GROUND"
UP_W = np.array([0.0, 0.0, 1.0])
MARKER = "@@ORCHARD_GROUND@@"

_GROUND_HELPER = r"""
import io, json, sys
import numpy as np
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1] + "/environment.usdc")
cache = UsdGeom.XformCache()
vertices, triangles, offset = [], [], 0
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh) or "/ground/" not in str(prim.GetPath()):
        continue
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    M = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    vertices.append((np.c_[pts, np.ones(len(pts))] @ M)[:, :3])
    off = 0
    for c in counts:
        face = idx[off:off + c]
        for k in range(1, c - 1):
            triangles.append((face[0] + offset, face[k] + offset, face[k + 1] + offset))
        off += c
    offset += len(pts)
buffer = io.BytesIO()
np.savez(buffer, v=np.concatenate(vertices), t=np.asarray(triangles, dtype=np.int64))
blob = buffer.getvalue()
sys.stdout.buffer.write(sys.argv[2].encode() + b"\n" + json.dumps({"npz_bytes": len(blob)}).encode() + b"\n" + blob)
"""


def stream_ground(export_dir, isaac_python):
    """``(vertices, triangles)`` of the ground relief in the export's frame, fan triangulated, in memory only.

    Call after ``known_map_planner.stream_export`` on the same directory: that call verifies environment.usdc
    against the export manifest."""
    export_dir = Path(export_dir).resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [str(isaac_python), "-c", _GROUND_HELPER, str(export_dir), MARKER], capture_output=True, env=env, timeout=3600
    )
    head = MARKER.encode() + b"\n"
    if done.returncode != 0 or head not in done.stdout:
        raise RuntimeError(
            f"ground streaming failed ({done.returncode}): {done.stderr.decode(errors='replace')[-600:]}"
        )
    line, _, blob = done.stdout[done.stdout.index(head) + len(head) :].partition(b"\n")
    if len(blob) != json.loads(line)["npz_bytes"]:
        raise RuntimeError("streamed ground geometry is truncated")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return np.asarray(data["v"], float), np.asarray(data["t"], np.int64)


def placed(ground, translation, rotation):
    """The ground relief with the orchard's rigid transform, as the renderer places it with the trees."""
    vertices, triangles = ground
    return np.asarray(vertices, float) @ np.asarray(rotation, float).T + np.asarray(translation, float), triangles


def vertical_gaps(index, points):
    """``(above_m, below_m)``: distance from each point up and down to the ground; ``inf`` where there is none.

    ``index`` is a ray index over the placed ground alone (``cast(origins, directions)`` returning distances
    first), as ``known_map_planner.Open3dMeshIndex`` and ``NumpyMeshIndex`` are."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    up = np.repeat(UP_W[None], len(points), 0)
    above = np.asarray(index.cast(points, up)[0], dtype=float)
    below = np.asarray(index.cast(points, -up)[0], dtype=float)
    return np.where(np.isfinite(above), above, np.inf), np.where(np.isfinite(below), below, np.inf)


def beneath(index, points):
    """One flag per point: the ground is above it and not below it."""
    above, below = vertical_gaps(index, points)
    return np.isfinite(above) & ~np.isfinite(below)
