#!/usr/bin/env python3
"""Extract the shadow casters of one recorded Isaac run into a private geometry cache.

Reads the run's composed ``scene.usda`` (run with the Isaac venv python, which
has ``pxr``) and writes ``shadow_casters.npz`` and ``shadow_casters.json``:
world-space triangles of both trees (selected spur included) and the orchard
(posts, wires, ground); the UR5e render meshes in their link frames with the
six revolute joints for forward kinematics; the sun direction (+Z of the world
transform of the orchard DistantLight) and the dome intensity and colour.

The cache holds licensed tree meshes and mock-pruner CAD, so it must stay in a
scratch directory: the tool refuses a cache directory inside the repository or
inside any ``artifacts/`` directory. Output bytes are deterministic, and the
JSON records the sha256 of the scene it came from. The jaw proxy is not
extracted; the diagnosis rebuilds it from each frame's recorded tool pose.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE_STEM = "shadow_casters"
ENV = "/World/envs/env_0"
ORCHARD = "/World/Orchard"
LINKS = (
    "ur5e__base_link_inertia",
    "ur5e__shoulder_link",
    "ur5e__upper_arm_link",
    "ur5e__forearm_link",
    "ur5e__wrist_1_link",
    "ur5e__wrist_2_link",
    "ur5e__wrist_3_link",
)


def refuse_cache_dir(cache_dir, repo_root=REPO):
    """Return the resolved cache directory, or raise if it could leak licensed geometry."""
    path = Path(cache_dir).resolve()
    repo = Path(repo_root).resolve()
    if path == repo or repo in path.parents:
        raise ValueError(f"Refusing a cache directory inside the repository: {path}")
    if "artifacts" in path.parts:
        raise ValueError(f"Refusing a cache directory inside an artifacts/ directory: {path}")
    return path


def write_npz(path, arrays):
    """np.savez_compressed with fixed zip timestamps, so equal arrays give equal bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def extract(scene_path):
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

    stage = Usd.Stage.Open(str(scene_path))
    everything = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def world(prim):
        return np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T  # USD rows -> column vectors

    def triangles(prim):
        points = np.array(prim.GetAttribute("points").Get(), dtype=float)
        counts = prim.GetAttribute("faceVertexCounts").Get()
        indices = np.array(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int64)
        faces, start = [], 0
        for count in counts:
            faces += [(indices[start], indices[start + k], indices[start + k + 1]) for k in range(1, count - 1)]
            start += count
        return points, np.array(faces, dtype=np.int64).reshape(-1, 3)

    def meshes(root):
        for prim in Usd.PrimRange(root, everything):
            imageable = UsdGeom.Imageable(prim)
            if prim.GetTypeName() == "Mesh" and imageable.ComputeVisibility() != "invisible":
                if imageable.ComputePurpose() == "default":
                    yield prim

    groups, counts = {}, {"static": 0, "robot": 0}
    vertices, faces, labels = ({"static": [], "robot": []} for _ in range(3))

    def add(kind, group, local_vertices, local_faces, **info):
        index = groups.setdefault((kind, group), {"kind": kind, "name": group, **info} | {"index": len(groups)})
        vertices[kind].append(local_vertices)
        faces[kind].append(local_faces + counts[kind])
        labels[kind].append(np.full(len(local_faces), index["index"], dtype=np.int32))
        counts[kind] += len(local_vertices)

    trees = [child for child in stage.GetPrimAtPath(ENV).GetChildren() if child.GetName().startswith("Tree")]
    orchard = stage.GetPrimAtPath(ORCHARD)
    if len(trees) < 2 or not orchard.IsValid():
        raise ValueError("Scene lacks the two trees or the orchard")
    for root in [*trees, orchard]:
        is_tree = root in trees
        for prim in meshes(root):
            relative = str(prim.GetPath())[len(str(root.GetPath())) + 1 :]
            group = "selected_spur" if "SelectedSpur" in relative else re.sub(r"[\d_]+$", "", relative.split("/")[0])
            points, tris = triangles(prim)
            transform = world(prim)
            add("static", group, points @ transform[:3, :3].T + transform[:3, 3], tris, is_tree=is_tree)

    robot = stage.GetPrimAtPath(f"{ENV}/Robot")
    frames = {}
    for prim in Usd.PrimRange(robot, everything):
        if prim.GetName() in (*LINKS, "ur5e__tool0"):
            frames.setdefault(prim.GetName(), world(prim))
    for prim in meshes(robot):
        path = str(prim.GetPath())
        owner = [link for link in LINKS if f"/{link}/" in path][-1]
        group = "tool" if "/ur5e__tool0/" in path else owner.removeprefix("ur5e__")
        points, tris = triangles(prim)
        transform = np.linalg.inv(frames[owner]) @ world(prim)
        add("robot", group, points @ transform[:3, :3].T + transform[:3, 3], tris, link=owner)

    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            rotations = [joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()]
            joints.append(
                {
                    "name": prim.GetName(),
                    "axis": joint.GetAxisAttr().Get(),
                    "body0": joint.GetBody0Rel().GetTargets()[0].name,
                    "body1": joint.GetBody1Rel().GetTargets()[0].name,
                    "pos0": list(joint.GetLocalPos0Attr().Get()),
                    "pos1": list(joint.GetLocalPos1Attr().Get()),
                    "rot0_wxyz": [rotations[0].GetReal(), *rotations[0].GetImaginary()],
                    "rot1_wxyz": [rotations[1].GetReal(), *rotations[1].GetImaginary()],
                }
            )
    order = {link: k for k, link in enumerate(LINKS)}
    joints.sort(key=lambda joint: order.get(joint["body1"], len(LINKS)))
    if [joint["body1"] for joint in joints] != list(LINKS[1:]):
        raise ValueError("Expected the six revolute joints of the UR5e chain")

    def lights(kind, root):
        return [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(root)) if prim.IsA(kind)]

    suns, domes = lights(UsdLux.DistantLight, ORCHARD), lights(UsdLux.DomeLight, "/World")
    if len(suns) != 1 or len(domes) != 1:
        raise ValueError(f"Expected one orchard sun and one dome, found {len(suns)} and {len(domes)}")
    sun, dome = suns[0], domes[0]
    towards_sun = world(sun)[:3, 2]
    arrays = {
        f"{kind}_{name}": np.concatenate(values).astype(dtype)
        for kind in ("static", "robot")
        for name, values, dtype in (
            ("vertices", vertices[kind], np.float64),
            ("faces", faces[kind], np.int64),
            ("groups", labels[kind], np.int32),
        )
    }
    meta = {
        "scene_sha256": hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),
        "groups": sorted(groups.values(), key=lambda group: group["index"]),
        "joints": joints,
        "link_world_default": {name: frames[name].tolist() for name in (LINKS[0], LINKS[-1], "ur5e__tool0")},
        "sun_path": str(sun.GetPath()),
        "sun_direction_to_sun_w": (towards_sun / np.linalg.norm(towards_sun)).tolist(),
        "sun_angle_deg": sun.GetAttribute("inputs:angle").Get(),
        "sun_intensity": sun.GetAttribute("inputs:intensity").Get(),
        "sun_color": list(sun.GetAttribute("inputs:color").Get()),
        "dome_path": str(dome.GetPath()),
        "dome_intensity": dome.GetAttribute("inputs:intensity").Get(),
        "dome_color": list(dome.GetAttribute("inputs:color").Get()),
    }
    return arrays, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path, help="Recorded run directory holding scene.usda")
    parser.add_argument("--cache-dir", required=True, type=Path, help="Scratch directory for the geometry cache")
    args = parser.parse_args()
    try:
        cache_dir = refuse_cache_dir(args.cache_dir)
    except ValueError as exc:
        parser.error(str(exc))
    arrays, meta = extract(args.run_dir / "scene.usda")
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_npz(cache_dir / f"{CACHE_STEM}.npz", arrays)
    (cache_dir / f"{CACHE_STEM}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps({"cache_dir": str(cache_dir), "scene_sha256": meta["scene_sha256"], "groups": len(meta["groups"])})
    )


if __name__ == "__main__":
    main()
