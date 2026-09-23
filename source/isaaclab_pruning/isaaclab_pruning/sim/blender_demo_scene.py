"""Original Blender orchard geometry with an explicit simulated-detachment piece.

No Blender runtime is required here. USD imports stay lazy so topology and target
selection can be tested on ordinary CPU installations. Jaw meshes are a visual
surrogate, not an actuated or calibrated representation of the physical pruner.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TREE_PATH = "/World/envs/env_0/Tree"
SECOND_TREE_PATH = "/World/envs/env_0/Tree1"
ORCHARD_PATH = "/World/Orchard"
PROXY_PATH = "/World/PruningJawProxy"


def gap_aligned_camera_mount(closing_axis_tool, radial_distance_m=0.14, z_offset_m=-0.025):
    """Fixed virtual camera outside the tool, looking between rather than across jaws.

    This uses the already-selected proxy attachment orientation once. It does
    not move with image targets or claim physical-camera calibration.
    """
    axis = np.asarray(closing_axis_tool, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(axis[2]) > 1e-6 or np.linalg.norm(axis) < 1e-8:
        raise ValueError("Closing axis must be finite, nonzero and in the tool XY plane")
    if not math.isfinite(radial_distance_m) or radial_distance_m <= 0 or not math.isfinite(z_offset_m):
        raise ValueError("Camera offsets must be finite and radial distance positive")
    lateral = np.cross([0, 0, 1], axis)
    lateral /= np.linalg.norm(lateral)
    if lateral[1] > 0 or (abs(lateral[1]) < 1e-8 and lateral[0] > 0):
        lateral *= -1
    offset = lateral * radial_distance_m
    offset[2] = z_offset_m
    return tuple(float(value) for value in offset)


@dataclass(frozen=True)
class FacePartition:
    point_indices: tuple[int, ...]
    face_indices: tuple[int, ...]
    corner_indices: tuple[int, ...]
    counts: tuple[int, ...]
    indices: tuple[int, ...]


def partition_faces(counts, indices, component_vertices) -> tuple[FacePartition, FacePartition]:
    """Partition complete connected components without duplicating source faces."""
    counts = tuple(int(value) for value in counts)
    indices = tuple(int(value) for value in indices)
    component = {int(value) for value in component_vertices}
    if not component or any(value < 3 for value in counts) or sum(counts) != len(indices):
        raise ValueError("Expected nonempty component and complete polygon topology")
    if any(value < 0 for value in indices) or not component.issubset(indices):
        raise ValueError("Component contains invalid or unused vertex indices")
    groups = [[], []]
    offset = 0
    for face_id, count in enumerate(counts):
        vertices = indices[offset : offset + count]
        member = [index in component for index in vertices]
        if any(member) and not all(member):
            raise ValueError("Selected vertices cross a polygon; not a connected component")
        groups[0 if all(member) else 1].append((face_id, tuple(range(offset, offset + count)), vertices))
        offset += count
    if not all(groups):
        raise ValueError("Selected component must leave both selected and remaining faces")
    results = []
    for faces in groups:
        points = tuple(sorted({index for _, _, vertices in faces for index in vertices}))
        remap = {old: new for new, old in enumerate(points)}
        results.append(
            FacePartition(
                points,
                tuple(face_id for face_id, _, _ in faces),
                tuple(index for _, corners, _ in faces for index in corners),
                tuple(len(vertices) for _, _, vertices in faces),
                tuple(remap[index] for _, _, vertices in faces for index in vertices),
            )
        )
    return tuple(results)


def select_component(
    manifest: dict, first_vertex: int = 7524, max_radius_m: float = 0.012, tree_index: int = 0
) -> dict:
    if type(tree_index) is not int or tree_index not in (0, 1):
        raise ValueError("tree_index must be 0 or 1")
    candidates = manifest["branch_geometry_candidates"][f"tree{tree_index}_SPUR"]["candidates"]
    matches = [item for item in candidates if item["component_first_vertex"] == first_vertex]
    if len(matches) != 1:
        raise ValueError(f"Expected one source spur component {first_vertex}")
    candidate = matches[0]
    radius = float(candidate["max_radius_m"])
    if not math.isfinite(radius) or not 0 < radius <= max_radius_m:
        raise ValueError("Source component exceeds the demo jaw geometry")
    for name in ("center_m", "axis"):
        value = np.asarray(candidate[name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid component {name}")
    if np.linalg.norm(candidate["axis"]) < 1e-10:
        raise ValueError("Component axis is degenerate")
    return candidate


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _filtered(values, interpolation, part: FacePartition):
    if interpolation == "constant":
        return list(values)
    selectors = {
        "vertex": part.point_indices,
        "varying": part.point_indices,
        "uniform": part.face_indices,
        "faceVarying": part.corner_indices,
    }
    if interpolation not in selectors:
        raise ValueError(f"Unsupported primvar interpolation: {interpolation}")
    return [values[index] for index in selectors[interpolation]]


def _split_mesh(stage, source_mesh, vertex_ids):
    from pxr import Gf, UsdGeom, UsdShade

    selected, remainder = partition_faces(
        source_mesh.GetFaceVertexCountsAttr().Get(), source_mesh.GetFaceVertexIndicesAttr().Get(), vertex_ids
    )
    source = source_mesh.GetPrim()
    points = source_mesh.GetPointsAttr().Get()
    center = np.asarray([list(points[index]) for index in selected.point_indices]).mean(axis=0)
    body = UsdGeom.Xform.Define(stage, source.GetParent().GetPath().AppendChild("SelectedSpur"))
    # The current Blender export authors its transform on the mesh's parent,
    # but preserve mesh-local transforms too. The selected body is its sibling,
    # so inheriting only the parent would otherwise displace the split piece.
    source_transform = UsdGeom.Xformable(source).GetLocalTransformation()
    if source_transform != Gf.Matrix4d(1):
        body.AddTransformOp().Set(source_transform)
    body.AddTranslateOp().Set(Gf.Vec3d(*center))
    body.SetResetXformStack(UsdGeom.Xformable(source).GetResetXformStack())
    piece = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("Mesh"))
    normals = source_mesh.GetNormalsAttr().Get()
    normal_interpolation = source_mesh.GetNormalsInterpolation()
    # Flatten indexed primvars first, then partition by their interpolation.
    primvars = [
        (var.GetPrimvarName(), var.GetTypeName(), var.GetInterpolation(), var.ComputeFlattened(), var.GetElementSize())
        for var in UsdGeom.PrimvarsAPI(source).GetPrimvars()
        if var.HasValue()
    ]
    for var_name, _, _, _, element_size in primvars:
        if element_size != 1:
            raise ValueError(f"Unsupported elementSize on source primvar {var_name}")
    for name in ("cornerIndices", "creaseIndices", "holeIndices"):
        if source.GetAttribute(name).HasAuthoredValue():
            raise ValueError(f"Cannot safely partition authored subdivision topology {name}")
    material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial()
    for mesh, part, translation in ((piece, selected, center), (source_mesh, remainder, np.zeros(3))):
        mesh_points = [Gf.Vec3f(*(np.asarray(points[index]) - translation)) for index in part.point_indices]
        mesh.CreatePointsAttr(mesh_points)
        mesh.CreateFaceVertexCountsAttr(list(part.counts))
        mesh.CreateFaceVertexIndicesAttr(list(part.indices))
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh_points))
        mesh.CreateSubdivisionSchemeAttr(source_mesh.GetSubdivisionSchemeAttr().Get())
        mesh.CreateDoubleSidedAttr(source_mesh.GetDoubleSidedAttr().Get())
        mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get())
        if normals:
            mesh.CreateNormalsAttr(_filtered(normals, normal_interpolation, part))
            mesh.SetNormalsInterpolation(normal_interpolation)
        for name, type_name, interpolation, values, _ in primvars:
            var = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(name, type_name, interpolation)
            var.Set(_filtered(values, interpolation, part))
            var.BlockIndices()
        if material:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return (
        body,
        piece,
        {
            "selected_faces": len(selected.counts),
            "remaining_faces": len(remainder.counts),
            "selected_vertices": len(selected.point_indices),
            "remaining_vertices": len(remainder.point_indices),
            "primvars_preserved": [str(item[0]) for item in primvars],
            "normal_interpolation": normal_interpolation,
            "source_faces_removed": True,
        },
    )


class BlenderDemoScene:
    """Runtime handles plus JSON evidence for the imported inspection fixture."""

    def __init__(self, stage, body, piece, target, evidence):
        self.stage = stage
        self.body = body
        self.piece = piece
        self.evidence = evidence
        self.target_position_w = tuple(target["position_w_m"])
        self.target_axis_w = tuple(target["axis_w"])
        self.target_radius_m = target["radius_m"]
        self.target_id = target["id"]
        self.selected_mesh_prim_path = str(piece.GetPath())
        self.detached = False
        self._physics_view = None
        self._body_view = None
        self._proxy_roll_rad = 0.0
        self._create_tool_proxy()

    def set_proxy_closing_axis_tool(self, axis_tool):
        """Set a fixed attachment roll; this does not rotate the physical CAD tool."""
        axis = np.asarray(axis_tool, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(axis[2]) > 1e-6 or np.linalg.norm(axis) < 1e-8:
            raise ValueError("Proxy closing axis must be nonzero and in the tool XY plane")
        axis /= np.linalg.norm(axis)
        self._proxy_roll_rad = float(np.arctan2(axis[1], axis[0]))
        self.evidence["visual_proxy_closing_axis_tool"] = axis.tolist()
        self.evidence["visual_proxy_roll_rad"] = self._proxy_roll_rad

    def initialize_physics_tracking(self, tensors=None):
        """Bind only after physics startup; read PhysX instead of stale USD/Fabric."""
        if tensors is None:
            import omni.physics.tensors as tensors

        self._physics_view = tensors.create_simulation_view("torch")
        self._physics_view.set_subspace_roots("/")
        self._body_view = self._physics_view.create_rigid_body_view(str(self.body.GetPath()))
        if self._body_view.count != 1:
            raise RuntimeError("Expected exactly one selected-spur PhysX rigid body")
        self.evidence["piece_pose_source"] = "PhysX rigid-body tensor view, world xyz + wxyz"
        return self.measured_piece_pose_wxyz()

    def measured_piece_pose_wxyz(self):
        if self._body_view is None:
            raise RuntimeError("Initialize physics tracking after scene startup first")
        raw = self._body_view.get_transforms()
        poses = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw)
        if poses.shape != (1, 7) or not np.isfinite(poses).all():
            raise RuntimeError("Selected spur PhysX pose is missing or nonfinite")
        # Physics tensors use xyzw. Public evidence follows the task's wxyz.
        return poses[0, [0, 1, 2, 6, 3, 4, 5]].astype(float).tolist()

    def _create_tool_proxy(self):
        from pxr import UsdGeom

        proxy = UsdGeom.Xform.Define(self.stage, PROXY_PATH)
        self._proxy_position = proxy.AddTranslateOp()
        self._proxy_rotation = proxy.AddOrientOp()
        proxy.GetPrim().SetCustomDataByKey("pruning:role", "visual_jaw_surrogate_not_actuated_cad")
        self._jaw_translations = []
        for name in ("LeftJaw", "RightJaw"):
            mesh = UsdGeom.Cube.Define(self.stage, f"{PROXY_PATH}/{name}")
            mesh.CreateSizeAttr(1)
            self._jaw_translations.append(mesh.AddTranslateOp())
            mesh.AddScaleOp().Set((0.006, 0.025, 0.030))
            mesh.CreateDisplayColorAttr([(0.42, 0.47, 0.53)])
        self.update_tool_proxy((0, 0, -10, 1, 0, 0, 0), 0)

    def update_tool_proxy(self, tool_pose_wxyz, closure_progress):
        from pxr import Gf

        pose = np.asarray(tool_pose_wxyz, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Tool pose must be finite xyz + wxyz")
        if not math.isfinite(closure_progress) or not 0 <= closure_progress <= 1:
            raise ValueError("Closure progress must be in [0,1]")
        norm = np.linalg.norm(pose[3:])
        if norm <= 1e-10:
            raise ValueError("Tool orientation is degenerate")
        quat = pose[3:] / norm
        self._proxy_position.Set(Gf.Vec3d(*pose[:3]))
        tool_quat = Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:]))
        roll_quat = Gf.Quatf(math.cos(self._proxy_roll_rad / 2), Gf.Vec3f(0, 0, math.sin(self._proxy_roll_rad / 2)))
        self._proxy_rotation.Set(tool_quat * roll_quat)
        gap = (1 - closure_progress) * 0.032 + closure_progress * 2 * self.target_radius_m
        for sign, op in zip((-1, 1), self._jaw_translations, strict=True):
            op.Set(Gf.Vec3d(sign * (gap + 0.006) / 2, 0, 0.070))
        return {"visual_gap_m": gap, "closure_progress": float(closure_progress), "model": "visual_surrogate"}

    def detach(self):
        """Release the existing kinematic piece to PhysX gravity once; never teleport."""
        from pxr import UsdPhysics

        if self.detached:
            return False
        body_api = UsdPhysics.RigidBodyAPI(self.body.GetPrim())
        body_api.GetKinematicEnabledAttr().Set(False)
        self.detached = True
        self.evidence["detachment_requested"] = True
        return True


def spawn_blender_demo_scene(
    stage,
    export_dir,
    target_position_w=(0.251, 0.634, 0.815),
    base_height=0.70,
    yaw_degrees=0,
    component_first_vertex=7524,
    tree_count=1,
    target_tree_index=0,
):
    """Import the original orchard and move one real spur into the robot workspace.

    ``base_height`` is recorded for the caller's mount; it does not deform or lift
    the terrain. Ground relief stays visual; the caller owns its physics plane.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if isinstance(tree_count, bool) or not isinstance(tree_count, int) or tree_count not in (1, 2):
        raise ValueError("tree_count must be 1 or 2")
    if type(target_tree_index) is not int or not 0 <= target_tree_index < tree_count:
        raise ValueError("Target tree must be among the imported distinct trees")
    object_name = f"tree{target_tree_index}_SPUR"
    export_dir = Path(export_dir).resolve()
    manifest = json.loads((export_dir / "manifest.json").read_text())
    listed = manifest["branch_geometry_candidates"][object_name]["candidates"]
    if any(item["component_first_vertex"] == component_first_vertex for item in listed):
        candidate = select_component(manifest, first_vertex=component_first_vertex, tree_index=target_tree_index)
    else:
        from isaaclab_pruning.sim.blender_component import component_from_export

        if target_tree_index != 0:
            raise ValueError("Unlisted tree1 components require a new geometry audit")
        measured = component_from_export(export_dir, component_first_vertex)
        candidate = select_component(
            {"branch_geometry_candidates": {"tree0_SPUR": {"candidates": [measured]}}},
            first_vertex=component_first_vertex,
        )
    target = np.asarray(target_position_w, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or not math.isfinite(yaw_degrees):
        raise ValueError("Target and orchard yaw must be finite")
    if manifest.get("units") != "meters" or manifest.get("up_axis") != "Z":
        raise ValueError("Export must explicitly use meters and Z-up")
    files = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    tree_sources = [(TREE_PATH, "tree0.usdc")]
    if tree_count == 2:
        if manifest.get("tree_count") != 2:
            raise ValueError("Two-tree mode requires a verified two-tree export, not a duplicated tree0")
        tree_sources.append((SECOND_TREE_PATH, "tree1.usdc"))
    sources = [*tree_sources, (ORCHARD_PATH, "environment.usdc")]
    for _, name in sources:
        if _hash(export_dir / name) != files[name]:
            raise ValueError(f"Export provenance mismatch: {name}")
    angle = math.radians(yaw_degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    translation = target - rotation @ np.asarray(candidate["center_m"])
    for path, filename in sources:
        if stage.GetPrimAtPath(path):
            raise ValueError(f"Refusing to replace existing scene prim {path}")
        prim = UsdGeom.Xform.Define(stage, path)
        prim.GetPrim().GetReferences().AddReference(str(export_dir / filename))
        prim.AddTranslateOp().Set(Gf.Vec3d(*translation))
        prim.AddRotateZOp().Set(float(yaw_degrees))
    target_tree_path = TREE_PATH if target_tree_index == 0 else SECOND_TREE_PATH
    matches = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target_tree_path))
        if prim.IsA(UsdGeom.Mesh) and prim.GetParent().GetName() == object_name
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one selected original spur mesh")
    source_mesh_path = str(matches[0].GetPath())
    source_mesh = UsdGeom.Mesh(matches[0])
    if not source_mesh:
        raise ValueError("Original spur mesh prim is missing")
    body, piece, partition_evidence = _split_mesh(stage, source_mesh, candidate["source_vertex_indices"])
    piece_transform = UsdGeom.XformCache().GetLocalToWorldTransform(piece.GetPrim())
    actual_center = np.asarray(
        [piece_transform.Transform(Gf.Vec3d(point)) for point in piece.GetPointsAttr().Get()]
    ).mean(axis=0)
    center_error = float(np.linalg.norm(actual_center - target))
    if center_error > 1e-5:
        raise ValueError(f"Selected spur does not match requested world target ({center_error:.6g} m error)")
    partition_evidence["actual_center_w_m"] = actual_center.tolist()
    partition_evidence["target_center_error_m"] = center_error
    selected_path = str(piece.GetPath())
    collision_paths = []
    for path, _ in sources:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()).startswith(f"{ORCHARD_PATH}/ground/"):
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            approximation = "convexHull" if str(prim.GetPath()) == selected_path else "none"
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
            collision_paths.append(str(prim.GetPath()))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateVelocityAttr(Gf.Vec3f(0))
    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.005)
    for prim in (body.GetPrim(), piece.GetPrim()):
        prim.SetCustomDataByKey("pruning:semantic", "spur")
        prim.SetCustomDataByKey("pruning:identity_source", "selected_original_blender_component")
    axis = rotation @ np.asarray(candidate["axis"])
    axis /= np.linalg.norm(axis)
    target_info = {
        "id": f"{object_name}_component_{component_first_vertex}",
        "position_w_m": target.tolist(),
        "axis_w": axis.tolist(),
        "radius_m": candidate["max_radius_m"],
        "source_vertex_ids": candidate["source_vertex_indices"],
        "source_component": candidate,
        "semantic": "spur",
        "identity_source": "known_scene_component_not_learned_classifier",
    }
    evidence = {
        "export_dir": str(export_dir),
        "manifest_sha256": _hash(export_dir / "manifest.json"),
        "source_blend_sha256": manifest["source_sha256"],
        "usd_sha256": {filename: files[filename] for _, filename in sources},
        "translation_w_m": translation.tolist(),
        "yaw_degrees": float(yaw_degrees),
        "robot_base_height_m": float(base_height),
        "tree_path": target_tree_path,
        "target_tree_index": target_tree_index,
        "tree_count": tree_count,
        "tree_paths": [path for path, _ in tree_sources],
        "tree_layout": "Distinct original source trees; shared rigid transform preserves relative placement",
        "orchard_path": ORCHARD_PATH,
        "source_mesh_path": source_mesh_path,
        "selected_mesh_path": selected_path,
        "target": target_info,
        "partition": partition_evidence,
        "collision_mesh_paths": collision_paths,
        "detachment_requested": False,
        "limitations": [
            "Jaw meshes are a visual surrogate at tool-local +Z 0.070 m, not actuated pruner CAD.",
            "Detachment switches an existing 5 g rigid piece from kinematic to dynamic; no wood fracture is modeled.",
            "Target identity and initialization use known Blender mesh topology, not a learned classifier.",
            "Original terrain relief is visual only; the caller's flat plane remains the ground collider.",
            "The original Blender procedural sky is not reproduced by this USD export.",
        ],
    }
    return BlenderDemoScene(stage, body, piece, target_info, evidence)
