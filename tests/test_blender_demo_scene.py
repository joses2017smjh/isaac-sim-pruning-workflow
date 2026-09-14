"""Topology and fail-closed target-selection checks without Isaac or Blender."""

from pathlib import Path

import numpy as np
import pytest


def test_piece_pose_reads_physics_xyzw_not_usd_and_validates_shape():
    from isaaclab_pruning.sim.blender_demo_scene import BlenderDemoScene

    class BodyView:
        count = 1

        def get_transforms(self):
            return np.array([[1, 2, 3, 0.1, 0.2, 0.3, 0.9]])

    scene = BlenderDemoScene.__new__(BlenderDemoScene)
    scene._body_view = None
    with pytest.raises(RuntimeError, match="Initialize physics tracking"):
        scene.measured_piece_pose_wxyz()
    scene._body_view = BodyView()
    assert scene.measured_piece_pose_wxyz() == pytest.approx([1, 2, 3, 0.9, 0.1, 0.2, 0.3])
    scene._body_view.get_transforms = lambda: np.array([[1, 2, float("nan"), 0, 0, 0, 1]])
    with pytest.raises(RuntimeError, match="nonfinite"):
        scene.measured_piece_pose_wxyz()


from isaaclab_pruning.sim.blender_demo_scene import partition_faces, select_component


@pytest.mark.parametrize("axis", [[1, 0, 0], [-0.856, 0.517, 0], [0.856, -0.517, 0], [0, 1, 0]])
def test_gap_aligned_camera_is_outside_tool_and_between_proxy_jaws(axis):
    from isaaclab_pruning.sim.blender_demo_scene import gap_aligned_camera_mount

    mount = np.asarray(gap_aligned_camera_mount(axis))
    assert np.linalg.norm(mount[:2]) == pytest.approx(0.14)
    assert mount[2] == -0.025
    assert mount[1] <= 0
    # Every point on a ray to the centered mouth stays at closing-axis zero,
    # inside even the fully closed 13.6 mm gap, rather than crossing a jaw.
    direction = np.asarray(axis) / np.linalg.norm(axis)
    for fraction in np.linspace(0, 1, 20):
        point = mount + fraction * (np.array([0, 0, 0.07]) - mount)
        assert abs(np.dot(point, direction)) < 1e-8


@pytest.mark.parametrize("axis", [[0, 0, 0], [1, 0, 0.1], [float("nan"), 0, 0], [1, 2]])
def test_gap_aligned_camera_rejects_invalid_axes(axis):
    from isaaclab_pruning.sim.blender_demo_scene import gap_aligned_camera_mount

    with pytest.raises(ValueError, match="Closing axis"):
        gap_aligned_camera_mount(axis)


def candidate_manifest(radius=0.0068):
    return {
        "branch_geometry_candidates": {
            "tree0_SPUR": {
                "candidates": [
                    {
                        "component_first_vertex": 7524,
                        "max_radius_m": radius,
                        "center_m": [1, 2, 3],
                        "axis": [0, 0, 1],
                    }
                ]
            }
        }
    }


def test_partition_preserves_source_face_and_corner_order_without_duplicates():
    selected, remaining = partition_faces([3, 4, 3], [0, 1, 2, 3, 4, 5, 6, 2, 1, 0], [3, 4, 5, 6])
    assert selected.point_indices == (3, 4, 5, 6)
    assert selected.face_indices == (1,)
    assert selected.corner_indices == (3, 4, 5, 6)
    assert selected.indices == (0, 1, 2, 3)
    assert remaining.face_indices == (0, 2)
    assert remaining.corner_indices == (0, 1, 2, 7, 8, 9)
    assert remaining.indices == (0, 1, 2, 2, 1, 0)
    assert set(selected.face_indices).isdisjoint(remaining.face_indices)


@pytest.mark.parametrize(
    "counts,indices,component",
    [
        ([3], [0, 1], [0]),
        ([2], [0, 1], [0]),
        ([3], [0, 1, 2], [0]),
        ([3], [0, 1, 2], [0, 1, 2]),
        ([3], [0, 1, 2], []),
        ([3], [0, 1, 2], [8]),
    ],
)
def test_invalid_or_unsplittable_components_fail(counts, indices, component):
    with pytest.raises(ValueError):
        partition_faces(counts, indices, component)


def test_selected_component_is_explicit_and_fits_demo_jaw_limit():
    assert select_component(candidate_manifest())["component_first_vertex"] == 7524
    with pytest.raises(ValueError):
        select_component(candidate_manifest(), first_vertex=1)


@pytest.mark.parametrize("radius", [-0.01, 0, 0.02, float("nan")])
def test_oversized_or_invalid_component_is_not_silently_substituted(radius):
    with pytest.raises(ValueError):
        select_component(candidate_manifest(radius))


@pytest.mark.parametrize("reset_stack", [False, True])
def test_usd_split_retains_mesh_local_transform_and_facevarying_data(reset_stack):
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from isaaclab_pruning.sim.blender_demo_scene import _split_mesh

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Tree")
    root.AddTranslateOp().Set((4, 5, 6))
    root.AddRotateZOp().Set(37)
    mesh = UsdGeom.Mesh.Define(stage, "/Tree/Source")
    mesh.AddTranslateOp().Set((0.2, 0.3, 0.4))
    mesh.AddRotateXOp().Set(23)
    mesh.SetResetXformStack(reset_stack)
    mesh.CreatePointsAttr([(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (1, 0, 0), (1.1, 0, 0), (1, 0.1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 6)
    mesh.SetNormalsInterpolation("faceVarying")
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    uv.Set([(0, 0), (1, 0), (0, 1)])
    uv.SetIndices([0, 1, 2, 0, 2, 1])
    material = UsdShade.Material.Define(stage, "/Bark")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    expected = np.asarray([world_transform.Transform(Gf.Vec3d(point)) for point in mesh.GetPointsAttr().Get()[:3]])

    body, piece, evidence = _split_mesh(stage, mesh, [0, 1, 2])

    actual_transform = UsdGeom.XformCache().GetLocalToWorldTransform(piece.GetPrim())
    actual = np.asarray([actual_transform.Transform(Gf.Vec3d(point)) for point in piece.GetPointsAttr().Get()])
    assert actual == pytest.approx(expected, abs=1e-7)
    assert body.GetResetXformStack() is reset_stack
    assert evidence["selected_faces"] == evidence["remaining_faces"] == 1
    for result, expected_uv in ((piece, [(0, 0), (1, 0), (0, 1)]), (mesh, [(0, 0), (0, 1), (1, 0)])):
        result_uv = UsdGeom.PrimvarsAPI(result).GetPrimvar("st")
        assert result_uv.GetInterpolation() == "faceVarying"
        assert not result_uv.IsIndexed()
        assert np.asarray(result_uv.Get()) == pytest.approx(np.asarray(expected_uv))
        assert np.isfinite(np.asarray(result.GetNormalsAttr().Get())).all()
        assert UsdShade.MaterialBindingAPI(result).ComputeBoundMaterial()[0].GetPath() == material.GetPath()


@pytest.mark.parametrize("yaw_degrees", [0, 31])
def test_actual_export_target_materials_and_collision_contract(yaw_degrees):
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_pruning.sim.blender_demo_scene import spawn_blender_demo_scene

    export_dir = Path(__file__).resolve().parents[1] / "artifacts" / "blender_scene" / "orchard_v1"
    if not (export_dir / "manifest.json").is_file():
        pytest.skip("Optional original Blender orchard export is not installed")
    target = np.asarray([0.25123947, 0.63413405, 0.81517712])
    stage = Usd.Stage.CreateInMemory()
    scene = spawn_blender_demo_scene(stage, export_dir, target_position_w=target, yaw_degrees=yaw_degrees)

    assert scene.evidence["partition"]["actual_center_w_m"] == pytest.approx(target, abs=1e-6)
    assert scene.evidence["partition"]["target_center_error_m"] < 1e-6
    body_position = UsdGeom.XformCache().GetLocalToWorldTransform(scene.body.GetPrim()).ExtractTranslation()
    assert list(body_position) == pytest.approx(target, abs=1e-6)
    assert UsdPhysics.RigidBodyAPI(scene.body).GetKinematicEnabledAttr().Get() is True
    assert len(scene.evidence["collision_mesh_paths"]) == 22
    for path in scene.evidence["collision_mesh_paths"]:
        assert "/ground/" not in path
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        expected_approximation = "convexHull" if path == scene.selected_mesh_prim_path else "none"
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == expected_approximation
    for path in (scene.evidence["source_mesh_path"], scene.selected_mesh_prim_path):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        corners = len(mesh.GetFaceVertexIndicesAttr().Get())
        normals = np.asarray(mesh.GetNormalsAttr().Get())
        assert normals.shape == (corners, 3)
        assert np.isfinite(normals).all()
        uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        assert uv.GetInterpolation() == "faceVarying"
        assert np.asarray(uv.Get()).shape == (corners, 2)
        assert np.isfinite(np.asarray(uv.Get())).all()
        assert str(UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath()).endswith("/mat_texture_0")
    assert scene.detach() is True
    assert scene.detach() is False
    assert UsdPhysics.RigidBodyAPI(scene.body).GetKinematicEnabledAttr().Get() is False


def test_two_original_tree_export_has_separate_materials_and_colliders():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_pruning.sim.blender_demo_scene import spawn_blender_demo_scene

    export_dir = Path(__file__).resolve().parents[1] / "artifacts/blender_scene/orchard_two_trees_v1"
    if not (export_dir / "manifest.json").is_file():
        pytest.skip("Optional original two-tree export is not installed")
    stage = Usd.Stage.CreateInMemory()
    scene = spawn_blender_demo_scene(stage, export_dir, yaw_degrees=150, component_first_vertex=8235, tree_count=2)
    assert scene.evidence["tree_count"] == 2
    assert len(scene.evidence["tree_paths"]) == 2
    assert scene.evidence["usd_sha256"]["tree0.usdc"] != scene.evidence["usd_sha256"]["tree1.usdc"]
    assert len(scene.evidence["collision_mesh_paths"]) == 25
    second = stage.GetPrimAtPath(scene.evidence["tree_paths"][1])
    meshes = [prim for prim in Usd.PrimRange(second) if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 3
    for prim in meshes:
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert str(prim.GetPath()) in scene.evidence["collision_mesh_paths"]
        assert UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        assert UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").HasValue()
    assert scene.evidence["partition"]["target_center_error_m"] < 1e-6
