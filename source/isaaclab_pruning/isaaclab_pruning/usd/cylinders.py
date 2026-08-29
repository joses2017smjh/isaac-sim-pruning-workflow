"""Author L-Py cylinder metadata as a semantic, collision-aware USD stage."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab_pruning.geometry.cylinders import (
    Cylinder,
    collision_enabled,
    load_cylinders,
    transform_cylinders,
)
from isaaclab_pruning.usd.bark import BARK_DIFFUSE, BARK_MATERIAL_NAME, BARK_ROUGHNESS, packaged_bark_texture

DEFAULT_TREE_TILT_X_DEG = -17.143
_DISPLAY_COLORS = {
    "trunk": (0.20, 0.35, 0.95),
    "branch": (0.15, 0.75, 0.20),
    "spur": (1.00, 0.44, 0.26),
    "nontrunk": (0.80, 0.25, 0.80),
    "other": (0.50, 0.50, 0.50),
}


def _require_pxr() -> tuple[Any, ...]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as error:
        raise RuntimeError(
            "USD authoring requires the `pxr` modules bundled with Isaac Sim. "
            "Run this command through Isaac Lab's Python interpreter."
        ) from error
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _prim_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "unnamed"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def _add_semantics(prim: Any, semantic_type: str, label: str, sdf: Any) -> None:
    schema_name = f"SemanticsLabelsAPI:{semantic_type}"
    prim.AddAppliedSchema(schema_name)
    attribute = prim.CreateAttribute(
        f"semantics:labels:{semantic_type}",
        sdf.ValueTypeNames.TokenArray,
        custom=False,
    )
    attribute.Set([label])


def _local_transform(tilt_x_deg: float, translation: Iterable[float]) -> np.ndarray:
    angle = math.radians(float(tilt_x_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )
    transform[:3, 3] = np.asarray(tuple(translation), dtype=np.float64)
    return transform


def write_cylinder_tree_usd(
    cylinders: Iterable[Cylinder],
    output_path: str | Path,
    *,
    tree_id: str,
    collision_classes: Iterable[str] = ("trunk", "branch"),
    active_cut_point: Iterable[float] | None = None,
    active_radius_m: float = 0.5,
    bind_bark: bool = True,
) -> dict[str, Any]:
    """Write one ``UsdGeom.Cylinder`` per metadata cylinder.

    ``UsdGeom.Cylinder`` is intentional: the Blender generator builds finite
    cylinders, not capsules. Using capsules changes the end geometry and breaks
    the millimetre-level Blender/Isaac depth comparison.
    """
    gf, sdf, usd, usd_geom, usd_physics, usd_shade = _require_pxr()
    records = list(cylinders)
    if not records:
        raise ValueError("Cannot write an empty cylinder tree.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = usd.Stage.CreateNew(str(destination))
    usd_geom.SetStageMetersPerUnit(stage, 1.0)
    usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)

    root_name = _prim_name(tree_id)
    root = usd_geom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    bark_material = None
    if bind_bark:
        usd_geom.Xform.Define(stage, f"/{root_name}/Looks")
        material_path = f"/{root_name}/Looks/{BARK_MATERIAL_NAME}"
        bark_material = usd_shade.Material.Define(stage, material_path)
        shader = usd_shade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(gf.Vec3f(*BARK_DIFFUSE))
        shader.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(BARK_ROUGHNESS)
        bark_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        texture = packaged_bark_texture()
        if texture.is_file():
            tex = usd_shade.Shader.Define(stage, f"{material_path}/diffuseTex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", sdf.ValueTypeNames.Asset).Set(str(texture.resolve()))
            shader.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )
        usd_shade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(bark_material)

    class_groups: dict[str, Any] = {}
    part_groups: dict[tuple[str, str], Any] = {}
    class_counts: Counter[str] = Counter()
    collision_count = 0

    for index, record in enumerate(records):
        organ_class = record.organ_class
        class_counts[organ_class] += 1
        if organ_class not in class_groups:
            class_path = f"/{root_name}/{_prim_name(organ_class)}"
            class_group = usd_geom.Xform.Define(stage, class_path)
            _add_semantics(class_group.GetPrim(), "class", organ_class, sdf)
            class_groups[organ_class] = class_group

        part_key = (organ_class, record.part_name)
        if part_key not in part_groups:
            part_path = f"{class_groups[organ_class].GetPath()}/{_prim_name(record.part_name)}"
            part_group = usd_geom.Xform.Define(stage, part_path)
            _add_semantics(part_group.GetPrim(), "class", organ_class, sdf)
            _add_semantics(part_group.GetPrim(), "instance", record.part_name or record.record_id, sdf)
            part_groups[part_key] = part_group

        cylinder_path = f"{part_groups[part_key].GetPath()}/cylinder_{index:05d}"
        cylinder = usd_geom.Cylinder.Define(stage, cylinder_path)
        cylinder.GetAxisAttr().Set(usd_geom.Tokens.z)
        cylinder.GetHeightAttr().Set(record.length)
        cylinder.GetRadiusAttr().Set(record.radius)
        cylinder.CreateDisplayColorAttr().Set([gf.Vec3f(*_DISPLAY_COLORS.get(organ_class, _DISPLAY_COLORS["other"]))])
        cylinder.GetPrim().CreateAttribute("pruning:recordId", sdf.ValueTypeNames.String, custom=True).Set(
            record.record_id
        )

        rotation = gf.Rotation(gf.Vec3d(0.0, 0.0, 1.0), gf.Vec3d(*record.orientation.tolist())).GetQuat()
        transformable = usd_geom.Xformable(cylinder.GetPrim())
        transformable.AddTranslateOp(precision=usd_geom.XformOp.PrecisionDouble).Set(
            gf.Vec3d(*record.centroid.tolist())
        )
        transformable.AddOrientOp(precision=usd_geom.XformOp.PrecisionDouble).Set(rotation)

        if bark_material is not None:
            usd_shade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(bark_material)
        if collision_enabled(
            record,
            classes=collision_classes,
            active_cut_point=active_cut_point,
            active_radius_m=active_radius_m,
        ):
            usd_physics.CollisionAPI.Apply(cylinder.GetPrim())
            collision_count += 1

    stage.GetRootLayer().Save()
    return {
        "tree_id": tree_id,
        "output_path": str(destination),
        "cylinders": len(records),
        "collision_cylinders": collision_count,
        "organ_counts": dict(sorted(class_counts.items())),
        "material": BARK_MATERIAL_NAME if bind_bark else None,
    }


def _tree_id_from_metadata(path: Path) -> str:
    suffix = "_metadata"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="L-Py *_metadata.json")
    parser.add_argument("output", type=Path, help="Output .usd or .usda path")
    parser.add_argument("--world-sidecar", type=Path, help="Optional full cylinders_world JSON")
    parser.add_argument("--tree-id", help="USD root name; defaults to the metadata filename")
    parser.add_argument(
        "--collision-class",
        action="append",
        dest="collision_classes",
        choices=("trunk", "branch", "spur", "nontrunk"),
        help="Organ class with global collision; repeat as needed (default: trunk, branch)",
    )
    parser.add_argument("--active-cut-point", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--active-radius-m", type=float, default=0.5)
    parser.add_argument(
        "--tilt-x-deg",
        type=float,
        default=DEFAULT_TREE_TILT_X_DEG,
        help="Local metadata tilt (ignored when --world-sidecar is supplied)",
    )
    parser.add_argument("--translate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cylinders = load_cylinders(args.metadata, world_sidecar_path=args.world_sidecar)
    if args.world_sidecar is None:
        cylinders = transform_cylinders(cylinders, _local_transform(args.tilt_x_deg, args.translate))
    result = write_cylinder_tree_usd(
        cylinders,
        args.output,
        tree_id=args.tree_id or _tree_id_from_metadata(args.metadata),
        collision_classes=args.collision_classes or ("trunk", "branch"),
        active_cut_point=args.active_cut_point,
        active_radius_m=args.active_radius_m,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
