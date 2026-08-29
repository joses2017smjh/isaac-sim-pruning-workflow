"""ASCII USDA cylinder trees. Does not require pxr / Isaac Sim."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab_pruning.geometry.cylinders import Cylinder, collision_enabled
from isaaclab_pruning.usd.bark import BARK_MATERIAL_NAME, bark_material_usda, packaged_bark_texture
from isaaclab_pruning.usd.cylinders import _DISPLAY_COLORS, _prim_name


def quat_wxyz_align_z(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion that rotates ``+Z`` onto ``direction`` (Gf.Rotation from-to)."""
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if norm < 1e-12:
        raise ValueError("Cannot align +Z to a zero vector.")
    target = target / norm
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if cosine < -1.0 + 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(source, target)
    axis = axis / float(np.linalg.norm(axis))
    half = 0.5 * math.acos(cosine)
    sine = math.sin(half)
    quaternion = (math.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine)
    if quaternion[0] < 0.0:
        quaternion = tuple(-value for value in quaternion)
    return (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))


def write_cylinder_tree_usda(
    cylinders: Iterable[Cylinder],
    output_path: str | Path,
    *,
    tree_id: str,
    collision_classes: Iterable[str] = ("trunk", "branch"),
    active_cut_point: Iterable[float] | None = None,
    active_radius_m: float = 0.5,
    bind_bark: bool = True,
    bark_texture: str | Path | None = None,
) -> dict[str, Any]:
    """Write one ``UsdGeom.Cylinder`` per metadata cylinder as ASCII USDA."""
    records = list(cylinders)
    if not records:
        raise ValueError("Cannot write an empty cylinder tree.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root_name = _prim_name(tree_id)
    root_path = f"/{root_name}"
    looks_path = f"{root_path}/Looks"
    material_path = f"{looks_path}/{BARK_MATERIAL_NAME}"

    grouped: dict[str, dict[str, list[tuple[int, Cylinder]]]] = defaultdict(lambda: defaultdict(list))
    class_order: list[str] = []
    part_order: dict[str, list[str]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    collision_count = 0
    for index, record in enumerate(records):
        if record.organ_class not in grouped:
            class_order.append(record.organ_class)
        if record.part_name not in grouped[record.organ_class]:
            part_order[record.organ_class].append(record.part_name)
        grouped[record.organ_class][record.part_name].append((index, record))
        class_counts[record.organ_class] += 1
        if collision_enabled(
            record,
            classes=collision_classes,
            active_cut_point=active_cut_point,
            active_radius_m=active_radius_m,
        ):
            collision_count += 1

    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root_name}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{root_name}"',
        "{",
    ]
    if bind_bark:
        lines.append(f"    rel material:binding = <{material_path}>")

    for organ_class in class_order:
        class_prim = _prim_name(organ_class)
        class_path = f"{root_path}/{class_prim}"
        lines.extend(
            [
                f'    def Xform "{class_prim}" (',
                '        prepend apiSchemas = ["SemanticsLabelsAPI:class"]',
                "    )",
                "    {",
                f'        token[] semantics:labels:class = ["{organ_class}"]',
            ]
        )
        if bind_bark:
            lines.append(f"        rel material:binding = <{material_path}>")
        for part_name in part_order[organ_class]:
            part_prim = _prim_name(part_name)
            instance_label = part_name or grouped[organ_class][part_name][0][1].record_id
            lines.extend(
                [
                    f'        def Xform "{part_prim}" (',
                    '            prepend apiSchemas = ["SemanticsLabelsAPI:class", "SemanticsLabelsAPI:instance"]',
                    "        )",
                    "        {",
                    f'            token[] semantics:labels:class = ["{organ_class}"]',
                    f'            token[] semantics:labels:instance = ["{instance_label}"]',
                ]
            )
            for index, record in grouped[organ_class][part_name]:
                collide = collision_enabled(
                    record,
                    classes=collision_classes,
                    active_cut_point=active_cut_point,
                    active_radius_m=active_radius_m,
                )
                color = _DISPLAY_COLORS.get(organ_class, _DISPLAY_COLORS["other"])
                quat = quat_wxyz_align_z(record.orientation)
                cx, cy, cz = (float(value) for value in record.centroid.tolist())
                header = f'            def Cylinder "cylinder_{index:05d}"'
                if collide:
                    header += ' (\n                prepend apiSchemas = ["PhysicsCollisionAPI"]\n            )'
                lines.extend(
                    [
                        header,
                        "            {",
                        '                uniform token axis = "Z"',
                        f"                double height = {float(record.length)}",
                        f"                color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]",
                        f'                custom string pruning:recordId = "{record.record_id}"',
                        f"                double radius = {float(record.radius)}",
                        f"                quatd xformOp:orient = ({quat[0]}, {quat[1]}, {quat[2]}, {quat[3]})",
                        f"                double3 xformOp:translate = ({cx}, {cy}, {cz})",
                        '                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
                    ]
                )
                if bind_bark:
                    lines.append(f"                rel material:binding = <{material_path}>")
                lines.append("            }")
            lines.append("        }")
        lines.append("    }")

    if bind_bark:
        texture = bark_texture
        if texture is None:
            packaged = packaged_bark_texture()
            texture = packaged if packaged.is_file() else None
        lines.extend(
            [
                '    def Xform "Looks"',
                "    {",
                *bark_material_usda(looks_path, texture_path=texture, indent="        "),
                "    }",
            ]
        )
    lines.extend(["}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return {
        "tree_id": tree_id,
        "output_path": str(destination),
        "cylinders": len(records),
        "collision_cylinders": collision_count,
        "organ_counts": dict(sorted(class_counts.items())),
        "writer": "ascii",
        "material": BARK_MATERIAL_NAME if bind_bark else None,
    }
