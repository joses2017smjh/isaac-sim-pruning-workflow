"""ASCII USDA authoring for orchard furniture. Does not require pxr."""

from __future__ import annotations

from pathlib import Path

from isaaclab_pruning.assets.orchard import OrchardLayout, build_v_trellis_layout
from isaaclab_pruning.usd.bark import bark_material_usda, packaged_bark_texture


def _fmt(values) -> str:
    return ", ".join(f"{float(value):.6f}" for value in values)


def write_orchard_usda(
    output_path: str | Path,
    layout: OrchardLayout | None = None,
    *,
    tree_usda_paths: list[str] | None = None,
) -> Path:
    """Write posts, wires, ground, and optional tree references as USDA."""
    layout = layout or build_v_trellis_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    gx, gy, gz = layout.ground_size_m
    lines = [
        "#usda 1.0",
        "(",
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "orchard"',
        "{",
        '    def Cube "ground"',
        "    {",
        f"        double3 xformOp:scale = ({gx}, {gy}, {gz})",
        "        double3 xformOp:translate = (0, 0, 0)",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        "        rel material:binding = </orchard/Looks/ground>",
        "    }",
    ]

    for post in layout.posts:
        cx, cy, cz = post.centroid.tolist()
        lines.extend(
            [
                f'    def Cylinder "{post.name}"',
                "    {",
                '        uniform token axis = "Z"',
                f"        double height = {post.height_m:.6f}",
                f"        double radius = {post.radius_m:.6f}",
                f"        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )

    for wire in layout.wires:
        cx, cy, cz = wire.centroid.tolist()
        lines.extend(
            [
                f'    def Cylinder "{wire.name}"',
                "    {",
                '        uniform token axis = "X"',
                f"        double height = {wire.length_m:.6f}",
                f"        double radius = {wire.radius_m:.6f}",
                f"        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
            ]
        )

    if tree_usda_paths:
        for index, (path, translation) in enumerate(zip(tree_usda_paths, layout.tree_translations, strict=False)):
            tx, ty, tz = translation
            lines.extend(
                [
                    f'    def Xform "tree_{index}" (',
                    f"        prepend references = @{path}@",
                    "    )",
                    "    {",
                    f"        double3 xformOp:translate = ({tx:.6f}, {ty:.6f}, {tz:.6f})",
                    '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                    "    }",
                ]
            )

    lines.extend(
        [
            '    def Xform "Looks"',
            "    {",
            f'        custom string pruning:baseline_bark = "{layout.baseline_bark}"',
            f"        custom double pruning:tree_tilt_x_deg = {layout.tree_tilt_x_deg:.6f}",
            f"        custom double pruning:dome_intensity = {float(layout.lighting['dome_intensity']):.6f}",
            f"        custom double pruning:sun_intensity = {float(layout.lighting['sun_intensity']):.6f}",
            f"        custom double3 pruning:sun_angle_deg = ({_fmt(layout.lighting['sun_angle_deg'])})",
            '        def Material "ground"',
            "        {",
            "            token outputs:surface.connect = </orchard/Looks/ground/PreviewSurface.outputs:surface>",
            '            def Shader "PreviewSurface"',
            "            {",
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = ({_fmt(layout.ground_color)})",
            "                float inputs:roughness = 0.900000",
            "                token outputs:surface",
            "            }",
            "        }",
            *bark_material_usda(
                "/orchard/Looks",
                texture_path=packaged_bark_texture() if packaged_bark_texture().is_file() else None,
                indent="        ",
            ),
            "    }",
            "}",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
