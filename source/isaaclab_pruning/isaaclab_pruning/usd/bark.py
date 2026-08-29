"""UsdPreviewSurface for the Blender orchard baseline bark.

The orchard generator's baseline texture is ``bark_brown_02``. Isaac's hydra
path needs a real ``UsdPreviewSurface`` (plus a light); a custom string
attribute is not a material. The diffuse albedo matches the packed 512² JPEG
sampled from the spur-depth orchard texture. Cylinder prims often lack UVs, so
the shader always has a fallback ``diffuseColor``.
"""

from __future__ import annotations

from pathlib import Path

BARK_MATERIAL_NAME = "bark_brown_02"
BARK_DIFFUSE = (0.345098, 0.219608, 0.125490)
BARK_ROUGHNESS = 0.85


def packaged_bark_texture() -> Path:
    """JPEG shipped with the package (downscaled from the orchard 4k albedo)."""
    return Path(__file__).resolve().parents[1] / "assets" / "textures" / "bark_brown_02_diff.jpg"


def bark_material_usda(
    looks_path: str,
    *,
    texture_path: str | Path | None = None,
    indent: str = "        ",
) -> list[str]:
    """ASCII USDA for ``Looks/<bark_brown_02>`` as UsdPreviewSurface."""
    material_path = f"{looks_path}/{BARK_MATERIAL_NAME}"
    shader_path = f"{material_path}/PreviewSurface"
    inner = indent + "    "
    shader_inner = indent + "        "
    resolved = Path(texture_path).resolve() if texture_path is not None else None
    use_texture = resolved is not None and resolved.is_file()

    lines = [
        f'{indent}def Material "{BARK_MATERIAL_NAME}"',
        f"{indent}{{",
        f"{indent}    token outputs:surface.connect = <{shader_path}.outputs:surface>",
        f'{inner}def Shader "PreviewSurface"',
        f"{inner}{{",
        f'{shader_inner}uniform token info:id = "UsdPreviewSurface"',
        (
            f"{shader_inner}color3f inputs:diffuseColor = "
            f"({BARK_DIFFUSE[0]:.6f}, {BARK_DIFFUSE[1]:.6f}, {BARK_DIFFUSE[2]:.6f})"
        ),
        f"{shader_inner}float inputs:roughness = {BARK_ROUGHNESS:.6f}",
        f"{shader_inner}token outputs:surface",
        f"{inner}}}",
    ]
    if use_texture:
        tex_path = f"{material_path}/diffuseTex"
        lines.insert(
            6,
            f"{shader_inner}color3f inputs:diffuseColor.connect = <{tex_path}.outputs:rgb>",
        )
        lines.extend(
            [
                f'{inner}def Shader "diffuseTex"',
                f"{inner}{{",
                f'{shader_inner}uniform token info:id = "UsdUVTexture"',
                f"{shader_inner}asset inputs:file = @{resolved}@",
                f"{shader_inner}float3 outputs:rgb",
                f"{inner}}}",
            ]
        )
    lines.append(f"{indent}}}")
    return lines
