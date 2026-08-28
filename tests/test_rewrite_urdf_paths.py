from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "rewrite_urdf_paths.py"
_SPEC = importlib.util.spec_from_file_location("rewrite_urdf_paths", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_REWRITE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWRITE)
rewrite_urdf = _REWRITE.rewrite_urdf


def test_rewrite_package_uri_to_absolute_path(tmp_path: Path) -> None:
    mesh = tmp_path / "pkg" / "meshes" / "link.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("solid", encoding="utf-8")
    urdf = '<mesh filename="package://demo_pkg/meshes/link.stl"/>'
    missing: list[str] = []
    rewritten = rewrite_urdf(urdf, {"demo_pkg": tmp_path / "pkg"}, missing)
    assert str(mesh.resolve()) in rewritten
    assert "package://" not in rewritten
    assert missing == []


def test_rewrite_reports_missing_mesh(tmp_path: Path) -> None:
    missing: list[str] = []
    rewrite_urdf(
        '<mesh filename="package://demo_pkg/meshes/missing.stl"/>',
        {"demo_pkg": tmp_path},
        missing,
    )
    assert missing
    assert "missing.stl" in missing[0]


def test_rewrite_luke_install_file_uri(tmp_path: Path) -> None:
    mesh = tmp_path / "pkg" / "meshes" / "MockPruner.STL"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid")
    urdf = (
        '<mesh filename="file:///home/luke/branch_detection_ws/install/'
        'demo_pkg/share/demo_pkg/meshes/MockPruner.STL"/>'
    )
    missing: list[str] = []
    rewritten = rewrite_urdf(urdf, {"demo_pkg": tmp_path / "pkg"}, missing)
    assert str(mesh.resolve()) in rewritten
    assert "file:///home/luke" not in rewritten
    assert missing == []
