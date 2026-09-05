#!/usr/bin/env python3
"""Rewrite package:// and unresolved mesh URIs in a URDF to absolute file paths.

Isaac's URDF importer cannot resolve ROS package URIs. This is the CPU-only
step; the GPU importer is hpc/inner/import_urdf.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PACKAGE_URI = re.compile(r"""((?:filename|filepath)=["'])package://([^/"']+)/([^"']+)(["'])""")
FILE_URI = re.compile(r"""((?:filename|filepath)=["'])file://([^"']+)(["'])""")
# Flattened URDFs from Luke's machines: file:///home/luke/.../install/pkg/share/pkg/rel
INSTALL_FILE_URI = re.compile(
    r"""((?:filename|filepath)=["'])file://[^"']*/install/([^/"']+)/share/\2/([^"']+)(["'])"""
)


def rewrite_urdf(text: str, package_map: dict[str, Path], missing: list[str]) -> str:
    def _package(match: re.Match[str]) -> str:
        quote_l, package, rel, quote_r = match.groups()
        root = package_map.get(package)
        if root is None:
            missing.append(f"package://{package}/{rel}")
            return match.group(0)
        resolved = (root / rel).resolve()
        if not resolved.is_file():
            missing.append(str(resolved))
        return f"{quote_l}{resolved}{quote_r}"

    def _install_file(match: re.Match[str]) -> str:
        quote_l, package, rel, quote_r = match.groups()
        root = package_map.get(package)
        if root is None:
            missing.append(f"file://install/{package}/{rel}")
            return match.group(0)
        resolved = (root / rel).resolve()
        if not resolved.is_file():
            missing.append(str(resolved))
        return f"{quote_l}{resolved}{quote_r}"

    rewritten = INSTALL_FILE_URI.sub(_install_file, text)
    rewritten = PACKAGE_URI.sub(_package, rewritten)

    def _file(match: re.Match[str]) -> str:
        quote_l, path, quote_r = match.groups()
        if not Path(path).is_file():
            missing.append(path)
        return f"{quote_l}{path}{quote_r}"

    return FILE_URI.sub(_file, rewritten)


def load_package_map(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[1]
    resolved: dict[str, Path] = {}
    for name, value in payload.items():
        mapped = Path(value).expanduser()
        if not mapped.is_absolute():
            mapped = repo / mapped
        resolved[name] = mapped.resolve()
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--package-map", type=Path, required=True)
    args = parser.parse_args(argv)

    missing: list[str] = []
    rewritten = rewrite_urdf(args.urdf.read_text(encoding="utf-8"), load_package_map(args.package_map), missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rewritten, encoding="utf-8")
    report = {"output": str(args.output), "missing_meshes": missing, "ok": not missing}
    print(json.dumps(report, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
