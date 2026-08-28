"""Compose the URDF importer's split payload layers for assertions.

The converter writes a 1 KB root composition layer and puts the robot in
``payloads/robot.usda``, PhysX in ``payloads/Physics/physx.usda``, and (for
some robots) separate base/sensor layers. Asserting against one file rediscovers
the ``mock_pruner__tool0`` miss. Concatenate ASCII payloads instead of opening
a USD stage — this helper is Isaac-free.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab_pruning.robot.ur5e_pruner import imported_usd_path


def imported_usd_dir(usd_path: str | Path | None = None) -> Path:
    path = Path(usd_path) if usd_path is not None else imported_usd_path()
    return path.parent if path.suffix else path


def imported_usd_payload_paths(usd_path: str | Path | None = None) -> tuple[Path, ...]:
    """Root composition layer plus every ASCII ``.usda`` under ``payloads/``."""
    root = Path(usd_path) if usd_path is not None else imported_usd_path()
    directory = imported_usd_dir(root)
    paths: list[Path] = []
    if root.is_file():
        paths.append(root)
    payloads = directory / "payloads"
    if payloads.is_dir():
        paths.extend(sorted(path for path in payloads.rglob("*.usda") if path.is_file() and path not in paths))
    return tuple(paths)


def load_imported_usd(usd_path: str | Path | None = None) -> str:
    """Return composed ASCII payload text. Raises if the import artifacts are missing."""
    paths = imported_usd_payload_paths(usd_path)
    if not paths:
        target = Path(usd_path) if usd_path is not None else imported_usd_path()
        raise FileNotFoundError(f"Imported USD payloads are missing under {imported_usd_dir(target)}")
    chunks: list[str] = []
    directory = imported_usd_dir(paths[0])
    for path in paths:
        try:
            relative = path.relative_to(directory)
        except ValueError:
            relative = path
        chunks.append(f"# --- {relative.as_posix()} ---\n")
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        if not chunks[-1].endswith("\n"):
            chunks.append("\n")
    return "".join(chunks)
