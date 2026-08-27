"""Validated access to L-Py cylinder metadata and world sidecars."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ORGAN_CLASSES = ("trunk", "branch", "spur", "nontrunk")


def _vector3(value: Any, *, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must be a finite three-vector, got {value!r}.")
    return vector


def _organ_class(part_name: str) -> str:
    prefix = part_name.split("_", maxsplit=1)[0].lower()
    return prefix if prefix in ORGAN_CLASSES else "other"


@dataclass(frozen=True)
class Cylinder:
    """One finite cylinder in metres."""

    record_id: str
    part_name: str
    centroid: np.ndarray
    orientation: np.ndarray
    radius: float
    length: float

    def __post_init__(self) -> None:
        centroid = _vector3(self.centroid, field="centroid")
        orientation = _vector3(self.orientation, field="orientation")
        orientation_norm = float(np.linalg.norm(orientation))
        if orientation_norm < 1e-12:
            raise ValueError(f"Cylinder {self.record_id!r} has a zero orientation.")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError(f"Cylinder {self.record_id!r} radius must be positive.")
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError(f"Cylinder {self.record_id!r} length must be positive.")
        object.__setattr__(self, "centroid", centroid)
        object.__setattr__(self, "orientation", orientation / orientation_norm)
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "length", float(self.length))

    @property
    def organ_class(self) -> str:
        return _organ_class(self.part_name)

    @classmethod
    def from_mapping(cls, record_id: str, record: dict[str, Any]) -> Cylinder:
        missing = {"centroid", "radius", "length"} - record.keys()
        if missing:
            raise ValueError(f"Cylinder {record_id!r} is missing {sorted(missing)}.")
        return cls(
            record_id=str(record_id),
            part_name=str(record.get("part_name", "")),
            centroid=record["centroid"],
            orientation=record.get("orientation", (0.0, 0.0, 1.0)),
            radius=float(record["radius"]),
            length=float(record["length"]),
        )


def _read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {source}: {error}") from error


def load_cylinders(
    metadata_path: str | Path,
    *,
    world_sidecar_path: str | Path | None = None,
) -> list[Cylinder]:
    """Load local metadata or a full ``cylinders_world`` sidecar.

    Per-frame annotations are intentionally rejected: their ``cylinders_world``
    field contains centroids only and cannot recover radius, length, or labels.
    """
    if world_sidecar_path is not None:
        records = _read_json(world_sidecar_path)
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("A world sidecar must be a list of full cylinder objects.")
        return [Cylinder.from_mapping(str(index), record) for index, record in enumerate(records)]

    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("cylinder_data"), dict):
        raise ValueError(f"{metadata_path} does not contain a cylinder_data object.")
    return [Cylinder.from_mapping(record_id, record) for record_id, record in metadata["cylinder_data"].items()]


def transform_cylinders(cylinders: Iterable[Cylinder], transform_cw: Any) -> list[Cylinder]:
    """Apply a rigid local-to-world transform to cylinders."""
    transform = np.asarray(transform_cw, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("Cylinder transforms must be rigid rotations plus translation.")
    translation = transform[:3, 3]
    return [
        Cylinder(
            record_id=cylinder.record_id,
            part_name=cylinder.part_name,
            centroid=rotation @ cylinder.centroid + translation,
            orientation=rotation @ cylinder.orientation,
            radius=cylinder.radius,
            length=cylinder.length,
        )
        for cylinder in cylinders
    ]


def cylinder_endpoints(cylinder: Cylinder) -> tuple[np.ndarray, np.ndarray]:
    """Return segment endpoints on the cylinder axis."""
    half_axis = 0.5 * cylinder.length * cylinder.orientation
    return cylinder.centroid - half_axis, cylinder.centroid + half_axis


def collision_enabled(
    cylinder: Cylinder,
    *,
    classes: Iterable[str] = ("trunk", "branch"),
    active_cut_point: Any | None = None,
    active_radius_m: float = 0.5,
) -> bool:
    """Select collision LOD: trunk/branch globally, thin organs near the cut."""
    enabled_classes = {name.lower() for name in classes}
    if cylinder.organ_class in enabled_classes:
        return True
    if active_cut_point is None:
        return False
    if active_radius_m <= 0:
        raise ValueError("active_radius_m must be positive.")
    point = _vector3(active_cut_point, field="active_cut_point")
    return bool(np.linalg.norm(cylinder.centroid - point) <= active_radius_m)
