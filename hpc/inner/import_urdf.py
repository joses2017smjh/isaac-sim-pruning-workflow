"""Import one provenance-verified URDF into an immutable USD asset directory.

The preflight in this module deliberately has no Isaac Sim imports. It checks
the content-addressed asset identity and the hash of the generated absolute
URDF, reserves a job-specific evidence report, and only then starts Kit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENVIRONMENT = (
    "PRUNING_ASSET_ID",
    "PRUNING_URDF",
    "PRUNING_URDF_PROVENANCE",
)
ACTIVE_UR_JOINTS = (
    "ur5e__shoulder_pan_joint",
    "ur5e__shoulder_lift_joint",
    "ur5e__elbow_joint",
    "ur5e__wrist_1_joint",
    "ur5e__wrist_2_joint",
    "ur5e__wrist_3_joint",
)
FRAME_NAMES = (
    "ur5e__base_link",
    "mock_pruner__base",
    "mock_pruner__camera0",
    "mock_pruner__tof0",
    "mock_pruner__tof1",
    "mock_pruner__tool0",
)
CONVERTER_OPTIONS = {
    "force_usd_conversion": True,
    "make_instanceable": True,
    "fix_base": True,
    "merge_fixed_joints": False,
    "self_collision": False,
}
_ASSET_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,190}[a-z0-9])?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ImportPreflightError(RuntimeError):
    """Raised when an import would not be reproducible or immutable."""


@dataclass(frozen=True)
class ImportPlan:
    """Fully validated paths and metadata needed to start the converter."""

    asset_id: str
    repository_root: Path
    urdf_path: Path
    urdf_sha256: str
    provenance_path: Path
    provenance_sha256: str
    provenance: dict[str, Any]
    usd_root: Path
    output_dir: Path
    usd_path: Path
    report_path: Path
    job_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ImportPreflightError(f"required environment variable is empty: {name}")
    return value


def _absolute_existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ImportPreflightError(f"{label} must be an absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ImportPreflightError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise ImportPreflightError(f"{label} is not a regular file: {resolved}")
    return resolved


def _resolve_provenance_output(path_value: object, repository_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ImportPreflightError("provenance outputs.isaac_absolute_urdf.path is missing")
    path = Path(path_value)
    if not path.is_absolute():
        path = repository_root / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ImportPreflightError(f"provenance absolute URDF does not exist: {path}") from error


def _safe_job_id(environment: Mapping[str, str]) -> str:
    value = environment.get("SLURM_JOB_ID", "").strip() or f"manual_{os.getpid()}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _report_path(environment: Mapping[str, str], repository_root: Path, job_id: str) -> Path:
    override = environment.get("BENCH_OUT", "").strip()
    path = Path(override) if override else repository_root / "docs" / "evidence" / f"urdf_import_{job_id}.json"
    if not path.is_absolute():
        raise ImportPreflightError(f"BENCH_OUT must be an absolute path: {path}")
    return path


def build_import_plan(
    environment: Mapping[str, str] | None = None,
    *,
    repository_root: Path | None = None,
) -> ImportPlan:
    """Validate all inputs and derive the one permitted USD destination.

    This function is safe to import and run in a CPU-only Python environment.
    No Kit, Isaac Sim, or USD modules are imported during preflight.
    """
    environment = os.environ if environment is None else environment
    for name in REQUIRED_ENVIRONMENT:
        _required(environment, name)

    asset_id = _required(environment, "PRUNING_ASSET_ID")
    if not _ASSET_ID_PATTERN.fullmatch(asset_id):
        raise ImportPreflightError(
            "PRUNING_ASSET_ID must contain only lowercase letters, digits, '_' or '-', "
            "must start and end with a letter or digit, and must be at most 192 characters"
        )

    root_value = environment.get("PRUNING_ROOT", "").strip()
    root = Path(root_value) if root_value else (repository_root or REPOSITORY_ROOT)
    if not root.is_absolute():
        raise ImportPreflightError(f"PRUNING_ROOT must be an absolute path: {root}")
    root = root.resolve(strict=True)

    urdf_path = _absolute_existing_file(_required(environment, "PRUNING_URDF"), "PRUNING_URDF")
    provenance_path = _absolute_existing_file(
        _required(environment, "PRUNING_URDF_PROVENANCE"),
        "PRUNING_URDF_PROVENANCE",
    )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ImportPreflightError(f"cannot read URDF provenance: {error}") from error
    if not isinstance(provenance, dict):
        raise ImportPreflightError("URDF provenance root must be a JSON object")
    if provenance.get("ok") is not True:
        raise ImportPreflightError("URDF provenance is not marked ok")
    if provenance.get("asset_id") != asset_id:
        raise ImportPreflightError(
            f"asset ID mismatch: environment={asset_id!r}, provenance={provenance.get('asset_id')!r}"
        )

    try:
        absolute_record = provenance["outputs"]["isaac_absolute_urdf"]
    except (KeyError, TypeError) as error:
        raise ImportPreflightError("provenance has no outputs.isaac_absolute_urdf record") from error
    if not isinstance(absolute_record, dict):
        raise ImportPreflightError("provenance outputs.isaac_absolute_urdf must be an object")

    recorded_path = _resolve_provenance_output(absolute_record.get("path"), root)
    if recorded_path != urdf_path:
        raise ImportPreflightError(f"absolute URDF path mismatch: environment={urdf_path}, provenance={recorded_path}")
    recorded_sha256 = str(absolute_record.get("sha256", "")).lower()
    if not _SHA256_PATTERN.fullmatch(recorded_sha256):
        raise ImportPreflightError("provenance absolute URDF SHA-256 is missing or malformed")
    actual_sha256 = sha256_file(urdf_path)
    if actual_sha256 != recorded_sha256:
        raise ImportPreflightError(
            f"absolute URDF SHA-256 mismatch: actual={actual_sha256}, provenance={recorded_sha256}"
        )
    recorded_bytes = absolute_record.get("bytes")
    if recorded_bytes is not None and recorded_bytes != urdf_path.stat().st_size:
        raise ImportPreflightError(
            f"absolute URDF size mismatch: actual={urdf_path.stat().st_size}, provenance={recorded_bytes}"
        )

    usd_root = root / "artifacts" / "usd"
    output_dir = usd_root / asset_id
    # Isaac Lab 3 / Isaac Sim 6 always writes a layered URDF asset beneath a
    # directory named for the input URDF stem, regardless of usd_file_name.
    # Model that public converter contract up front instead of discovering it
    # only after an otherwise successful GPU conversion.
    converter_stem = urdf_path.stem
    usd_path = output_dir / converter_stem / f"{converter_stem}.usda"
    if output_dir.exists() or output_dir.is_symlink():
        raise ImportPreflightError(f"immutable USD destination already exists: {output_dir}")

    job_id = _safe_job_id(environment)
    report_path = _report_path(environment, root, job_id)
    if report_path.exists() or report_path.is_symlink():
        raise ImportPreflightError(f"evidence report already exists and will not be overwritten: {report_path}")

    return ImportPlan(
        asset_id=asset_id,
        repository_root=root,
        urdf_path=urdf_path,
        urdf_sha256=actual_sha256,
        provenance_path=provenance_path,
        provenance_sha256=sha256_file(provenance_path),
        provenance=provenance,
        usd_root=usd_root,
        output_dir=output_dir,
        usd_path=usd_path,
        report_path=report_path,
        job_id=job_id,
    )


def _path_for_report(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return str(path)


def _environment_report(environment: Mapping[str, str]) -> dict[str, Any]:
    return {
        "stack": {
            "bhl_stack": environment.get("BHL_STACK"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
        "job": {
            "id": environment.get("SLURM_JOB_ID"),
            "name": environment.get("SLURM_JOB_NAME"),
            "array_job_id": environment.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": environment.get("SLURM_ARRAY_TASK_ID"),
            "cluster": environment.get("SLURM_CLUSTER_NAME"),
            "partition": environment.get("SLURM_JOB_PARTITION"),
            "submit_dir": environment.get("SLURM_SUBMIT_DIR"),
        },
        "node": {
            "hostname": socket.gethostname(),
            "slurmd_nodename": environment.get("SLURMD_NODENAME"),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def initial_report(plan: ImportPlan, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the pending report that is persisted before Kit starts."""
    environment = os.environ if environment is None else environment
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pending",
        "ok": False,
        "imported": False,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "asset_id": plan.asset_id,
        "input": {
            "urdf": _path_for_report(plan.urdf_path, plan.repository_root),
            "urdf_sha256": plan.urdf_sha256,
            "provenance": _path_for_report(plan.provenance_path, plan.repository_root),
            "provenance_sha256": plan.provenance_sha256,
            "provenance_asset_id_verified": True,
            "absolute_urdf_sha256_verified": True,
        },
        "output": {
            "directory": _path_for_report(plan.output_dir, plan.repository_root),
            "root_layer": _path_for_report(plan.usd_path, plan.repository_root),
            "files": [],
        },
        "converter_config": {
            "asset_path": str(plan.urdf_path),
            "usd_dir": str(plan.output_dir),
            "usd_file_name": str(plan.usd_path.relative_to(plan.output_dir)),
            **CONVERTER_OPTIONS,
        },
        "stage_validation": None,
    }
    report.update(_environment_report(environment))
    return report


def write_report(path: Path, report: Mapping[str, Any], *, replace: bool) -> None:
    """Write JSON atomically, refusing to replace the initial evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (path.exists() or path.is_symlink()):
        raise ImportPreflightError(f"evidence report already exists and will not be overwritten: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inventory_output(output_dir: Path, repository_root: Path) -> list[dict[str, Any]]:
    """Hash every converter output file in deterministic path order."""
    return [
        {
            "path": _path_for_report(path, repository_root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in output_dir.rglob("*") if candidate.is_file())
    ]


def _matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _rotation_rows(matrix: Any) -> list[list[float]]:
    rotation = matrix.ExtractRotationMatrix()
    return [[float(rotation[row][column]) for column in range(3)] for row in range(3)]


def _vector_values(vector: Any) -> list[float]:
    return [float(vector[index]) for index in range(3)]


def _relative_matrix(cache: Any, prim: Any, ancestor: Any) -> Any:
    result = cache.ComputeRelativeTransform(prim, ancestor)
    return result[0] if isinstance(result, tuple) else result


def _assert_close(actual: list[float], expected: list[float], label: str, tolerance: float = 1.0e-6) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance) for left, right in zip(actual, expected)
    ):
        raise RuntimeError(f"{label} mismatch: actual={actual}, expected={expected}")


def inspect_stage(usd_path: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Open and validate the composed USD stage produced by the converter."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"USD stage could not be opened: {usd_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"USD metersPerUnit must be 1, got {meters_per_unit}")
    if up_axis.upper() != "Z":
        raise RuntimeError(f"USD up axis must be Z, got {up_axis!r}")

    prims_by_name: dict[str, list[Any]] = {}
    slider_paths: list[str] = []
    for prim in stage.Traverse():
        prims_by_name.setdefault(prim.GetName(), []).append(prim)
        if "linear_slider" in str(prim.GetPath()).lower():
            slider_paths.append(str(prim.GetPath()))
    if slider_paths:
        raise RuntimeError(f"linear-slider prims are forbidden: {slider_paths}")

    joint_records = []
    for name in ACTIVE_UR_JOINTS:
        matches = prims_by_name.get(name, [])
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {name!r} prim, found {len(matches)}")
        prim = matches[0]
        type_name = str(prim.GetTypeName())
        if type_name != "PhysicsRevoluteJoint":
            raise RuntimeError(f"{name} must be PhysicsRevoluteJoint, got {type_name!r}")
        joint_records.append({"name": name, "path": str(prim.GetPath()), "type": type_name})

    frame_prims: dict[str, Any] = {}
    for name in FRAME_NAMES:
        matches = prims_by_name.get(name, [])
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {name!r} prim, found {len(matches)}")
        prim = matches[0]
        if not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"{name} must be Xformable, got {prim.GetTypeName()!s}")
        frame_prims[name] = prim

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_prim = frame_prims["mock_pruner__base"]
    fixed_frames = provenance.get("validation", {}).get("fixed_frames", {})
    frame_records: dict[str, Any] = {}
    for name, prim in frame_prims.items():
        world_matrix = cache.GetLocalToWorldTransform(prim)
        record: dict[str, Any] = {
            "path": str(prim.GetPath()),
            "type": str(prim.GetTypeName()),
            "world_matrix": _matrix_rows(world_matrix),
            "world_translation_m": _vector_values(world_matrix.ExtractTranslation()),
        }
        if name.startswith("mock_pruner__") and name != "mock_pruner__base":
            relative_matrix = _relative_matrix(cache, prim, base_prim)
            record["relative_to_mock_pruner_base"] = {
                "matrix": _matrix_rows(relative_matrix),
                "translation_m": _vector_values(relative_matrix.ExtractTranslation()),
                "rotation_matrix": _rotation_rows(relative_matrix),
            }
            provenance_key = f"mock_pruner__base_to_{name.removeprefix('mock_pruner__')}"
            expected = fixed_frames.get(provenance_key)
            if not isinstance(expected, dict):
                raise RuntimeError(f"provenance has no expected transform for {provenance_key}")
            expected_translation = [float(value) for value in expected.get("translation_m", [])]
            expected_rpy = [float(value) for value in expected.get("rpy_rad", [])]
            _assert_close(
                record["relative_to_mock_pruner_base"]["translation_m"],
                expected_translation,
                f"{name} translation",
            )
            if len(expected_rpy) != 3:
                raise RuntimeError(f"provenance RPY for {provenance_key} must contain three values")
            if all(math.isclose(value, 0.0, abs_tol=1.0e-12) for value in expected_rpy):
                actual_rotation = record["relative_to_mock_pruner_base"]["rotation_matrix"]
                identity = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
                for row, (actual_row, identity_row) in enumerate(zip(actual_rotation, identity)):
                    _assert_close(actual_row, identity_row, f"{name} rotation row {row}")
            record["provenance_transform_verified"] = True
        frame_records[name] = record

    return {
        "ok": True,
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "active_ur_joint_count": len(joint_records),
        "active_ur_joints": joint_records,
        "linear_slider_prim_paths": slider_paths,
        "frames": frame_records,
        "used_layers": sorted(layer.identifier for layer in stage.GetUsedLayers()),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _best_effort_failure_path(environment: Mapping[str, str]) -> Path | None:
    try:
        root_value = environment.get("PRUNING_ROOT", "").strip()
        root = (Path(root_value) if root_value else REPOSITORY_ROOT).resolve(strict=True)
        return _report_path(environment, root, _safe_job_id(environment))
    except (ImportPreflightError, OSError):
        return None


def _emit(report: Mapping[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)


def main() -> int:
    """Run preflight, conversion, validation, and evidence capture."""
    environment = os.environ
    try:
        plan = build_import_plan(environment)
    except Exception as error:  # noqa: BLE001
        failure = {
            "schema_version": 1,
            "status": "failed_preflight",
            "ok": False,
            "imported": False,
            "updated_at": _utc_now(),
            "reason": f"{type(error).__name__}: {error}",
        }
        failure.update(_environment_report(environment))
        path = _best_effort_failure_path(environment)
        if path is not None and not path.exists():
            write_report(path, failure, replace=False)
        _emit(failure)
        return 2

    report = initial_report(plan, environment)
    write_report(plan.report_path, report, replace=False)

    app = None
    try:
        plan.usd_root.mkdir(parents=True, exist_ok=True)
        # mkdir is the immutable destination reservation: concurrent/repeated
        # imports of the same content address cannot pass this operation.
        plan.output_dir.mkdir()

        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        cfg = UrdfConverterCfg(
            asset_path=str(plan.urdf_path),
            usd_dir=str(plan.output_dir),
            usd_file_name=str(plan.usd_path.relative_to(plan.output_dir)),
            **CONVERTER_OPTIONS,
        )
        converter = UrdfConverter(cfg)
        converter_path = Path(converter.usd_path).resolve(strict=False)
        if converter_path != plan.usd_path.resolve(strict=False):
            raise RuntimeError(f"converter returned unexpected USD path: {converter_path} != {plan.usd_path}")
        if not plan.usd_path.is_file():
            raise RuntimeError(f"converter did not create the required root layer: {plan.usd_path}")

        report["stage_validation"] = inspect_stage(plan.usd_path, plan.provenance)
        report["output"]["files"] = inventory_output(plan.output_dir, plan.repository_root)
        report["stack"].update(
            {
                "isaaclab_version": _package_version("isaaclab"),
                "isaacsim_version": _package_version("isaacsim"),
            }
        )
        report.update(
            {
                "status": "complete",
                "ok": True,
                "imported": True,
                "updated_at": _utc_now(),
            }
        )
        write_report(plan.report_path, report, replace=True)
        _emit(report)
        return 0
    except Exception as error:  # noqa: BLE001
        report.update(
            {
                "status": "failed",
                "ok": False,
                "imported": False,
                "updated_at": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        if plan.output_dir.is_dir():
            report["output"]["files"] = inventory_output(plan.output_dir, plan.repository_root)
        write_report(plan.report_path, report, replace=True)
        _emit(report)
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
