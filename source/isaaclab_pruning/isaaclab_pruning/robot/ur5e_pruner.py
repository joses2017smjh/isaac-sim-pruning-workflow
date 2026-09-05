"""UR5e + mock-pruner articulation constants. Isaac Lab is not required."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PACKAGE = "isaaclab_pruning.config.robot"
ALLOW_STALE_USD_ENV = "PRUNING_ALLOW_STALE_USD"
RUNTIME_USD_ENV = "PRUNING_USD"
RUNTIME_USD_EVIDENCE_ENV = "PRUNING_USD_EVIDENCE"
RUNTIME_ASSET_ID_ENV = "PRUNING_ASSET_ID"


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    lower: float
    upper: float
    stiffness: float
    damping: float
    effort_limit: float | None = None


@dataclass(frozen=True)
class Ur5ePrunerSpec:
    physics_eef_body: str
    control_tool_frame: str
    control_tool_translation_in_physics_body_m: tuple[float, float, float]
    control_tool_quaternion_wxyz_in_physics_body: tuple[float, float, float, float]
    slider_held_fixed: bool
    action_dim: int
    arm_joints: tuple[JointSpec, ...]
    slider_joint: JointSpec
    ik_method: str
    ik_lambda: float
    ik_relative_mode: bool
    mouth_half_extents_m: tuple[float, float, float]
    mouth_offset_m: tuple[float, float, float]
    failure_half_extents_m: tuple[float, float, float]
    failure_offset_m: tuple[float, float, float]
    cutter_source: str
    perpendicularity_tolerance_deg: float
    joint_names_expr: tuple[str, ...]

    @property
    def eef_body(self) -> str:
        """Compatibility alias for the rigid body used to obtain pose/Jacobian state."""
        return self.physics_eef_body


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def imported_usd_path(cfg: dict[str, Any] | None = None) -> Path:
    """Configured USD path, or the explicitly selected content-addressed asset."""
    payload = cfg if cfg is not None else load_ur5e_pruner_config()
    override = os.environ.get(RUNTIME_USD_ENV)
    if override:
        return Path(override)
    return repository_root() / str(payload["usd"]["relative_path"])


def load_ur5e_pruner_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        source = resources.files(_CONFIG_PACKAGE).joinpath("ur5e_pruner.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(source)
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _fixed_float_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(component) for component in value)
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values; got {len(result)}.")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_recorded_path(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Runtime USD evidence has no {label} path.")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _validate_runtime_usd_evidence(usd_path: Path, evidence_path: Path, asset_id: str) -> None:
    """Bind one runtime USD byte-for-byte to a successful importer report."""
    root = repository_root()
    if not usd_path.is_absolute() or not evidence_path.is_absolute():
        raise RuntimeError("Runtime USD and evidence paths must be absolute.")
    if not usd_path.is_file():
        raise RuntimeError(f"Selected runtime USD is missing: {usd_path}")
    if not evidence_path.is_file():
        raise RuntimeError(f"Selected runtime USD evidence is missing: {evidence_path}")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read runtime USD evidence: {error}") from error
    if evidence.get("ok") is not True or evidence.get("status") != "complete" or evidence.get("imported") is not True:
        raise RuntimeError("Runtime USD evidence is not a successful completed import.")
    if evidence.get("asset_id") != asset_id:
        raise RuntimeError(f"Runtime asset ID mismatch: selected={asset_id!r}, evidence={evidence.get('asset_id')!r}.")
    root_layer = _resolve_recorded_path(evidence.get("output", {}).get("root_layer"), root, "output.root_layer")
    if root_layer != usd_path.resolve():
        raise RuntimeError(f"Runtime USD path mismatch: selected={usd_path.resolve()}, evidence={root_layer}.")
    inventory = evidence.get("output", {}).get("files", [])
    matches = [
        entry
        for entry in inventory
        if isinstance(entry, dict)
        and _resolve_recorded_path(entry.get("path"), root, "output.files[]") == usd_path.resolve()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Runtime USD evidence must inventory its root layer exactly once; found {len(matches)}.")
    recorded_hash = str(matches[0].get("sha256", ""))
    actual_hash = _sha256_file(usd_path)
    if recorded_hash != actual_hash:
        raise RuntimeError(f"Runtime USD SHA-256 mismatch: actual={actual_hash}, evidence={recorded_hash}.")
    validation = evidence.get("stage_validation", {})
    required_frames = {
        "ur5e__base_link",
        "mock_pruner__base",
        "mock_pruner__camera0",
        "mock_pruner__tof0",
        "mock_pruner__tof1",
        "mock_pruner__tool0",
    }
    if validation.get("ok") is not True or validation.get("active_ur_joint_count") != 6:
        raise RuntimeError("Runtime USD stage validation is missing or did not verify six UR joints.")
    if set(validation.get("frames", {})) != required_frames:
        raise RuntimeError("Runtime USD stage validation does not contain the exact required frame set.")
    if validation.get("linear_slider_prim_paths"):
        raise RuntimeError("Runtime USD unexpectedly contains the held-fixed linear slider.")


def assert_runtime_usd_ready(
    cfg: dict[str, Any] | None = None,
    *,
    explicit_asset: bool = False,
    usd_path: str | Path | None = None,
) -> None:
    """Require a successful import report for every non-diagnostic runtime USD.

    ``explicit_asset`` remains for call-site compatibility, but is not itself a
    trust decision: an arbitrary ``PRUNING_USD`` path must also have a matching
    asset ID and immutable importer evidence.
    """
    payload = cfg if cfg is not None else load_ur5e_pruner_config()
    stale = bool(payload.get("usd", {}).get("reimport_required", False))
    diagnostic_override = os.environ.get(ALLOW_STALE_USD_ENV) == "1"
    selected_path = Path(usd_path) if usd_path is not None else imported_usd_path(payload)
    if diagnostic_override and stale and not explicit_asset:
        return
    if stale and not explicit_asset:
        raise RuntimeError(
            "Configured UR5e USD is a stale generated snapshot. Select a reviewed asset with "
            f"{RUNTIME_USD_ENV}, {RUNTIME_ASSET_ID_ENV}, and {RUNTIME_USD_EVIDENCE_ENV}. "
            f"Set {ALLOW_STALE_USD_ENV}=1 only for importer diagnostics."
        )
    usd_cfg = payload.get("usd", {})
    evidence_value = os.environ.get(RUNTIME_USD_EVIDENCE_ENV) or usd_cfg.get("import_evidence")
    asset_id = os.environ.get(RUNTIME_ASSET_ID_ENV) or usd_cfg.get("asset_id")
    if not evidence_value or not asset_id:
        raise RuntimeError(
            f"Runtime USD selection requires {RUNTIME_ASSET_ID_ENV} and {RUNTIME_USD_EVIDENCE_ENV} "
            "(or their promoted config equivalents)."
        )
    evidence_path = Path(str(evidence_value))
    if not evidence_path.is_absolute():
        evidence_path = repository_root() / evidence_path
    _validate_runtime_usd_evidence(selected_path, evidence_path, str(asset_id))


def load_ur5e_pruner_spec(path: str | Path | None = None) -> Ur5ePrunerSpec:
    cfg = load_ur5e_pruner_config(path)
    actuators = cfg["actuators"]
    slider = cfg["joints"]["slider"][0]
    arm = [
        JointSpec(
            name=joint["name"],
            joint_type=joint["type"],
            lower=float(joint["lower_rad"]),
            upper=float(joint["upper_rad"]),
            stiffness=float(actuators["arm"]["stiffness"]),
            damping=float(actuators["arm"]["damping"]),
            effort_limit=float(joint["effort_limit"]),
        )
        for joint in cfg["joints"]["arm"]
    ]
    slider_spec = JointSpec(
        name=slider["name"],
        joint_type=slider["type"],
        lower=float(slider["lower_m"]),
        upper=float(slider["upper_m"]),
        stiffness=float(slider["stiffness"]),
        damping=float(slider["damping"]),
    )
    cutter = cfg["cutter"]
    physics_eef_body = str(cfg.get("physics_eef_body", cfg.get("eef_body", "")))
    if not physics_eef_body:
        raise ValueError("Robot config must define physics_eef_body.")
    control_tool_frame = str(cfg.get("control_tool_frame", cfg.get("eef_body", physics_eef_body)))
    control_tool_translation = _fixed_float_tuple(
        cfg.get("control_tool_translation_in_physics_body_m", (0.0, 0.0, 0.0)),
        3,
        "control_tool_translation_in_physics_body_m",
    )
    control_tool_quaternion = _fixed_float_tuple(
        cfg.get("control_tool_quaternion_wxyz_in_physics_body", (1.0, 0.0, 0.0, 0.0)),
        4,
        "control_tool_quaternion_wxyz_in_physics_body",
    )
    if sum(component * component for component in control_tool_quaternion) <= 1.0e-24:
        raise ValueError("control_tool_quaternion_wxyz_in_physics_body must be nonzero.")
    return Ur5ePrunerSpec(
        physics_eef_body=physics_eef_body,
        control_tool_frame=control_tool_frame,
        control_tool_translation_in_physics_body_m=control_tool_translation,
        control_tool_quaternion_wxyz_in_physics_body=control_tool_quaternion,
        slider_held_fixed=bool(cfg["slider_held_fixed"]),
        action_dim=int(cfg["action_dim"]),
        arm_joints=tuple(arm),
        slider_joint=slider_spec,
        ik_method=str(cfg["ik"]["method"]),
        ik_lambda=float(cfg["ik"]["lambda_val"]),
        ik_relative_mode=bool(cfg["ik"]["use_relative_mode"]),
        mouth_half_extents_m=tuple(float(v) for v in cutter["mouth_half_extents_m"]),
        mouth_offset_m=tuple(float(v) for v in cutter.get("mouth_offset_m", (0.0, 0.0, 0.0))),
        failure_half_extents_m=tuple(float(v) for v in cutter["failure_half_extents_m"]),
        failure_offset_m=tuple(float(v) for v in cutter["failure_offset_m"]),
        cutter_source=str(cutter["source"]),
        perpendicularity_tolerance_deg=float(cutter["perpendicularity_tolerance_deg"]),
        joint_names_expr=tuple(str(expr) for expr in actuators["arm"]["joint_names_expr"]),
    )
