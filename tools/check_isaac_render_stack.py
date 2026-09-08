#!/usr/bin/env python3
"""Validate the pinned render stack before launching Kit; preserve failures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path


def verify_versions(lock: dict, versions: dict[str, str], python_minor: str) -> None:
    if python_minor != lock["python_minor"]:
        raise RuntimeError(f"Python {python_minor} != pinned {lock['python_minor']}")
    for name, expected in lock["packages"].items():
        if versions.get(name) != expected:
            raise RuntimeError(f"{name} {versions.get(name)!r} != pinned {expected!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "ok": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "step_id": os.environ.get("SLURM_STEP_ID"),
        "node": platform.node(),
        "python": sys.version,
        "interpreter": sys.executable,
    }
    code = 1
    try:
        lock = json.loads(args.lock.read_text())
        versions = {name: importlib.metadata.version(name) for name in lock["packages"]}
        report["packages"] = versions
        verify_versions(lock, versions, f"{sys.version_info.major}.{sys.version_info.minor}")
        if os.environ.get("BHL_STACK") != lock["stack"]:
            raise RuntimeError("Select the pinned render stack with BHL_STACK=v60")
        container = Path(lock["container"]["path"])
        with container.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        report["container_sha256"] = digest
        if digest != lock["container"]["sha256"]:
            raise RuntimeError("Container digest does not match the environment lock")
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        report["gpu_driver"] = gpu.stdout.strip().splitlines()
        report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        root = Path(__file__).resolve().parents[1]
        sources = list((root / "source/isaaclab_pruning/isaaclab_pruning").rglob("*.py"))
        sources.extend((root / "source/isaaclab_pruning/isaaclab_pruning/config").rglob("*.yaml"))
        sources.extend((root / "hpc/inner").glob("*.py"))
        report["source_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)
        }
        sys.path.insert(0, str(root / "source/isaaclab_pruning"))
        from isaaclab_pruning.robot import assert_runtime_usd_ready

        assert_runtime_usd_ready()
        report["asset_gate"] = "passed"
        report["ok"] = True
        code = 0
    except Exception:
        report["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
