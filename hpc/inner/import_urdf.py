"""Import the rewritten UR5e + mock-pruner URDF to USD on the v60 stack.

Requires Gate 0. Lab 3's converter lives at isaaclab.sim.converters; if that
import moves, fail loudly instead of pretending the USD exists. This URDF is
the BDS flatten (Amiga + UR5e + mock-pruner), not the slider.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

out_json = Path(os.environ.get("BENCH_OUT", "/tmp/pruning_urdf_import.json"))
out_json.parent.mkdir(parents=True, exist_ok=True)


def _write(report: dict, code: int) -> None:
    text = json.dumps(report, indent=2, default=str)
    print(text, flush=True)
    out_json.write_text(text + "\n", encoding="utf-8")
    app.close()
    raise SystemExit(code)


try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
except ImportError as error:  # pragma: no cover
    _write({"imported": False, "reason": f"UrdfConverter unavailable: {error}"}, 1)

urdf = Path(os.environ["PRUNING_URDF"])
out_dir = Path(os.environ["PRUNING_USD_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

try:
    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(out_dir),
        usd_file_name="ur5e_pruner.usda",
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
    )
    converter = UrdfConverter(cfg)
    usd_path = Path(converter.usd_path)
except Exception as error:  # noqa: BLE001
    _write({"imported": False, "urdf": str(urdf), "reason": repr(error)}, 1)

report = {
    "imported": bool(usd_path.is_file()),
    "urdf": str(urdf),
    "usd": str(usd_path),
    "bytes": usd_path.stat().st_size if usd_path.is_file() else 0,
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
}
_write(report, 0 if report["imported"] else 1)
