#!/usr/bin/env python3
"""Independent application gate for a fresh Isaac rendering bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate_bundle(root: Path) -> dict:
    report = json.loads((root / "report.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    frames = json.loads((root / "frames.json").read_text())["frames"]
    if preflight.get("ok") is not True or report.get("ok") is not True:
        raise ValueError("Preflight or rendering report did not pass")
    if report.get("rendering_ok") is not True:
        raise ValueError("No successful rendering evidence")
    if len(frames) < 30 or report.get("frame_count") != len(frames):
        raise ValueError("Incomplete frame bundle")
    for index in (0, len(frames) // 2, len(frames) - 1):
        for kind in ("overview", "wrist"):
            with Image.open(root / "frames" / f"{kind}_{index:05d}.png") as image:
                image.load()
                if image.width < 64 or image.height < 64:
                    raise ValueError("Rendered camera image is too small")
                if np.asarray(image.convert("RGB"), dtype=float).std() < 1.0:
                    raise ValueError("Rendered camera image is flat")
        depth = np.load(root / "frames" / f"depth_{index:05d}.npy", allow_pickle=False)
        if depth.ndim != 2 or not (np.isfinite(depth) & (depth > 0)).any():
            raise ValueError("Rendered depth is absent or invalid")
    return {"ok": True, "frame_count": len(frames), "task_outcome": report.get("task_outcome")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.root), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
