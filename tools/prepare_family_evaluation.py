"""Validate paired rendering evidence and materialize a read-only evaluation plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from depth_generalization import sha256
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    frames = []
    sources = []
    for name in inputs["render_manifests"]:
        path = Path(name)
        data = json.loads(path.read_text())
        assert data["ok"]
        assert len(data["frames"]) == 24
        by_view = {}
        for original in data["frames"]:
            f = dict(original)
            # v1/v2 pilots stored continuous coordinates measured from image edges.
            # Preserve renders/predictions; correct the pixel-center metadata explicitly.
            if "pixel_coordinate_convention" not in f:
                f["K"] = [list(row) for row in f["K"]]
                f["K"][0][2] -= 0.5
                f["K"][1][2] -= 0.5
                f["pixel_coordinate_convention"] = "zero-based pixel centers; Blender edge coordinates minus 0.5"
                f["annotation_correction"] = "Half-pixel conversion from preserved edge-coordinate pilot manifest"
                pixel = f.get("projected_target_pixel_xy")
                f["target_visible"] = False
                f["target_pixel_xy"] = None
                if pixel is not None:
                    pixel = [v - 0.5 for v in pixel]
                    f["projected_target_pixel_xy"] = pixel
                    rx, ry, rz = f["camera_rotation_euler"]
                    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
                    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
                    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                    camera_point = (Rz @ Ry @ Rx).T @ (np.array(data["target"]["centroid"]) - f["camera_location"])
                    optical_z = -float(camera_point[2])
                    projected = np.asarray(f["K"]) @ (camera_point * [1, -1, -1])
                    assert np.max(np.abs(projected[:2] / projected[2] - pixel)) < 0.002
                    f["target_expected_optical_z_m"] = optical_z
                    gt = np.load(f["depth"])
                    x, y = [round(v) for v in pixel]
                    if 0 <= x < gt.shape[1] and 0 <= y < gt.shape[0]:
                        f["target_visible"] = bool(
                            abs(float(gt[y, x]) - optical_z) <= 2 * data["target"]["radius"] + 0.01
                        )
                    if f["target_visible"]:
                        f["target_pixel_xy"] = pixel
            rgb = np.asarray(Image.open(f["rgb"]).convert("RGB"))
            depth = np.load(f["depth"], allow_pickle=False)
            mask = np.asarray(Image.open(f["mask"])) > 0
            assert rgb.shape[:2] == depth.shape == mask.shape
            assert rgb.std() > 3 and mask.any()
            assert np.isfinite(np.asarray(f["K"])).all() and f["K"][0][0] > 0
            # GT and masks are physically shared paths for unchanged geometry/view.
            by_view.setdefault(f["view_id"], []).append(f)
            frames.append(f)
        assert len(by_view) == 6
        for view in by_view.values():
            assert {f["lighting"] for f in view} == {"source", "morning", "noon", "evening"}
            assert len({f["depth"] for f in view}) == len({f["mask"] for f in view}) == 1
            assert len({sha256(f["rgb"]) for f in view}) == 4
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "tree_id": data["tree_id"],
                "geometry_sha256": data["geometry_sha256"],
                "K": data["K"],
            }
        )
    plan = {
        "schema_version": 1,
        "scope": inputs["scope"],
        "source_renders": sources,
        "frames": frames,
        "depth_convention": "Cycles optical-Z metres verified by 2m frontoparallel plane at center and off axis",
        "target_scope": (
            "Fixed geometric spur segment, no outcome-driven replacement; invisible projections are explicit failures"
        ),
        "temporal_scope": "Static paired views; cross-view changes are not temporal stability measurements",
    }
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
    print(json.dumps({"ok": True, "trees": len(sources), "frames": len(frames)}))


if __name__ == "__main__":
    main()
