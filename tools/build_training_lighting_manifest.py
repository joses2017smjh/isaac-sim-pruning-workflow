#!/usr/bin/env python3
"""Build the fine-tune manifest from the rendered-lighting batches: rendered rows plus the surviving rows.

A rendered frame enters only when its re-rendered geometry reproduced the
surviving depth and mask it reuses (``ok`` in the render manifest). Every
registered frame is accounted for in the summary: rendered and kept, rendered
and failed the check, or never rendered. Depth and mask paths always point to
the surviving companion files; the RGB of a rendered row is the new render.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_lighting as tl  # noqa: E402
from queue_family_matrix import sha256  # noqa: E402

COLUMNS = ["bark", "tree", "set_id", "shot", "view", "lighting", "rgb_path", "depth_path", "mask_path"]


def render_manifests(batches):
    """Every tree's render manifest in the given batches, pilot trees included, keyed by tree."""
    found = {}
    for batch in batches:
        plan = json.loads((Path(batch) / "plan.json").read_text())
        for tree_id, entry in plan["trees"].items():
            if entry["status"] == "rendered_pilot":
                path = Path(entry["render_manifest"])
            else:
                path = Path(batch) / "render" / tree_id / "render_manifest.json"
            found.setdefault(tree_id, {"registered": len(entry["frames"]), "manifest": path})
            if path.is_file() and entry["status"] == "to_render":
                found[tree_id] = {"registered": len(entry["frames"]), "manifest": path}
    return found


def build(batches, surviving_manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rendered_rows, per_tree, lumas = [], [], []
    for tree_id, entry in sorted(render_manifests(batches).items()):
        path = entry["manifest"]
        record = {"tree": tree_id, "registered": entry["registered"], "manifest": str(path), "rendered": 0, "kept": 0}
        if path.is_file():
            data = json.loads(path.read_text())
            if data.get("smoke"):
                raise ValueError(f"Smoke render cannot enter a training manifest: {path}")
            record["manifest_sha256"] = sha256(path)
            for frame in data["frames"]:
                record["rendered"] += 1
                if not frame.get("ok"):
                    continue
                record["kept"] += 1
                lumas.append(frame["mean_luma"])
                rendered_rows.append(
                    {
                        "bark": frame["bark"],
                        "tree": frame["tree"],
                        "set_id": frame["set_id"],
                        "shot": frame["shot"],
                        "view": frame["view"],
                        "lighting": frame["lighting_label"],
                        "rgb_path": frame["rgb"],
                        "depth_path": frame["depth_path"],
                        "mask_path": frame["mask_path"],
                    }
                )
        per_tree.append(record)
    surviving = [{**row, "lighting": "legacy_source"} for row in tl.surviving_rows(surviving_manifest)]
    for name, rows in (("rendered", rendered_rows), ("combined", surviving + rendered_rows)):
        with (output_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    registered = sum(t["registered"] for t in per_tree)
    summary = {
        "schema_version": 1,
        "surviving_manifest": str(surviving_manifest),
        "surviving_manifest_sha256": sha256(surviving_manifest),
        "surviving_rows": len(surviving),
        "registered_frames": registered,
        "rendered_frames": sum(t["rendered"] for t in per_tree),
        "kept_frames": len(rendered_rows),
        "not_rendered": registered - sum(t["rendered"] for t in per_tree),
        "failed_geometry_check": sum(t["rendered"] - t["kept"] for t in per_tree),
        "kept_luma": {
            "mean": float(np.mean(lumas)) if lumas else None,
            "fraction_at_or_below_90": float(np.mean(np.asarray(lumas) <= 90)) if lumas else None,
        },
        "per_tree": per_tree,
        "outputs": {name: sha256(output_dir / f"{name}.csv") for name in ("rendered", "combined")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--surviving-manifest", type=Path, required=True, help="The companion's spur_train.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build(args.batch, args.surviving_manifest, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_tree"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
