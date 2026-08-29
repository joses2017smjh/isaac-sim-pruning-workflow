#!/usr/bin/env python3
"""Convert a directory of L-Py metadata JSON files to per-tree ASCII USDA.

Login-node safe: does not import pxr. Skips held-out Envy identities unless
``--include-held-out``. UFO trees are converted (eval policy remains untouched).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.assets import load_tree_splits  # noqa: E402
from isaaclab_pruning.geometry.cylinders import load_cylinders, transform_cylinders  # noqa: E402
from isaaclab_pruning.usd.ascii_tree import write_cylinder_tree_usda  # noqa: E402
from isaaclab_pruning.usd.bark import packaged_bark_texture  # noqa: E402
from isaaclab_pruning.usd.cylinders import DEFAULT_TREE_TILT_X_DEG, _local_transform  # noqa: E402


def _tree_id(metadata: Path) -> str:
    return metadata.stem.removesuffix("_metadata")


def convert_one(metadata: Path, output: Path, *, tilt_x_deg: float) -> dict:
    cylinders = transform_cylinders(
        load_cylinders(metadata),
        _local_transform(tilt_x_deg, (0.0, 0.0, 0.0)),
    )
    texture = packaged_bark_texture()
    return write_cylinder_tree_usda(
        cylinders,
        output,
        tree_id=_tree_id(metadata),
        bark_texture=texture if texture.is_file() else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--tilt-x-deg", type=float, default=DEFAULT_TREE_TILT_X_DEG)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    splits = load_tree_splits()
    files = sorted(args.metadata_dir.glob("*_metadata.json"))
    if not files:
        raise SystemExit(f"No *_metadata.json files in {args.metadata_dir}")

    converted: list[dict] = []
    skipped: list[str] = []
    for metadata in files:
        tree_id = _tree_id(metadata)
        if not args.include_held_out and splits.is_held_out(tree_id):
            skipped.append(tree_id)
            continue
        summary = convert_one(
            metadata,
            args.output_dir / f"{tree_id}.usda",
            tilt_x_deg=args.tilt_x_deg,
        )
        converted.append(summary)

    envy = [item for item in converted if "envy" in item["tree_id"]]
    ufo = [item for item in converted if "ufo" in item["tree_id"]]
    report = {
        "ok": True,
        "writer": "ascii_usda",
        "material": "bark_brown_02",
        "usdpreviewsurface": True,
        "tilt_x_deg": args.tilt_x_deg,
        "metadata_dir": str(args.metadata_dir),
        "output_dir": str(args.output_dir),
        "count": len(converted),
        "envy": len(envy),
        "ufo": len(ufo),
        "skipped_held_out": skipped,
        "trees": converted,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(json.dumps({key: report[key] for key in report if key != "trees"}, indent=2))
    destination = args.manifest
    if destination is None:
        destination = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "trees_converted_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
