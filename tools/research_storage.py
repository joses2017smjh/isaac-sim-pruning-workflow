#!/usr/bin/env python3
"""Measure storage before a bounded pruning experiment; refuse unsafe projections."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: Warning line, in bytes of `du -sx` apparent size on the user's share.
#: Corrected on 2026-09-27 to the share's real limit: the share belongs to Lustre
#: project 30762, whose block quota is 1.5 TiB soft and 2 TiB hard with a grace
#: time of 4 weeks 2 days (`lfs quota -p 30762 /nfs/hpc/share`; `df` reports the
#: same 1,649,267,441,664-byte size). The earlier text said `lfs quota` showed no
#: limit; that was the user quota, which is unset. The line had been raised to
#: 1.6 TB and then 1.7 TB on 2026-09-23 with the user's approval, both above the
#: soft quota without anyone knowing it. On 2026-09-27 the user chose to let the
#: already-queued perception round run past the soft quota and free space later,
#: and asked that future batches check the real quota. Later on 2026-09-27, when
#: approving the rendered-lighting training batch, the user chose the hard limit
#: as the line: the home quota is the one that blocks their HPC use, and the
#: hpc-share's binding limit is 2 TiB hard. The soft quota is still recorded in
#: every assessment so the grace-clock risk stays visible.
SOFT_QUOTA_BYTES = 1_649_267_441_664
#: The project's hard block limit, 2 TiB. Writes fail at once above it.
HARD_LIMIT_BYTES = 2_199_023_255_552
#: Refuse when used + estimate + reserve reaches the hard limit (the reserve is added in ``assess``).
WARNING_BYTES = HARD_LIMIT_BYTES


def assess(used, additional, reserve=20_000_000_000, warning=WARNING_BYTES, hard=HARD_LIMIT_BYTES):
    if min(used, additional, reserve) < 0:
        raise ValueError("Storage byte counts must be nonnegative")
    projected = used + additional + reserve
    return {
        "known_used_bytes": used,
        "estimated_output_bytes": additional,
        "safety_reserve_bytes": reserve,
        "projected_with_reserve_bytes": projected,
        "warning_bytes": warning,
        "hard_limit_bytes": hard,
        "soft_quota_bytes": SOFT_QUOTA_BYTES,
        "over_soft_quota": projected >= SOFT_QUOTA_BYTES,
        "ok": projected < warning and projected < hard,
    }


def measure(path):
    result = subprocess.run(["du", "-sx", "-B1", str(path)], capture_output=True, text=True, timeout=900)
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"No storage measurement for {path}: {result.stderr}")
    used = int(lines[-1].split()[0])
    warnings = result.stderr.strip().splitlines()
    # The only pre-reviewed incomplete entries are inaccessible Kit screenshot
    # cache locations. Keep the scan incomplete flag and reserve 20 GB regardless.
    known = all(
        "/kit/data/documents/Kit/shared/screenshots" in line and "Permission denied" in line for line in warnings
    )
    return {
        "path": str(path),
        "bytes": used,
        "complete": result.returncode == 0,
        "warnings": warnings,
        "only_known_kit_screenshot_warnings": known,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--additional-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-known-kit-warnings",
        action="store_true",
        help="Keep 20 GB uncertainty reserve; never suppress other scan failures",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    share = measure(args.share)
    home_df = subprocess.run(["df", "-B1", str(args.home)], capture_output=True, text=True, check=True).stdout
    project = measure(args.project)
    report = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "share": share,
        "project": project,
        "home_df": home_df,
        "projection": assess(share["bytes"], args.additional_bytes),
        "policy": "No heavy home writes; reserve 20 GB; refuse at warning even below hard limit.",
    }
    scan_ok = share["complete"] or (args.allow_known_kit_warnings and share["only_known_kit_screenshot_warnings"])
    report["ok"] = scan_ok and project["complete"] and report["projection"]["ok"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
