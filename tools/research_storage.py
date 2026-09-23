#!/usr/bin/env python3
"""Measure storage before a bounded pruning experiment; refuse unsafe projections."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: Warning line, in bytes of `du -sx` apparent size on the user's share. This is
#: the repository's own policy, not a filesystem quota: `lfs quota` reports no
#: enforced limit on /nfs/hpc/share. Raised from 1.5 TB to 1.6 TB on 2026-09-23
#: with the user's approval, after the share measured 1.548 TB (Lustre block
#: count 1.404 TB) and the registered Envy/UFO matrix, about 0.4 GB, was refused.
#: The hard limit is unchanged.
WARNING_BYTES = 1_600_000_000_000
HARD_LIMIT_BYTES = 2_000_000_000_000


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
