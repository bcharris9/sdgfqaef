#!/usr/bin/env python3
"""Run LTspice simulations for one lab's generated variants.

Simple wrapper around pipeline/run_ltspice_batch.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTspice batch for one lab")
    parser.add_argument("--lab", required=True, help="Lab stem, e.g. lab9_task2")
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument("--ltspice-bin", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--keep-raw", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lab_dir = args.out_root / args.lab
    variants_dir = lab_dir / "variants"
    results_dir = lab_dir / "sim_results"
    manifest = lab_dir / "sim_manifest.jsonl"

    if not variants_dir.exists():
        raise FileNotFoundError(f"Variants dir not found: {variants_dir}")

    cmd = [
        sys.executable,
        "pipeline/run_ltspice_batch.py",
        "--variants-dir",
        str(variants_dir),
        "--results-dir",
        str(results_dir),
        "--manifest",
        str(manifest),
        "--ltspice-bin",
        args.ltspice_bin,
        "--max-workers",
        str(args.max_workers),
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if args.keep_raw:
        cmd.append("--keep-raw")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

