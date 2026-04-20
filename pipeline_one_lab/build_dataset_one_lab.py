#!/usr/bin/env python3
"""Build training dataset for one lab from its manifests.

Simple wrapper around pipeline/build_dataset.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-lab dataset")
    parser.add_argument("--lab", required=True, help="Lab stem, e.g. lab9_task2")
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lab_dir = args.out_root / args.lab

    variant_manifest = lab_dir / "variant_manifest.jsonl"
    sim_manifest = lab_dir / "sim_manifest.jsonl"
    out_file = lab_dir / "training_dataset.jsonl"

    if not variant_manifest.exists():
        raise FileNotFoundError(f"Missing variant manifest: {variant_manifest}")
    if not sim_manifest.exists():
        raise FileNotFoundError(f"Missing sim manifest: {sim_manifest}")

    cmd = [
        sys.executable,
        "pipeline/build_dataset.py",
        "--variant-manifest",
        str(variant_manifest),
        "--sim-manifest",
        str(sim_manifest),
        "--out",
        str(out_file),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

