#!/usr/bin/env python3
"""Generate variants for exactly one lab .asc file.

This is a simple wrapper around pipeline/generate_variants.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate variants for one lab task")
    parser.add_argument("--lab", required=True, help="Lab stem, e.g. lab9_task2 (matches lab9_task2.asc)")
    parser.add_argument("--asc-dir", type=Path, default=Path("LTSpice_files"))
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument("--variants-per-circuit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--ltspice-bin", type=str, required=True)
    parser.add_argument("--weight-param-drift", type=float, default=0.30)
    parser.add_argument("--weight-missing-component", type=float, default=0.12)
    parser.add_argument("--weight-pin-open", type=float, default=0.12)
    parser.add_argument("--weight-swapped-nodes", type=float, default=0.18)
    parser.add_argument("--weight-short-between-nodes", type=float, default=0.08)
    parser.add_argument("--weight-resistor-value-swap", type=float, default=0.20)
    parser.add_argument("--weight-resistor-wrong-value", type=float, default=0.15)
    parser.add_argument("--vsource-min", type=float, default=-5.0)
    parser.add_argument("--vsource-max", type=float, default=5.0)
    parser.add_argument("--param-drift-vsource-prob", type=float, default=0.45)
    parser.add_argument(
        "--param-drift-allow-resistor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow resistor components to be sampled for param_drift",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    asc_path = args.asc_dir / f"{args.lab}.asc"
    if not asc_path.exists():
        raise FileNotFoundError(f"Could not find {asc_path}")

    lab_dir = args.out_root / args.lab
    input_dir = lab_dir / "input_asc"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asc_path, input_dir / asc_path.name)

    cmd = [
        sys.executable,
        "pipeline/generate_variants.py",
        "--asc-dir",
        str(input_dir),
        "--out-dir",
        str(lab_dir),
        "--variants-per-circuit",
        str(args.variants_per_circuit),
        "--seed",
        str(args.seed),
        "--max-workers",
        str(args.max_workers),
        "--ltspice-bin",
        args.ltspice_bin,
        "--weight-param-drift",
        str(args.weight_param_drift),
        "--weight-missing-component",
        str(args.weight_missing_component),
        "--weight-pin-open",
        str(args.weight_pin_open),
        "--weight-swapped-nodes",
        str(args.weight_swapped_nodes),
        "--weight-short-between-nodes",
        str(args.weight_short_between_nodes),
        "--weight-resistor-value-swap",
        str(args.weight_resistor_value_swap),
        "--weight-resistor-wrong-value",
        str(args.weight_resistor_wrong_value),
        "--vsource-min",
        str(args.vsource_min),
        "--vsource-max",
        str(args.vsource_max),
        "--param-drift-vsource-prob",
        str(args.param_drift_vsource_prob),
        "--param-drift-allow-resistor"
        if args.param_drift_allow_resistor
        else "--no-param-drift-allow-resistor",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
