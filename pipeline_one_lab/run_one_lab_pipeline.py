#!/usr/bin/env python3
"""Run the full one-lab pipeline: generate, simulate, build dataset, split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-lab end-to-end data pipeline")
    parser.add_argument("--lab", required=True, help="Lab stem, e.g. lab9_task2")
    parser.add_argument("--asc-dir", type=Path, default=Path("LTSpice_files"))
    parser.add_argument("--ltspice-bin", type=str, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument("--variants-per-circuit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-subdir",
        type=str,
        default="finetune_small",
        help="Output subdirectory under each lab folder for finetune files.",
    )
    parser.add_argument("--include-failed-sims", action="store_true")
    parser.add_argument("--use-golden", action=argparse.BooleanOptionalAction, default=True)
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
    )
    parser.add_argument("--measurement-noise-sigma", type=float, default=0.0)
    parser.add_argument("--measurement-noise-prob", type=float, default=1.0)
    parser.add_argument("--max-measurements", type=int, default=24)
    parser.add_argument("--max-deltas", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument(
        "--measurement-stat-mode",
        choices=["full", "max_only", "max_rms"],
        default="max_only",
    )
    parser.add_argument(
        "--prefer-voltage-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--voltage-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--map-resistor-param-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--drop-noop-faults",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--input-mode",
        choices=["full", "delta_only", "delta_plus_measured"],
        default="full",
    )
    parser.add_argument(
        "--output-mode",
        choices=["diag_fix", "faulttype_diag_fix"],
        default="diag_fix",
    )
    parser.add_argument(
        "--canonicalize-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-variant-id",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-lab-id",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")


def main() -> int:
    args = parse_args()

    run(
        [
            sys.executable,
            "pipeline_one_lab/generate_variants_one_lab.py",
            "--lab",
            args.lab,
            "--asc-dir",
            str(args.asc_dir),
            "--out-root",
            str(args.out_root),
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
    )

    sim_cmd = [
        sys.executable,
        "pipeline_one_lab/run_ltspice_batch_one_lab.py",
        "--lab",
        args.lab,
        "--out-root",
        str(args.out_root),
        "--ltspice-bin",
        args.ltspice_bin,
        "--max-workers",
        str(args.max_workers),
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if args.keep_raw:
        sim_cmd.append("--keep-raw")
    run(sim_cmd)

    run(
        [
            sys.executable,
            "pipeline_one_lab/build_dataset_one_lab.py",
            "--lab",
            args.lab,
            "--out-root",
            str(args.out_root),
        ]
    )

    if args.use_golden:
        golden_cmd = [
            sys.executable,
            "pipeline_one_lab/build_golden_one_lab.py",
            "--lab",
            args.lab,
            "--out-root",
            str(args.out_root),
            "--ltspice-bin",
            args.ltspice_bin,
            "--timeout-sec",
            str(args.timeout_sec),
        ]
        if args.keep_raw:
            golden_cmd.append("--keep-raw")
        run(golden_cmd)

    split_cmd = [
        sys.executable,
        "pipeline_one_lab/prepare_finetune_one_lab.py",
        "--lab",
        args.lab,
        "--out-root",
        str(args.out_root),
        "--seed",
        str(args.seed),
        "--val-ratio",
        str(args.val_ratio),
        "--split-subdir",
        str(args.split_subdir),
        "--measurement-noise-sigma",
        str(args.measurement_noise_sigma),
        "--measurement-noise-prob",
        str(args.measurement_noise_prob),
        "--max-measurements",
        str(args.max_measurements),
        "--max-deltas",
        str(args.max_deltas),
        "--max-chars",
        str(args.max_chars),
        "--measurement-stat-mode",
        args.measurement_stat_mode,
        "--prefer-voltage-keys" if args.prefer_voltage_keys else "--no-prefer-voltage-keys",
        "--voltage-only" if args.voltage_only else "--no-voltage-only",
        "--map-resistor-param-drift"
        if args.map_resistor_param_drift
        else "--no-map-resistor-param-drift",
        "--drop-noop-faults" if args.drop_noop_faults else "--no-drop-noop-faults",
        "--canonicalize-output" if args.canonicalize_output else "--no-canonicalize-output",
        "--include-variant-id" if args.include_variant_id else "--no-include-variant-id",
        "--include-lab-id" if args.include_lab_id else "--no-include-lab-id",
        "--input-mode",
        args.input_mode,
        "--output-mode",
        args.output_mode,
    ]
    if args.include_failed_sims:
        split_cmd.append("--include-failed-sims")
    if args.use_golden:
        split_cmd.append("--use-golden")
    run(split_cmd)

    print(f"Done. One-lab outputs are in: {args.out_root / args.lab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
