#!/usr/bin/env python3
"""Run the one-lab pipeline across a set of golden circuits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-lab pipeline for many golden circuits")
    parser.add_argument("--asc-dir", type=Path, default=Path("LTSpice_files"))
    parser.add_argument(
        "--lab-prefix",
        type=str,
        default="",
        help="Only include .asc stems that start with this prefix",
    )
    parser.add_argument(
        "--labs",
        type=str,
        default="",
        help="Comma-separated lab stems. If omitted, all .asc files in --asc-dir are used.",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=0,
        help="If > 0, require exactly this many selected labs",
    )
    parser.add_argument("--ltspice-bin", type=str, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument("--variants-per-circuit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=240)
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


def detect_labs(asc_dir: Path, prefix: str) -> list[str]:
    stems = sorted(p.stem for p in asc_dir.glob("*.asc"))
    if not prefix:
        return stems
    return [s for s in stems if s.startswith(prefix)]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")


def main() -> int:
    args = parse_args()

    if args.labs.strip():
        labs = [x.strip() for x in args.labs.split(",") if x.strip()]
    else:
        labs = detect_labs(args.asc_dir, args.lab_prefix.strip())

    if not labs:
        raise RuntimeError("No labs to process.")
    if args.expect_count > 0 and len(labs) != args.expect_count:
        raise RuntimeError(
            f"Selected {len(labs)} labs, but --expect-count={args.expect_count}. "
            f"asc_dir={args.asc_dir} prefix={args.lab_prefix!r}"
        )

    print(f"Golden set labs ({len(labs)}): {', '.join(labs)}")

    for i, lab in enumerate(labs, start=1):
        print(f"\n=== [{i}/{len(labs)}] {lab} ===")
        cmd = [
            sys.executable,
            "pipeline_one_lab/run_one_lab_pipeline.py",
            "--lab",
            lab,
            "--asc-dir",
            str(args.asc_dir),
            "--ltspice-bin",
            args.ltspice_bin,
            "--out-root",
            str(args.out_root),
            "--variants-per-circuit",
            str(args.variants_per_circuit),
            "--seed",
            str(args.seed + (i * 1009)),
            "--max-workers",
            str(args.max_workers),
            "--timeout-sec",
            str(args.timeout_sec),
            "--val-ratio",
            str(args.val_ratio),
            "--split-subdir",
            str(args.split_subdir),
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
        if args.keep_raw:
            cmd.append("--keep-raw")
        if args.include_failed_sims:
            cmd.append("--include-failed-sims")
        if args.use_golden:
            cmd.append("--use-golden")
        else:
            cmd.append("--no-use-golden")
        run(cmd)

    print(f"\nDone. All one-lab outputs are under: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
