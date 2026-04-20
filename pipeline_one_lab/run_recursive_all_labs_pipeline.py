#!/usr/bin/env python3
"""Run the one-lab pipeline recursively across all .asc files under a root.

This is the practical path for datasets organized by lab folders (including nested
task folders like LTSpice_files/Lab4/Task */*.asc).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-lab pipeline for all recursive LTspice .asc files")
    p.add_argument("--asc-root", type=Path, default=Path("LTSpice_files"))
    p.add_argument(
        "--ltspice-bin",
        type=str,
        default="",
        help="Path to LTspice executable. If omitted, script tries common install paths.",
    )
    p.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab_all"))
    p.add_argument("--split-subdir", type=str, default="finetune_all")
    p.add_argument("--variants-per-circuit", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=22)
    p.add_argument("--timeout-sec", type=int, default=240)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--include-failed-sims", action="store_true")
    p.add_argument("--use-golden", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log failing circuits and continue processing the rest.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-circuits", type=int, default=0, help="If > 0, process only first N circuits")
    p.add_argument(
        "--exclude-stem-prefix",
        action="append",
        default=["draft"],
        help="Case-insensitive stem prefix to skip (can be repeated). Default skips Draft*.asc.",
    )
    p.add_argument(
        "--exclude-path-substring",
        action="append",
        default=[],
        help="Case-insensitive path substring to skip (can be repeated).",
    )
    p.add_argument("--merge-after", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--merge-dest",
        type=Path,
        default=None,
        help="Merged finetune destination. Default: <out-root>/merged_<split-subdir>",
    )

    # Fault mix / pipeline controls (forwarded to run_one_lab_pipeline.py)
    p.add_argument("--weight-param-drift", type=float, default=0.30)
    p.add_argument("--weight-missing-component", type=float, default=0.12)
    p.add_argument("--weight-pin-open", type=float, default=0.12)
    p.add_argument("--weight-swapped-nodes", type=float, default=0.18)
    p.add_argument("--weight-short-between-nodes", type=float, default=0.08)
    p.add_argument("--weight-resistor-value-swap", type=float, default=0.20)
    p.add_argument("--weight-resistor-wrong-value", type=float, default=0.15)
    p.add_argument("--vsource-min", type=float, default=-5.0)
    p.add_argument("--vsource-max", type=float, default=5.0)
    p.add_argument("--param-drift-vsource-prob", type=float, default=0.45)
    p.add_argument(
        "--param-drift-allow-resistor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--measurement-noise-sigma", type=float, default=0.0)
    p.add_argument("--measurement-noise-prob", type=float, default=1.0)
    p.add_argument("--max-measurements", type=int, default=24)
    p.add_argument("--max-deltas", type=int, default=24)
    p.add_argument("--max-chars", type=int, default=2200)
    p.add_argument("--measurement-stat-mode", choices=["full", "max_only", "max_rms"], default="max_only")
    p.add_argument("--prefer-voltage-keys", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--voltage-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--map-resistor-param-drift", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-noop-faults", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input-mode", choices=["full", "delta_only", "delta_plus_measured"], default="delta_plus_measured")
    p.add_argument("--output-mode", choices=["diag_fix", "faulttype_diag_fix"], default="faulttype_diag_fix")
    p.add_argument("--canonicalize-output", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-variant-id", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-lab-id", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def detect_ltspice_bin(user_value: str) -> str:
    if user_value:
        return user_value
    candidates = [
        r"C:\Program Files\ADI\LTspice\LTspice.exe",
        r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
        "LTspice",
        "ltspice",
    ]
    for c in candidates:
        if ("\\" in c or "/" in c) and Path(c).exists():
            return c
        if shutil.which(c):
            return c
    return ""


def discover_asc_files(args: argparse.Namespace) -> list[Path]:
    root = args.asc_root
    if not root.exists():
        raise FileNotFoundError(f"asc root not found: {root}")

    excl_stem = [s.lower() for s in (args.exclude_stem_prefix or []) if str(s).strip()]
    excl_path = [s.lower() for s in (args.exclude_path_substring or []) if str(s).strip()]

    files: list[Path] = []
    for p in sorted(root.rglob("*.asc")):
        stem_low = p.stem.lower()
        rel_low = str(p.relative_to(root)).replace("/", "\\").lower()
        if any(stem_low.startswith(prefix) for prefix in excl_stem):
            continue
        if any(substr in rel_low for substr in excl_path):
            continue
        files.append(p)

    if args.max_circuits > 0:
        files = files[: args.max_circuits]
    return files


def run(cmd: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")


def build_one_lab_cmd(args: argparse.Namespace, asc_path: Path, idx: int, ltspice_bin: str) -> list[str]:
    lab = asc_path.stem
    cmd = [
        sys.executable,
        "pipeline_one_lab/run_one_lab_pipeline.py",
        "--lab",
        lab,
        "--asc-dir",
        str(asc_path.parent),
        "--ltspice-bin",
        ltspice_bin,
        "--out-root",
        str(args.out_root),
        "--split-subdir",
        args.split_subdir,
        "--variants-per-circuit",
        str(args.variants_per_circuit),
        "--seed",
        str(args.seed + (idx * 1009)),
        "--max-workers",
        str(args.max_workers),
        "--timeout-sec",
        str(args.timeout_sec),
        "--val-ratio",
        str(args.val_ratio),
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
        "--input-mode",
        args.input_mode,
        "--output-mode",
        args.output_mode,
        "--prefer-voltage-keys" if args.prefer_voltage_keys else "--no-prefer-voltage-keys",
        "--voltage-only" if args.voltage_only else "--no-voltage-only",
        "--map-resistor-param-drift" if args.map_resistor_param_drift else "--no-map-resistor-param-drift",
        "--drop-noop-faults" if args.drop_noop_faults else "--no-drop-noop-faults",
        "--canonicalize-output" if args.canonicalize_output else "--no-canonicalize-output",
        "--include-variant-id" if args.include_variant_id else "--no-include-variant-id",
        "--include-lab-id" if args.include_lab_id else "--no-include-lab-id",
        "--param-drift-allow-resistor" if args.param_drift_allow_resistor else "--no-param-drift-allow-resistor",
    ]
    if args.keep_raw:
        cmd.append("--keep-raw")
    if args.include_failed_sims:
        cmd.append("--include-failed-sims")
    cmd.append("--use-golden" if args.use_golden else "--no-use-golden")
    return cmd


def main() -> int:
    args = parse_args()
    asc_files = discover_asc_files(args)
    if not asc_files:
        raise RuntimeError("No .asc files found after filtering.")

    stems = [p.stem for p in asc_files]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Duplicate .asc stems detected across folders; rename or flatten first.")

    ltspice_bin = detect_ltspice_bin(args.ltspice_bin)
    if not ltspice_bin and not args.dry_run:
        raise FileNotFoundError(
            "LTspice executable not found. Pass --ltspice-bin explicitly.\n"
            "Expected e.g. C:\\Program Files\\ADI\\LTspice\\LTspice.exe"
        )

    args.out_root.mkdir(parents=True, exist_ok=True)

    source_manifest = []
    for p in asc_files:
        source_manifest.append(
            {
                "lab": p.stem,
                "asc_path": str(p),
                "asc_relpath": str(p.relative_to(args.asc_root)),
                "asc_parent": str(p.parent),
            }
        )
    (args.out_root / "source_circuit_manifest.json").write_text(
        json.dumps({"count": len(source_manifest), "circuits": source_manifest}, indent=2),
        encoding="utf-8",
    )

    print(f"Discovered circuits: {len(asc_files)}")
    if ltspice_bin:
        print(f"LTspice: {ltspice_bin}")
    if args.dry_run:
        print("Mode: dry-run (commands only)")

    failures: list[dict[str, str | int]] = []
    completed = 0
    skipped = 0
    for i, asc_path in enumerate(asc_files, start=1):
        lab = asc_path.stem
        done_marker = args.out_root / lab / args.split_subdir / "train_instruct.jsonl"
        if args.resume and done_marker.exists():
            print(f"[{i}/{len(asc_files)}] skip (exists): {lab}")
            skipped += 1
            continue
        print(f"\n=== [{i}/{len(asc_files)}] {lab} ===")
        print(f"Source: {asc_path}")
        cmd = build_one_lab_cmd(args, asc_path, i, ltspice_bin or "LTspice")
        try:
            run(cmd, args.dry_run)
            completed += 1
        except Exception as e:
            msg = str(e)
            failures.append(
                {
                    "index": i,
                    "lab": lab,
                    "asc_path": str(asc_path),
                    "error": msg,
                }
            )
            print(f"[ERROR] {lab}: {msg}")
            if not args.continue_on_error:
                raise
            continue

    failure_path = args.out_root / "failed_circuits.json"
    failure_payload = {
        "total_selected": len(asc_files),
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }
    failure_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")
    print(
        "Summary: "
        f"selected={len(asc_files)} completed={completed} skipped={skipped} failed={len(failures)}"
    )
    print(f"Failure log: {failure_path}")

    if args.merge_after:
        merge_dest = args.merge_dest or (args.out_root / f"merged_{args.split_subdir}")
        merge_cmd = [
            sys.executable,
            "pipeline_one_lab/merge_finetune_sets.py",
            "--out-root",
            str(args.out_root),
            "--finetune-subdir",
            args.split_subdir,
            "--dest-dir",
            str(merge_dest),
            "--shuffle",
        ]
        run(merge_cmd, args.dry_run)
        print(f"Merged finetune set: {merge_dest}")

    print(f"Done. Outputs under: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
