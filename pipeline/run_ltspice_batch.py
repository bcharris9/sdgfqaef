#!/usr/bin/env python3
"""Run LTspice batch simulations for generated netlist variants."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTspice simulations in batch")
    parser.add_argument("--variants-dir", type=Path, default=Path("pipeline/out/variants"))
    parser.add_argument("--results-dir", type=Path, default=Path("pipeline/out/sim_results"))
    parser.add_argument("--manifest", type=Path, default=Path("pipeline/out/sim_manifest.jsonl"))
    parser.add_argument("--ltspice-bin", type=str, default="ltspice")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep LTspice .raw artifacts and record raw_path when present",
    )
    return parser.parse_args()


def validate_ltspice_bin(bin_path: str) -> str:
    if "/" in bin_path or "\\" in bin_path:
        if Path(bin_path).exists():
            return bin_path
        raise FileNotFoundError(
            "LTspice executable not found at: "
            f"{bin_path}\n"
            "Pass a valid path with --ltspice-bin."
        )

    resolved = shutil.which(bin_path)
    if resolved:
        return resolved

    raise FileNotFoundError(
        "LTspice executable not found on PATH. "
        "Pass --ltspice-bin with full executable path."
    )


def parse_measurements(log_path: Path) -> dict[str, float | str]:
    if not log_path.exists():
        return {}

    out: dict[str, float | str] = {}
    # Handles common LTspice log formats, including:
    #  - key=value
    #  - meas_name: expr=value [FROM ... TO ...]
    meas_re = re.compile(
        r"^\s*([A-Za-z0-9_.$-]+)\s*[:=].*?=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
    )
    kv_re = re.compile(r"^\s*([A-Za-z0-9_.$-]+)\s*=\s*(\S+)")

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.lower().startswith(("warning", "error")):
            continue

        m = meas_re.match(stripped)
        if m:
            key = m.group(1).strip().replace(" ", "_")
            value = m.group(2)
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
            continue

        m2 = kv_re.match(stripped)
        if m2:
            key = m2.group(1).strip().replace(" ", "_")
            value = m2.group(2).strip()
            if value.endswith(","):
                value = value[:-1]
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value

    return out


def run_sim(ltspice_bin: str, netlist: Path, timeout_sec: int) -> tuple[int, str, str]:
    cmd = [ltspice_bin, "-b", str(netlist)]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Simulation timeout after {timeout_sec}s"


def find_raw_artifact(netlist: Path) -> Path | None:
    stem = netlist.stem
    parent = netlist.parent
    raw_files = sorted(parent.glob(f"{stem}*.raw"))
    if not raw_files:
        return None
    return raw_files[0]


def cleanup_artifacts_for_variant(netlist: Path) -> None:
    stem = netlist.stem
    parent = netlist.parent

    # LTspice can emit different raw artifacts, e.g. <name>.raw, <name>.op.raw, <name>.tran.raw.
    for raw_file in parent.glob(f"{stem}*.raw"):
        try:
            raw_file.unlink()
        except FileNotFoundError:
            pass

    # LTspice may also create a per-schematic database file.
    db_file = parent / f"{stem}.db"
    if db_file.exists():
        try:
            db_file.unlink()
        except FileNotFoundError:
            pass


def simulate_one(
    netlist: Path,
    ltspice_bin: str,
    timeout_sec: int,
    results_dir: Path,
    keep_raw: bool,
) -> dict:
    variant_id = netlist.stem
    rc, stdout, stderr = run_sim(ltspice_bin, netlist, timeout_sec)

    src_log = netlist.with_suffix(".log")
    dst_log = results_dir / f"{variant_id}.log"

    if src_log.exists():
        shutil.move(str(src_log), dst_log)

    raw_file = find_raw_artifact(netlist)

    # Keep raw artifacts only when requested.
    if not keep_raw:
        cleanup_artifacts_for_variant(netlist)

    measurements = parse_measurements(dst_log)
    return {
        "variant_id": variant_id,
        "return_code": rc,
        "success": rc == 0,
        "log_path": str(dst_log) if dst_log.exists() else None,
        "raw_path": str(raw_file) if (keep_raw and raw_file and raw_file.exists()) else None,
        "measurements": measurements,
        "stderr": stderr.strip()[:2000],
        "stdout": stdout.strip()[:2000],
    }


def main() -> int:
    args = parse_args()
    ltspice_bin = validate_ltspice_bin(args.ltspice_bin)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    variant_files = sorted(args.variants_dir.glob("*.cir"))
    if not variant_files:
        print(f"No variants found in {args.variants_dir}")
        return 1

    workers = max(1, args.max_workers)
    with args.manifest.open("w", encoding="utf-8") as mf:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    simulate_one,
                    netlist,
                    ltspice_bin,
                    args.timeout_sec,
                    args.results_dir,
                    args.keep_raw,
                )
                for netlist in variant_files
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                mf.write(json.dumps(row) + "\n")

    print(f"Wrote simulation manifest: {args.manifest}")
    print(f"Simulated {len(variant_files)} variants with max_workers={workers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
