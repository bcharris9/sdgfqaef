#!/usr/bin/env python3
"""Build JSONL training dataset by joining variant + simulation manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training dataset JSONL")
    parser.add_argument("--variant-manifest", type=Path, default=Path("pipeline/out/variant_manifest.jsonl"))
    parser.add_argument("--sim-manifest", type=Path, default=Path("pipeline/out/sim_manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("pipeline/out/training_dataset.jsonl"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_measurements(measurements: dict) -> str:
    if not measurements:
        return "No parsed measurements."
    parts = [f"{k}: {v}" for k, v in sorted(measurements.items())]
    return "; ".join(parts)


def build_target_text(fault: dict) -> str:
    ftype = fault.get("fault_type", "unknown")
    comp = fault.get("component", "unknown_component")

    if ftype == "param_drift":
        return (
            f"Diagnosis: parameter drift in {comp}. "
            f"Fix: restore value from {fault.get('new_value')} toward {fault.get('old_value')}."
        )
    if ftype == "missing_component":
        return f"Diagnosis: missing component {comp}. Fix: reinsert component {comp} with correct value/model."
    if ftype == "pin_open":
        return f"Diagnosis: open connection on {comp}. Fix: reconnect opened pin to its intended node."
    if ftype == "swapped_nodes":
        return f"Diagnosis: swapped terminals on {comp}. Fix: swap the two nodes back."
    if ftype == "short_between_nodes":
        return "Diagnosis: unintended short between nodes. Fix: remove short and restore proper wiring."
    if ftype == "resistor_value_swap":
        comps = fault.get("components", [])
        if len(comps) >= 2:
            return (
                f"Diagnosis: resistor values were swapped between {comps[0]} and {comps[1]}. "
                f"Fix: restore each resistor to its intended value."
            )
        return "Diagnosis: resistor values were swapped. Fix: restore each resistor to its intended value."
    if ftype == "resistor_wrong_value":
        return (
            f"Diagnosis: wrong resistor value on {comp}. "
            f"Fix: change value from {fault.get('new_value')} back toward {fault.get('old_value')}."
        )
    return "Diagnosis: unknown fault. Fix: inspect connectivity and component values."


def main() -> int:
    args = parse_args()

    variant_rows = load_jsonl(args.variant_manifest)
    sim_rows = load_jsonl(args.sim_manifest)
    sim_by_id = {row["variant_id"]: row for row in sim_rows}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.out.open("w", encoding="utf-8") as out:
        for v in variant_rows:
            variant_id = v["variant_id"]
            s = sim_by_id.get(variant_id)
            if s is None:
                continue

            fault = v.get("fault", {})
            measurements = s.get("measurements", {})

            prompt = (
                "You are debugging an LTspice circuit. "
                f"Variant ID: {variant_id}. "
                f"Circuit source: {v.get('source_netlist')}. "
                f"Simulation success: {s.get('success')}. "
                f"Measurements: {format_measurements(measurements)}"
            )
            completion = build_target_text(fault)

            row = {
                "variant_id": variant_id,
                "prompt": prompt,
                "completion": completion,
                "fault": fault,
                "measurements": measurements,
                "sim_success": s.get("success"),
                "raw_path": s.get("raw_path"),
                "log_path": s.get("log_path"),
            }
            out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
