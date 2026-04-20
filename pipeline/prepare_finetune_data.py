#!/usr/bin/env python3
"""Prepare compact SFT datasets from training_dataset.jsonl for small models.

This script is designed for edge-friendly fine-tuning workflows (e.g. Jetson Orin Nano):
- Shorter prompts to reduce token/memory cost.
- Circuit-level train/validation split to reduce leakage.
- Output formats for common SFT pipelines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare compact fine-tune datasets")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("pipeline/out/training_dataset.jsonl"),
        help="Input training dataset JSONL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pipeline/out/finetune_small"),
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split",
    )
    parser.add_argument(
        "--val-circuits",
        type=int,
        default=2,
        help="Number of full circuits to reserve for validation",
    )
    parser.add_argument(
        "--include-failed-sims",
        action="store_true",
        help="Include rows where sim_success is false",
    )
    parser.add_argument(
        "--max-measurements",
        type=int,
        default=12,
        help="Max measurement key/value pairs included in prompt",
    )
    parser.add_argument(
        "--voltage-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only voltage-related measurement keys in prompts",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Hard cap for user prompt length",
    )
    parser.add_argument(
        "--measurement-noise-sigma",
        type=float,
        default=0.0,
        help="Relative Gaussian noise sigma applied to numeric measurements (e.g. 0.01 = 1%).",
    )
    parser.add_argument(
        "--measurement-noise-prob",
        type=float,
        default=1.0,
        help="Probability of applying noise to each numeric measurement.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You diagnose LTspice circuit faults from compact simulation summaries. "
            "Return a concise diagnosis and fix."
        ),
        help="System prompt for chat format",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def circuit_id_from_variant(variant_id: str) -> str:
    if "__" not in variant_id:
        return variant_id
    return variant_id.split("__", 1)[0]


def compact_measurements(measurements: dict, max_measurements: int, voltage_only: bool) -> str:
    if not measurements:
        return "none"

    keys = sorted(measurements.keys())
    if voltage_only:
        keys = [k for k in keys if k.lower().startswith("v_")]
        if not keys:
            return "none"

    # Keep metadata keys first if present; fill remainder alphabetically.
    priority = ["solver", "method", "temp", "tnom"]
    ordered: list[str] = []
    for key in priority:
        if key in measurements:
            ordered.append(key)
    for key in keys:
        if key not in ordered:
            ordered.append(key)

    selected = ordered[:max(1, max_measurements)]
    parts: list[str] = []
    for key in selected:
        parts.append(f"{key}={measurements.get(key)}")

    return "; ".join(parts)


def apply_measurement_noise(
    measurements: dict,
    rng: random.Random,
    sigma: float,
    prob: float,
    voltage_only: bool,
) -> dict:
    if not measurements or sigma <= 0:
        return dict(measurements or {})

    out: dict = dict(measurements)
    prob = max(0.0, min(1.0, prob))
    for k, v in list(out.items()):
        if voltage_only and not str(k).lower().startswith("v_"):
            continue
        if not isinstance(v, (int, float)):
            continue
        if rng.random() > prob:
            continue
        scale = max(abs(float(v)), 1e-6)
        out[k] = float(v) + rng.gauss(0.0, sigma * scale)
    return out


def build_user_prompt(
    row: dict,
    max_measurements: int,
    max_chars: int,
    voltage_only: bool,
    noise_sigma: float,
    noise_prob: float,
    rng: random.Random,
) -> str:
    variant_id = row.get("variant_id", "unknown")
    circuit_id = circuit_id_from_variant(variant_id)
    sim_success = row.get("sim_success", False)
    noisy_meas = apply_measurement_noise(
        row.get("measurements", {}),
        rng,
        max(0.0, noise_sigma),
        noise_prob,
        voltage_only,
    )
    meas_text = compact_measurements(noisy_meas, max_measurements, voltage_only)

    text = (
        f"Circuit: {circuit_id}\n"
        f"SimSuccess: {sim_success}\n"
        f"Measurements: {meas_text}\n"
        "Task: diagnose the fault and propose the fix."
    )
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def to_openai_chat_row(system_prompt: str, user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def to_instruct_row(user_text: str, assistant_text: str) -> dict:
    return {
        "instruction": "Diagnose the LTspice fault and provide a fix.",
        "input": user_text,
        "output": assistant_text,
    }


def split_by_circuit(rows: list[dict], val_circuits: int, seed: int) -> tuple[list[dict], list[dict], list[str]]:
    circuit_ids = sorted({circuit_id_from_variant(r.get("variant_id", "")) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(circuit_ids)

    n_val = max(1, min(val_circuits, max(1, len(circuit_ids) - 1)))
    val_set = set(circuit_ids[:n_val])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for row in rows:
        cid = circuit_id_from_variant(row.get("variant_id", ""))
        if cid in val_set:
            val_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, val_rows, sorted(val_set)


def main() -> int:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    rows = load_jsonl(args.input_path)
    if not rows:
        raise RuntimeError("Input dataset is empty")

    if not args.include_failed_sims:
        rows = [r for r in rows if r.get("sim_success") is True]
        if not rows:
            raise RuntimeError("No rows left after filtering failed simulations")

    train_raw, val_raw, val_circuit_ids = split_by_circuit(rows, args.val_circuits, args.seed)

    train_chat: list[dict] = []
    val_chat: list[dict] = []
    train_instruct: list[dict] = []
    val_instruct: list[dict] = []
    noise_rng = random.Random(args.seed + 99991)

    for raw_row in train_raw:
        user_text = build_user_prompt(
            raw_row,
            args.max_measurements,
            args.max_chars,
            args.voltage_only,
            args.measurement_noise_sigma,
            args.measurement_noise_prob,
            noise_rng,
        )
        assistant_text = raw_row.get("completion", "").strip()
        train_chat.append(to_openai_chat_row(args.system_prompt, user_text, assistant_text))
        train_instruct.append(to_instruct_row(user_text, assistant_text))

    for raw_row in val_raw:
        user_text = build_user_prompt(
            raw_row,
            args.max_measurements,
            args.max_chars,
            args.voltage_only,
            args.measurement_noise_sigma,
            args.measurement_noise_prob,
            noise_rng,
        )
        assistant_text = raw_row.get("completion", "").strip()
        val_chat.append(to_openai_chat_row(args.system_prompt, user_text, assistant_text))
        val_instruct.append(to_instruct_row(user_text, assistant_text))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_chat_path = args.out_dir / "train_chat.jsonl"
    val_chat_path = args.out_dir / "val_chat.jsonl"
    train_instruct_path = args.out_dir / "train_instruct.jsonl"
    val_instruct_path = args.out_dir / "val_instruct.jsonl"
    meta_path = args.out_dir / "split_meta.json"

    write_jsonl(train_chat_path, train_chat)
    write_jsonl(val_chat_path, val_chat)
    write_jsonl(train_instruct_path, train_instruct)
    write_jsonl(val_instruct_path, val_instruct)

    meta = {
        "input_path": str(args.input_path),
        "include_failed_sims": bool(args.include_failed_sims),
        "seed": args.seed,
        "val_circuits": args.val_circuits,
        "val_circuit_ids": val_circuit_ids,
        "max_measurements": args.max_measurements,
        "voltage_only": bool(args.voltage_only),
        "max_chars": args.max_chars,
        "measurement_noise_sigma": args.measurement_noise_sigma,
        "measurement_noise_prob": args.measurement_noise_prob,
        "train_rows": len(train_raw),
        "val_rows": len(val_raw),
        "outputs": {
            "train_chat": str(train_chat_path),
            "val_chat": str(val_chat_path),
            "train_instruct": str(train_instruct_path),
            "val_instruct": str(val_instruct_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote train chat: {train_chat_path}")
    print(f"Wrote val chat: {val_chat_path}")
    print(f"Wrote train instruct: {train_instruct_path}")
    print(f"Wrote val instruct: {val_instruct_path}")
    print(f"Wrote split metadata: {meta_path}")
    print(f"Rows used: {len(rows)} (train={len(train_raw)}, val={len(val_raw)})")
    print(f"Validation circuits: {', '.join(val_circuit_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
