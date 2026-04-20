"""Extract one instruct eval row into a standalone /debug API JSON payload."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STAT_PRIORITY = {"avg": 0, "max": 1, "rms": 2, "min": 3, "pp": 4}
MEASUREMENT_STATS = {"max", "min", "rms", "avg", "pp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read one instruct-format eval row and write a standalone JSON payload "
            "you can POST directly to /debug."
        )
    )
    p.add_argument("--data-file", type=Path, required=True, help="Tagged instruct JSONL eval/train file.")
    p.add_argument("--row-index", type=int, default=0, help="Zero-based row index to extract.")
    p.add_argument("--out-file", type=Path, required=True, help="Where to write the request JSON payload.")
    p.add_argument(
        "--meta-file",
        type=Path,
        default=Path(""),
        help="Optional sidecar JSON with the source row, parsed sections, and expected answer.",
    )
    p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to set strict=true in the generated API payload.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_key_value_blob(blob: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in (blob or "").split(";"):
        bit = part.strip()
        if not bit or "=" not in bit:
            continue
        key, raw_val = bit.split("=", 1)
        key = key.strip().lower()
        raw_val = raw_val.strip()
        try:
            out[key] = float(raw_val)
        except Exception:
            out[key] = raw_val
    return out


def parse_input_sections(input_text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "circuit_name": None,
        "variant_id": None,
        "lab_name": None,
        "task": None,
        "sim_success": None,
        "measured": {},
        "deltas_vs_golden": {},
        "golden_measurements": {},
        "other_lines": [],
    }
    text = (input_text or "").strip()
    if not text:
        return parsed

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("Circuit:"):
            parsed["circuit_name"] = line.split(":", 1)[1].strip() or None
        elif line.startswith("Variant:") or line.startswith("Variant ID:"):
            parsed["variant_id"] = line.split(":", 1)[1].strip() or None
        elif line.startswith("Lab:"):
            parsed["lab_name"] = line.split(":", 1)[1].strip() or None
        elif lower.startswith("simsuccess:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in {"true", "1", "yes"}:
                parsed["sim_success"] = True
            elif value in {"false", "0", "no"}:
                parsed["sim_success"] = False
            else:
                parsed["sim_success"] = value
        elif line.startswith("Measured:"):
            parsed["measured"] = parse_key_value_blob(line.split(":", 1)[1])
        elif line.startswith("DeltasVsGolden:"):
            parsed["deltas_vs_golden"] = parse_key_value_blob(line.split(":", 1)[1])
        elif line.startswith("GoldenMeasurements:"):
            parsed["golden_measurements"] = parse_key_value_blob(line.split(":", 1)[1])
        elif line.startswith("Task:"):
            parsed["task"] = line.split(":", 1)[1].strip() or None
        else:
            parsed["other_lines"].append(line)
    return parsed


def measurement_subject_to_display_name(subject: str) -> str:
    token = (subject or "").strip()
    if token.startswith("_"):
        token = "-" + token[1:]
    return token.upper()


def normalize_fault_type_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    alias = {
        "param_drift": "param_drift",
        "missing_component": "missing_component",
        "pin_open": "pin_open",
        "swapped_nodes": "swapped_nodes",
        "short_between_nodes": "short_between_nodes",
        "resistor_value_swap": "resistor_value_swap",
        "resistor_wrong_value": "resistor_wrong_value",
    }
    return alias.get(key, "unknown")


def parse_expected_output(output_text: str) -> dict[str, Any]:
    raw = (output_text or "").strip()
    fault_type = "unknown"
    m = re.search(r"^FaultType:\s*([A-Za-z_]+)\s*$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        fault_type = normalize_fault_type_label(m.group(1))
    return {"raw_text": raw, "fault_type": fault_type}


def _group_measurements(
    measured: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float], dict[str, dict[str, float]]]:
    node_voltages: dict[str, float] = {}
    node_measurements: dict[str, dict[str, float]] = {}
    source_currents: dict[str, float] = {}
    source_current_measurements: dict[str, dict[str, float]] = {}

    best_voltage_choice: dict[str, tuple[int, float]] = {}
    best_source_choice: dict[str, tuple[int, float]] = {}

    for key, value in measured.items():
        if not isinstance(value, (int, float)):
            continue
        m = re.match(r"^([vi])_(.+?)_(avg|max|min|rms|pp)$", str(key), flags=re.IGNORECASE)
        if not m:
            continue
        kind = m.group(1).lower()
        subject = measurement_subject_to_display_name(m.group(2))
        stat = m.group(3).lower()
        if stat not in MEASUREMENT_STATS:
            continue
        priority = STAT_PRIORITY.get(stat, 999)
        if kind == "v":
            node_measurements.setdefault(subject, {})[stat] = float(value)
            prev = best_voltage_choice.get(subject)
            if prev is None or priority < prev[0]:
                best_voltage_choice[subject] = (priority, float(value))
        elif kind == "i" and subject.startswith("V"):
            source_current_measurements.setdefault(subject, {})[stat] = float(value)
            prev = best_source_choice.get(subject)
            if prev is None or priority < prev[0]:
                best_source_choice[subject] = (priority, float(value))

    for subject, (_priority, value) in sorted(best_voltage_choice.items()):
        node_voltages[subject] = value
    for subject, (_priority, value) in sorted(best_source_choice.items()):
        source_currents[subject] = value

    return node_voltages, node_measurements, source_currents, source_current_measurements


def build_debug_payload(parsed_input: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    measured = dict(parsed_input.get("measured") or {})
    node_voltages, node_measurements, source_currents, source_current_measurements = _group_measurements(measured)
    circuit_name = (
        parsed_input.get("circuit_name")
        or parsed_input.get("lab_name")
        or parsed_input.get("variant_id")
        or "__fill_in_circuit_name__"
    )
    return {
        "circuit_name": circuit_name,
        "node_voltages": node_voltages,
        "node_measurements": node_measurements,
        "source_currents": source_currents,
        "source_current_measurements": source_current_measurements,
        "measurement_overrides": measured,
        "temp": measured.get("temp", 27.0),
        "tnom": measured.get("tnom", 27.0),
        "strict": bool(strict),
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.data_file)
    if not rows:
        raise SystemExit(f"No rows found in {args.data_file}")
    if args.row_index < 0 or args.row_index >= len(rows):
        raise SystemExit(f"--row-index must be between 0 and {len(rows) - 1}")

    row = rows[args.row_index]
    parsed_input = parse_input_sections(str(row.get("input", "")))
    payload = build_debug_payload(parsed_input, strict=args.strict)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote payload: {args.out_file}")
    print(f"Circuit: {payload['circuit_name']}")

    meta_file = args.meta_file if str(args.meta_file) else None
    if meta_file:
        meta = {
            "source_data_file": str(args.data_file),
            "row_index": args.row_index,
            "instruction": row.get("instruction"),
            "parsed_input": parsed_input,
            "expected": parse_expected_output(str(row.get("output", ""))),
            "payload_file": str(args.out_file),
        }
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote metadata: {meta_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
