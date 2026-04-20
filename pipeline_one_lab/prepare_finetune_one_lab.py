#!/usr/bin/env python3
"""Prepare train/val fine-tune files for one lab task.

Unlike the multi-circuit splitter, this uses row-level splitting because a
single lab has only one circuit id.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one-lab fine-tune splits")
    parser.add_argument("--lab", required=True, help="Lab stem, e.g. lab9_task2")
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument(
        "--split-subdir",
        type=str,
        default="finetune_small",
        help="Output subdirectory name under each lab folder for generated splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--include-failed-sims", action="store_true")
    parser.add_argument("--max-measurements", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--use-golden", action="store_true")
    parser.add_argument("--max-deltas", type=int, default=24)
    parser.add_argument(
        "--measurement-stat-mode",
        choices=["full", "max_only", "max_rms"],
        default="max_only",
        help="Which statistic suffixes to keep from *_max/*_min/*_rms measurement keys.",
    )
    parser.add_argument(
        "--prefer-voltage-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize v_* keys before i_* keys when truncating measurements/deltas.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["full", "delta_only", "delta_plus_measured"],
        default="full",
        help="Prompt content mode. delta_only focuses on Golden-vs-Measured deltas.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["diag_fix", "faulttype_diag_fix"],
        default="diag_fix",
        help="Assistant target style. faulttype_diag_fix adds a first line: FaultType: <class>.",
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
        "--voltage-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only voltage-related measurement keys in prompts and golden deltas",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You diagnose LTspice circuit faults from compact simulation summaries. Return a concise diagnosis and fix.",
    )
    parser.add_argument(
        "--ambiguity-policy",
        choices=["none", "drop", "majority"],
        default="none",
        help=(
            "How to handle identical measurement signatures that map to multiple fault classes. "
            "drop removes all conflicting signatures; majority keeps only rows from the majority class."
        ),
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance training rows per class after split using up/down sampling.",
    )
    parser.add_argument(
        "--balance-target-per-class",
        type=int,
        default=0,
        help="If > 0 and --balance-classes, use this as per-class target count; otherwise use max class count.",
    )
    parser.add_argument(
        "--balance-max-per-class",
        type=int,
        default=0,
        help="If > 0 and --balance-classes, cap the target per-class count at this value.",
    )
    parser.add_argument(
        "--canonicalize-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit class-level canonical diagnosis/fix text instead of variant-specific completion text.",
    )
    parser.add_argument(
        "--include-variant-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include variant id in prompt input text.",
    )
    parser.add_argument(
        "--include-lab-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include stable lab/circuit id in prompt input text (derived from variant id prefix).",
    )
    parser.add_argument(
        "--map-resistor-param-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map param_drift rows on resistor components to resistor_wrong_value class.",
    )
    parser.add_argument(
        "--drop-noop-faults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows where old_value and new_value are effectively identical.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def format_measurement_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_suffix_priority(key: str, stat_mode: str) -> tuple[int, bool]:
    low = key.lower()
    if low.endswith("_max"):
        suffix_rank = 0
    elif low.endswith("_rms"):
        suffix_rank = 1
    elif low.endswith("_min"):
        suffix_rank = 2
    else:
        suffix_rank = 3

    if stat_mode == "full":
        allowed = True
    elif stat_mode == "max_only":
        allowed = low.endswith("_max") or ("_" not in low)
    else:  # max_rms
        allowed = low.endswith("_max") or low.endswith("_rms") or ("_" not in low)
    return suffix_rank, allowed


def measurement_group_priority(key: str, prefer_voltage_keys: bool) -> int:
    low = key.lower()
    if low.startswith("v_"):
        return 0 if prefer_voltage_keys else 1
    if low.startswith("i_"):
        return 1 if prefer_voltage_keys else 0
    if low in {"temp", "tnom"}:
        return 2
    if low in {"method", "solver"}:
        return 3
    return 4


def ordered_measurement_keys(
    measurements: dict,
    max_measurements: int,
    voltage_only: bool,
    stat_mode: str,
    prefer_voltage_keys: bool,
) -> list[str]:
    if not measurements:
        return []

    ranked: list[tuple[tuple[int, int, str], str]] = []
    for key in measurements.keys():
        low = str(key).lower()
        if voltage_only and not low.startswith("v_"):
            continue
        suffix_rank, allowed = metric_suffix_priority(low, stat_mode)
        if not allowed:
            continue
        group_rank = measurement_group_priority(low, prefer_voltage_keys)
        ranked.append(((group_rank, suffix_rank, low), str(key)))

    ranked.sort(key=lambda item: item[0])
    keys = [k for _, k in ranked]
    return keys[: max(1, max_measurements)]


def compact_measurements(
    measurements: dict,
    max_measurements: int,
    voltage_only: bool,
    stat_mode: str,
    prefer_voltage_keys: bool,
) -> str:
    if not measurements:
        return "none"
    keys = ordered_measurement_keys(
        measurements,
        max_measurements,
        voltage_only,
        stat_mode,
        prefer_voltage_keys,
    )
    if not keys:
        return "none"
    return "; ".join(f"{k}={format_measurement_value(measurements.get(k))}" for k in keys)


def compact_deltas(
    measurements: dict,
    golden: dict,
    max_deltas: int,
    voltage_only: bool,
    stat_mode: str,
    prefer_voltage_keys: bool,
) -> str:
    if not measurements or not golden:
        return "none"
    out: list[str] = []
    keys = ordered_measurement_keys(
        measurements,
        max_deltas,
        voltage_only,
        stat_mode,
        prefer_voltage_keys,
    )
    for k in keys:
        if k not in golden:
            continue
        v = measurements.get(k)
        g = golden.get(k)
        if isinstance(v, (int, float)) and isinstance(g, (int, float)):
            out.append(f"{k}_delta={v - g:.6g}")
    return "; ".join(out) if out else "none"


def normalize_fault_type(value: str) -> str:
    key = (value or "").strip().lower()
    allowed = {
        "param_drift",
        "missing_component",
        "pin_open",
        "swapped_nodes",
        "short_between_nodes",
        "resistor_value_swap",
        "resistor_wrong_value",
    }
    if key in allowed:
        return key
    return "unknown"


def infer_fault_type_from_text(text: str) -> str:
    t = (text or "").lower()
    if "wrong resistor value" in t:
        return "resistor_wrong_value"
    if "resistor values were swapped" in t or ("swapped" in t and "resistor" in t):
        return "resistor_value_swap"
    if "parameter drift" in t:
        return "param_drift"
    if "missing component" in t:
        return "missing_component"
    if "open connection" in t:
        return "pin_open"
    if "swapped terminals" in t:
        return "swapped_nodes"
    if "short between nodes" in t or "unintended short" in t:
        return "short_between_nodes"
    return "unknown"


def infer_fault_type_from_row(row: dict, map_resistor_param_drift: bool = False) -> str:
    fault = row.get("fault", {}) or {}
    fault_type = normalize_fault_type(str(fault.get("fault_type", "")))
    if fault_type == "param_drift" and map_resistor_param_drift:
        component = str(fault.get("component", "")).strip().upper()
        if component.startswith("R"):
            return "resistor_wrong_value"
    if fault_type != "unknown":
        return fault_type
    return infer_fault_type_from_text((row.get("completion", "") or "").strip())


def values_effectively_equal(old_value: object, new_value: object) -> bool:
    old_s = str(old_value).strip()
    new_s = str(new_value).strip()
    try:
        old_f = float(old_s)
        new_f = float(new_s)
    except (TypeError, ValueError):
        return old_s.lower() == new_s.lower()
    tol = max(1e-12, 1e-6 * max(1.0, abs(old_f), abs(new_f)))
    return abs(old_f - new_f) <= tol


def row_has_noop_fault(row: dict) -> bool:
    fault = row.get("fault", {}) or {}
    fault_type = normalize_fault_type(str(fault.get("fault_type", "")))
    if fault_type not in {"param_drift", "resistor_wrong_value"}:
        return False
    if "old_value" not in fault or "new_value" not in fault:
        return False
    return values_effectively_equal(fault.get("old_value"), fault.get("new_value"))


def canonical_completion_for_fault(fault_type: str) -> str:
    templates = {
        "param_drift": (
            "Diagnosis: parameter drift in one or more components. "
            "Fix: restore drifted parameter values to their intended targets."
        ),
        "missing_component": (
            "Diagnosis: missing component in the circuit path. "
            "Fix: reinsert the missing component with the intended value/model."
        ),
        "pin_open": (
            "Diagnosis: open connection on a component terminal. "
            "Fix: reconnect the opened pin to its intended node."
        ),
        "swapped_nodes": (
            "Diagnosis: swapped terminals on a component/source. "
            "Fix: swap the two connections back to their intended nodes."
        ),
        "short_between_nodes": (
            "Diagnosis: unintended short between nodes. "
            "Fix: remove the short and restore proper wiring."
        ),
        "resistor_value_swap": (
            "Diagnosis: resistor values were swapped between two resistors. "
            "Fix: restore each resistor to its intended value."
        ),
        "resistor_wrong_value": (
            "Diagnosis: wrong resistor value on one resistor. "
            "Fix: change that resistor back to its intended value."
        ),
    }
    return templates.get(
        fault_type,
        "Diagnosis: unknown fault class from provided measurements. Fix: inspect wiring and component values.",
    )


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
    golden_measurements: dict | None,
    max_deltas: int,
    voltage_only: bool,
    measurement_stat_mode: str,
    prefer_voltage_keys: bool,
    noise_sigma: float,
    noise_prob: float,
    rng: random.Random,
    input_mode: str,
    include_variant_id: bool,
    include_lab_id: bool,
) -> str:
    variant_id = row.get("variant_id", "unknown")
    sim_success = row.get("sim_success", False)
    lab_id = str(variant_id).split("__", 1)[0] if "__" in str(variant_id) else str(variant_id)
    variant_line = f"Variant: {variant_id}\n" if include_variant_id else ""
    lab_line = f"Lab: {lab_id}\n" if include_lab_id else ""
    noisy_meas = apply_measurement_noise(
        row.get("measurements", {}),
        rng,
        max(0.0, noise_sigma),
        noise_prob,
        voltage_only,
    )
    meas_text = compact_measurements(
        noisy_meas,
        max_measurements,
        voltage_only,
        measurement_stat_mode,
        prefer_voltage_keys,
    )
    if golden_measurements:
        g_text = compact_measurements(
            golden_measurements,
            max_measurements,
            voltage_only,
            measurement_stat_mode,
            prefer_voltage_keys,
        )
        d_text = compact_deltas(
            noisy_meas,
            golden_measurements,
            max_deltas,
            voltage_only,
            measurement_stat_mode,
            prefer_voltage_keys,
        )
        if input_mode == "delta_only":
            text = (
                f"{variant_line}"
                f"{lab_line}"
                f"SimSuccess: {sim_success}\n"
                f"DeltasVsGolden: {d_text}\n"
                "Task: choose the most likely fault class and provide diagnosis/fix."
            )
        elif input_mode == "delta_plus_measured":
            text = (
                f"{variant_line}"
                f"{lab_line}"
                f"SimSuccess: {sim_success}\n"
                f"DeltasVsGolden: {d_text}\n"
                f"Measured: {meas_text}\n"
                "Task: choose the most likely fault class and provide diagnosis/fix."
            )
        else:
            text = (
                f"{variant_line}"
                f"{lab_line}"
                f"SimSuccess: {sim_success}\n"
                f"GoldenMeasurements: {g_text}\n"
                f"Measured: {meas_text}\n"
                f"DeltasVsGolden: {d_text}\n"
                "Task: diagnose the fault and propose the fix."
            )
    else:
        if input_mode in {"delta_only", "delta_plus_measured"}:
            text = (
                f"{variant_line}"
                f"{lab_line}"
                f"SimSuccess: {sim_success}\n"
                f"Measurements: {meas_text}\n"
                "Task: diagnose the fault and propose the fix."
            )
        else:
            text = (
                f"{variant_line}"
                f"{lab_line}"
                f"SimSuccess: {sim_success}\n"
                f"Measurements: {meas_text}\n"
                "Task: diagnose the fault and propose the fix."
            )
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def to_chat(system_prompt: str, user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def to_instruct(user_text: str, assistant_text: str, output_mode: str) -> dict:
    if output_mode == "faulttype_diag_fix":
        instruction = (
            "Classify the LTspice fault and provide a fix. "
            "Return exactly two lines: FaultType then Diagnosis/Fix. "
            "Use resistor_wrong_value when a resistor value is incorrect. "
            "Use param_drift for non-resistor parameter drift (for example source/supply drift)."
        )
    else:
        instruction = "Diagnose the LTspice fault and provide a fix."
    return {
        "instruction": instruction,
        "input": user_text,
        "output": assistant_text,
    }


def build_assistant_output(
    row: dict,
    output_mode: str,
    canonicalize_output: bool,
    map_resistor_param_drift: bool,
) -> str:
    fault_type = infer_fault_type_from_row(row, map_resistor_param_drift)
    if canonicalize_output:
        base = canonical_completion_for_fault(fault_type)
    else:
        base = (row.get("completion", "") or "").strip()
    if output_mode == "diag_fix":
        return base
    return f"FaultType: {fault_type}\n{base}"


def build_signature_text(
    row: dict,
    golden_measurements: dict | None,
    max_measurements: int,
    max_deltas: int,
    voltage_only: bool,
    measurement_stat_mode: str,
    prefer_voltage_keys: bool,
    input_mode: str,
) -> str:
    measurements = row.get("measurements", {}) or {}
    if golden_measurements:
        deltas = compact_deltas(
            measurements,
            golden_measurements,
            max_deltas,
            voltage_only,
            measurement_stat_mode,
            prefer_voltage_keys,
        )
        if input_mode == "delta_only":
            return f"deltas:{deltas}"
        measured = compact_measurements(
            measurements,
            max_measurements,
            voltage_only,
            measurement_stat_mode,
            prefer_voltage_keys,
        )
        if input_mode == "delta_plus_measured":
            return f"deltas:{deltas}|measured:{measured}"
        golden = compact_measurements(
            golden_measurements,
            max_measurements,
            voltage_only,
            measurement_stat_mode,
            prefer_voltage_keys,
        )
        return f"golden:{golden}|measured:{measured}|deltas:{deltas}"
    measured = compact_measurements(
        measurements,
        max_measurements,
        voltage_only,
        measurement_stat_mode,
        prefer_voltage_keys,
    )
    return f"measured:{measured}"


def apply_ambiguity_policy(
    rows: list[dict],
    ambiguity_policy: str,
    golden_measurements: dict | None,
    max_measurements: int,
    max_deltas: int,
    voltage_only: bool,
    measurement_stat_mode: str,
    prefer_voltage_keys: bool,
    input_mode: str,
    map_resistor_param_drift: bool,
) -> tuple[list[dict], dict]:
    if ambiguity_policy == "none":
        return rows, {
            "kept_rows": len(rows),
            "dropped_rows": 0,
            "ambiguous_signatures": 0,
            "policy": ambiguity_policy,
        }

    by_sig: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sig = build_signature_text(
            row,
            golden_measurements,
            max_measurements,
            max_deltas,
            voltage_only,
            measurement_stat_mode,
            prefer_voltage_keys,
            input_mode,
        )
        by_sig[sig].append(row)

    kept: list[dict] = []
    dropped = 0
    ambiguous = 0
    for sig_rows in by_sig.values():
        class_counts = Counter(
            infer_fault_type_from_row(r, map_resistor_param_drift) for r in sig_rows
        )
        if len(class_counts) <= 1:
            kept.extend(sig_rows)
            continue
        ambiguous += 1
        if ambiguity_policy == "drop":
            dropped += len(sig_rows)
            continue

        # majority policy
        top_count = max(class_counts.values())
        top_classes = sorted(k for k, v in class_counts.items() if v == top_count)
        keep_class = top_classes[0]
        for row in sig_rows:
            if infer_fault_type_from_row(row, map_resistor_param_drift) == keep_class:
                kept.append(row)
            else:
                dropped += 1

    stats = {
        "kept_rows": len(kept),
        "dropped_rows": dropped,
        "ambiguous_signatures": ambiguous,
        "policy": ambiguity_policy,
    }
    return kept, stats


def balance_rows_by_class(
    rows: list[dict],
    rng: random.Random,
    target_per_class: int,
    max_per_class: int,
    map_resistor_param_drift: bool,
) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[infer_fault_type_from_row(row, map_resistor_param_drift)].append(row)

    if not grouped:
        return rows, {"balanced": False, "target_per_class": 0, "class_counts_before": {}}

    before = {k: len(v) for k, v in sorted(grouped.items())}
    target = target_per_class if target_per_class > 0 else max(before.values())
    if max_per_class > 0:
        target = min(target, max_per_class)
    target = max(1, target)

    balanced: list[dict] = []
    for cls, cls_rows in grouped.items():
        n = len(cls_rows)
        if n >= target:
            balanced.extend(rng.sample(cls_rows, target))
        else:
            balanced.extend(cls_rows)
            needed = target - n
            for _ in range(needed):
                balanced.append(rng.choice(cls_rows))

    rng.shuffle(balanced)
    after_counts = Counter(
        infer_fault_type_from_row(r, map_resistor_param_drift) for r in balanced
    )
    stats = {
        "balanced": True,
        "target_per_class": target,
        "class_counts_before": before,
        "class_counts_after": dict(sorted(after_counts.items())),
    }
    return balanced, stats


def main() -> int:
    args = parse_args()
    lab_dir = args.out_root / args.lab
    in_file = lab_dir / "training_dataset.jsonl"
    out_dir = lab_dir / args.split_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(f"Missing input dataset: {in_file}")

    rows = load_jsonl(in_file)
    if not args.include_failed_sims:
        rows = [r for r in rows if r.get("sim_success") is True]
    if not rows:
        raise RuntimeError("No rows to split.")

    dropped_noop_rows = 0
    if args.drop_noop_faults:
        before = len(rows)
        rows = [r for r in rows if not row_has_noop_fault(r)]
        dropped_noop_rows = before - len(rows)
        if not rows:
            raise RuntimeError("No rows remain after dropping no-op faults.")

    golden_measurements: dict | None = None
    if args.use_golden:
        golden_path = lab_dir / "golden" / "golden_measurements.json"
        if not golden_path.exists():
            raise FileNotFoundError(
                f"--use-golden is set, but missing file: {golden_path}. "
                "Run build_golden_one_lab.py first."
            )
        golden_measurements = json.loads(golden_path.read_text(encoding="utf-8"))

    rows, ambiguity_stats = apply_ambiguity_policy(
        rows,
        args.ambiguity_policy,
        golden_measurements,
        args.max_measurements,
        args.max_deltas,
        args.voltage_only,
        args.measurement_stat_mode,
        args.prefer_voltage_keys,
        args.input_mode,
        args.map_resistor_param_drift,
    )
    if not rows:
        raise RuntimeError("No rows remain after ambiguity policy filtering.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_count = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows:
        raise RuntimeError("Training split is empty. Lower --val-ratio.")

    balance_stats = {
        "balanced": False,
        "target_per_class": 0,
        "class_counts_before": dict(
            sorted(
                Counter(
                    infer_fault_type_from_row(r, args.map_resistor_param_drift) for r in train_rows
                ).items()
            )
        ),
        "class_counts_after": {},
    }
    if args.balance_classes:
        train_rows, balance_stats = balance_rows_by_class(
            train_rows,
            rng,
            args.balance_target_per_class,
            args.balance_max_per_class,
            args.map_resistor_param_drift,
        )
    if not balance_stats.get("class_counts_after"):
        balance_stats["class_counts_after"] = dict(
            sorted(
                Counter(
                    infer_fault_type_from_row(r, args.map_resistor_param_drift) for r in train_rows
                ).items()
            )
        )

    train_chat: list[dict] = []
    val_chat: list[dict] = []
    train_instruct: list[dict] = []
    val_instruct: list[dict] = []
    noise_rng = random.Random(args.seed + 99991)

    for row in train_rows:
        user = build_user_prompt(
            row,
            args.max_measurements,
            args.max_chars,
            golden_measurements,
            args.max_deltas,
            args.voltage_only,
            args.measurement_stat_mode,
            args.prefer_voltage_keys,
            args.measurement_noise_sigma,
            args.measurement_noise_prob,
            noise_rng,
            args.input_mode,
            args.include_variant_id,
            args.include_lab_id,
        )
        out = build_assistant_output(
            row,
            args.output_mode,
            args.canonicalize_output,
            args.map_resistor_param_drift,
        )
        train_chat.append(to_chat(args.system_prompt, user, out))
        train_instruct.append(to_instruct(user, out, args.output_mode))

    for row in val_rows:
        user = build_user_prompt(
            row,
            args.max_measurements,
            args.max_chars,
            golden_measurements,
            args.max_deltas,
            args.voltage_only,
            args.measurement_stat_mode,
            args.prefer_voltage_keys,
            args.measurement_noise_sigma,
            args.measurement_noise_prob,
            noise_rng,
            args.input_mode,
            args.include_variant_id,
            args.include_lab_id,
        )
        out = build_assistant_output(
            row,
            args.output_mode,
            args.canonicalize_output,
            args.map_resistor_param_drift,
        )
        val_chat.append(to_chat(args.system_prompt, user, out))
        val_instruct.append(to_instruct(user, out, args.output_mode))

    write_jsonl(out_dir / "train_chat.jsonl", train_chat)
    write_jsonl(out_dir / "val_chat.jsonl", val_chat)
    write_jsonl(out_dir / "train_instruct.jsonl", train_instruct)
    write_jsonl(out_dir / "val_instruct.jsonl", val_instruct)

    meta = {
        "lab": args.lab,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "include_failed_sims": bool(args.include_failed_sims),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "source_rows": len(rows),
        "source_file": str(in_file),
        "use_golden": bool(args.use_golden),
        "max_measurements": args.max_measurements,
        "max_chars": args.max_chars,
        "max_deltas": args.max_deltas,
        "measurement_stat_mode": args.measurement_stat_mode,
        "prefer_voltage_keys": bool(args.prefer_voltage_keys),
        "voltage_only": bool(args.voltage_only),
        "measurement_noise_sigma": args.measurement_noise_sigma,
        "measurement_noise_prob": args.measurement_noise_prob,
        "input_mode": args.input_mode,
        "output_mode": args.output_mode,
        "ambiguity_policy": args.ambiguity_policy,
        "ambiguity_stats": ambiguity_stats,
        "balance_classes": bool(args.balance_classes),
        "balance_target_per_class": args.balance_target_per_class,
        "balance_max_per_class": args.balance_max_per_class,
        "balance_stats": balance_stats,
        "canonicalize_output": bool(args.canonicalize_output),
        "include_variant_id": bool(args.include_variant_id),
        "map_resistor_param_drift": bool(args.map_resistor_param_drift),
        "drop_noop_faults": bool(args.drop_noop_faults),
        "dropped_noop_rows": dropped_noop_rows,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote one-lab fine-tune files in: {out_dir}")
    print(f"Rows: train={len(train_rows)} val={len(val_rows)}")
    print(
        "Ambiguity:"
        f" policy={args.ambiguity_policy}"
        f" ambiguous_signatures={ambiguity_stats['ambiguous_signatures']}"
        f" dropped_rows={ambiguity_stats['dropped_rows']}"
    )
    if args.drop_noop_faults:
        print(f"Dropped no-op rows: {dropped_noop_rows}")
    print(f"Train class counts: {json.dumps(balance_stats['class_counts_after'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
