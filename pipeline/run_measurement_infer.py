#!/usr/bin/env python3
"""Interactive inference tool for manually entered circuit measurements.

This script is designed to mirror the same prompt shape used for LoRA training:
it loads one instruct-format template row, asks for the required measurement
values, computes deltas versus a golden measurement file, builds the final input
text, and runs model generation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA inference from manually entered measurements")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name used for LoRA training",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("pipeline/out/qwen15b_lab4_task2_part1_5_lora_v8_measured512"),
        help="Directory containing trained LoRA adapter",
    )
    parser.add_argument(
        "--template-file",
        type=Path,
        default=Path(
            "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/train_instruct.jsonl"
        ),
        help="Instruct JSONL used as prompt template source",
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=0,
        help="Row index to use from template-file",
    )
    parser.add_argument(
        "--golden-file",
        type=Path,
        default=Path(
            "pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/golden/golden_measurements.json"
        ),
        help="Golden measurement JSON used to compute deltas",
    )
    parser.add_argument(
        "--measure",
        action="append",
        default=[],
        help="Provide measurement overrides as key=value (repeatable)",
    )
    parser.add_argument(
        "--delta",
        action="append",
        default=[],
        help="Provide delta overrides as key=value (repeatable)",
    )
    parser.add_argument(
        "--variant-id",
        type=str,
        default="manual_case",
        help="Variant string used if template includes Variant line",
    )
    parser.add_argument(
        "--sim-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Value for SimSuccess line",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt. Requires all needed values from --measure/--delta/defaults.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Build and print the prompt input text without loading/running the model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--enforce-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize output to the expected diagnosis format",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output JSON path with assembled input and prediction",
    )
    return parser.parse_args()


def choose_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_keyvals(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in (text or "").split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid key=value item: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def format_value(value: Any) -> str:
    fval = to_float(value)
    if fval is None:
        return str(value)
    return f"{fval:.6g}"


def parse_template_input(input_text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in (input_text or "").splitlines()]
    measured_pairs: list[tuple[str, str]] = []
    delta_pairs: list[tuple[str, str]] = []
    golden_pairs: list[tuple[str, str]] = []
    has_variant = False
    has_sim_success = False
    has_measured = False
    has_deltas = False
    has_golden = False
    task_line = "Task: choose the most likely fault class and provide diagnosis/fix."
    for line in lines:
        if line.startswith("Variant:"):
            has_variant = True
        elif line.startswith("SimSuccess:"):
            has_sim_success = True
        elif line.startswith("Measured:"):
            has_measured = True
            measured_pairs = parse_keyvals(line.split(":", 1)[1].strip())
        elif line.startswith("Measurements:"):
            has_measured = True
            measured_pairs = parse_keyvals(line.split(":", 1)[1].strip())
        elif line.startswith("DeltasVsGolden:"):
            has_deltas = True
            delta_pairs = parse_keyvals(line.split(":", 1)[1].strip())
        elif line.startswith("GoldenMeasurements:"):
            has_golden = True
            golden_pairs = parse_keyvals(line.split(":", 1)[1].strip())
        elif line.startswith("Task:"):
            task_line = line
    return {
        "lines": lines,
        "measured_pairs": measured_pairs,
        "delta_pairs": delta_pairs,
        "golden_pairs": golden_pairs,
        "has_variant": has_variant,
        "has_sim_success": has_sim_success,
        "has_measured": has_measured,
        "has_deltas": has_deltas,
        "has_golden": has_golden,
        "task_line": task_line,
    }


def is_numeric_key(key: str, default_value: str | None) -> bool:
    if default_value is not None and to_float(default_value) is not None:
        return True
    low = key.lower()
    if low.startswith("v_") or low.startswith("i_"):
        return True
    if low in {"temp", "tnom"}:
        return True
    if low.endswith("_max") or low.endswith("_min") or low.endswith("_rms"):
        return True
    if low.endswith("_delta"):
        return True
    return False


def prompt_value(
    key: str,
    default: str | None,
    expect_numeric: bool,
    interactive: bool,
) -> str:
    if not interactive:
        if default is None:
            raise ValueError(f"Missing value for key '{key}' in non-interactive mode")
        return default

    while True:
        prompt = f"{key}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        raw = input(prompt).strip()
        if not raw:
            if default is not None:
                return default
            print("Value required.")
            continue
        if expect_numeric and to_float(raw) is None:
            print("Please enter a numeric value.")
            continue
        return raw


def build_prompt(instruction: str, input_text: str, response_style: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if response_style == "faulttype_diag_fix":
        response_header = (
            "### Response:\n"
            "Return exactly two lines.\n"
            "FaultType: one of param_drift|missing_component|pin_open|swapped_nodes|"
            "short_between_nodes|resistor_value_swap|resistor_wrong_value.\n"
            "Diagnosis: concise sentence. Fix: concise sentence.\n"
        )
    else:
        response_header = (
            "### Response:\n"
            "Return exactly one line in this format: Diagnosis: <text>. Fix: <text>.\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{input_text}\n\n"
        f"{response_header}"
    )


def force_output_format(text: str, response_style: str) -> str:
    raw = (text or "").strip()
    if not raw:
        if response_style == "faulttype_diag_fix":
            return (
                "FaultType: unknown\n"
                "Diagnosis: unknown fault. Fix: inspect wiring and component values."
            )
        return "Diagnosis: unknown fault. Fix: inspect wiring and component values."

    clean = re.sub(r"\s+", " ", raw).strip()
    low = clean.lower()
    dpos = low.find("diagnosis:")
    fpos = low.find("fix:")
    if dpos >= 0 and fpos >= 0:
        if fpos > dpos:
            diagnosis = clean[dpos + len("diagnosis:"):fpos].strip(" .;")
            fix = clean[fpos + len("fix:"):].strip(" .;")
        else:
            fix = clean[fpos + len("fix:"):dpos].strip(" .;")
            diagnosis = clean[dpos + len("diagnosis:"):].strip(" .;")
    else:
        parts = [p.strip(" .;") for p in re.split(r"[.!?]+", clean) if p.strip()]
        diagnosis = parts[0] if parts else "unknown fault"
        fix = parts[1] if len(parts) > 1 else "inspect wiring and component values"

    diagnosis = diagnosis or "unknown fault"
    fix = fix or "inspect wiring and component values"
    if response_style == "faulttype_diag_fix":
        fault = "unknown"
        for line in raw.splitlines():
            if line.lower().startswith("faulttype:"):
                candidate = line.split(":", 1)[1].strip()
                fault = normalize_fault_label(candidate)
                break
        if fault == "unknown":
            fault_match = re.search(r"faulttype:\s*([a-z_ \-]+)", clean, flags=re.IGNORECASE)
            if fault_match:
                fault = normalize_fault_label(fault_match.group(1))
        return f"FaultType: {fault}\nDiagnosis: {diagnosis}. Fix: {fix}."
    return f"Diagnosis: {diagnosis}. Fix: {fix}."


def normalize_fault_label(value: str) -> str:
    key = (value or "").strip().lower()
    key = key.split()[0] if key else ""
    alias = {
        "param_drift": "param_drift",
        "parameter_drift": "param_drift",
        "missing_component": "missing_component",
        "pin_open": "pin_open",
        "swapped_nodes": "swapped_nodes",
        "short_between_nodes": "short_between_nodes",
        "resistor_value_swap": "resistor_value_swap",
        "resistor_wrong_value": "resistor_wrong_value",
    }
    return alias.get(key, "unknown")


def load_model(model_name: str, adapter_dir: Path, device: str, dtype: torch.dtype) -> Any:
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig.parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.to(device)
    return model


def generate_one(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    # Match training/eval tokenization (no automatic BOS/EOS injection).
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    if "token_type_ids" not in forward_params and "token_type_ids" in inputs:
        inputs.pop("token_type_ids", None)
    if "token_type_ids" in forward_params and "token_type_ids" not in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_beams=max(1, num_beams),
            repetition_penalty=max(1.0, repetition_penalty),
            no_repeat_ngram_size=max(0, no_repeat_ngram_size),
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()

    if not args.template_file.exists():
        raise FileNotFoundError(f"Template file not found: {args.template_file}")
    if not args.adapter_dir.exists() and not args.prompt_only:
        raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")

    rows = load_jsonl(args.template_file)
    if not rows:
        raise RuntimeError(f"No rows in template file: {args.template_file}")
    if args.template_index < 0 or args.template_index >= len(rows):
        raise IndexError(f"template-index out of range: {args.template_index} (rows={len(rows)})")
    row = rows[args.template_index]

    instruction = (row.get("instruction", "") or "").strip()
    template_input = (row.get("input", "") or "").strip()
    template_output = (row.get("output", "") or "").strip().lower()
    response_style = "faulttype_diag_fix" if template_output.startswith("faulttype:") else "diag_fix"

    parsed = parse_template_input(template_input)
    measured_defaults = dict(parsed["measured_pairs"])
    delta_defaults = dict(parsed["delta_pairs"])
    golden_defaults = dict(parsed["golden_pairs"])

    golden_measurements: dict[str, Any] = {}
    if args.golden_file.exists():
        golden_measurements = json.loads(args.golden_file.read_text(encoding="utf-8"))

    measure_overrides = parse_overrides(args.measure)
    delta_overrides = parse_overrides(args.delta)
    interactive = not args.non_interactive

    measured_values: dict[str, str] = {}
    measured_order = [k for k, _ in parsed["measured_pairs"]]
    for key in measured_order:
        if key in measure_overrides:
            measured_values[key] = measure_overrides[key]
            continue
        default = None
        if key in golden_measurements:
            default = format_value(golden_measurements[key])
        elif key in measured_defaults:
            default = measured_defaults[key]
        expected_numeric = is_numeric_key(key, default)
        measured_values[key] = prompt_value(key, default, expected_numeric, interactive)

    delta_values: dict[str, str] = {}
    delta_order = [k for k, _ in parsed["delta_pairs"]]
    for key in delta_order:
        if key in delta_overrides:
            delta_values[key] = delta_overrides[key]
            continue
        base_key = key[:-6] if key.endswith("_delta") else key
        m_val = to_float(measured_values.get(base_key))
        g_val = to_float(golden_measurements.get(base_key))
        if m_val is not None and g_val is not None:
            delta_values[key] = format_value(m_val - g_val)
            continue
        default = delta_defaults.get(key, "0")
        expected_numeric = is_numeric_key(key, default)
        delta_values[key] = prompt_value(key, default, expected_numeric, interactive)

    golden_values: dict[str, str] = {}
    for key, default_val in parsed["golden_pairs"]:
        if key in golden_measurements:
            golden_values[key] = format_value(golden_measurements[key])
        else:
            golden_values[key] = default_val

    rebuilt_lines: list[str] = []
    for line in parsed["lines"]:
        if line.startswith("Variant:"):
            rebuilt_lines.append(f"Variant: {args.variant_id}")
        elif line.startswith("SimSuccess:"):
            rebuilt_lines.append(f"SimSuccess: {bool(args.sim_success)}")
        elif line.startswith("DeltasVsGolden:"):
            text = "; ".join(f"{k}={delta_values[k]}" for k in delta_order)
            rebuilt_lines.append(f"DeltasVsGolden: {text}")
        elif line.startswith("Measured:"):
            text = "; ".join(f"{k}={measured_values[k]}" for k in measured_order)
            rebuilt_lines.append(f"Measured: {text}")
        elif line.startswith("Measurements:"):
            text = "; ".join(f"{k}={measured_values[k]}" for k in measured_order)
            rebuilt_lines.append(f"Measurements: {text}")
        elif line.startswith("GoldenMeasurements:"):
            g_order = [k for k, _ in parsed["golden_pairs"]]
            text = "; ".join(f"{k}={golden_values[k]}" for k in g_order)
            rebuilt_lines.append(f"GoldenMeasurements: {text}")
        else:
            rebuilt_lines.append(line)

    input_text = "\n".join(rebuilt_lines).strip()
    prompt = build_prompt(instruction, input_text, response_style)

    print("\n=== Assembled Input ===")
    print(input_text)

    if args.prompt_only:
        if args.save_json is not None:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_json.write_text(
                json.dumps(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "response_style": response_style,
                        "measured_values": measured_values,
                        "delta_values": delta_values,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nSaved prompt payload: {args.save_json}")
        return 0

    device, dtype = choose_device()
    print(f"\ndevice={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_name, args.adapter_dir, device, dtype)
    print(f"model_device={next(model.parameters()).device}")

    pred = generate_one(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    if args.enforce_format:
        pred = force_output_format(pred, response_style)

    print("\n=== Model Prediction ===")
    print(pred)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "prediction": pred,
                    "response_style": response_style,
                    "measured_values": measured_values,
                    "delta_values": delta_values,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved output JSON: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
