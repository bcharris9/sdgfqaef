#!/usr/bin/env python3
"""Generate faulted LTspice netlist variants from .asc files.

Workflow:
1) Export each .asc to .net with LTspice CLI (unless --skip-netlist-export).
2) Create N faulted variants per circuit from the base .net.
3) Write variant metadata to JSONL for downstream simulation/dataset steps.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TWO_PIN_PREFIXES = {"R", "C", "L", "V", "I", "D"}
VALUE_MULTIPLIERS = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}
DEFAULT_PASSIVE_VALUES = {
    "R": "1k",
    "C": "1u",
    "L": "1m",
}
DEFAULT_SOURCE_VALUES = {
    "V": "5",
    "I": "1m",
}
MEAS_NODE_ARITY_BY_PREFIX = {
    "R": 2,
    "C": 2,
    "L": 2,
    "V": 2,
    "I": 2,
    "D": 2,
    "B": 2,  # behavioral source
    "F": 2,  # CCCS (controlling source is not a node)
    "H": 2,  # CCVS (controlling source is not a node)
    "W": 2,  # current-controlled switch
    "J": 3,  # JFET
    "Q": 3,  # BJT (substrate node optional; omitted here to avoid model-name false positives)
    "E": 4,  # VCVS
    "G": 4,  # VCCS
    "S": 4,  # voltage-controlled switch
    "T": 4,  # transmission line
    "O": 4,  # lossy transmission line
    "M": 4,  # MOSFET
}
WAVEFORM_SOURCE_KEYWORDS = {
    "PULSE",
    "SINE",
    "PWL",
    "EXP",
    "SFFM",
    "AM",
    "FM",
    "TRNOISE",
    "NOISE",
}


@dataclass
class Component:
    line_index: int
    name: str
    n1: str
    n2: str
    value: str


@dataclass
class FaultConfig:
    weights: dict[str, float]
    vsource_min: float
    vsource_max: float
    param_drift_vsource_prob: float
    param_drift_allow_resistor: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate faulted LTspice netlists")
    parser.add_argument("--asc-dir", type=Path, default=Path("LTSpice_files"))
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline/out"))
    parser.add_argument("--variants-per-circuit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--ltspice-bin", type=str, default="")
    parser.add_argument("--skip-netlist-export", action="store_true")
    parser.add_argument("--weight-param-drift", type=float, default=0.30)
    parser.add_argument("--weight-missing-component", type=float, default=0.12)
    parser.add_argument("--weight-pin-open", type=float, default=0.12)
    parser.add_argument("--weight-swapped-nodes", type=float, default=0.18)
    parser.add_argument("--weight-short-between-nodes", type=float, default=0.08)
    parser.add_argument("--weight-resistor-value-swap", type=float, default=0.20)
    parser.add_argument("--weight-resistor-wrong-value", type=float, default=0.15)
    parser.add_argument("--vsource-min", type=float, default=-5.0)
    parser.add_argument("--vsource-max", type=float, default=5.0)
    parser.add_argument(
        "--param-drift-vsource-prob",
        type=float,
        default=0.45,
        help="When param_drift is selected, probability of choosing a voltage source if available",
    )
    parser.add_argument(
        "--param-drift-allow-resistor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow resistor components to be chosen for param_drift faults",
    )
    return parser.parse_args()


def detect_ltspice_bin(user_value: str) -> str:
    if user_value:
        return user_value

    candidates = [
        "ltspice",
        "LTspice",
        "/Applications/LTspice.app/Contents/MacOS/LTspice",
        "/mnt/c/Program Files/ADI/LTspice/LTspice.exe",
        "/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
        r"C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe",
        r"C:\\Program Files\\ADI\\LTspice\\LTspice.exe",
    ]
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        if Path(candidate).exists():
            return candidate
    return "ltspice"


def validate_ltspice_bin(bin_path: str) -> str:
    # If user passed an explicit filesystem path, require that it exists.
    if "/" in bin_path or "\\" in bin_path:
        if Path(bin_path).exists():
            return bin_path
        raise FileNotFoundError(
            "LTspice executable not found at: "
            f"{bin_path}\n"
            "Pass a valid path with --ltspice-bin, e.g. "
            "'/mnt/c/Program Files/ADI/LTspice/LTspice.exe' on WSL."
        )

    # Otherwise treat it as a command name on PATH.
    resolved = shutil.which(bin_path)
    if resolved:
        return resolved

    raise FileNotFoundError(
        "LTspice executable not found on PATH.\n"
        "Install LTspice or pass --ltspice-bin with full path, e.g. "
        "'/mnt/c/Program Files/ADI/LTspice/LTspice.exe' (WSL) "
        "or 'C:\\Program Files\\ADI\\LTspice\\LTspice.exe' (Windows)."
    )


def run_cmd(cmd: list[str], timeout_sec: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec, check=False)


def export_netlists(asc_dir: Path, base_netlist_dir: Path, ltspice_bin: str) -> list[Path]:
    asc_files = sorted(asc_dir.glob("*.asc"))
    if not asc_files:
        raise FileNotFoundError(f"No .asc files found in {asc_dir}")

    base_netlist_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for asc in asc_files:
        if asc.stat().st_size == 0:
            print(f"[WARN] Empty .asc skipped: {asc}")
            continue

        cmd = [ltspice_bin, "-netlist", str(asc)]
        result = run_cmd(cmd)
        candidate_net = asc.with_suffix(".net")
        target_net = base_netlist_dir / f"{asc.stem}.net"

        if result.returncode == 0 and candidate_net.exists():
            shutil.move(str(candidate_net), target_net)
            generated.append(target_net)
            continue

        if target_net.exists():
            generated.append(target_net)
            continue

        print(f"[WARN] Failed to export netlist for {asc.name}")
        if result.stderr:
            print(result.stderr.strip())

    if not generated:
        raise RuntimeError(
            "No base netlists generated. Set --ltspice-bin or run with --skip-netlist-export "
            "after manually creating .net files in pipeline/out/base_netlists."
        )
    return generated


def load_existing_netlists(base_netlist_dir: Path) -> list[Path]:
    nets = sorted(base_netlist_dir.glob("*.net"))
    if not nets:
        raise FileNotFoundError(f"No .net files found in {base_netlist_dir}")
    return nets


def parse_components(lines: list[str]) -> tuple[list[Component], list[str]]:
    comps: list[Component] = []
    nodes: set[str] = set()

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith(";") or line.startswith("."):
            continue
        tokens = line.split()
        if len(tokens) < 4:
            continue
        name = tokens[0]
        prefix = name[0].upper()
        if prefix not in TWO_PIN_PREFIXES:
            continue

        n1, n2 = tokens[1], tokens[2]
        value = " ".join(tokens[3:])
        comps.append(Component(idx, name, n1, n2, value))

        if n1 != "0":
            nodes.add(n1)
        if n2 != "0":
            nodes.add(n2)

    return comps, sorted(nodes)


def strip_inline_comment_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if tok.startswith(";"):
            break
        out.append(tok)
    return out


def is_param_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    if "=" in t:
        return True
    return t.upper().startswith("PARAMS:")


def extract_measure_nodes_from_tokens(tokens: list[str]) -> list[str]:
    tokens = strip_inline_comment_tokens(tokens)
    if len(tokens) < 3:
        return []
    name = tokens[0]
    if not name:
        return []
    prefix = name[0].upper()

    if prefix in MEAS_NODE_ARITY_BY_PREFIX:
        n = MEAS_NODE_ARITY_BY_PREFIX[prefix]
        return tokens[1 : 1 + n]

    # Subcircuit / macro-model instances: Xname n1 n2 ... subckt_name [params...]
    # Heuristic: nodes run until the token before the first param assignment,
    # otherwise all tokens before the final token (subckt name).
    if prefix in {"X", "A"}:
        body = tokens[1:]
        if len(body) < 2:
            return []
        cut = len(body)
        for i, tok in enumerate(body):
            if is_param_token(tok):
                cut = i
                break
        if cut < len(body):
            node_end = max(0, cut - 1)
        else:
            node_end = max(0, len(body) - 1)
        return body[:node_end]

    return []


def collect_measurement_nodes_and_vsources(lines: list[str]) -> tuple[list[str], list[str]]:
    nodes: set[str] = set()
    vsrc: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith(";") or line.startswith("."):
            continue
        tokens = line.split()
        if not tokens:
            continue

        name = tokens[0]
        if not name:
            continue
        prefix = name[0].upper()
        if prefix == "V" and len(tokens) >= 3:
            vsrc.append(name)

        for node in extract_measure_nodes_from_tokens(tokens):
            if not node or node == "0":
                continue
            nodes.add(node)

    return sorted(nodes), sorted(set(vsrc))


def sanitize_base_netlist(lines: list[str]) -> tuple[list[str], list[str]]:
    sanitized = list(lines)
    notes: list[str] = []
    inline_libs: set[str] = set()
    inline_lib_model_hints: dict[str, str] = {}

    for idx, raw in enumerate(lines):
        # LTspice exports on Windows can contain mojibake when read as UTF-8 later
        # (e.g. "Âµ", "Â§"). Normalize common cases before token processing.
        normalized_raw = (
            raw.replace("Âµ", "u")
            .replace("µ", "u")
            .replace("Â§", "")
            .replace("§", "")
        )
        if normalized_raw != raw:
            sanitized[idx] = normalized_raw
            notes.append("Normalized non-ASCII export characters in netlist line")

        line = normalized_raw.strip()
        if not line or line.startswith("*") or line.startswith(";") or line.startswith("."):
            continue
        body, sep, comment = normalized_raw.rstrip("\n").partition(";")
        tokens = body.split()
        if len(tokens) < 4:
            continue

        rewritten = False
        name = tokens[0]
        clean_name = re.sub(r"[^A-Za-z0-9_]", "", name)
        if clean_name and clean_name != name:
            tokens[0] = clean_name
            name = clean_name
            rewritten = True
            notes.append(f"Sanitized component name {raw.strip().split()[0]} -> {clean_name}")

        prefix = name[0].upper()

        # Some symbol exports inject a library filename before the subckt/model token on X lines:
        #   XU2 ... LTC1.lib LM339
        # LTspice expects only the subckt name in the call line.
        if prefix in {"X", "A"} and len(tokens) >= 3 and tokens[-2].lower().endswith(".lib"):
            dropped = tokens.pop(-2)
            inline_libs.add(dropped)
            if len(tokens) >= 2:
                inline_lib_model_hints[dropped] = tokens[-1]
            rewritten = True
            notes.append(f"Removed inline library token {dropped} from {name}")

        if prefix not in DEFAULT_PASSIVE_VALUES and prefix not in DEFAULT_SOURCE_VALUES:
            if rewritten:
                rebuilt = " ".join(tokens)
                if sep:
                    rebuilt += f" ;{comment}"
                sanitized[idx] = rebuilt + "\n"
            continue

        # LTspice can export missing values as bare prefix tokens (e.g. "R1 ... R", "V1 ... V").
        if tokens[3].upper() == prefix:
            if prefix in DEFAULT_PASSIVE_VALUES:
                default_value = DEFAULT_PASSIVE_VALUES[prefix]
            else:
                # Heuristic for split rails: node names containing '-' default negative.
                if prefix == "V" and (tokens[1].startswith("-") or "-" in tokens[1]):
                    default_value = "-5"
                else:
                    default_value = DEFAULT_SOURCE_VALUES[prefix]
            tokens[3:] = [default_value]
            rebuilt = " ".join(tokens)
            if sep:
                rebuilt += f" ;{comment}"
            sanitized[idx] = rebuilt + "\n"
            notes.append(
                f"Replaced missing {prefix} value on {tokens[0]} with default {default_value}"
            )
        elif rewritten:
            rebuilt = " ".join(tokens)
            if sep:
                rebuilt += f" ;{comment}"
            sanitized[idx] = rebuilt + "\n"

    if inline_libs:
        existing_lib_lines = [
            raw.strip().lower()
            for raw in sanitized
            if raw.strip().lower().startswith(".lib")
        ]
        for libtok in sorted(inline_libs):
            libtok_l = libtok.lower()
            if any(libtok_l in line for line in existing_lib_lines):
                continue
            lib_arg = libtok
            model_hint = inline_lib_model_hints.get(libtok, "")
            # Lab4 Task 3 exports inline "LTC1.lib" for LM339, but the actual model file
            # is often in the user's LTspice documents folder as LM339.lib.
            if libtok_l == "ltc1.lib" and model_hint:
                model_lib = f"{model_hint}.lib"
                home = Path.home()
                model_candidates = [
                    home / "OneDrive" / "Documents" / "LTspice" / model_lib,
                    home / "Documents" / "LTspice" / model_lib,
                ]
                for cand in model_candidates:
                    if cand.exists():
                        lib_arg = f'"{cand}"'
                        notes.append(
                            f"Resolved inline library token {libtok} to local model file {cand}"
                        )
                        break
            insert_before_end(sanitized, f".lib {lib_arg}")
            notes.append(f"Added library directive from inline token: .lib {lib_arg}")

    return sanitized, notes


def has_analysis_directive(lines: list[str]) -> bool:
    analysis_prefixes = (".op", ".tran", ".ac", ".dc", ".noise", ".tf", ".pz", ".step")
    for raw in lines:
        stripped = raw.strip().lower()
        if any(stripped.startswith(prefix) for prefix in analysis_prefixes):
            return True
    return False


def has_directive(lines: list[str], directive: str) -> bool:
    prefix = directive.strip().lower()
    for raw in lines:
        if raw.strip().lower().startswith(prefix):
            return True
    return False


def ensure_analysis_directive(lines: list[str]) -> tuple[list[str], list[str]]:
    updated = list(lines)
    notes: list[str] = []

    if not has_directive(updated, ".op"):
        insert_before_end(updated, ".op")
        notes.append("Added default analysis directive: .op")

    if not has_directive(updated, ".tran"):
        insert_before_end(updated, ".tran 0 10m 0 10u")
        notes.append("Added default analysis directive: .tran 0 10m 0 10u")

    return updated, notes


def safe_measure_name(token: str) -> str:
    # Preserve common punctuation where possible, but exclude '-' because LTspice
    # rejects it in .meas labels (e.g. V_-VCC causes a syntax error).
    cleaned = re.sub(r"[^A-Za-z0-9_.$]", "_", token)
    if not cleaned:
        return "x"
    if cleaned[0].isdigit():
        return f"n_{cleaned}"
    return cleaned


def ensure_measurement_directives(lines: list[str]) -> tuple[list[str], list[str]]:
    updated = list(lines)
    notes: list[str] = []

    # Avoid duplicating our autogenerated measurement block on repeated runs.
    if any(raw.strip().startswith("; AUTO_MEAS_BEGIN") for raw in updated):
        return updated, notes

    nodes, voltage_sources = collect_measurement_nodes_and_vsources(updated)

    directives: list[str] = []
    directives.append("; AUTO_MEAS_BEGIN")
    directives.append(".save V(*)")
    for vsrc in voltage_sources:
        directives.append(f".save I({vsrc})")

    for node in nodes:
        node_id = safe_measure_name(node)
        directives.append(f".meas op V_{node_id} FIND V({node})")
        directives.append(f".meas tran V_{node_id}_MAX MAX V({node})")
        directives.append(f".meas tran V_{node_id}_MIN MIN V({node})")
        directives.append(f".meas tran V_{node_id}_RMS RMS V({node})")

    for vsrc in voltage_sources:
        src_id = safe_measure_name(vsrc)
        directives.append(f".meas op I_{src_id} FIND I({vsrc})")
        directives.append(f".meas tran I_{src_id}_MAX MAX I({vsrc})")
        directives.append(f".meas tran I_{src_id}_MIN MIN I({vsrc})")
        directives.append(f".meas tran I_{src_id}_RMS RMS I({vsrc})")

    directives.append("; AUTO_MEAS_END")

    for directive in directives:
        insert_before_end(updated, directive)

    notes.append(
        "Added autogenerated .save/.meas directives for node voltages and voltage-source currents"
    )
    return updated, notes


def parse_spice_number(token: str) -> float | None:
    token = token.strip()
    m = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([a-zA-Z]*)$", token)
    if not m:
        return None
    value = float(m.group(1))
    suffix = m.group(2).lower()

    if suffix in VALUE_MULTIPLIERS:
        return value * VALUE_MULTIPLIERS[suffix]
    return None


def format_spice_number(value: float) -> str:
    return f"{value:.6g}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def is_voltage_source(comp: Component) -> bool:
    return bool(comp.name) and comp.name[0].upper() == "V"


def source_value_head_token(value: str) -> str:
    return ((value or "").strip().split() or [""])[0]


def is_waveform_defined_source(comp: Component) -> bool:
    if not is_voltage_source(comp):
        return False
    head = source_value_head_token(comp.value).upper()
    if not head:
        return False
    # Handles tokens like "SINE(...", "PULSE(...", "PWL(..."
    return any(head.startswith(k) for k in WAVEFORM_SOURCE_KEYWORDS)


def values_effectively_equal(old_value: str, new_value: str) -> bool:
    old_token = (old_value or "").split()[0] if old_value else ""
    new_token = (new_value or "").split()[0] if new_value else ""
    old_num = parse_spice_number(old_token)
    new_num = parse_spice_number(new_token)
    if old_num is not None and new_num is not None:
        tol = max(1e-12, 1e-6 * max(1.0, abs(old_num), abs(new_num)))
        return abs(old_num - new_num) <= tol
    return old_token.strip().lower() == new_token.strip().lower()


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def replace_line(lines: list[str], idx: int, new_line: str) -> None:
    lines[idx] = new_line if new_line.endswith("\n") else new_line + "\n"


def insert_before_end(lines: list[str], new_line: str) -> None:
    for idx, raw in enumerate(lines):
        if raw.strip().lower() == ".end":
            lines.insert(idx, new_line if new_line.endswith("\n") else new_line + "\n")
            return
    lines.append(new_line if new_line.endswith("\n") else new_line + "\n")


def build_fault_config(args: argparse.Namespace) -> FaultConfig:
    vmin = safe_float(args.vsource_min, -5.0)
    vmax = safe_float(args.vsource_max, 5.0)
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    weights = {
        "param_drift": max(0.0, safe_float(args.weight_param_drift, 0.0)),
        "missing_component": max(0.0, safe_float(args.weight_missing_component, 0.0)),
        "pin_open": max(0.0, safe_float(args.weight_pin_open, 0.0)),
        "swapped_nodes": max(0.0, safe_float(args.weight_swapped_nodes, 0.0)),
        "short_between_nodes": max(0.0, safe_float(args.weight_short_between_nodes, 0.0)),
        "resistor_value_swap": max(0.0, safe_float(args.weight_resistor_value_swap, 0.0)),
        "resistor_wrong_value": max(0.0, safe_float(args.weight_resistor_wrong_value, 0.0)),
    }
    if sum(weights.values()) <= 0:
        raise ValueError("All fault weights are zero. Provide at least one positive fault weight.")

    vsrc_prob = clamp(safe_float(args.param_drift_vsource_prob, 0.45), 0.0, 1.0)

    return FaultConfig(
        weights=weights,
        vsource_min=vmin,
        vsource_max=vmax,
        param_drift_vsource_prob=vsrc_prob,
        param_drift_allow_resistor=bool(args.param_drift_allow_resistor),
    )


def generate_single_variant(
    base_lines: list[str],
    comps: list[Component],
    nodes: list[str],
    rng: random.Random,
    variant_index: int,
    fault_config: FaultConfig,
) -> tuple[list[str], dict[str, Any]]:
    if not comps:
        raise ValueError("Netlist has no supported 2-pin components")

    lines = list(base_lines)
    resistors = [c for c in comps if c.name and c.name[0].upper() == "R"]

    available: list[str] = []
    available_weights: list[float] = []
    driftable_for_param = [
        c
        for c in comps
        if not (is_voltage_source(c) and is_waveform_defined_source(c))
    ]

    for fault_type, weight in fault_config.weights.items():
        if weight <= 0:
            continue
        if fault_type == "param_drift" and not driftable_for_param:
            continue
        if fault_type == "short_between_nodes" and len(nodes) < 2:
            continue
        if fault_type == "resistor_value_swap" and len(resistors) < 2:
            continue
        if fault_type == "resistor_wrong_value" and len(resistors) < 1:
            continue
        available.append(fault_type)
        available_weights.append(weight)

    if not available:
        raise RuntimeError("No valid fault types available for this circuit with current settings.")

    fault_type = rng.choices(population=available, weights=available_weights, k=1)[0]

    detail: dict[str, Any] = {"fault_type": fault_type}

    if fault_type in {"missing_component", "pin_open", "swapped_nodes"}:
        comp = rng.choice(comps)
        detail["component"] = comp.name
        detail["original_nodes"] = [comp.n1, comp.n2]

    if fault_type == "param_drift":
        vsrc_comps = [c for c in driftable_for_param if is_voltage_source(c)]
        if fault_config.param_drift_allow_resistor:
            drift_pool = list(driftable_for_param)
        else:
            drift_pool = [
                c for c in driftable_for_param if not c.name.upper().startswith("R")
            ]
            if not drift_pool:
                drift_pool = list(driftable_for_param)

        if not drift_pool:
            raise RuntimeError("No driftable components available for param_drift")

        if vsrc_comps and rng.random() < fault_config.param_drift_vsource_prob:
            comp = rng.choice(vsrc_comps)
        else:
            comp = rng.choice(drift_pool)
        detail["component"] = comp.name
        detail["original_nodes"] = [comp.n1, comp.n2]
        original = parse_spice_number(comp.value.split()[0])
        factor = rng.choice([0.5, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0])

        if original is None:
            if comp.name[0].upper() == "R":
                new_value = rng.choice(["100", "1k", "10k", "100k"])
            elif comp.name[0].upper() == "C":
                new_value = rng.choice(["10p", "100p", "1n", "10n", "100n", "1u"])
            elif comp.name[0].upper() == "L":
                new_value = rng.choice(["1u", "10u", "100u", "1m"])
            elif is_voltage_source(comp):
                if comp.n1.startswith("-") or "-" in comp.n1:
                    sampled = rng.uniform(fault_config.vsource_min, 0.0)
                elif comp.n1.upper().endswith("VCC"):
                    sampled = rng.uniform(0.0, fault_config.vsource_max)
                else:
                    sampled = rng.uniform(fault_config.vsource_min, fault_config.vsource_max)
                new_value = format_spice_number(clamp(sampled, fault_config.vsource_min, fault_config.vsource_max))
            else:
                new_value = comp.value
        else:
            proposed = original * factor
            if is_voltage_source(comp):
                proposed = clamp(proposed, fault_config.vsource_min, fault_config.vsource_max)
            new_value = format_spice_number(proposed)

        if values_effectively_equal(comp.value, new_value):
            if original is not None:
                if is_voltage_source(comp):
                    if original >= fault_config.vsource_max:
                        forced = max(fault_config.vsource_min, fault_config.vsource_max - 0.5)
                    elif original <= fault_config.vsource_min:
                        forced = min(fault_config.vsource_max, fault_config.vsource_min + 0.5)
                    else:
                        forced = clamp(
                            original + (0.5 if original <= 0 else -0.5),
                            fault_config.vsource_min,
                            fault_config.vsource_max,
                        )
                else:
                    forced = original * (0.8 if abs(original) > 1e-12 else 2.0)
                new_value = format_spice_number(forced)
            else:
                # Last-resort non-numeric fallback.
                if is_voltage_source(comp):
                    new_value = format_spice_number(
                        clamp(0.0, fault_config.vsource_min, fault_config.vsource_max)
                    )
                else:
                    prefix = comp.name[0].upper()
                    if prefix == "R":
                        new_value = "2k"
                    elif prefix == "C":
                        new_value = "100n"
                    elif prefix == "L":
                        new_value = "10m"
                    else:
                        new_value = "1"

        replace_line(lines, comp.line_index, f"{comp.name} {comp.n1} {comp.n2} {new_value}")
        detail["factor"] = factor
        detail["old_value"] = comp.value
        detail["new_value"] = new_value
        if is_voltage_source(comp):
            detail["bounds"] = [fault_config.vsource_min, fault_config.vsource_max]

    elif fault_type == "missing_component":
        comp = next(c for c in comps if c.name == detail["component"])
        replace_line(lines, comp.line_index, f"; FAULT_REMOVED {base_lines[comp.line_index].strip()}")

    elif fault_type == "pin_open":
        comp = next(c for c in comps if c.name == detail["component"])
        floating = f"__open_{comp.name}_{variant_index}"
        replace_line(lines, comp.line_index, f"{comp.name} {floating} {comp.n2} {comp.value}")
        detail["opened_pin"] = comp.n1
        detail["new_pin"] = floating

    elif fault_type == "swapped_nodes":
        comp = next(c for c in comps if c.name == detail["component"])
        replace_line(lines, comp.line_index, f"{comp.name} {comp.n2} {comp.n1} {comp.value}")

    elif fault_type == "short_between_nodes":
        n1, n2 = rng.sample(nodes, 2)
        short_name = f"RSHORT_{variant_index}"
        insert_before_end(lines, f"{short_name} {n1} {n2} 1u")
        detail["nodes"] = [n1, n2]
        detail["component"] = short_name

    elif fault_type == "resistor_value_swap":
        r1, r2 = rng.sample(resistors, 2)
        replace_line(lines, r1.line_index, f"{r1.name} {r1.n1} {r1.n2} {r2.value}")
        replace_line(lines, r2.line_index, f"{r2.name} {r2.n1} {r2.n2} {r1.value}")
        detail["components"] = [r1.name, r2.name]
        detail["values_before"] = {r1.name: r1.value, r2.name: r2.value}
        detail["values_after"] = {r1.name: r2.value, r2.name: r1.value}

    elif fault_type == "resistor_wrong_value":
        r = rng.choice(resistors)
        old_value = r.value
        parsed = parse_spice_number(old_value.split()[0]) if old_value else None
        if parsed is None:
            candidates = ["100", "220", "470", "1k", "2.2k", "4.7k", "10k", "22k", "47k", "100k"]
            new_value = rng.choice(candidates)
            if new_value == old_value:
                new_value = "1k"
        else:
            factor = rng.choice([0.1, 0.2, 0.5, 2.0, 5.0, 10.0])
            new_value = format_spice_number(parsed * factor)
            if new_value == old_value:
                new_value = format_spice_number(parsed * 2.0)

        replace_line(lines, r.line_index, f"{r.name} {r.n1} {r.n2} {new_value}")
        detail["component"] = r.name
        detail["old_value"] = old_value
        detail["new_value"] = new_value

    return lines, detail


def generate_variants_for_netlist(
    netlist_path: Path,
    variants_dir: Path,
    variants_per_circuit: int,
    seed: int,
    fault_config: FaultConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)

    raw_base_lines = netlist_path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    base_lines, sanitize_notes = sanitize_base_netlist(raw_base_lines)
    base_lines, analysis_notes = ensure_analysis_directive(base_lines)
    base_lines, measurement_notes = ensure_measurement_directives(base_lines)

    for note in sanitize_notes + analysis_notes + measurement_notes:
        warnings.append(f"[WARN] {netlist_path.name}: {note}")

    comps, nodes = parse_components(base_lines)
    if not comps:
        warnings.append(f"[WARN] Skipped {netlist_path.name}: no supported components found")
        return rows, warnings

    for i in range(variants_per_circuit):
        variant_lines, detail = generate_single_variant(
            base_lines,
            comps,
            nodes,
            rng,
            i,
            fault_config,
        )
        variant_id = f"{netlist_path.stem}__v{i:04d}"
        variant_path = variants_dir / f"{variant_id}.cir"
        variant_path.write_text("".join(variant_lines), encoding="utf-8")

        rows.append(
            {
                "variant_id": variant_id,
                "source_netlist": str(netlist_path),
                "variant_netlist": str(variant_path),
                "fault": detail,
            }
        )

    return rows, warnings


def main() -> int:
    args = parse_args()
    fault_config = build_fault_config(args)

    out_dir = args.out_dir
    base_netlist_dir = out_dir / "base_netlists"
    variants_dir = out_dir / "variants"
    manifest_path = out_dir / "variant_manifest.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    ltspice_bin = validate_ltspice_bin(detect_ltspice_bin(args.ltspice_bin))

    if args.skip_netlist_export:
        base_netlists = load_existing_netlists(base_netlist_dir)
    else:
        base_netlists = export_netlists(args.asc_dir, base_netlist_dir, ltspice_bin)

    workers = max(1, args.max_workers)
    with manifest_path.open("w", encoding="utf-8") as mf:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx, netlist_path in enumerate(base_netlists):
                circuit_seed = args.seed + (idx * 1000003)
                futures.append(
                    executor.submit(
                        generate_variants_for_netlist,
                        netlist_path,
                        variants_dir,
                        args.variants_per_circuit,
                        circuit_seed,
                        fault_config,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                rows, warnings = future.result()
                for warning in warnings:
                    print(warning)
                for row in rows:
                    mf.write(json.dumps(row) + "\n")

    print(f"Wrote variants to: {variants_dir}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Generated variants with max_workers={workers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

