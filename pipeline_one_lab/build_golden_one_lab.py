#!/usr/bin/env python3
"""Build and simulate a golden (unfaulted) circuit for one lab."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


DEFAULT_PASSIVE_VALUES = {"R": "1k", "C": "1u", "L": "1m"}
DEFAULT_SOURCE_VALUES = {"V": "5", "I": "1m"}
TWO_PIN_PREFIXES = {"R", "C", "L", "V", "I", "D"}
MEAS_NODE_ARITY_BY_PREFIX = {
    "R": 2,
    "C": 2,
    "L": 2,
    "V": 2,
    "I": 2,
    "D": 2,
    "B": 2,
    "F": 2,
    "H": 2,
    "W": 2,
    "J": 3,
    "Q": 3,
    "E": 4,
    "G": 4,
    "S": 4,
    "T": 4,
    "O": 4,
    "M": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a one-lab golden circuit")
    parser.add_argument("--lab", required=True, help="Lab stem, e.g. lab9_task2")
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument("--ltspice-bin", required=True, type=str)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--keep-raw", action="store_true")
    return parser.parse_args()


def insert_before_end(lines: list[str], new_line: str) -> None:
    full = new_line if new_line.endswith("\n") else new_line + "\n"
    for i, raw in enumerate(lines):
        if raw.strip().lower() == ".end":
            lines.insert(i, full)
            return
    lines.append(full)


def has_directive(lines: list[str], directive: str) -> bool:
    d = directive.strip().lower()
    return any(raw.strip().lower().startswith(d) for raw in lines)


def sanitize_base(lines: list[str]) -> list[str]:
    out = list(lines)
    inline_libs: set[str] = set()
    inline_lib_model_hints: dict[str, str] = {}
    for i, raw in enumerate(lines):
        normalized_raw = (
            raw.replace("Âµ", "u")
            .replace("µ", "u")
            .replace("Â§", "")
            .replace("§", "")
        )
        s = normalized_raw.strip()
        if not s or s.startswith(("*", ";", ".")):
            continue
        body, sep, comment = normalized_raw.rstrip("\n").partition(";")
        tokens = body.split()
        if len(tokens) < 4:
            continue
        name = tokens[0]
        clean_name = re.sub(r"[^A-Za-z0-9_]", "", name)
        if clean_name and clean_name != name:
            tokens[0] = clean_name
            name = clean_name
        prefix = name[0].upper()
        if prefix in {"X", "A"} and len(tokens) >= 3 and tokens[-2].lower().endswith(".lib"):
            dropped = tokens.pop(-2)
            inline_libs.add(dropped)
            if len(tokens) >= 2:
                inline_lib_model_hints[dropped] = tokens[-1]
            rebuilt = " ".join(tokens)
            if sep:
                rebuilt += f" ;{comment}"
            out[i] = rebuilt + "\n"
            continue
        if prefix not in DEFAULT_PASSIVE_VALUES and prefix not in DEFAULT_SOURCE_VALUES:
            if normalized_raw != raw or clean_name != raw.strip().split()[0]:
                rebuilt = " ".join(tokens)
                if sep:
                    rebuilt += f" ;{comment}"
                out[i] = rebuilt + "\n"
            continue
        if tokens[3].upper() == prefix:
            if prefix in DEFAULT_PASSIVE_VALUES:
                v = DEFAULT_PASSIVE_VALUES[prefix]
            else:
                v = "-5" if (prefix == "V" and (tokens[1].startswith("-") or "-" in tokens[1])) else DEFAULT_SOURCE_VALUES[prefix]
            tokens[3:] = [v]
            rebuilt = " ".join(tokens)
            if sep:
                rebuilt += f" ;{comment}"
            out[i] = rebuilt + "\n"
        elif normalized_raw != raw or clean_name != raw.strip().split()[0]:
            rebuilt = " ".join(tokens)
            if sep:
                rebuilt += f" ;{comment}"
            out[i] = rebuilt + "\n"
    if inline_libs:
        existing_lib_lines = [
            raw.strip().lower()
            for raw in out
            if raw.strip().lower().startswith(".lib")
        ]
        for libtok in sorted(inline_libs):
            libtok_l = libtok.lower()
            if any(libtok_l in line for line in existing_lib_lines):
                continue
            lib_arg = libtok
            model_hint = inline_lib_model_hints.get(libtok, "")
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
                        break
            insert_before_end(out, f".lib {lib_arg}")
    return out


def parse_nodes_and_vsources(lines: list[str]) -> tuple[list[str], list[str]]:
    nodes: set[str] = set()
    vsrc: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith(("*", ";", ".")):
            continue
        t = s.split()
        if not t:
            continue
        name = t[0]
        prefix = name[0].upper()
        if prefix == "V" and len(t) >= 3:
            vsrc.append(name)

        # Strip inline ';' comments before node parsing.
        body: list[str] = []
        for tok in t:
            if tok.startswith(";"):
                break
            body.append(tok)
        t = body
        if len(t) < 3:
            continue

        node_tokens: list[str] = []
        if prefix in MEAS_NODE_ARITY_BY_PREFIX:
            node_tokens = t[1 : 1 + MEAS_NODE_ARITY_BY_PREFIX[prefix]]
        elif prefix in {"X", "A"}:
            rest = t[1:]
            if len(rest) >= 2:
                cut = len(rest)
                for i, tok in enumerate(rest):
                    if "=" in tok or tok.upper().startswith("PARAMS:"):
                        cut = i
                        break
                if cut < len(rest):
                    node_end = max(0, cut - 1)
                else:
                    node_end = max(0, len(rest) - 1)
                node_tokens = rest[:node_end]

        for n in node_tokens:
            if n and n != "0":
                nodes.add(n)
    return sorted(nodes), sorted(vsrc)


def safe_name(x: str) -> str:
    # Keep .meas labels readable but valid; '-' in labels breaks LTspice parsing.
    z = re.sub(r"[^A-Za-z0-9_.$]", "_", x)
    if not z:
        return "x"
    if z[0].isdigit():
        return f"n_{z}"
    return z


def add_analysis_and_meas(lines: list[str]) -> list[str]:
    out = list(lines)
    if not has_directive(out, ".op"):
        insert_before_end(out, ".op")
    if not has_directive(out, ".tran"):
        insert_before_end(out, ".tran 0 10m 0 10u")
    if any(raw.strip().startswith("; AUTO_MEAS_BEGIN") for raw in out):
        return out

    nodes, vsrc = parse_nodes_and_vsources(out)
    insert_before_end(out, "; AUTO_MEAS_BEGIN")
    insert_before_end(out, ".save V(*)")
    for s in vsrc:
        insert_before_end(out, f".save I({s})")
    for n in nodes:
        nid = safe_name(n)
        insert_before_end(out, f".meas op V_{nid} FIND V({n})")
        insert_before_end(out, f".meas tran V_{nid}_MAX MAX V({n})")
        insert_before_end(out, f".meas tran V_{nid}_MIN MIN V({n})")
        insert_before_end(out, f".meas tran V_{nid}_RMS RMS V({n})")
    for s in vsrc:
        sid = safe_name(s)
        insert_before_end(out, f".meas op I_{sid} FIND I({s})")
        insert_before_end(out, f".meas tran I_{sid}_MAX MAX I({s})")
        insert_before_end(out, f".meas tran I_{sid}_MIN MIN I({s})")
        insert_before_end(out, f".meas tran I_{sid}_RMS RMS I({s})")
    insert_before_end(out, "; AUTO_MEAS_END")
    return out


def parse_measurements(log_path: Path) -> dict[str, float | str]:
    if not log_path.exists():
        return {}
    out: dict[str, float | str] = {}
    meas_re = re.compile(r"^\s*([A-Za-z0-9_.$-]+)\s*[:=].*?=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b")
    kv_re = re.compile(r"^\s*([A-Za-z0-9_.$-]+)\s*=\s*(\S+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        m = meas_re.match(s)
        if m:
            k, v = m.group(1), m.group(2)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
            continue
        m2 = kv_re.match(s)
        if m2:
            k, v = m2.group(1), m2.group(2).rstrip(",")
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main() -> int:
    args = parse_args()
    lab_dir = args.out_root / args.lab
    base_net = lab_dir / "base_netlists" / f"{args.lab}.net"
    if not base_net.exists():
        raise FileNotFoundError(f"Base netlist not found: {base_net}")

    golden_dir = lab_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    lines = base_net.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    lines = sanitize_base(lines)
    lines = add_analysis_and_meas(lines)

    cir_path = golden_dir / f"{args.lab}__golden.cir"
    log_path = golden_dir / f"{args.lab}__golden.log"
    cir_path.write_text("".join(lines), encoding="utf-8")

    proc = subprocess.run(
        [args.ltspice_bin, "-b", str(cir_path)],
        text=True,
        capture_output=True,
        timeout=args.timeout_sec,
        check=False,
    )

    src_log = cir_path.with_suffix(".log")
    if src_log.exists():
        src_log.replace(log_path)

    measurements = parse_measurements(log_path)
    raw_path = None
    raws = sorted(golden_dir.glob(f"{cir_path.stem}*.raw"))
    if raws:
        raw_path = str(raws[0])
    if not args.keep_raw:
        for rf in raws:
            rf.unlink(missing_ok=True)
        raw_path = None

    manifest = {
        "variant_id": f"{args.lab}__golden",
        "return_code": proc.returncode,
        "success": proc.returncode == 0,
        "log_path": str(log_path) if log_path.exists() else None,
        "raw_path": raw_path,
        "measurements": measurements,
        "stderr": (proc.stderr or "").strip()[:2000],
        "stdout": (proc.stdout or "").strip()[:2000],
        "netlist_path": str(cir_path),
    }

    (golden_dir / "golden_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (golden_dir / "golden_measurements.json").write_text(
        json.dumps(manifest.get("measurements", {}), indent=2),
        encoding="utf-8",
    )

    print(f"Wrote golden manifest: {golden_dir / 'golden_manifest.json'}")
    print(f"Wrote golden measurements: {golden_dir / 'golden_measurements.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

