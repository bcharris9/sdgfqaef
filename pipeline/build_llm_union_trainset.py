#!/usr/bin/env python3
"""Build a large instruct-format LLM train/val set from multiple merged finetune sets.

This script is designed for the all-labs LLM workflow:
- Merge existing merged instruct sets (v2/v3/v4/v5, etc.).
- Optionally include each source set's validation rows into the training pool.
- Optionally boost selected fault classes by duplication.
- Optionally add measurement-jittered copies (Measured + DeltasVsGolden consistency preserved).
- Create a fresh internal train/val split for LoRA training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAULT_RE = re.compile(r"^faulttype:\s*(.+)$", re.I | re.M)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build union LLM instruct trainset from multiple merged finetune sets")
    p.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        default=[],
        help="Merged finetune directory containing train_instruct.jsonl and val_instruct.jsonl (repeatable).",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Output directory for train_instruct.jsonl / val_instruct.jsonl",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1, help="Internal validation fraction from final pool.")
    p.add_argument(
        "--include-source-val-in-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, include each source val_instruct.jsonl in the merged training pool.",
    )
    p.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deduplicate exact rows by (instruction,input,output).",
    )
    p.add_argument(
        "--boost",
        action="append",
        default=[],
        help="Class multiplier in CLASS=M form. Repeats exact row copies before jitter (e.g. pin_open=3).",
    )
    p.add_argument("--default-multiplier", type=int, default=1)
    p.add_argument(
        "--jitter-copies",
        type=int,
        default=0,
        help="How many additional jittered copies to create per pooled row.",
    )
    p.add_argument(
        "--jitter-sigma",
        type=float,
        default=0.005,
        help="Relative Gaussian sigma for jitter on numeric measurement/delta features (0.005 = 0.5%%).",
    )
    p.add_argument(
        "--jitter-prob",
        type=float,
        default=1.0,
        help="Per-feature probability of applying jitter.",
    )
    p.add_argument(
        "--jitter-max-abs",
        type=float,
        default=1e9,
        help="Skip jitter for absolute values larger than this (protect pathological short-current values).",
    )
    p.add_argument(
        "--jitter-classes",
        type=str,
        default="",
        help="Comma-separated classes eligible for jitter copies. Empty = all classes.",
    )
    p.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional JSON report path.",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def extract_fault_type(row: dict[str, Any]) -> str:
    out = str(row.get("output", "") or "")
    m = FAULT_RE.search(out)
    if not m:
        return "unknown"
    return m.group(1).strip()


def parse_boosts(boost_items: list[str], default_multiplier: int) -> dict[str, int]:
    boosts: dict[str, int] = {}
    for item in boost_items:
        if "=" not in item:
            raise ValueError(f"Invalid --boost '{item}'. Expected CLASS=M")
        k, v = item.split("=", 1)
        cls = k.strip()
        mult = int(v.strip())
        if not cls or mult < 1:
            raise ValueError(f"Invalid --boost '{item}'")
        boosts[cls] = mult
    boosts["__default__"] = max(1, int(default_multiplier))
    return boosts


def _parse_kv_segment(segment: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in segment.split(";"):
        s = item.strip()
        if not s or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _format_num(v: float) -> str:
    return f"{v:.12g}"


def _is_jitter_key(k: str) -> bool:
    key = (k or "").strip().lower()
    if key in {"temp", "tnom"}:
        return False
    if key.startswith("v_") or key.startswith("i_"):
        return True
    if key.endswith("_delta"):
        base = key[: -len("_delta")]
        return base.startswith("v_") or base.startswith("i_")
    return False


def jitter_input_text(
    input_text: str,
    rng: random.Random,
    sigma: float,
    prob: float,
    max_abs: float,
) -> str:
    if sigma <= 0:
        return input_text

    lines = str(input_text or "").splitlines()
    measured_idx = None
    deltas_idx = None
    measured_pairs: list[tuple[str, str]] = []
    delta_pairs: list[tuple[str, str]] = []

    for i, line in enumerate(lines):
        if line.startswith("Measured:"):
            measured_idx = i
            measured_pairs = _parse_kv_segment(line.split(":", 1)[1].strip())
        elif line.startswith("DeltasVsGolden:"):
            deltas_idx = i
            delta_pairs = _parse_kv_segment(line.split(":", 1)[1].strip())

    if measured_idx is None and deltas_idx is None:
        return input_text

    # Shared additive jitter by base feature key to preserve delta/measured consistency.
    # Example: v_n001_max and v_n001_max_delta get the same additive perturbation.
    eps_by_base: dict[str, float] = {}

    def jitter_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for k, raw_v in pairs:
            lk = k.lower()
            if not _is_jitter_key(lk):
                out.append((k, raw_v))
                continue
            try:
                v = float(raw_v)
            except Exception:
                out.append((k, raw_v))
                continue
            if abs(v) > max_abs or rng.random() > prob:
                out.append((k, raw_v))
                continue
            base = lk[:-6] if lk.endswith("_delta") else lk
            if base not in eps_by_base:
                scale = max(abs(v), 1e-6)
                eps_by_base[base] = rng.gauss(0.0, sigma * scale)
            out.append((k, _format_num(v + eps_by_base[base])))
        return out

    if deltas_idx is not None and delta_pairs:
        delta_pairs = jitter_pairs(delta_pairs)
        lines[deltas_idx] = "DeltasVsGolden: " + "; ".join(f"{k}={v}" for k, v in delta_pairs)
    if measured_idx is not None and measured_pairs:
        measured_pairs = jitter_pairs(measured_pairs)
        lines[measured_idx] = "Measured: " + "; ".join(f"{k}={v}" for k, v in measured_pairs)

    return "\n".join(lines)


def stratified_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[extract_fault_type(row)].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for cls, cls_rows in by_class.items():
        rng.shuffle(cls_rows)
        if len(cls_rows) <= 1:
            train.extend(cls_rows)
            continue
        n_val = int(round(len(cls_rows) * max(0.0, min(0.5, val_ratio))))
        n_val = max(1, min(len(cls_rows) - 1, n_val))
        val.extend(cls_rows[:n_val])
        train.extend(cls_rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("instruction", "") or ""),
        str(row.get("input", "") or ""),
        str(row.get("output", "") or ""),
    )


def main() -> int:
    args = parse_args()
    if not args.source_dir:
        raise RuntimeError("Provide at least one --source-dir")

    rng = random.Random(args.seed)
    boosts = parse_boosts(args.boost, args.default_multiplier)
    jitter_classes = {
        x.strip() for x in str(args.jitter_classes).split(",") if x.strip()
    } if str(args.jitter_classes).strip() else None

    pooled: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []

    for src in args.source_dir:
        ti = src / "train_instruct.jsonl"
        vi = src / "val_instruct.jsonl"
        if not ti.exists():
            raise FileNotFoundError(f"Missing {ti}")
        train_rows = load_jsonl(ti)
        val_rows = load_jsonl(vi) if (args.include_source_val_in_pool and vi.exists()) else []
        pooled.extend(train_rows)
        pooled.extend(val_rows)
        source_stats.append(
            {
                "source_dir": str(src),
                "train_rows_added": len(train_rows),
                "val_rows_added": len(val_rows),
                "total_added": len(train_rows) + len(val_rows),
            }
        )

    before_merge = len(pooled)

    if args.dedup:
        uniq: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in pooled:
            uniq[row_key(row)] = row
        pooled = list(uniq.values())

    after_dedup = len(pooled)

    # Exact copy boosts
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pooled:
        by_class[extract_fault_type(row)].append(row)

    boosted: list[dict[str, Any]] = []
    for cls, rows in by_class.items():
        mult = boosts.get(cls, boosts["__default__"])
        boosted.extend(rows)
        extra = (mult - 1) * len(rows)
        for _ in range(extra):
            boosted.append(rng.choice(rows))

    # Measurement jitter augmentation
    augmented: list[dict[str, Any]] = list(boosted)
    if args.jitter_copies > 0 and args.jitter_sigma > 0:
        for row in boosted:
            cls = extract_fault_type(row)
            if jitter_classes is not None and cls not in jitter_classes:
                continue
            for _ in range(args.jitter_copies):
                new_row = dict(row)
                new_row["input"] = jitter_input_text(
                    str(row.get("input", "")),
                    rng=rng,
                    sigma=float(args.jitter_sigma),
                    prob=float(args.jitter_prob),
                    max_abs=float(args.jitter_max_abs),
                )
                augmented.append(new_row)

    train_rows, val_rows = stratified_split(augmented, args.val_ratio, args.seed + 101)

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dest_dir / "train_instruct.jsonl", train_rows)
    write_jsonl(args.dest_dir / "val_instruct.jsonl", val_rows)

    cls_before = Counter(extract_fault_type(r) for r in pooled)
    cls_boosted = Counter(extract_fault_type(r) for r in boosted)
    cls_final = Counter(extract_fault_type(r) for r in augmented)
    cls_train = Counter(extract_fault_type(r) for r in train_rows)
    cls_val = Counter(extract_fault_type(r) for r in val_rows)

    report = {
        "source_stats": source_stats,
        "settings": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "include_source_val_in_pool": bool(args.include_source_val_in_pool),
            "dedup": bool(args.dedup),
            "boosts": {k: v for k, v in boosts.items() if k != "__default__"},
            "default_multiplier": boosts["__default__"],
            "jitter_copies": int(args.jitter_copies),
            "jitter_sigma": float(args.jitter_sigma),
            "jitter_prob": float(args.jitter_prob),
            "jitter_max_abs": float(args.jitter_max_abs),
            "jitter_classes": sorted(jitter_classes) if jitter_classes is not None else [],
        },
        "rows": {
            "before_merge": before_merge,
            "after_dedup": after_dedup,
            "after_boost": len(boosted),
            "after_jitter": len(augmented),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "class_counts": {
            "pooled": dict(sorted(cls_before.items())),
            "boosted": dict(sorted(cls_boosted.items())),
            "augmented": dict(sorted(cls_final.items())),
            "train": dict(sorted(cls_train.items())),
            "val": dict(sorted(cls_val.items())),
        },
    }
    report_path = args.report_file or (args.dest_dir / "build_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote_train={args.dest_dir / 'train_instruct.jsonl'}")
    print(f"wrote_val={args.dest_dir / 'val_instruct.jsonl'}")
    print(f"wrote_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

