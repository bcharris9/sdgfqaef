#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create an oversampled instruct JSONL focused on specified fault classes."
    )
    p.add_argument("--in-file", required=True, type=Path, help="Input instruct JSONL")
    p.add_argument("--out-file", required=True, type=Path, help="Output instruct JSONL")
    p.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional JSON report with before/after class counts",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--boost",
        action="append",
        default=[],
        help=(
            "Class boost in CLASS=M form. M is total multiplier (e.g., pin_open=3). "
            "Can be repeated."
        ),
    )
    p.add_argument(
        "--default-multiplier",
        type=int,
        default=1,
        help="Multiplier for classes not listed in --boost (default: 1)",
    )
    return p.parse_args()


def extract_fault_type(row: dict) -> str:
    out = str(row.get("output", "") or "")
    for line in out.splitlines():
        if line.lower().startswith("faulttype:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def parse_boosts(boost_items: list[str], default_multiplier: int) -> dict[str, int]:
    boosts: dict[str, int] = {}
    for item in boost_items:
        if "=" not in item:
            raise ValueError(f"Invalid --boost '{item}'. Expected CLASS=M")
        cls, mult_text = item.split("=", 1)
        cls = cls.strip()
        mult = int(mult_text.strip())
        if not cls:
            raise ValueError(f"Invalid --boost '{item}'. Empty class name")
        if mult < 1:
            raise ValueError(f"Invalid --boost '{item}'. Multiplier must be >= 1")
        boosts[cls] = mult
    boosts["__default__"] = max(1, int(default_multiplier))
    return boosts


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    boosts = parse_boosts(args.boost, args.default_multiplier)

    rows: list[dict] = []
    by_class: dict[str, list[dict]] = defaultdict(list)
    before = Counter()

    with args.in_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            cls = extract_fault_type(row)
            before[cls] += 1
            by_class[cls].append(row)

    augmented: list[dict] = []
    for cls, cls_rows in by_class.items():
        mult = boosts.get(cls, boosts["__default__"])
        # Keep all originals, then add sampled duplicates if multiplier > 1.
        augmented.extend(cls_rows)
        extra = (mult - 1) * len(cls_rows)
        for _ in range(extra):
            augmented.append(rng.choice(cls_rows))

    rng.shuffle(augmented)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8", newline="\n") as f:
        for row in augmented:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")

    after = Counter(extract_fault_type(r) for r in augmented)
    report = {
        "input_file": str(args.in_file),
        "output_file": str(args.out_file),
        "seed": args.seed,
        "boosts": {k: v for k, v in boosts.items() if k != "__default__"},
        "default_multiplier": boosts["__default__"],
        "rows_before": sum(before.values()),
        "rows_after": sum(after.values()),
        "class_counts_before": dict(sorted(before.items())),
        "class_counts_after": dict(sorted(after.items())),
    }
    print(json.dumps(report, indent=2))

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote_report={args.report_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
