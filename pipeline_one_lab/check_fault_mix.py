#!/usr/bin/env python3
"""Summarize fault-type distribution from variant manifest JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check fault-type mix in variant manifests")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Single variant_manifest.jsonl path",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("pipeline/out_one_lab"),
        help="Root directory that contains per-lab folders",
    )
    parser.add_argument(
        "--lab-prefix",
        type=str,
        default="",
        help="Only include labs whose folder name starts with this prefix",
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


def find_manifests(out_root: Path, lab_prefix: str) -> list[Path]:
    manifests: list[Path] = []
    for p in sorted(out_root.glob("*/variant_manifest.jsonl")):
        lab = p.parent.name
        if lab_prefix and not lab.startswith(lab_prefix):
            continue
        manifests.append(p)
    return manifests


def main() -> int:
    args = parse_args()

    manifests: list[Path]
    if args.manifest is not None:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        manifests = [args.manifest]
    else:
        manifests = find_manifests(args.out_root, args.lab_prefix.strip())
        if not manifests:
            raise FileNotFoundError(
                f"No manifests found under {args.out_root} "
                f"(lab_prefix={args.lab_prefix!r})"
            )

    counts: Counter[str] = Counter()
    by_lab: dict[str, Counter[str]] = {}
    total = 0

    for manifest in manifests:
        rows = load_jsonl(manifest)
        lab = manifest.parent.name
        local = Counter()
        for row in rows:
            ftype = ((row.get("fault") or {}).get("fault_type")) or "unknown"
            counts[ftype] += 1
            local[ftype] += 1
            total += 1
        by_lab[lab] = local

    print(f"manifests={len(manifests)} total_variants={total}")
    print("overall_fault_mix:")
    for fault_type, n in counts.most_common():
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {fault_type:24s} {n:8d}  {pct:6.2f}%")

    print("\nper_lab_totals:")
    for lab in sorted(by_lab.keys()):
        lab_total = sum(by_lab[lab].values())
        print(f"  {lab}: {lab_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

