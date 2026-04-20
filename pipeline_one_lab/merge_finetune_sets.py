#!/usr/bin/env python3
"""Merge per-lab fine-tune JSONL files into one combined dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one-lab finetune sets")
    parser.add_argument("--out-root", type=Path, default=Path("pipeline/out_one_lab"))
    parser.add_argument(
        "--labs",
        type=str,
        default="",
        help="Comma-separated lab stems. If omitted, all folders under --out-root are used.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("pipeline/out_one_lab/merged_finetune"),
    )
    parser.add_argument(
        "--finetune-subdir",
        type=str,
        default="finetune_small",
        help="Per-lab subdirectory name to merge from (default: finetune_small).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def detect_labs(out_root: Path) -> list[str]:
    return sorted(p.name for p in out_root.iterdir() if p.is_dir())


def read_jsonl(path: Path) -> list[dict]:
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


def main() -> int:
    args = parse_args()
    if args.labs.strip():
        labs = [x.strip() for x in args.labs.split(",") if x.strip()]
    else:
        labs = detect_labs(args.out_root)
    if not labs:
        raise RuntimeError("No labs found to merge.")

    train_instruct: list[dict] = []
    val_instruct: list[dict] = []
    train_chat: list[dict] = []
    val_chat: list[dict] = []
    used_labs: list[str] = []

    for lab in labs:
        base = args.out_root / lab / args.finetune_subdir
        ti = base / "train_instruct.jsonl"
        vi = base / "val_instruct.jsonl"
        tc = base / "train_chat.jsonl"
        vc = base / "val_chat.jsonl"
        if not (ti.exists() and vi.exists() and tc.exists() and vc.exists()):
            continue

        train_instruct.extend(read_jsonl(ti))
        val_instruct.extend(read_jsonl(vi))
        train_chat.extend(read_jsonl(tc))
        val_chat.extend(read_jsonl(vc))
        used_labs.append(lab)

    if not used_labs:
        raise RuntimeError("No valid finetune_small sets found.")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(train_instruct)
        rng.shuffle(val_instruct)
        rng.shuffle(train_chat)
        rng.shuffle(val_chat)

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dest_dir / "train_instruct.jsonl", train_instruct)
    write_jsonl(args.dest_dir / "val_instruct.jsonl", val_instruct)
    write_jsonl(args.dest_dir / "train_chat.jsonl", train_chat)
    write_jsonl(args.dest_dir / "val_chat.jsonl", val_chat)

    meta = {
        "used_labs": used_labs,
        "train_rows": len(train_instruct),
        "val_rows": len(val_instruct),
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "out_root": str(args.out_root),
    }
    (args.dest_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Merged labs: {', '.join(used_labs)}")
    print(f"Rows: train={len(train_instruct)} val={len(val_instruct)}")
    print(f"Wrote merged set: {args.dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
