#!/usr/bin/env python3
"""Iteratively retrain a LoRA adapter against unseen eval chunks until streak target.

Loop:
1) Evaluate current adapter on next unseen chunk.
2) If class_match >= threshold, increment streak.
3) Else reset streak, add failed chunk rows into augmented train set, warm-start retrain.
4) Continue until streak target reached or chunks exhausted/max iterations reached.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAULT_RE = re.compile(r"^faulttype:\s*(.+)$", re.I | re.M)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative unseen eval + warm-start LoRA retraining")
    p.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument(
        "--initial-adapter-dir",
        type=Path,
        required=True,
        help="Starting adapter directory",
    )
    p.add_argument("--base-train-file", type=Path, required=True)
    p.add_argument("--base-val-file", type=Path, required=True)
    p.add_argument("--chunk-dir", type=Path, required=True)
    p.add_argument("--chunk-pattern", type=str, default="eval_chunk_*.jsonl")
    p.add_argument("--knn-ref-file", type=Path, required=True)
    p.add_argument("--decode-mode", type=str, default="score_classes_knn")
    p.add_argument("--knn-k", type=int, default=3)
    p.add_argument("--knn-alpha", type=float, default=0.2)
    p.add_argument("--knn-weighted-vote", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-pct", type=float, default=90.0)
    p.add_argument("--target-streak", type=int, default=5)
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--sample-cap", type=int, default=0, help="Optional max samples per chunk for quick runs.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)

    # Retrain knobs
    p.add_argument("--retrain-max-steps", type=int, default=120)
    p.add_argument("--retrain-num-epochs", type=float, default=1.0)
    p.add_argument("--retrain-learning-rate", type=float, default=1.2e-4)
    p.add_argument("--retrain-max-length", type=int, default=512)
    p.add_argument("--retrain-train-batch-size", type=int, default=1)
    p.add_argument("--retrain-eval-batch-size", type=int, default=1)
    p.add_argument("--retrain-grad-accum", type=int, default=8)
    p.add_argument("--retrain-eval-steps", type=int, default=60)
    p.add_argument("--retrain-save-steps", type=int, default=60)
    p.add_argument("--retrain-logging-steps", type=int, default=10)

    # How hard to upweight failed chunks
    p.add_argument("--failed-default-multiplier", type=int, default=3)
    p.add_argument(
        "--failed-class-multiplier",
        action="append",
        default=[],
        help="Override per class, CLASS=M (repeatable), e.g. pin_open=6",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def fault_type_of_row(row: dict[str, Any]) -> str:
    out = str(row.get("output", "") or "")
    m = FAULT_RE.search(out)
    if not m:
        return "unknown"
    return m.group(1).strip()


def parse_multiplier_overrides(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --failed-class-multiplier '{item}'")
        k, v = item.split("=", 1)
        cls = k.strip()
        mul = int(v.strip())
        if not cls or mul < 1:
            raise ValueError(f"Invalid --failed-class-multiplier '{item}'")
        out[cls] = mul
    return out


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


@dataclass
class IterResult:
    iteration: int
    chunk_path: str
    class_pct: float
    class_count: int
    samples: int
    streak_after: int
    retrained: bool
    adapter_dir: str
    report_file: str
    preds_file: str
    train_file: str


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    if not args.python_exe.exists():
        raise FileNotFoundError(f"Python exe not found: {args.python_exe}")
    if not args.initial_adapter_dir.exists():
        raise FileNotFoundError(f"Initial adapter dir not found: {args.initial_adapter_dir}")
    if not args.base_train_file.exists():
        raise FileNotFoundError(f"Base train file not found: {args.base_train_file}")
    if not args.base_val_file.exists():
        raise FileNotFoundError(f"Base val file not found: {args.base_val_file}")
    if not args.knn_ref_file.exists():
        raise FileNotFoundError(f"KNN ref file not found: {args.knn_ref_file}")

    chunk_paths = sorted(args.chunk_dir.glob(args.chunk_pattern))
    if not chunk_paths:
        raise RuntimeError(f"No chunk files found under {args.chunk_dir} with pattern {args.chunk_pattern}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_train_rows = load_jsonl(args.base_train_file)
    multiplier_overrides = parse_multiplier_overrides(args.failed_class_multiplier)

    failed_memory_rows: list[dict[str, Any]] = []
    current_adapter = args.initial_adapter_dir
    streak = 0
    iter_idx = 0
    chunk_idx = 0
    history: list[IterResult] = []

    while iter_idx < args.max_iterations and chunk_idx < len(chunk_paths):
        iter_idx += 1
        chunk = chunk_paths[chunk_idx]
        chunk_idx += 1
        chunk_rows = load_jsonl(chunk)
        chunk_n = len(chunk_rows)
        if chunk_n == 0:
            raise RuntimeError(f"Chunk has no rows: {chunk}")

        iter_dir = args.out_dir / f"iter_{iter_idx:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        pred_file = iter_dir / "eval_predictions.jsonl"
        report_file = iter_dir / "eval_report.json"
        cmd_eval = [
            str(args.python_exe),
            "pipeline/test_lora_model.py",
            "--model-name",
            args.model_name,
            "--adapter-dir",
            str(current_adapter),
            "--data-file",
            str(chunk),
            "--out-file",
            str(pred_file),
            "--report-file",
            str(report_file),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            "0",
            "--decode-mode",
            args.decode_mode,
            "--knn-ref-file",
            str(args.knn_ref_file),
            "--knn-k",
            str(args.knn_k),
            "--knn-alpha",
            str(args.knn_alpha),
            "--enforce-format",
            "--use-prerules",
        ]
        if args.knn_weighted_vote:
            cmd_eval.append("--knn-weighted-vote")
        else:
            cmd_eval.append("--no-knn-weighted-vote")
        eval_samples = args.sample_cap if args.sample_cap > 0 else chunk_n
        cmd_eval.extend(["--max-samples", str(eval_samples)])

        env = dict(os.environ)
        env["TQDM_DISABLE"] = "1"
        run_cmd(cmd_eval, cwd=Path("."), env=env)

        rep = json.loads(report_file.read_text(encoding="utf-8"))
        class_pct = float(rep["class_match"]["pct"])
        class_count = int(rep["class_match"]["count"])
        samples = int(rep["samples"])

        retrained = False
        train_file_used = str(args.base_train_file)
        if class_pct >= args.target_pct:
            streak += 1
        else:
            streak = 0
            retrained = True

            # Add failed chunk rows to memory with class-targeted multipliers.
            for row in chunk_rows:
                cls = fault_type_of_row(row)
                mult = multiplier_overrides.get(cls, args.failed_default_multiplier)
                for _ in range(max(1, mult)):
                    failed_memory_rows.append(row)

            aug_rows = list(base_train_rows) + failed_memory_rows
            rnd = random.Random(args.seed + iter_idx)
            rnd.shuffle(aug_rows)
            train_aug = iter_dir / "train_aug.jsonl"
            write_jsonl(train_aug, aug_rows)
            train_file_used = str(train_aug)

            next_adapter = iter_dir / "adapter"
            cmd_train = [
                str(args.python_exe),
                "pipeline/train_lora.py",
                "--model-name",
                args.model_name,
                "--train-file",
                str(train_aug),
                "--val-file",
                str(args.base_val_file),
                "--output-dir",
                str(next_adapter),
                "--init-adapter-dir",
                str(current_adapter),
                "--max-length",
                str(args.retrain_max_length),
                "--learning-rate",
                str(args.retrain_learning_rate),
                "--num-epochs",
                str(args.retrain_num_epochs),
                "--max-steps",
                str(args.retrain_max_steps),
                "--train-batch-size",
                str(args.retrain_train_batch_size),
                "--eval-batch-size",
                str(args.retrain_eval_batch_size),
                "--grad-accum",
                str(args.retrain_grad_accum),
                "--eval-steps",
                str(args.retrain_eval_steps),
                "--save-steps",
                str(args.retrain_save_steps),
                "--logging-steps",
                str(args.retrain_logging_steps),
                "--seed",
                str(args.seed + iter_idx),
                "--gradient-checkpointing",
            ]
            run_cmd(cmd_train, cwd=Path("."), env=env)
            current_adapter = next_adapter

        history.append(
            IterResult(
                iteration=iter_idx,
                chunk_path=str(chunk),
                class_pct=class_pct,
                class_count=class_count,
                samples=samples,
                streak_after=streak,
                retrained=retrained,
                adapter_dir=str(current_adapter),
                report_file=str(report_file),
                preds_file=str(pred_file),
                train_file=train_file_used,
            )
        )

        print(
            f"iter={iter_idx} chunk={chunk.name} class_match={class_pct:.2f}% "
            f"({class_count}/{samples}) streak={streak} retrained={retrained}"
        )

        if streak >= args.target_streak:
            break

    out_state = {
        "target_pct": args.target_pct,
        "target_streak": args.target_streak,
        "achieved_streak": streak,
        "iterations": iter_idx,
        "chunks_used": chunk_idx,
        "chunks_available": len(chunk_paths),
        "success": streak >= args.target_streak,
        "current_adapter": str(current_adapter),
        "history": [r.__dict__ for r in history],
    }
    state_path = args.out_dir / "iter_state.json"
    state_path.write_text(json.dumps(out_state, indent=2), encoding="utf-8")
    print(f"wrote_state={state_path}")
    if out_state["success"]:
        print("result=SUCCESS")
        return 0
    print("result=INCOMPLETE")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
