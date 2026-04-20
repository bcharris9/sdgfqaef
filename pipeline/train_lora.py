#!/usr/bin/env python3
"""LoRA fine-tuning script for compact instruction datasets.

Expected dataset schema (JSONL):
  {"instruction": "...", "input": "...", "output": "..."}
"""

from __future__ import annotations

import argparse
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small LoRA adapter on instruct JSONL data")
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-4b-it",
        help="Base model checkpoint",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("pipeline/out/finetune_small/train_instruct.jsonl"),
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("pipeline/out/finetune_small/val_instruct.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/out/lora_small"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If > 0, only use this many training rows (for quick smoke runs).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="If > 0, only use this many validation rows (for quick smoke runs).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, stop after this many optimizer steps (overrides epoch length).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to LoRA-wrap if present",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--init-adapter-dir",
        type=Path,
        default=Path(""),
        help=(
            "Optional existing LoRA adapter directory to continue training from. "
            "When set, LoRA config args are ignored and the adapter config is reused."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume interrupted training.",
    )
    parser.add_argument(
        "--response-style",
        choices=["auto", "diag_fix", "faulttype_diag_fix"],
        default="auto",
        help=(
            "Expected assistant output style in the dataset. "
            "auto infers from training outputs."
        ),
    )
    parser.add_argument(
        "--prompt-truncation",
        choices=["tail", "head_tail"],
        default="head_tail",
        help=(
            "How to truncate overlength prompts. "
            "tail keeps the last prompt tokens only. "
            "head_tail keeps both beginning and end of the prompt."
        ),
    )
    parser.add_argument(
        "--prompt-format-hint",
        type=str,
        default="",
        help="Optional extra instruction appended to the response-format header.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def detect_available_target_modules(model: torch.nn.Module, candidates: list[str]) -> list[str]:
    names = set()
    for module_name, _ in model.named_modules():
        last = module_name.split(".")[-1]
        if last in candidates:
            names.add(last)
    return sorted(names)


def detect_response_style(rows: Any) -> str:
    for row in rows:
        output = str((row or {}).get("output", "")).strip().lower()
        if not output:
            continue
        if output.startswith("faulttype:"):
            return "faulttype_diag_fix"
        return "diag_fix"
    return "diag_fix"


def build_prompt(
    instruction: str,
    input_text: str,
    response_style: str,
    prompt_format_hint: str,
) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()

    if response_style == "faulttype_diag_fix":
        hint = (
            "Return exactly two lines.\n"
            "FaultType: one of param_drift|missing_component|pin_open|swapped_nodes|"
            "short_between_nodes|resistor_value_swap|resistor_wrong_value.\n"
            "Diagnosis: concise sentence. Fix: concise sentence."
        )
    else:
        hint = (
            "Return exactly one line in this format: "
            "Diagnosis: <text>. Fix: <text>."
        )

    extra = (prompt_format_hint or "").strip()
    if extra:
        hint = f"{hint}\n{extra}"

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{hint}\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{hint}\n"
    )


def preprocess_row(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    response_style: str,
    prompt_format_hint: str,
    prompt_truncation: str,
) -> dict[str, list[int]]:
    prompt = build_prompt(
        row.get("instruction", ""),
        row.get("input", ""),
        response_style,
        prompt_format_hint,
    )
    output = (row.get("output", "") or "").strip()
    if output:
        output = output + tokenizer.eos_token
    else:
        output = tokenizer.eos_token

    prompt_tok = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_tok["input_ids"]
    prompt_tti = prompt_tok.get("token_type_ids", [0] * len(prompt_ids))

    output_tok = tokenizer(output, add_special_tokens=False)
    output_ids = output_tok["input_ids"]
    output_tti = output_tok.get("token_type_ids", [0] * len(output_ids))
    if not output_ids:
        output_ids = [tokenizer.eos_token_id]
        output_tti = [0]

    # Always preserve supervised response tokens.
    if len(output_ids) >= max_length:
        output_ids = output_ids[:max_length]
        output_tti = output_tti[:max_length]
        kept_prompt_ids: list[int] = []
        kept_prompt_tti: list[int] = []
    else:
        prompt_budget = max_length - len(output_ids)
        if len(prompt_ids) <= prompt_budget:
            kept_prompt_ids = prompt_ids
            kept_prompt_tti = prompt_tti
        elif prompt_truncation == "tail":
            kept_prompt_ids = prompt_ids[-prompt_budget:]
            kept_prompt_tti = prompt_tti[-len(kept_prompt_ids):]
        else:
            # Keep both prompt header/context and latest measurements/task text.
            head_keep = max(1, int(prompt_budget * 0.35))
            tail_keep = max(1, prompt_budget - head_keep)
            if head_keep + tail_keep > prompt_budget:
                tail_keep = prompt_budget - head_keep
            if tail_keep <= 0:
                tail_keep = 1
                head_keep = prompt_budget - tail_keep
            kept_prompt_ids = prompt_ids[:head_keep] + prompt_ids[-tail_keep:]
            kept_prompt_tti = prompt_tti[:head_keep] + prompt_tti[-tail_keep:]

    full_ids = kept_prompt_ids + output_ids
    full_token_type_ids = kept_prompt_tti + output_tti
    attention_mask = [1] * len(full_ids)
    labels = ([-100] * len(kept_prompt_ids)) + output_ids

    return {
        "input_ids": full_ids,
        "token_type_ids": full_token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class SupervisedDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            row = {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            if "token_type_ids" in f:
                row["token_type_ids"] = f["token_type_ids"]
            input_features.append(row)
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        max_len = batch["input_ids"].shape[1]

        labels = []
        for feature in features:
            row = feature["labels"]
            if len(row) < max_len:
                row = row + ([-100] * (max_len - len(row)))
            labels.append(row)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if not args.train_file.exists():
        raise FileNotFoundError(f"Missing train file: {args.train_file}")
    if not args.val_file.exists():
        raise FileNotFoundError(f"Missing val file: {args.val_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = choose_dtype()
    # Newer transformers prefers "dtype"; many stable versions still use "torch_dtype".
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    from_pretrained_sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    target_modules: list[str]
    init_adapter = str(args.init_adapter_dir).strip()
    if init_adapter:
        if not args.init_adapter_dir.exists():
            raise FileNotFoundError(f"Init adapter dir not found: {args.init_adapter_dir}")
        peft_load_kwargs: dict[str, Any] = {}
        peft_sig = inspect.signature(PeftModel.from_pretrained).parameters
        if "is_trainable" in peft_sig:
            peft_load_kwargs["is_trainable"] = True
        model = PeftModel.from_pretrained(model, str(args.init_adapter_dir), **peft_load_kwargs)

        # Compatibility fallback for older PEFT versions without is_trainable support.
        if "is_trainable" not in peft_sig:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        peft_cfg = model.peft_config.get("default")
        if peft_cfg and getattr(peft_cfg, "target_modules", None):
            target_modules = sorted(list(peft_cfg.target_modules))
        else:
            target_modules = []
        print(f"Loaded init adapter: {args.init_adapter_dir}")
    else:
        target_candidates = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        target_modules = detect_available_target_modules(model, target_candidates)
        if not target_modules:
            raise RuntimeError(
                "No matching LoRA target modules found in model. "
                f"Candidates were: {target_candidates}"
            )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    files = {"train": str(args.train_file), "validation": str(args.val_file)}
    raw_ds = load_dataset("json", data_files=files)

    if args.max_train_samples > 0:
        train_n = min(args.max_train_samples, len(raw_ds["train"]))
        raw_ds["train"] = raw_ds["train"].select(range(train_n))
    if args.max_val_samples > 0:
        val_n = min(args.max_val_samples, len(raw_ds["validation"]))
        raw_ds["validation"] = raw_ds["validation"].select(range(val_n))

    response_style = args.response_style
    if response_style == "auto":
        response_style = detect_response_style(raw_ds["train"])
    print(f"response_style={response_style}")

    def estimate_truncation(split: Any, split_name: str) -> None:
        rows = len(split)
        if rows == 0:
            print(f"truncation_stats {split_name}=0")
            return
        truncated = 0
        prompt_lens: list[int] = []
        output_lens: list[int] = []
        for row in split:
            prompt = build_prompt(
                row.get("instruction", ""),
                row.get("input", ""),
                response_style,
                args.prompt_format_hint,
            )
            output = (row.get("output", "") or "").strip()
            if output:
                output = output + tokenizer.eos_token
            else:
                output = tokenizer.eos_token
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            output_len = len(tokenizer(output, add_special_tokens=False)["input_ids"])
            prompt_lens.append(prompt_len)
            output_lens.append(output_len)
            if output_len >= args.max_length or prompt_len > max(0, args.max_length - output_len):
                truncated += 1
        avg_prompt = sum(prompt_lens) / rows
        avg_output = sum(output_lens) / rows
        pct = 100.0 * truncated / rows
        print(
            "truncation_stats "
            f"{split_name}_rows={rows} "
            f"{split_name}_avg_prompt_tokens={avg_prompt:.1f} "
            f"{split_name}_avg_output_tokens={avg_output:.1f} "
            f"{split_name}_truncated_rows={truncated} "
            f"{split_name}_truncated_pct={pct:.2f}"
        )

    estimate_truncation(raw_ds["train"], "train")
    estimate_truncation(raw_ds["validation"], "val")

    def map_fn(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        out_input_ids: list[list[int]] = []
        out_token_type_ids: list[list[int]] = []
        out_attention: list[list[int]] = []
        out_labels: list[list[int]] = []
        n = len(batch["instruction"])
        for i in range(n):
            row = {
                "instruction": batch["instruction"][i],
                "input": batch.get("input", [""] * n)[i],
                "output": batch["output"][i],
            }
            ex = preprocess_row(
                row,
                tokenizer,
                args.max_length,
                response_style,
                args.prompt_format_hint,
                args.prompt_truncation,
            )
            out_input_ids.append(ex["input_ids"])
            out_token_type_ids.append(ex["token_type_ids"])
            out_attention.append(ex["attention_mask"])
            out_labels.append(ex["labels"])
        return {
            "input_ids": out_input_ids,
            "token_type_ids": out_token_type_ids,
            "attention_mask": out_attention,
            "labels": out_labels,
        }

    tokenized = raw_ds.map(
        map_fn,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing",
    )

    print(
        f"dataset_sizes train={len(tokenized['train'])} "
        f"val={len(tokenized['validation'])}"
    )

    def supervised_stats(split: Any) -> tuple[int, int, float]:
        rows = len(split)
        if rows == 0:
            return 0, 0, 0.0
        rows_with_zero = 0
        total_supervised = 0
        for labels in split["labels"]:
            n = sum(1 for t in labels if t != -100)
            total_supervised += n
            if n == 0:
                rows_with_zero += 1
        return rows, rows_with_zero, total_supervised / rows

    tr_rows, tr_zero, tr_avg = supervised_stats(tokenized["train"])
    va_rows, va_zero, va_avg = supervised_stats(tokenized["validation"])
    print(
        "supervision_stats "
        f"train_zero_label_rows={tr_zero}/{tr_rows} train_avg_supervised_tokens={tr_avg:.1f} "
        f"val_zero_label_rows={va_zero}/{va_rows} val_avg_supervised_tokens={va_avg:.1f}"
    )

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": (dtype == torch.float16),
        "bf16": (dtype == torch.bfloat16),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "seed": args.seed,
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"
    else:
        raise RuntimeError(
            "Your transformers version does not expose evaluation strategy args. "
            "Please update transformers to a recent 4.x release."
        )

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=SupervisedDataCollator(tokenizer=tokenizer),
    )

    model.print_trainable_parameters()
    resume_ckpt = args.resume_from_checkpoint.strip() or None
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")
    print(f"LoRA target modules: {target_modules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
