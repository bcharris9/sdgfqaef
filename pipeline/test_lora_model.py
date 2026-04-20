#!/usr/bin/env python3
"""Run LoRA adapter inference and evaluation on instruct JSONL data."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on instruct JSONL")
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-4b-it",
        help="Base model name used for LoRA training",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("pipeline/out/lora_small"),
        help="Directory containing trained LoRA adapter",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("pipeline/out/finetune_small/val_instruct.jsonl"),
        help="Instruct-format JSONL with instruction/input/output fields",
    )
    parser.add_argument("--out-file", type=Path, default=Path("pipeline/out/lora_small_eval.jsonl"))
    parser.add_argument("--max-samples", type=int, default=200)
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
        help="Force prediction into: 'Diagnosis: ... Fix: ...' format during eval",
    )
    parser.add_argument(
        "--response-style",
        choices=["auto", "diag_fix", "faulttype_diag_fix"],
        default="auto",
        help="Expected output style. auto detects from target rows.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=[
            "generate",
            "score_classes",
            "score_faulttype",
            "score_classes_knn",
            "score_faulttype_knn",
            "knn_only",
        ],
        default="generate",
        help=(
            "generate: normal text generation. "
            "score_classes: score canonical outputs for all fault classes and pick best. "
            "score_faulttype: score only the FaultType line, then emit canonical diagnosis/fix. "
            "score_classes_knn: score classes and add KNN class priors from reference measurements. "
            "score_faulttype_knn: score only FaultType lines and add KNN class priors. "
            "knn_only: predict class from KNN on measurement features only, then emit canonical diagnosis/fix."
        ),
    )
    parser.add_argument(
        "--knn-ref-file",
        type=Path,
        default=Path(""),
        help="Optional instruct JSONL used as KNN reference set (used by score_classes_knn mode).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=3,
        help="Number of nearest neighbors for KNN priors in score_classes_knn mode.",
    )
    parser.add_argument(
        "--knn-alpha",
        type=float,
        default=1.0,
        help="Weight for KNN prior penalty when using score_classes_knn mode.",
    )
    parser.add_argument(
        "--knn-weighted-vote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-distance weighted KNN voting.",
    )
    parser.add_argument(
        "--knn-standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply z-score standardization before computing KNN distances.",
    )
    parser.add_argument(
        "--knn-eps",
        type=float,
        default=1e-9,
        help="Small epsilon to stabilize inverse-distance weighted KNN voting.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path(""),
        help="Optional JSON file to write class-level metrics summary.",
    )
    parser.add_argument(
        "--use-prerules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply deterministic high-confidence rules before model/KNN decoding.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
            "Choose the single best fault family: parameter drift, missing component, open connection, "
            "swapped terminals, short between nodes, resistor values swapped, wrong resistor value.\n"
        )

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            f"{response_header}"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        f"{response_header}"
    )


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def has_diag_and_fix(text: str) -> bool:
    low = (text or "").lower()
    return ("diagnosis:" in low) and ("fix:" in low)


def normalize_fault_type_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    alias = {
        "param_drift": "param_drift",
        "parameter_drift": "param_drift",
        "parameter drift": "param_drift",
        "param drift": "param_drift",
        "missing_component": "missing_component",
        "missing component": "missing_component",
        "pin_open": "pin_open",
        "open_connection": "pin_open",
        "open connection": "pin_open",
        "swapped_nodes": "swapped_nodes",
        "swapped_terminals": "swapped_nodes",
        "swapped terminals": "swapped_nodes",
        "nodes swapped": "swapped_nodes",
        "short_between_nodes": "short_between_nodes",
        "short between nodes": "short_between_nodes",
        "resistor_value_swap": "resistor_value_swap",
        "resistor value swap": "resistor_value_swap",
        "resistor values swapped": "resistor_value_swap",
        "resistor_wrong_value": "resistor_wrong_value",
        "wrong resistor value": "resistor_wrong_value",
    }
    return alias.get(key, "unknown")


def force_diag_fix_format(text: str, response_style: str) -> str:
    raw = (text or "").strip()
    if not raw:
        if response_style == "faulttype_diag_fix":
            return (
                "FaultType: unknown\n"
                "Diagnosis: unknown fault. Fix: inspect wiring and component values."
            )
        return "Diagnosis: unknown fault. Fix: inspect wiring and component values."

    # Normalize whitespace for robust extraction.
    clean = re.sub(r"\s+", " ", raw).strip()

    # Optional explicit fault type line.
    fault_type = "unknown"
    m = re.search(r"faulttype:\s*([a-z_ \-]+)", clean, flags=re.IGNORECASE)
    if m:
        fault_type = normalize_fault_type_label(m.group(1))

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
        diagnosis = diagnosis or "unknown fault"
        fix = fix or "inspect wiring and component values"
        if response_style == "faulttype_diag_fix":
            if fault_type == "unknown":
                fault_type = classify_fault_text(f"Diagnosis: {diagnosis}. Fix: {fix}.")
            return f"FaultType: {fault_type}\nDiagnosis: {diagnosis}. Fix: {fix}."
        return f"Diagnosis: {diagnosis}. Fix: {fix}."

    # Class-only outputs: promote to canonical diagnosis/fix for stable scoring.
    if response_style == "faulttype_diag_fix" and fault_type != "unknown":
        return f"FaultType: {fault_type}\n{canonical_completion_for_fault(fault_type)}"

    # Fallback: use first sentence as diagnosis and second as fix when possible.
    sentences = [s.strip(" .;") for s in re.split(r"[.!?]+", clean) if s.strip()]
    diagnosis = sentences[0] if sentences else "unknown fault"
    fix = sentences[1] if len(sentences) > 1 else "inspect wiring and component values"
    if response_style == "faulttype_diag_fix":
        if fault_type == "unknown":
            fault_type = classify_fault_text(f"Diagnosis: {diagnosis}. Fix: {fix}.")
        return f"FaultType: {fault_type}\nDiagnosis: {diagnosis}. Fix: {fix}."
    return f"Diagnosis: {diagnosis}. Fix: {fix}."


def classify_fault_text(text: str) -> str:
    m = re.search(r"faulttype:\s*([a-z_ \-]+)", (text or ""), flags=re.IGNORECASE)
    if m:
        label = normalize_fault_type_label(m.group(1))
        if label != "unknown":
            return label

    t = (text or "").lower()
    if "wrong resistor value" in t:
        return "resistor_wrong_value"
    if "resistor values were swapped" in t or ("swapped" in t and "resistor" in t):
        return "resistor_value_swap"
    if "parameter drift" in t or "parameter change" in t:
        return "param_drift"
    if "missing component" in t:
        return "missing_component"
    if "open connection" in t:
        return "pin_open"
    if "swapped terminals" in t or "nodes were swapped" in t:
        return "swapped_nodes"
    if "short between nodes" in t or "unintended short" in t:
        return "short_between_nodes"
    return "unknown"


FAULT_TYPE_ORDER = [
    "param_drift",
    "missing_component",
    "pin_open",
    "swapped_nodes",
    "short_between_nodes",
    "resistor_value_swap",
    "resistor_wrong_value",
]


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
        "Diagnosis: unknown fault. Fix: inspect wiring and component values.",
    )


def prerule_fault_type(input_text: str) -> str | None:
    text = (input_text or "").strip()
    if not text:
        return None

    # This signature is unambiguous in the current data: no parsed measurements.
    # It consistently maps to a missing-component class.
    has_measured_none = False
    has_delta_none = False
    sim_success_false = False
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low == "measured: none":
            has_measured_none = True
        elif low == "deltasvsgolden: none":
            has_delta_none = True
        elif low == "simsuccess: false":
            sim_success_false = True

    if has_measured_none and (has_delta_none or sim_success_false):
        return "missing_component"
    return None


def parse_measurement_features(input_text: str) -> dict[str, float]:
    text = (input_text or "").strip()
    if not text:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        low_line = stripped.lower()
        if line.startswith("Lab:"):
            lab = line.split(":", 1)[1].strip().lower()
            if lab:
                # Preserve sign information (e.g. ..._-4 vs ..._4) to avoid collisions.
                lab = lab.replace("-", "__neg__")
                safe_lab = re.sub(r"[^a-z0-9_]+", "_", lab).strip("_")
                if safe_lab:
                    out[f"lab__{safe_lab}"] = 1.0
            continue
        if low_line.startswith("simsuccess:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in {"true", "1", "yes"}:
                out["sim_success"] = 1.0
            elif val in {"false", "0", "no"}:
                out["sim_success"] = 0.0
            continue
        if not (
            line.startswith("Measured:")
            or line.startswith("DeltasVsGolden:")
            or line.startswith("GoldenMeasurements:")
        ):
            continue
        prefix, payload = line.split(":", 1)
        prefix_low = prefix.strip().lower()
        payload = payload.strip()
        if payload.lower() == "none":
            if prefix_low == "measured":
                out["measured_none"] = 1.0
                out["measured_count"] = 0.0
            elif prefix_low == "deltasvsgolden":
                out["deltas_none"] = 1.0
                out["deltas_count"] = 0.0
            elif prefix_low == "goldenmeasurements":
                out["golden_count"] = 0.0
            continue
        numeric_count = 0
        for item in payload.split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            key, raw_val = item.split("=", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            try:
                out[key] = float(raw_val)
                numeric_count += 1
            except Exception:
                continue
        if prefix_low == "measured":
            out["measured_none"] = 0.0
            out["measured_count"] = float(numeric_count)
        elif prefix_low == "deltasvsgolden":
            out["deltas_none"] = 0.0
            out["deltas_count"] = float(numeric_count)
        elif prefix_low == "goldenmeasurements":
            out["golden_count"] = float(numeric_count)
    return out


def parse_lab_id(input_text: str) -> str | None:
    text = (input_text or "").strip()
    if not text:
        return None
    for line in text.splitlines():
        if not line.startswith("Lab:"):
            continue
        lab = line.split(":", 1)[1].strip().lower()
        if not lab:
            return None
        # Preserve sign information (e.g. ..._-4 vs ..._4) to avoid collisions.
        lab = lab.replace("-", "__neg__")
        safe_lab = re.sub(r"[^a-z0-9_]+", "_", lab).strip("_")
        return safe_lab or None
    return None


def build_knn_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feats: list[dict[str, float]] = []
    classes: list[str] = []
    labs: list[str | None] = []
    key_set: set[str] = set()
    for row in rows:
        input_text = str(row.get("input", ""))
        f = parse_measurement_features(input_text)
        if not f:
            continue
        c = classify_fault_text(str(row.get("output", "")))
        if c == "unknown":
            continue
        feats.append(f)
        classes.append(c)
        labs.append(parse_lab_id(input_text))
        key_set.update(f.keys())
    keys = sorted(key_set)
    vectors: list[list[float]] = []
    for f in feats:
        vectors.append([f.get(k, 0.0) for k in keys])
    means: list[float] = []
    stds: list[float] = []
    if vectors and keys:
        n = float(len(vectors))
        for i in range(len(keys)):
            m = 0.0
            for row in vectors:
                m += row[i]
            m /= n
            means.append(m)
        for i in range(len(keys)):
            var = 0.0
            for row in vectors:
                d = row[i] - means[i]
                var += d * d
            if len(vectors) > 1:
                var /= float(len(vectors) - 1)
            s = math.sqrt(var)
            if s < 1e-12:
                s = 1.0
            stds.append(s)
    zvectors: list[list[float]] = []
    if vectors:
        for row in vectors:
            zrow: list[float] = []
            for i, val in enumerate(row):
                zrow.append((val - means[i]) / stds[i])
            zvectors.append(zrow)

    lab_to_indices: dict[str, list[int]] = {}
    for i, lab in enumerate(labs):
        if not lab:
            continue
        lab_to_indices.setdefault(lab, []).append(i)

    return {
        "keys": keys,
        "vectors": vectors,
        "zvectors": zvectors,
        "classes": classes,
        "labs": labs,
        "lab_to_indices": lab_to_indices,
        "means": means,
        "stds": stds,
    }


def knn_class_probs(
    index: dict[str, Any],
    input_text: str,
    k: int,
    weighted_vote: bool,
    standardize: bool,
    eps: float,
) -> dict[str, float]:
    keys: list[str] = index.get("keys", [])
    vectors: list[list[float]] = index.get("vectors", [])
    zvectors: list[list[float]] = index.get("zvectors", [])
    classes: list[str] = index.get("classes", [])
    means: list[float] = index.get("means", [])
    stds: list[float] = index.get("stds", [])
    lab_to_indices: dict[str, list[int]] = index.get("lab_to_indices", {})
    if not keys or not vectors or not classes:
        p = 1.0 / float(max(1, len(FAULT_TYPE_ORDER)))
        return {c: p for c in FAULT_TYPE_ORDER}

    f = parse_measurement_features(input_text)
    q_lab = parse_lab_id(input_text)
    q = [f.get(key, 0.0) for key in keys]
    if standardize and means and stds and len(means) == len(q):
        q = [(q[i] - means[i]) / stds[i] for i in range(len(q))]
        ref_vectors = zvectors if zvectors else vectors
    else:
        ref_vectors = vectors

    candidate_indices: list[int] | None = None
    if q_lab and lab_to_indices:
        same_lab = lab_to_indices.get(q_lab)
        if same_lab:
            candidate_indices = same_lab

    if candidate_indices is None:
        candidate_indices = list(range(len(ref_vectors)))

    kk = max(1, min(int(k), len(candidate_indices)))
    all_d: list[tuple[float, int]] = []
    for idx in candidate_indices:
        ref = ref_vectors[idx]
        s = 0.0
        for i, rv in enumerate(ref):
            d = q[i] - rv
            s += d * d
        all_d.append((s, idx))
    all_d.sort(key=lambda x: x[0])
    top = all_d[:kk]

    votes: dict[str, float] = {}
    total_vote = 0.0
    for dist2, idx in top:
        c = classes[idx]
        if c == "unknown":
            continue
        if weighted_vote:
            w = 1.0 / (dist2 + max(1e-12, float(eps)))
        else:
            w = 1.0
        votes[c] = votes.get(c, 0.0) + w
        total_vote += w

    if total_vote <= 0.0:
        p = 1.0 / float(max(1, len(FAULT_TYPE_ORDER)))
        return {c: p for c in FAULT_TYPE_ORDER}

    probs: dict[str, float] = {}
    for c in FAULT_TYPE_ORDER:
        probs[c] = votes.get(c, 0.0) / total_vote
    return probs


def knn_penalties(probs: dict[str, float], alpha: float) -> dict[str, float]:
    out: dict[str, float] = {}
    a = max(0.0, float(alpha))
    for c in FAULT_TYPE_ORDER:
        p = max(1e-8, float(probs.get(c, 0.0)))
        out[c] = a * (-math.log(p))
    return out


def build_class_candidates(response_style: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for fault_type in FAULT_TYPE_ORDER:
        body = canonical_completion_for_fault(fault_type)
        if response_style == "faulttype_diag_fix":
            text = f"FaultType: {fault_type}\n{body}"
        else:
            text = body
        items.append((fault_type, text))
    return items


def build_faulttype_only_candidates(response_style: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for fault_type in FAULT_TYPE_ORDER:
        if response_style == "faulttype_diag_fix":
            text = f"FaultType: {fault_type}\n"
        else:
            text = f"{fault_type}\n"
        items.append((fault_type, text))
    return items


def score_output_candidate(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_ids: list[int],
    prompt_tti: list[int] | None,
    candidate_text: str,
    needs_token_type_ids: bool,
) -> float:
    suffix = tokenizer.eos_token or ""
    candidate = (candidate_text or "").strip() + suffix
    cand_tok = tokenizer(candidate, add_special_tokens=False)
    cand_ids = cand_tok.get("input_ids", [])
    if not cand_ids:
        cand_ids = [tokenizer.eos_token_id]
    input_ids = prompt_ids + cand_ids
    labels = ([-100] * len(prompt_ids)) + cand_ids

    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }
    if needs_token_type_ids:
        pt = prompt_tti if prompt_tti is not None else [0] * len(prompt_ids)
        ct = cand_tok.get("token_type_ids", [0] * len(cand_ids))
        token_type_ids = pt + ct
        batch["token_type_ids"] = torch.tensor([token_type_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        loss = model(**batch).loss
    return float(loss.item())


def predict_by_class_scoring(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    response_style: str,
    class_penalties: dict[str, float] | None = None,
) -> str:
    prompt_tok = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_tok.get("input_ids", [])
    prompt_tti = prompt_tok.get("token_type_ids")
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    needs_token_type_ids = "token_type_ids" in forward_params

    best_text = ""
    best_score = float("inf")
    for fault_type, candidate_text in build_class_candidates(response_style):
        s = score_output_candidate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_ids=prompt_ids,
            prompt_tti=prompt_tti,
            candidate_text=candidate_text,
            needs_token_type_ids=needs_token_type_ids,
        )
        if class_penalties:
            s += float(class_penalties.get(fault_type, 0.0))
        if s < best_score:
            best_score = s
            best_text = candidate_text
    return best_text


def predict_by_faulttype_scoring(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    response_style: str,
    class_penalties: dict[str, float] | None = None,
) -> str:
    prompt_tok = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_tok.get("input_ids", [])
    prompt_tti = prompt_tok.get("token_type_ids")
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    needs_token_type_ids = "token_type_ids" in forward_params

    best_fault_type = "unknown"
    best_score = float("inf")
    for fault_type, candidate_text in build_faulttype_only_candidates(response_style):
        s = score_output_candidate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_ids=prompt_ids,
            prompt_tti=prompt_tti,
            candidate_text=candidate_text,
            needs_token_type_ids=needs_token_type_ids,
        )
        if class_penalties:
            s += float(class_penalties.get(fault_type, 0.0))
        if s < best_score:
            best_score = s
            best_fault_type = fault_type

    body = canonical_completion_for_fault(best_fault_type)
    if response_style == "faulttype_diag_fix":
        return f"FaultType: {best_fault_type}\n{body}"
    return body


def predict_by_knn_only(
    input_text: str,
    response_style: str,
    knn_index: dict[str, Any],
    k: int,
    weighted_vote: bool,
    standardize: bool,
    eps: float,
) -> str:
    probs = knn_class_probs(
        index=knn_index,
        input_text=input_text,
        k=k,
        weighted_vote=weighted_vote,
        standardize=standardize,
        eps=eps,
    )
    best_fault_type = max(FAULT_TYPE_ORDER, key=lambda c: probs.get(c, 0.0))
    body = canonical_completion_for_fault(best_fault_type)
    if response_style == "faulttype_diag_fix":
        return f"FaultType: {best_fault_type}\n{body}"
    return body


def load_model(model_name: str, adapter_dir: Path, device: str, dtype: torch.dtype):
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
    # Match train-time tokenization path (no automatic BOS/EOS injection).
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    forward_params = set(inspect.signature(model.forward).parameters.keys())
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
    gen_ids = output_ids[prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    device, dtype = choose_device()
    print(f"device={device} dtype={dtype}")

    needs_model = args.decode_mode in {
        "generate",
        "score_classes",
        "score_faulttype",
        "score_classes_knn",
        "score_faulttype_knn",
    }
    tokenizer: Any = None
    model: Any = None
    if needs_model:
        if not args.adapter_dir.exists():
            raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(args.model_name, args.adapter_dir, device, dtype)
        print(f"model_device={next(model.parameters()).device}")

    rows = load_jsonl(args.data_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    total = len(rows)
    if total == 0:
        raise RuntimeError("No samples to evaluate.")
    response_style = args.response_style
    if response_style == "auto":
        first_target = (rows[0].get("output", "") or "").strip().lower()
        if first_target.startswith("faulttype:"):
            response_style = "faulttype_diag_fix"
        else:
            response_style = "diag_fix"

    knn_index: dict[str, Any] | None = None
    if args.decode_mode in {"score_classes_knn", "score_faulttype_knn", "knn_only"}:
        if not str(args.knn_ref_file).strip():
            raise ValueError("--knn-ref-file is required for decode-mode=score_classes_knn/score_faulttype_knn/knn_only")
        if not args.knn_ref_file.exists():
            raise FileNotFoundError(f"KNN reference file not found: {args.knn_ref_file}")
        ref_rows = load_jsonl(args.knn_ref_file)
        knn_index = build_knn_index(ref_rows)
        print(
            "knn_index "
            f"rows={len(ref_rows)} "
            f"vectors={len(knn_index.get('vectors', []))} "
            f"features={len(knn_index.get('keys', []))} "
            f"k={max(1, args.knn_k)} alpha={args.knn_alpha} "
            f"weighted={args.knn_weighted_vote} standardized={args.knn_standardize}"
        )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    exact_match = 0
    format_ok = 0
    class_match = 0
    prerule_hits = 0
    pred_class_counts: dict[str, int] = {}
    target_class_counts: dict[str, int] = {}
    match_counts_by_target: dict[str, int] = {}

    with args.out_file.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt = build_prompt(row.get("instruction", ""), row.get("input", ""), response_style)
            target = (row.get("output", "") or "").strip()
            input_text = str(row.get("input", ""))
            pre_class = prerule_fault_type(input_text) if args.use_prerules else None
            if pre_class:
                prerule_hits += 1
                body = canonical_completion_for_fault(pre_class)
                if response_style == "faulttype_diag_fix":
                    pred = f"FaultType: {pre_class}\n{body}"
                else:
                    pred = body
            elif args.decode_mode == "score_classes":
                pred = predict_by_class_scoring(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    response_style=response_style,
                )
            elif args.decode_mode == "score_classes_knn":
                probs = knn_class_probs(
                    knn_index or {},
                    input_text,
                    args.knn_k,
                    args.knn_weighted_vote,
                    args.knn_standardize,
                    args.knn_eps,
                )
                penalties = knn_penalties(probs, args.knn_alpha)
                pred = predict_by_class_scoring(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    response_style=response_style,
                    class_penalties=penalties,
                )
            elif args.decode_mode == "score_faulttype_knn":
                probs = knn_class_probs(
                    knn_index or {},
                    input_text,
                    args.knn_k,
                    args.knn_weighted_vote,
                    args.knn_standardize,
                    args.knn_eps,
                )
                penalties = knn_penalties(probs, args.knn_alpha)
                pred = predict_by_faulttype_scoring(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    response_style=response_style,
                    class_penalties=penalties,
                )
            elif args.decode_mode == "knn_only":
                pred = predict_by_knn_only(
                    input_text=input_text,
                    response_style=response_style,
                    knn_index=knn_index or {},
                    k=args.knn_k,
                    weighted_vote=args.knn_weighted_vote,
                    standardize=args.knn_standardize,
                    eps=args.knn_eps,
                )
            elif args.decode_mode == "score_faulttype":
                pred = predict_by_faulttype_scoring(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    response_style=response_style,
                )
            else:
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
            if args.enforce_format and args.decode_mode == "generate":
                pred = force_diag_fix_format(pred, response_style)

            pred_class = classify_fault_text(pred)
            target_class = classify_fault_text(target)
            pred_class_counts[pred_class] = pred_class_counts.get(pred_class, 0) + 1
            target_class_counts[target_class] = target_class_counts.get(target_class, 0) + 1
            is_class_match = pred_class == target_class
            if is_class_match:
                class_match += 1
                match_counts_by_target[target_class] = match_counts_by_target.get(target_class, 0) + 1

            if normalize_text(pred) == normalize_text(target):
                exact_match += 1
            if has_diag_and_fix(pred):
                format_ok += 1

            rec = {
                "index": idx - 1,
                "prediction": pred,
                "target": target,
                "exact_match": normalize_text(pred) == normalize_text(target),
                "format_ok": has_diag_and_fix(pred),
                "pred_class": pred_class,
                "target_class": target_class,
                "class_match": is_class_match,
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

            if idx <= 3:
                print(f"[sample {idx}]")
                print(f"pred:   {pred}")
                print(f"target: {target}")

    em_pct = 100.0 * exact_match / total
    fmt_pct = 100.0 * format_ok / total
    class_pct = 100.0 * class_match / total
    print(f"samples={total}")
    print(f"exact_match={exact_match}/{total} ({em_pct:.2f}%)")
    print(f"format_ok={format_ok}/{total} ({fmt_pct:.2f}%)")
    print(f"class_match={class_match}/{total} ({class_pct:.2f}%)")
    print(f"prerule_hits={prerule_hits}")
    print(f"pred_class_counts={json.dumps(pred_class_counts, sort_keys=True)}")
    print(f"target_class_counts={json.dumps(target_class_counts, sort_keys=True)}")
    print(f"wrote_predictions={args.out_file}")

    if str(args.report_file).strip():
        report = {
            "samples": total,
            "response_style": response_style,
            "exact_match": {"count": exact_match, "pct": em_pct},
            "format_ok": {"count": format_ok, "pct": fmt_pct},
            "class_match": {"count": class_match, "pct": class_pct},
            "prerule_hits": prerule_hits,
            "pred_class_counts": pred_class_counts,
            "target_class_counts": target_class_counts,
            "class_recall_by_target": {
                k: {
                    "matched": match_counts_by_target.get(k, 0),
                    "total": target_class_counts.get(k, 0),
                    "pct": (
                        100.0 * match_counts_by_target.get(k, 0) / target_class_counts.get(k, 1)
                    ),
                }
                for k in sorted(target_class_counts.keys())
            },
        }
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote_report={args.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
