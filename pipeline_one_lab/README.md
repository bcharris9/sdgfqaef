# One-Lab Pipeline (Beginner)

Use this when you want to test one lab task at a time.
This pipeline now supports a **golden circuit** reference by default.

## At A Glance

- `run_one_lab_pipeline.py`
  Best single entry point for one circuit end-to-end.
- `run_recursive_all_labs_pipeline.py`
  Runs the one-lab flow across every discovered `.asc` file.
- `run_golden_set_pipeline.py`
  Runs a chosen group of labs as one golden-set job.
- `build_golden_one_lab.py`
  Builds the golden reference circuit and measurements.
- `prepare_finetune_one_lab.py`
  Shapes examples into prompt/response training files.
- `merge_finetune_sets.py`
  Merges multiple one-lab splits into one combined fine-tune set.

## Recommended Reading Order

If you are trying to understand this folder, open files in this order:

1. `run_one_lab_pipeline.py`
2. `prepare_finetune_one_lab.py`
3. `generate_variants_one_lab.py`
4. `build_golden_one_lab.py`
5. `build_dataset_one_lab.py`
6. `run_recursive_all_labs_pipeline.py`

Default fault mix is tuned for your stated priorities:
- `param_drift`: 30%
- `resistor_wrong_value`: 15%
- `resistor_value_swap`: 20% (common mistake emphasis)
- `swapped_nodes`: 18%
- `missing_component`: 12%
- `pin_open`: 12%
- `short_between_nodes`: 8%

Voltage-source drift is clamped to `[-5, 5]` by default.

For higher classification accuracy on breadboard-style inputs, the current
defaults also:
- drop no-op faults (`old_value == new_value`),
- map resistor-based `param_drift` into `resistor_wrong_value`,
- prioritize voltage features and use `_max` stats in prompts.

## Full pipeline in one command

```powershell
py -3 pipeline_one_lab/run_one_lab_pipeline.py `
  --lab lab9_task2 `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --out-root pipeline/out_one_lab `
  --variants-per-circuit 300 `
  --max-workers 6 `
  --timeout-sec 240 `
  --keep-raw `
  --val-ratio 0.2 `
  --input-mode delta_plus_measured `
  --output-mode faulttype_diag_fix
```

Outputs will be in:

`pipeline/out_one_lab/lab9_task2/`

Golden files will be in:

`pipeline/out_one_lab/lab9_task2/golden/`

## Step-by-step commands

1) Generate variants

```powershell
py -3 pipeline_one_lab/generate_variants_one_lab.py `
  --lab lab9_task2 `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
```

2) Run LTspice sims

```powershell
py -3 pipeline_one_lab/run_ltspice_batch_one_lab.py `
  --lab lab9_task2 `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --timeout-sec 240 `
  --keep-raw
```

3) Build merged dataset

```powershell
py -3 pipeline_one_lab/build_dataset_one_lab.py --lab lab9_task2
```

4) Prepare fine-tune split

```powershell
py -3 pipeline_one_lab/prepare_finetune_one_lab.py `
  --lab lab9_task2 `
  --val-ratio 0.2 `
  --use-golden
```

Fine-tune files will be in:

`pipeline/out_one_lab/lab9_task2/finetune_small/`

If you want to disable golden comparison in prompts, run:

```powershell
py -3 pipeline_one_lab/run_one_lab_pipeline.py `
  --lab lab9_task2 `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --no-use-golden
```

## Run all golden circuits (example: all 10 .asc files)

```powershell
py -3 pipeline_one_lab/run_golden_set_pipeline.py `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --out-root pipeline/out_one_lab `
  --variants-per-circuit 600 `
  --max-workers 6 `
  --timeout-sec 240 `
  --keep-raw `
  --val-ratio 0.2 `
  --use-golden
```

For your current set in `LTSpice_files/Lab4` with names starting `lab4_task2_part1_`:

```powershell
py -3 pipeline_one_lab/run_golden_set_pipeline.py `
  --asc-dir LTSpice_files/Lab4 `
  --lab-prefix "lab4_task2_part1_" `
  --expect-count 10 `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --out-root pipeline/out_one_lab/lab4_task2_part1_set `
  --variants-per-circuit 1000 `
  --max-workers 6 `
  --timeout-sec 240 `
  --keep-raw `
  --val-ratio 0.2 `
  --use-golden `
  --input-mode delta_plus_measured `
  --output-mode faulttype_diag_fix `
  --vsource-min -5 `
  --vsource-max 5 `
  --weight-resistor-value-swap 0.20 `
  --weight-resistor-wrong-value 0.15 `
  --measurement-noise-sigma 0.01 `
  --measurement-noise-prob 0.5
```

If you only want a specific list of labs:

```powershell
py -3 pipeline_one_lab/run_golden_set_pipeline.py `
  --labs "lab4_task2_part1_a,lab4_task2_part1_b" `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
```

## Merge all one-lab splits into one training set

```powershell
py -3 pipeline_one_lab/merge_finetune_sets.py `
  --out-root pipeline/out_one_lab `
  --dest-dir pipeline/out_one_lab/merged_finetune `
  --shuffle
```

Then train your model using:

`pipeline/out_one_lab/merged_finetune/train_instruct.jsonl`

and

`pipeline/out_one_lab/merged_finetune/val_instruct.jsonl`

## Codex Handoff (lab4_task2_part1_5 focus)

Use this section to resume quickly in a future session.

### Current best-known run snapshot

- Focus lab: `lab4_task2_part1_5`
- Focus output root: `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/`
- Current best adapter: `pipeline/out/qwen15b_lab4_task2_part1_5_lora_v12_len384`
- Current best eval report: `pipeline/out/qwen15b_lab4_task2_part1_5_eval_full_v12_len384_defaultfix_report.json`
- Report summary:
  - `samples=201`
  - `class_match=91.04% (183/201)`
  - main weakness: `resistor_wrong_value` recall (`43/61 = 70.49%`)

### Most important files and why they matter

- `pipeline_one_lab/run_one_lab_pipeline.py`
  - One-lab orchestrator: variants -> LTspice batch -> dataset -> golden -> finetune split.
- `pipeline_one_lab/generate_variants_one_lab.py`
  - One-lab wrapper for fault generation; controls fault mix and V source drift bounds.
- `pipeline/generate_variants.py`
  - Core fault injection engine (if adding new fault families, edit here).
- `pipeline/run_ltspice_batch.py`
  - Batch LTspice runner used by one-lab and multi-lab flows.
- `pipeline_one_lab/build_golden_one_lab.py`
  - Builds the golden `.cir`, injects `.op/.tran/.meas`, and writes `golden_measurements.json`.
- `pipeline_one_lab/build_dataset_one_lab.py`
  - Converts variant + sim manifests into `training_dataset.jsonl`.
- `pipeline_one_lab/prepare_finetune_one_lab.py`
  - Most important data-shaping stage:
    - input format (`full`, `delta_only`, `delta_plus_measured`)
    - output format (`diag_fix`, `faulttype_diag_fix`)
    - ambiguity handling, class balancing, canonical output labels
    - optional measurement noise injection
- `pipeline/train_lora.py`
  - LoRA trainer. Supports response style detection/forcing and prompt-format hints.
- `pipeline/test_lora_model.py`
  - Offline eval on instruct JSONL. Writes predictions JSONL and class-level report JSON.
- `pipeline/run_measurement_infer.py`
  - Manual-input inference from measured node values, using template + golden deltas.

### Most important generated artifacts

- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/variant_manifest.jsonl`
  - Fault truth per generated variant.
- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/sim_manifest.jsonl`
  - LTspice sim status + parsed measurements per variant.
- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/training_dataset.jsonl`
  - Consolidated supervised rows before finetune formatting.
- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/train_instruct.jsonl`
  - Training file consumed by `pipeline/train_lora.py`.
- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/val_instruct.jsonl`
  - Validation file for both in-training eval and offline eval.
- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/finetune_small/split_meta.json`
  - Ground truth for split configuration and class balancing.
- `pipeline/out_one_lab/lab4_task2_part1_5_focus/lab4_task2_part1_5/golden/golden_measurements.json`
  - Golden reference values used for delta features and manual inference.

### Important pitfalls already discovered

- Decoding settings matter a lot:
  - Keep `repetition_penalty=1.0` and `no_repeat_ngram_size=0` for this task.
  - Aggressive anti-repeat settings previously caused major collapse.
- Keep eval/infer tokenization aligned with training:
  - `add_special_tokens=False` is already enforced in eval/infer scripts.
- If class collapse happens, first inspect:
  - `split_meta.json` for balance/ambiguity config,
  - eval report class counts and class recall,
  - whether prompt style and response style match (`faulttype_diag_fix`).

### PowerShell run scripts (added for handoff)

These scripts are in `pipeline/`:

- `pipeline/run_lab4_task2_part1_5_focus_rebuild.ps1`
  - Rebuilds data pipeline for this single circuit and re-runs strict finetune split shaping.
- `pipeline/run_lab4_task2_part1_5_focus_train.ps1`
  - Trains a new LoRA adapter (Qwen2.5-1.5B default).
- `pipeline/run_lab4_task2_part1_5_focus_eval.ps1`
  - Runs offline eval and writes prediction + report files.
- `pipeline/run_lab4_task2_part1_5_focus_infer.ps1`
  - Interactive/manual measurement inference against trained adapter.
- `pipeline/run_lab4_task2_part1_5_focus_all.ps1`
  - Rebuild -> Train -> Eval chain in one command.

Run them with:

```powershell
powershell -ExecutionPolicy Bypass -File .\pipeline\run_lab4_task2_part1_5_focus_rebuild.ps1
powershell -ExecutionPolicy Bypass -File .\pipeline\run_lab4_task2_part1_5_focus_train.ps1 -RunTag v13
powershell -ExecutionPolicy Bypass -File .\pipeline\run_lab4_task2_part1_5_focus_eval.ps1 -RunTag v13
powershell -ExecutionPolicy Bypass -File .\pipeline\run_lab4_task2_part1_5_focus_infer.ps1 -RunTag v13
```

Or all-in-one:

```powershell
powershell -ExecutionPolicy Bypass -File .\pipeline\run_lab4_task2_part1_5_focus_all.ps1 -RunTag v13
```
