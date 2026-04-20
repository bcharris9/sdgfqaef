# LTspice Dataset Pipeline (PowerShell Only)

Run these commands from Windows PowerShell.

## At A Glance

- `generate_variants.py`
  Core fault injection and variant generation.
- `run_ltspice_batch.py`
  Batch LTspice execution and measurement collection.
- `build_dataset.py`
  Merge manifests into a training dataset JSONL.
- `prepare_finetune_data.py`
  Convert raw dataset rows into train/validation prompt files.
- `train_lora.py`
  Fine-tune a model.
- `test_lora_model.py`
  Run offline evaluation and write reports.
- `run_measurement_infer.py`
  Manual inference from measured node values.
- `build_llm_union_trainset.py`
  Merge multiple merged instruct sets into one larger train/val set.
- `make_gapfix_oversample.py`
  Build a targeted oversampled dataset for weak fault classes.
- `iterate_unseen_retrain.py`
  Iterative evaluate-and-retrain loop for unseen chunk performance.
- `run_lab4_task2_part1_5_focus_*.ps1`
  Convenience wrappers for the current focused single-circuit workflow.

## What In This Folder Is Generated?

These are usually run artifacts rather than source code:

- `out/`
- `out_one_lab/`
- `out_one_lab_all*/`
- `tmp_*/`
- `*.log`

## 0) Open project folder

```powershell
cd C:\ft
```

## 1) Clean previous output (optional)

```powershell
Remove-Item -Recurse -Force .\pipeline\out -ErrorAction SilentlyContinue
```

## 2) Generate faulted variants (parallel)

```powershell
py -3 pipeline\generate_variants.py `
  --asc-dir LTSpice_files `
  --out-dir pipeline\out `
  --variants-per-circuit 300 `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --max-workers 22
```

## 3) Run LTspice simulations (parallel)

```powershell
py -3 pipeline\run_ltspice_batch.py `
  --variants-dir pipeline\out\variants `
  --results-dir pipeline\out\sim_results `
  --manifest pipeline\out\sim_manifest.jsonl `
  --ltspice-bin "C:\Users\bchar\AppData\Local\Programs\ADI\LTspice\LTspice.exe" `
  --max-workers 22 `
  --keep-raw
```

## 4) Build training dataset JSONL

```powershell
py -3 pipeline\build_dataset.py `
  --variant-manifest pipeline\out\variant_manifest.jsonl `
  --sim-manifest pipeline\out\sim_manifest.jsonl `
  --out pipeline\out\training_dataset.jsonl
```

## 5) Quick verification

```powershell
Get-Item .\pipeline\out\training_dataset.jsonl | Select-Object FullName,Length
(Get-Content .\pipeline\out\training_dataset.jsonl | Measure-Object -Line).Lines
Get-Content .\pipeline\out\training_dataset.jsonl -TotalCount 3
```

## One-Lab Focus + Handoff

For your current single-circuit workflow (`lab4_task2_part1_5`) and full handoff notes, see:

- `pipeline_one_lab/README.md` (section: `Codex Handoff (lab4_task2_part1_5 focus)`)

PowerShell helpers for that flow are in:

- `pipeline/run_lab4_task2_part1_5_focus_rebuild.ps1`
- `pipeline/run_lab4_task2_part1_5_focus_train.ps1`
- `pipeline/run_lab4_task2_part1_5_focus_eval.ps1`
- `pipeline/run_lab4_task2_part1_5_focus_infer.ps1`
- `pipeline/run_lab4_task2_part1_5_focus_all.ps1`
