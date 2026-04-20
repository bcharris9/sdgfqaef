# Project Map

This file is the quick orientation guide for the tracked repo.

## What Is Actually Source Code?

Most of the real source lives in:

- `gpt/Capstone/`
- `circuit_debug_api/`
- `pipeline/`
- `pipeline_one_lab/`
- `LTSpice_files/`

Most of the large or noisy directories in the workspace are generated outputs,
local experiments, or archived snapshots.

## Top-Level Roles

- `gpt/Capstone/`
  Retrieval-augmented lab-manual assistant and validation scripts.
- `circuit_debug_api/`
  Final packaged API for chat plus circuit-debug inference.
- `LTSpice_files/`
  Input circuits. These are the `.asc` files that feed every downstream step.
- `pipeline/`
  Low-level data and model scripts plus a few convenience PowerShell wrappers.
- `pipeline_one_lab/`
  Higher-level orchestration layer for one-circuit and recursive workflows.

## Core Flow

The normal data path is:

1. `pipeline/generate_variants.py`
   Export base netlists and inject synthetic faults.
2. `pipeline/run_ltspice_batch.py`
   Run LTspice across generated variants and collect measurements.
3. `pipeline/build_dataset.py`
   Merge variant and simulation manifests into supervised rows.
4. `pipeline/prepare_finetune_data.py`
   Convert dataset rows into train/validation prompt-response JSONL.
5. `pipeline/train_lora.py`
   Fine-tune a model.
6. `pipeline/test_lora_model.py`
   Run offline evaluation and write reports.
7. `pipeline/run_measurement_infer.py`
   Do manual inference from measured node values.

## Key Files By Area

### In `pipeline/`

- `generate_variants.py`
  Core fault injection engine. Start here if you need new fault families or
  want to change how circuit corruption works.
- `run_ltspice_batch.py`
  Batch LTspice runner used by multiple workflows.
- `build_dataset.py`
  Builds the base training dataset from manifests.
- `prepare_finetune_data.py`
  Shapes examples into the prompt/response format used for training.
- `train_lora.py`
  LoRA training entry point.
- `test_lora_model.py`
  Offline evaluation and report generation.
- `run_measurement_infer.py`
  Manual, measurement-driven inference path.
- `build_llm_union_trainset.py`
  Builds a larger instruct-format train/val set by merging multiple merged
  finetune sets, with optional boosting and jitter.
- `make_gapfix_oversample.py`
  Creates a class-oversampled instruct JSONL for targeted gap-fixing runs.
- `iterate_unseen_retrain.py`
  Runs iterative evaluate-then-warm-start retraining against unseen eval chunks.
- `run_lab4_task2_part1_5_focus_*.ps1`
  Convenience wrappers for the current focused single-circuit workflow.

### In `pipeline_one_lab/`

- `run_one_lab_pipeline.py`
  Best single entry point for one circuit end-to-end.
- `run_recursive_all_labs_pipeline.py`
  Best entry point when rebuilding across all discovered `.asc` files.
- `run_golden_set_pipeline.py`
  Middle ground for grouped golden-set runs.
- `generate_variants_one_lab.py`
  One-lab wrapper around the core fault generator.
- `run_ltspice_batch_one_lab.py`
  One-lab simulation runner.
- `build_golden_one_lab.py`
  Builds the golden reference circuit and measurement file.
- `build_dataset_one_lab.py`
  Converts one-lab manifests into dataset rows.
- `prepare_finetune_one_lab.py`
  Most important place for prompt shaping, balancing, ambiguity handling, and
  output formatting in the one-lab flow.
- `merge_finetune_sets.py`
  Merges per-lab train/val splits into one combined dataset.

### In `gpt/Capstone/`

- `server.py`
  Main RAG/chat server for manual-aware question answering.
- `embed.py`
  Embedding/build path for the manual retrieval store.
- `supabase.py`
  Supabase integration helpers.
- `validate_rag.py`
  RAG validation script and quality checks.
- `chat_terminal_client.py`
  Lightweight client for exercising the chat flow.
- `MANUALS/`
  Source lab manuals and reference PDFs used by the RAG system.

### In `circuit_debug_api/`

- `server.py`
  Main FastAPI entry point for debug and chat endpoints.
- `runtime.py`
  Runtime inference path for the circuit-debug system.
- `hybrid_runtime.py`
  Hybrid inference path combining LLM and KNN-style assets.
- `build_runtime_assets.py`
  Packages runtime assets for deployment.
- `build_hybrid_assets.py`
  Packages hybrid adapter/KNN assets for deployment.
- `demo_payloads/`
  Reproducible sample requests for demonstrations.

## Recommended Reading Order

If you are new to the repo, open files in this order:

1. `README.md`
2. `gpt/Capstone/server.py`
3. `circuit_debug_api/server.py`
4. `pipeline_one_lab/run_one_lab_pipeline.py`
5. `pipeline/generate_variants.py`
6. `pipeline/train_lora.py`
7. `pipeline/test_lora_model.py`

## Generated Directory Naming

- `pipeline/out/`
  Older multi-lab outputs, eval reports, checkpoints, and ad hoc experiment data.
- `pipeline/out_one_lab/`
  Focused one-lab outputs.
- `pipeline/out_one_lab_all*/`
  Recursive all-lab experiment runs.
- `pipeline/tmp_*/`
  Temporary curated subsets of source circuits.
- `__pycache__/`
  Python cache files.
- `cleanup_archive_*/`
  Historical snapshots and old work copies that are useful for you locally but
  usually not part of the clean professor-facing narrative.

## If You Want To Change...

- RAG/manual chat behavior:
  `gpt/Capstone/server.py`
- Final API behavior:
  `circuit_debug_api/server.py`, `circuit_debug_api/runtime.py`, and `circuit_debug_api/hybrid_runtime.py`
- Fault generation:
  `pipeline/generate_variants.py`
- Measurement extraction or sim execution:
  `pipeline/run_ltspice_batch.py`
- Prompt shape or label format:
  `pipeline/prepare_finetune_data.py` and `pipeline_one_lab/prepare_finetune_one_lab.py`
- One-circuit orchestration:
  `pipeline_one_lab/run_one_lab_pipeline.py`
- Recursive all-circuit orchestration:
  `pipeline_one_lab/run_recursive_all_labs_pipeline.py`
- Training settings:
  `pipeline/train_lora.py`
- Evaluation behavior:
  `pipeline/test_lora_model.py`
