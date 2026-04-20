# LTSpice Fault Pipeline Workspace

This repository tracks the source circuits plus the scripts used to generate
faulted LTspice variants, run simulations, build training data, and train or
evaluate diagnosis models.

## Start Here

- [docs/project-map.md](docs/project-map.md)
  Repo map and "what file should I open first?" guide.
- [pipeline_one_lab/README.md](pipeline_one_lab/README.md)
  Best entry point for the current one-circuit and recursive all-labs flows.
- [pipeline/README.md](pipeline/README.md)
  Core pipeline scripts, training helpers, and focused PowerShell wrappers.

## Top-Level Layout

- `gpt/`
  RAG/manual-assistant work for the chat side of the final system.
- `circuit_debug_api/`
  Final API/runtime packaging for the circuit-debug product.
- `LTSpice_files/`
  Canonical source `.asc` circuits grouped by lab and task.
- `pipeline/`
  Core scripts for variant generation, LTspice batch simulation, dataset
  building, fine-tune prep, training, evaluation, and inference.
- `pipeline_one_lab/`
  Orchestrators for one-lab, golden-set, and recursive all-labs workflows.

## Generated Vs Source

The source code lives mostly in `pipeline/` and `pipeline_one_lab/`.

For the final product story, the most important architecture folders are:

- `gpt/Capstone/` for the RAG/manual Q&A system
- `circuit_debug_api/` for the deployed debug/chat API
- `LTSpice_files/`, `pipeline/`, and `pipeline_one_lab/` for the data and model pipeline

Generated artifacts are expected under paths such as:

- `pipeline/out/`
- `pipeline/out_one_lab/`
- `pipeline/out_one_lab_all*/`
- `pipeline/tmp_*/`
- `__pycache__/`
- LTspice simulation byproducts like `.raw`, `.op.raw`, `.log`, `.net`, and `.db`

If Git still shows old output folders, that usually means the files were
tracked historically and need a separate untracking pass. The navigation docs
here are meant to make the live source tree easier to read without making risky
data-deletion changes.

## Common Entry Points

1. Rebuild one circuit:
   `pipeline_one_lab/run_one_lab_pipeline.py`
2. Rebuild every circuit recursively:
   `pipeline_one_lab/run_recursive_all_labs_pipeline.py`
3. Train an adapter:
   `pipeline/train_lora.py`
4. Evaluate an adapter:
   `pipeline/test_lora_model.py`
5. Run manual measured-node inference:
   `pipeline/run_measurement_infer.py`
6. Review the RAG server:
   `gpt/Capstone/server.py`
7. Review the packaged API:
   `circuit_debug_api/server.py`
