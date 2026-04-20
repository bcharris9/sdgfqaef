# Circuit Debug API

FastAPI wrapper for your LTSpice-trained circuit fault system.

This folder is now packaged so it can run on its own after you upload just `circuit_debug_api/`.
The only major dependency that is not vendored into Git is the Qwen base model itself. By default
the API will download/load `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face. If you want to use a
local snapshot instead, set:

```powershell
$env:CIRCUIT_DEBUG_BASE_MODEL = "C:\\path\\to\\Qwen2.5-1.5B-Instruct"
```

Default backend:

- **LLM + KNN hybrid** (Qwen LoRA adapter + KNN class priors)
- Uses the same hybrid scoring path as `pipeline/test_lora_model.py` (`score_classes_knn`)

Fallback backend (if hybrid assets are not present):

- Tabular XGBoost classifier

Endpoints:

- `POST /chat` : ask lab-manual questions and let the server infer or auto-select the lab context
- `POST /chat/{lab_number}` : ask lab-manual questions for a specific lab
- `GET /circuits` : list all supported (golden) circuit names
- `GET /circuits/{circuit_name}/nodes` : list required node names (plus optional source current names)
- `POST /debug` : submit measured values and receive a predicted fault class + diagnosis/fix text

## Directory Contents

- `server.py` : FastAPI app and endpoints
- `runtime.py` : model loading, feature engineering, inference logic
- `build_runtime_assets.py` : packages tabular model + catalog assets
- `build_hybrid_assets.py` : packages LoRA adapter + KNN reference/index assets for hybrid API mode
- `client_example.py` : example client hitting all endpoints
- `chat_terminal_client.py` : interactive terminal chat client for `POST /chat` or `POST /chat/{lab_number}`
- `export_merged_debug_model.py` : merges the debug LoRA adapter into a standalone model directory for quantization/deployment
- `run_backend_jetson.sh` : starts the local Jetson chat server and FastAPI app together with readiness checks
- `run_chat_vllm_jetson.sh` : launches a local Jetson vLLM server for the fast `/chat` path
- `student_interactive_client.py` : interactive terminal client that prompts for node values one at a time
- `test_chat_endpoint.py` : Python smoke test for `POST /chat`
- `smoke_test_api.ps1` : PowerShell smoke test for API startup + `/debug` client flow
- `demo_payloads/` : ready-to-submit real simulated measurement payloads (for reproducible demos)
- `requirements.txt` : Python deps for API + client
- `make_venv.sh` : Bash script to create `.venv312` and install deps
- `run_api.ps1` : PowerShell start script (Windows)
- `assets/` : tabular assets + circuit catalog
- `assets_hybrid/` : optional hybrid assets (LoRA adapter copy, KNN ref/index, hybrid config) when you choose to package the LLM+KNN path
- `packaged_golden_root/` : local copy of the golden measurement files used by the packaged catalog
- `packaged_reports/` : local copy of the selected best-model eval report used for auto-pick metadata

## Install Dependencies

```powershell
.\\.venv312\\Scripts\\python.exe -m pip install -r .\\requirements.txt
```

On Linux/Jetson, prefer `./make_venv.sh` or set `TMPDIR` before installing manually because `/tmp`
is often a small tmpfs and large wheel downloads can fail with `OSError: [Errno 28] No space left on device`.

```bash
mkdir -p ./.tmp
TMPDIR="$PWD/.tmp" python -m pip install -r ./requirements.txt
```

## Build Runtime Assets (one-time or after model updates)

The default rebuild path is now self-contained and uses the packaged files already inside
`circuit_debug_api/`.

```powershell
.\\.venv312\\Scripts\\python.exe .\\build_runtime_assets.py
```

## Build Hybrid Assets (LLM + KNN)

This copies the selected LoRA adapter into the API directory and prebuilds a KNN index from the training instruct JSONL.
Some lean Git snapshots intentionally omit `assets_hybrid/` because the packaged hybrid assets are large. If that directory is missing, run this step to rebuild it.

The builder now defaults to the packaged adapter, packaged KNN reference file, and packaged report
inside `circuit_debug_api/`. It does not need the rest of the repo for a normal rebuild.

```powershell
.\\.venv312\\Scripts\\python.exe .\\build_hybrid_assets.py
```

Force a specific adapter (disable auto-pick):

```powershell
.\\.venv312\\Scripts\\python.exe .\\circuit_debug_api\\build_hybrid_assets.py `
  --auto-pick-best False `
  --adapter-dir .\\assets_hybrid\\adapter
```

## Run the API

```powershell
.\\.venv312\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8001
```

If you start `server.py` directly in one process while `/chat` uses a local Ollama or
OpenAI-compatible server on the same machine, the API now applies conservative shared-GPU defaults
for the hybrid `/debug` model (`device_map` plus GPU/CPU memory caps and offload folder). That
keeps `/debug` from consuming the whole GPU and breaking later `/chat` turns. If you want different
limits, set the `CIRCUIT_DEBUG_*` memory env vars explicitly or run an isolated debug worker.

Or (for a fresh best-model selection from the latest reports):

```powershell
powershell -ExecutionPolicy Bypass -File .\\circuit_debug_api\\run_api.ps1 -RefreshModel
```

`run_api.ps1` will auto-build both tabular and hybrid assets if they are missing and uses auto-pick for the best hybrid model.

## Jetson Performance Mode

For a fully local Jetson Orin Nano Super deployment, keep this FastAPI app on the device and run a local model server for `POST /chat`.

First time on the Jetson:

1. Make sure Docker is installed and the Docker daemon is running on the Jetson.
2. Create the virtual environment and install Python dependencies if you have not already:

```bash
./make_venv.sh
source .venv312/bin/activate
```

3. Make the runner scripts executable:

```bash
chmod +x ./run_backend_jetson.sh ./run_chat_vllm_jetson.sh
```

4. Start the full backend:

```bash
./run_backend_jetson.sh
```

The Jetson vLLM launcher now defaults to `RedHatAI/Qwen2.5-3B-quantized.w4a16` with a memory-safe
profile for small-memory devices:

- `LAB_CHAT_GPU_MEMORY_UTILIZATION=0.45`
- `LAB_CHAT_MAX_MODEL_LEN=1024`
- `LAB_CHAT_MAX_NUM_SEQS=1`
- `LAB_CHAT_MAX_NUM_BATCHED_TOKENS=128`
- `LAB_CHAT_ENFORCE_EAGER=1`

The Jetson backend now defaults to the Ollama Cloud 120B chat path, so the usual startup command is just:

```bash
./run_backend_jetson.sh
```

By default this uses `gpt-oss:120b-cloud` for chat, keeps embeddings local through Ollama on
`127.0.0.1:11434`, and keeps `/debug` on the Jetson GPU worker.

The chat path now allows a longer default completion budget (`LAB_CHAT_MAX_NEW_TOKENS=512`) and
automatically issues one short continuation request (`LAB_CHAT_CONTINUATION_MAX_TOKENS=384`) if the
remote model stops because it hit the length cap. That makes cut-off answers much less likely in the
terminal client.

The Hugging Face `/debug` path now runs in an isolated worker process on `127.0.0.1:8002` by
default, and the main API proxies debug requests to it. This keeps the heavy debug-model load out
of the main FastAPI process.

The API now also performs a cheap golden-value precheck before `/debug` inference. If all required
submitted node measurements are already within the configured tolerance of the packaged golden
values, it returns a `golden_match` response immediately instead of spending GPU time on the fault
model. The default threshold is 5% relative error, with small absolute floors for near-zero values
(`LAB_DEBUG_GOLDEN_MATCH_REL_TOL=0.05`, `LAB_DEBUG_GOLDEN_MATCH_ABS_VOLTAGE=0.05`,
`LAB_DEBUG_GOLDEN_MATCH_ABS_CURRENT=1e-4`).

Automatic debug prewarm is now on by default when chat is using the remote OpenAI-compatible path,
so the first student `/debug` request does not have to wait for the full worker model load. You can
override this with `LAB_DEBUG_PREWARM=0` or `LAB_DEBUG_PREWARM=1`.

If `./artifacts/debug_merged/model.safetensors` exists, the isolated worker automatically uses
that merged debug model and defaults to a GPU-capable debug path: `LAB_DEBUG_WORKER_BACKEND=container`
with `CIRCUIT_DEBUG_DEVICE=auto`. The container image is built locally from
`[docker/debug_worker.Dockerfile](/home/team28-404/Desktop/LLM-main/docker/debug_worker.Dockerfile)`
on top of the Jetson-native `dustynv/l4t-pytorch:r36.4.0` image by default. You can still force
the worker onto a specific device or runtime
with `LAB_DEBUG_WORKER_DEVICE=cpu|cuda` and `LAB_DEBUG_WORKER_BACKEND=host|container`.
The first containerized run is large and can take several minutes because Docker needs to download
the Jetson PyTorch base image; later runs reuse the cached image.

The worker now also uses conservative Hugging Face loading limits by default:

- `CIRCUIT_DEBUG_USE_DEVICE_MAP=1`
- `CIRCUIT_DEBUG_MAX_CPU_MEMORY=2GiB`
- `CIRCUIT_DEBUG_MAX_GPU_MEMORY=2GiB`
- `CIRCUIT_DEBUG_OFFLOAD_FOLDER=.debug_offload`

On constrained Jetson GPU loads, the debug model loader also skips Transformers' CUDA allocator
warmup whenever a GPU memory cap is in play. That warmup is only a load-speed optimization, and on
some PyTorch/Jetson combinations it can crash with NVML allocator assertions even when the actual
offloaded model load would fit.

That lets Transformers spill part of the model to disk instead of trying to fill all system RAM
while it loads. If you need to tune the balance later, you can raise or lower those values when
starting `./run_backend_jetson.sh`.

If you want to switch to the Jetson-local vLLM container path explicitly:

```bash
LAB_CHAT_BACKEND=openai_compat LAB_CHAT_MANAGE_SERVER=1 ./run_backend_jetson.sh
```

That path still keeps the smaller vLLM profile we used for Jetson stability instead of trying to
reserve a huge 128k context window on an 8 GB-class device.

If you want to override the profile, for example to force an even smaller fixed context:

```bash
LAB_CHAT_GPU_MEMORY_UTILIZATION=0.4 LAB_CHAT_MAX_MODEL_LEN=768 LAB_CHAT_MAX_NUM_BATCHED_TOKENS=96 ./run_backend_jetson.sh
```

If you want to be explicit, you can still start the local Ollama path with:

```bash
LAB_CHAT_BACKEND=ollama ./run_backend_jetson.sh
```

The backend runner will use Ollama Cloud for chat by default and will still pull the local
embedding model set by `LAB_EMBED_MODEL` so retrieval can work. If you want a different remote chat
model, set `LAB_CHAT_MODEL`.

If the Jetson GPU still cannot load the Ollama chat model, start a dedicated CPU-only Ollama server
on a separate port:

```bash
LAB_CHAT_BACKEND=ollama LAB_OLLAMA_FORCE_CPU=1 ./run_backend_jetson.sh
```

This is slower, but it avoids the Jetson GPU allocation failures we saw with local Ollama GPU
loading. The CPU-only path uses a smaller default context length and its own Ollama process on
`127.0.0.1:11435`.

If Docker is not available on the device, you can still bring the API up without the containerized
chat server by using the in-process Transformers fallback:

```bash
LAB_CHAT_BACKEND=transformers ./run_backend_jetson.sh
```

This path skips Docker entirely. If you already have an OpenAI-compatible chat server running
somewhere else, keep `LAB_CHAT_BACKEND=openai_compat` and set `LAB_CHAT_MANAGE_SERVER=0`.

For a split deployment where chat is remote, embeddings stay local on the Jetson, and `/debug`
still uses the Jetson GPU worker, use:

```bash
LAB_CHAT_BACKEND=openai_compat \
LAB_CHAT_MANAGE_SERVER=0 \
LAB_CHAT_BASE_URL=https://your-chat-endpoint.example/v1 \
LAB_CHAT_API_KEY=your_api_key \
LAB_OLLAMA_BASE_URL=http://127.0.0.1:11434 \
LAB_DEBUG_WORKER_DEVICE=auto \
./run_backend_jetson.sh
```

The runner now makes this combination first-class: it will not require Docker just to use a remote
chat endpoint, and it will still bring up local Ollama for embeddings when `LAB_OLLAMA_BASE_URL`
points at the Jetson. In that remote-chat mode, the default chat model is now
`gpt-oss:120b-cloud`; override `LAB_CHAT_MODEL` only if you want a different remote model.

The backend runner now auto-loads `.env`, `.env.local`, and `.env.jetson.local` from the repo
root before computing defaults. For example, this is a good place to keep the Ollama Cloud key
without hardcoding it into the script body:

```bash
# .env.jetson.local
OLLAMA_API_KEY=your_api_key_here
```

Startup will print `Chat API key: configured` when the runner picked it up successfully.

What happens on the first run:

- The script checks whichever chat runtime you selected.
- For `openai_compat`, it checks Docker and pulls the Jetson vLLM image if needed.
- For `ollama`, it tries to start the Ollama service and pulls the requested model if needed.
- If runtime assets are missing, it builds them.
- It starts the local chat model server and waits for it to become ready.
- It starts FastAPI and waits for `GET /health`.

This keeps all inference on the machine. The Jetson-local chat defaults are baked into `server.py`, and the lab-manual retrieval path defaults to manual version `v2`.

Notes:

- The first run can take significantly longer than later runs because Docker images and model weights may need to be downloaded and cached.
- If startup fails, check `_backend_logs/chat_server.log` and `_backend_logs/api_server.log`.
- The backend runner will print the final API URL when everything is ready.

For `POST /debug`, the current hybrid runtime still works in-process. If you want to prepare the fine-tuned debug model for later quantization, first export a merged model:

```bash
python ./export_merged_debug_model.py --output-dir ./artifacts/debug_merged
```

The code also supports loading a standalone merged debug model instead of composing the base model with the LoRA adapter at startup.

## Bash / Linux/macOS Quickstart

If you are using Bash (macOS/Linux, or Git Bash/WSL), these commands assume you already activated the venv:
`source .venv312/bin/activate`

Create the virtual environment (recommended):

```bash
./make_venv.sh
```

Install dependencies:

```bash
mkdir -p ./.tmp
TMPDIR="$PWD/.tmp" python -m pip install -r ./requirements.txt
```

Build runtime assets:

```bash
python ./build_runtime_assets.py
```

Skip this step if `./assets/model_bundle.joblib` already exists (it does in this demo folder).

Build hybrid assets:

```bash
python ./build_hybrid_assets.py
```

Skip this step if `./assets_hybrid/hybrid_config.json` already exists. In lean source-only snapshots, `assets_hybrid/` may be absent until you build it.

Run the API:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8001
```

Example client:

```bash
python ./client_example.py --demo-use-golden-values --demo-offset-node N001 --demo-offset-volts 0.5
```

Interactive student client:

```bash
python ./student_interactive_client.py
```

Interactive chat client:

```bash
python ./chat_terminal_client.py --base-url http://127.0.0.1:8001
```

Query endpoints from Bash:

```bash
curl -s http://127.0.0.1:8001/circuits | jq
curl -s http://127.0.0.1:8001/circuits/Lab9_2/nodes | jq
```

Submit a JSON payload from Bash:

```bash
curl -s http://127.0.0.1:8001/debug \
  -H 'Content-Type: application/json' \
  --data @./student_lab9_2_payload.json | jq
```

## Example Client (uses all endpoints)

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py --demo-use-golden-values --demo-offset-node N001 --demo-offset-volts 0.5
```

## Interactive Student Client (one measurement at a time)

This client first prompts the student to choose a lab, then a circuit within that lab, then prompts for measurements one by one.

```powershell
.\\.venv312\\Scripts\\python.exe .\\student_interactive_client.py
```

Flow in terminal:

1. Choose lab (example: `Lab4` or `4`)
2. Choose circuit from that lab
3. Enter each node voltage one at a time
4. Optionally enter source currents
5. Submit and receive diagnosis

Optional flags:

- `--circuit Lab9_2` : skip circuit selection prompt
- `--ask-source-currents` : also prompt for optional source currents
- `--save-payload .\\student_payload.json` : save the submitted payload
- `--show-golden` : instructor/demo mode only
- `--no-strict` : allow missing nodes (not recommended)

## Interactive Chat Client (terminal Q&A)

This client lets a student type a question in the terminal and prints the chat answer. In interactive mode it prompts for an optional lab number once; if you skip it, the server will infer or auto-select the lab.

```powershell
.\\.venv312\\Scripts\\python.exe .\\chat_terminal_client.py --base-url http://127.0.0.1:8001
```

Optional one-shot question:

```powershell
.\\.venv312\\Scripts\\python.exe .\\chat_terminal_client.py `
  --base-url http://127.0.0.1:8001 `
  --lab-number 1 `
  --question "What does Lab 1 procedure require?"
```

Exit commands in interactive mode: `/quit`, `/exit`.

## Chat Endpoint Test (server.py)

Run the chat endpoint smoke test:

```powershell
.\\.venv312\\Scripts\\python.exe .\\test_chat_endpoint.py --base-url http://127.0.0.1:8001
```

Strict mode (require valid-question call to return `200` with `answer`):

```powershell
.\\.venv312\\Scripts\\python.exe .\\test_chat_endpoint.py `
  --base-url http://127.0.0.1:8001 `
  --require-answer
```

This test checks:

- `POST /chat` exists and uses a JSON request body
- valid `{"question":"..."}` request
- empty-question behavior
- missing required `question` field (`422`)

## Full API Smoke Test (PowerShell)

`smoke_test_api.ps1` starts `uvicorn`, waits for `/health`, runs `client_example.py`, and prints server log tails:

```powershell
powershell -ExecutionPolicy Bypass -File .\\smoke_test_api.ps1
```

## Specific Circuit Demo (real simulated case)

This uses a real simulated variant from `Lab9_2` (`Lab9_2__v0022`) and submits the measured node voltages/source currents to the API.

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\demo_payloads\\Lab9_2__v0022_request.json
```

Reference metadata / expected injected fault for that demo case:

- `demo_payloads/Lab9_2__v0022_expected.json`

## Additional Real Demo Payloads (held-out simulated eval rows)

These are extra reproducible demos extracted from held-out simulated eval rows and aligned to a saved hybrid eval run.

- Index: `demo_payloads/demo_index.json`
- Each demo has:
  - `*_request.json` (send to `POST /debug`)
  - `*_expected.json` (saved target label / provenance)

Included classes:

- `param_drift`
- `resistor_value_swap`
- `resistor_wrong_value`
- `missing_component`
- `short_between_nodes`
- `swapped_nodes`
- `pin_open`

Run any one demo:

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\demo_payloads\\evalrow_0001__Lab1_2A_2_0__param_drift_request.json
```

Run another demo:

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\demo_payloads\\evalrow_0049__lab4_task2_part1_-3__short_between_nodes_request.json
```

## Student Breadboard Workflow (exact endpoint flow)

This is the intended real use path when a student has breadboard measurements.

### 1) Start the API

```powershell
.\\.venv312\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8001
```

### 2) Get the valid circuit names (pick the golden circuit the student is building)

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/circuits | ConvertTo-Json -Depth 5
```

### 3) Get the exact node names required for that circuit

Example for `Lab9_2`:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/circuits/Lab9_2/nodes | ConvertTo-Json -Depth 10
```

Use the returned `nodes[].node_name` list as the measurement checklist.

### 4) Take measurements on the breadboard (what to enter)

- Measure each listed node voltage relative to the breadboard ground node.
- Enter values using the exact node names from the API.
- If the circuit is time-varying, this API currently expects the same convention as training: `*_max` features (peak/max values).
- `source_currents` are optional, but if you can measure them they usually improve accuracy.
- Keep `temp` and `tnom` at defaults unless you intentionally changed them.

### 5) Fill a payload JSON file with the student measurements

Example (`student_lab9_2_payload.json`):

```json
{
  "circuit_name": "Lab9_2",
  "node_voltages": {
    "N001": 0.95,
    "N002": 0.18,
    "N003": 5.0,
    "N004": 1.0,
    "N005": 0.18,
    "N006": -5.0
  },
  "source_currents": {},
  "temp": 27.0,
  "tnom": 27.0,
  "strict": true
}
```

Notes:

- `strict: true` will fail if any required node is missing (recommended for students).
- If you do not know source currents, leave `source_currents` as `{}`.

### 6) Submit the measurements for debugging

Using the example client:

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\student_lab9_2_payload.json
```

Using the interactive student client (recommended for manual breadboard entry):

```powershell
.\\.venv312\\Scripts\\python.exe .\\student_interactive_client.py `
  --circuit Lab9_2 `
  --ask-source-currents `
  --save-payload .\\student_lab9_2_payload.json
```

Or directly with PowerShell:

```powershell
$body = Get-Content .\\student_lab9_2_payload.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8001/debug -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

### 7) Read the response

Main fields:

- `fault_type` : predicted fault class
- `confidence` : model confidence (not a guarantee)
- `diagnosis` / `fix` : human-readable guidance
- `missing_required_nodes` : nodes you forgot to provide (when `strict=false`)

Notes:

- Demo mode is only to show endpoint usage. Exact golden values are not a real fault case.
- `--payload-file` mode is the recommended way to demo a specific real simulated case.
- For best accuracy, provide all nodes from `GET /circuits/{name}/nodes`.
- Supplying source currents (if available) improves accuracy.
- `GET /health` reports which backend is active (`llm_knn_hybrid` or `tabular_xgboost`).
- `GET /model` returns the exact selected adapter paths and eval report metrics used to pick the model.

To force a fresh best-model selection and reload after new training/eval artifacts are produced:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/admin/refresh-model -Method Post | ConvertTo-Json -Depth 12
```

Or restart with refresh enabled:

```powershell
powershell -ExecutionPolicy Bypass -File .\\circuit_debug_api\\run_api.ps1 -RefreshModel
```

## POST /debug Request Shape

```json
{
  "circuit_name": "Lab1_1_0",
  "node_voltages": {
    "N001": 9.0,
    "N002": 5.0
  },
  "source_currents": {
    "V1": -0.00185
  },
  "strict": true
}
```
