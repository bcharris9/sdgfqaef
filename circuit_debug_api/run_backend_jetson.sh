#!/usr/bin/env bash
set -euo pipefail

# Start the full Jetson backend stack: local chat server first, then FastAPI.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# Load optional local env files before any defaults are computed.
# Later files override earlier ones, so .env.jetson.local wins.
for env_file in "${ROOT}/.env" "${ROOT}/.env.local" "${ROOT}/.env.jetson.local"; do
  if [[ -f "${env_file}" ]]; then
    if [[ ! -s "${env_file}" ]]; then
      echo "Config file is present but empty: ${env_file}"
      continue
    fi
    echo "Loading config from ${env_file}"
    set -a
    # shellcheck disable=SC1090
    . "${env_file}"
    set +a
  fi
done

PY="${LAB_PYTHON:-${ROOT}/.venv312/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="python"
fi

API_HOST="${LAB_API_HOST:-127.0.0.1}"
API_PORT="${LAB_API_PORT:-8001}"
CHAT_HOST="${LAB_CHAT_HOST:-127.0.0.1}"
CHAT_PORT="${LAB_CHAT_PORT:-8001}"
OLLAMA_HOST="${LAB_OLLAMA_HOST:-127.0.0.1}"
OLLAMA_GPU_PORT="${LAB_OLLAMA_PORT:-11434}"
OLLAMA_CPU_PORT="${LAB_OLLAMA_CPU_PORT:-11435}"
REQUESTED_CHAT_BACKEND="${LAB_CHAT_BACKEND:-openai_compat}"
CHAT_MANAGE_SERVER="${LAB_CHAT_MANAGE_SERVER:-0}"
CHAT_BASE_URL_DEFAULT="http://${CHAT_HOST}:${CHAT_PORT}/v1"
REMOTE_OPENAI_COMPAT_BASE_URL_DEFAULT="${LAB_OPENAI_COMPAT_REMOTE_BASE_URL:-https://ollama.com/v1}"
OLLAMA_FORCE_CPU="${LAB_OLLAMA_FORCE_CPU:-0}"
if [[ "${OLLAMA_FORCE_CPU}" == "1" ]]; then
  ACTIVE_OLLAMA_PORT="${OLLAMA_CPU_PORT}"
else
  ACTIVE_OLLAMA_PORT="${OLLAMA_GPU_PORT}"
fi
OLLAMA_BASE_URL_DEFAULT="http://${OLLAMA_HOST}:${ACTIVE_OLLAMA_PORT}/v1"
OLLAMA_API_BASE_URL_DEFAULT="http://${OLLAMA_HOST}:${ACTIVE_OLLAMA_PORT}"
export LAB_CHAT_BACKEND="${REQUESTED_CHAT_BACKEND}"
if [[ "${REQUESTED_CHAT_BACKEND}" == "ollama" ]]; then
  export LAB_CHAT_BASE_URL="${LAB_CHAT_BASE_URL:-${OLLAMA_BASE_URL_DEFAULT}}"
else
  if [[ "${CHAT_MANAGE_SERVER}" == "0" ]]; then
    export LAB_CHAT_BASE_URL="${LAB_CHAT_BASE_URL:-${REMOTE_OPENAI_COMPAT_BASE_URL_DEFAULT}}"
  else
    export LAB_CHAT_BASE_URL="${LAB_CHAT_BASE_URL:-${CHAT_BASE_URL_DEFAULT}}"
  fi
fi
export LAB_OLLAMA_BASE_URL="${LAB_OLLAMA_BASE_URL:-${OLLAMA_API_BASE_URL_DEFAULT}}"
DEFAULT_VLLM_CHAT_MODEL="RedHatAI/Qwen2.5-3B-quantized.w4a16"
DEFAULT_REMOTE_OLLAMA_CLOUD_MODEL="${LAB_OPENAI_COMPAT_REMOTE_MODEL:-gpt-oss:120b-cloud}"
DEFAULT_OLLAMA_CHAT_MODEL="${LAB_OLLAMA_DEFAULT_MODEL:-qwen2.5:3b}"
DEFAULT_TRANSFORMERS_CHAT_MODEL="${CIRCUIT_DEBUG_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DEFAULT_MERGED_DEBUG_MODEL_DIR="${LAB_DEBUG_MERGED_MODEL_DIR:-${ROOT}/artifacts/debug_merged}"
OLLAMA_EMBED_MODEL="${LAB_EMBED_MODEL:-mxbai-embed-large}"
OLLAMA_CONTEXT_LENGTH="${LAB_OLLAMA_CONTEXT_LENGTH:-4096}"
OLLAMA_MAX_LOADED_MODELS="${LAB_OLLAMA_MAX_LOADED_MODELS:-1}"
OLLAMA_NUM_PARALLEL="${LAB_OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_KEEP_ALIVE="${LAB_OLLAMA_KEEP_ALIVE:-5m}"
if [[ -z "${LAB_CHAT_MODEL:-}" ]]; then
  if [[ "${REQUESTED_CHAT_BACKEND}" == "openai_compat" ]]; then
    if [[ "${CHAT_MANAGE_SERVER}" == "0" ]]; then
      export LAB_CHAT_MODEL="${DEFAULT_REMOTE_OLLAMA_CLOUD_MODEL}"
    else
      export LAB_CHAT_MODEL="${DEFAULT_VLLM_CHAT_MODEL}"
    fi
  elif [[ "${REQUESTED_CHAT_BACKEND}" == "ollama" ]]; then
    export LAB_CHAT_MODEL="${DEFAULT_OLLAMA_CHAT_MODEL}"
  else
    export LAB_CHAT_MODEL="${DEFAULT_TRANSFORMERS_CHAT_MODEL}"
  fi
else
  export LAB_CHAT_MODEL
fi
export LAB_CHAT_API_KEY="${LAB_CHAT_API_KEY:-${OLLAMA_API_KEY:-}}"
export LAB_MANUAL_VERSION="${LAB_MANUAL_VERSION:-v2}"
if [[ -n "${CIRCUIT_DEBUG_MERGED_MODEL_DIR:-}" ]]; then
  export CIRCUIT_DEBUG_MERGED_MODEL_DIR
elif [[ -f "${DEFAULT_MERGED_DEBUG_MODEL_DIR}/model.safetensors" ]]; then
  export CIRCUIT_DEBUG_MERGED_MODEL_DIR="${DEFAULT_MERGED_DEBUG_MODEL_DIR}"
fi
DEBUG_USING_MERGED_MODEL=0
if [[ -n "${CIRCUIT_DEBUG_MERGED_MODEL_DIR:-}" ]]; then
  DEBUG_USING_MERGED_MODEL=1
fi
if [[ -n "${CIRCUIT_DEBUG_DEVICE:-}" ]]; then
  CIRCUIT_DEBUG_DEVICE_DEFAULT="${CIRCUIT_DEBUG_DEVICE}"
elif [[ "${DEBUG_USING_MERGED_MODEL}" == "1" ]]; then
  CIRCUIT_DEBUG_DEVICE_DEFAULT="auto"
else
  CIRCUIT_DEBUG_DEVICE_DEFAULT="cpu"
fi
export CIRCUIT_DEBUG_DEVICE="${CIRCUIT_DEBUG_DEVICE_DEFAULT}"
export CIRCUIT_DEBUG_CPU_DTYPE="${CIRCUIT_DEBUG_CPU_DTYPE:-float16}"
export CIRCUIT_DEBUG_LOCAL_FILES_ONLY="${CIRCUIT_DEBUG_LOCAL_FILES_ONLY:-1}"
export CIRCUIT_DEBUG_LOW_CPU_MEM_USAGE="${CIRCUIT_DEBUG_LOW_CPU_MEM_USAGE:-1}"
if [[ -z "${CIRCUIT_DEBUG_USE_DEVICE_MAP:-}" ]]; then
  if [[ "${CIRCUIT_DEBUG_DEVICE_DEFAULT}" == "cpu" && "${DEBUG_USING_MERGED_MODEL}" != "1" ]]; then
    export CIRCUIT_DEBUG_USE_DEVICE_MAP=0
  else
    export CIRCUIT_DEBUG_USE_DEVICE_MAP=1
  fi
else
  export CIRCUIT_DEBUG_USE_DEVICE_MAP
fi
export CIRCUIT_DEBUG_MAX_CPU_MEMORY="${CIRCUIT_DEBUG_MAX_CPU_MEMORY:-2GiB}"
export CIRCUIT_DEBUG_MAX_GPU_MEMORY="${CIRCUIT_DEBUG_MAX_GPU_MEMORY:-2GiB}"
export CIRCUIT_DEBUG_OFFLOAD_FOLDER="${CIRCUIT_DEBUG_OFFLOAD_FOLDER:-${ROOT}/.debug_offload}"
mkdir -p "${CIRCUIT_DEBUG_OFFLOAD_FOLDER}"

VLLM_IMAGE="${LAB_CHAT_VLLM_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
GPU_MEMORY_UTILIZATION="${LAB_CHAT_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_MAX_MODEL_LEN="${LAB_CHAT_MAX_MODEL_LEN:-1024}"
VLLM_MAX_NUM_SEQS="${LAB_CHAT_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${LAB_CHAT_MAX_NUM_BATCHED_TOKENS:-128}"
VLLM_ENFORCE_EAGER="${LAB_CHAT_ENFORCE_EAGER:-1}"
VLLM_DISABLE_CASCADE_ATTN="${LAB_CHAT_DISABLE_CASCADE_ATTN:-1}"
CHAT_READY_ATTEMPTS="${LAB_CHAT_READY_ATTEMPTS:-1800}"
API_READY_ATTEMPTS="${LAB_API_READY_ATTEMPTS:-300}"
DEBUG_WORKER_ENABLED="${LAB_DEBUG_WORKER_ENABLED:-1}"
DEBUG_WORKER_HOST="${LAB_DEBUG_WORKER_HOST:-127.0.0.1}"
DEBUG_WORKER_PORT="${LAB_DEBUG_WORKER_PORT:-8002}"
DEBUG_WORKER_READY_ATTEMPTS="${LAB_DEBUG_WORKER_READY_ATTEMPTS:-300}"
if [[ -n "${LAB_DEBUG_WORKER_BACKEND:-}" ]]; then
  DEBUG_WORKER_BACKEND="${LAB_DEBUG_WORKER_BACKEND}"
elif [[ "${DEBUG_USING_MERGED_MODEL}" == "1" ]]; then
  DEBUG_WORKER_BACKEND="container"
else
  DEBUG_WORKER_BACKEND="host"
fi
DEBUG_WORKER_DEVICE="${LAB_DEBUG_WORKER_DEVICE:-${CIRCUIT_DEBUG_DEVICE_DEFAULT}}"
DEBUG_WORKER_CONTAINER_IMAGE="${LAB_DEBUG_WORKER_IMAGE:-lab-debug-worker:jetson}"
DEBUG_WORKER_CONTAINER_BASE_IMAGE="${LAB_DEBUG_WORKER_BASE_IMAGE:-dustynv/l4t-pytorch:r36.4.0}"
DEBUG_WORKER_CONTAINER_NAME="${LAB_DEBUG_WORKER_CONTAINER_NAME:-lab-debug-worker-${USER}}"
if [[ "${DEBUG_WORKER_BACKEND}" != "container" ]] && [[ "${DEBUG_WORKER_DEVICE}" == "cuda" || "${DEBUG_WORKER_DEVICE}" == "gpu" ]]; then
  if ! "${PY}" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "Requested debug worker CUDA, but ${PY} does not report CUDA as available."
    echo "Falling back to debug worker device: auto"
    DEBUG_WORKER_DEVICE="auto"
  fi
fi
DEBUG_WORKER_URL_DEFAULT="http://${DEBUG_WORKER_HOST}:${DEBUG_WORKER_PORT}"
if [[ -n "${LAB_DEBUG_PREWARM:-}" ]]; then
  DEBUG_PREWARM="${LAB_DEBUG_PREWARM}"
elif [[ "${REQUESTED_CHAT_BACKEND}" == "openai_compat" && "${CHAT_MANAGE_SERVER}" == "0" ]]; then
  DEBUG_PREWARM="1"
else
  DEBUG_PREWARM="0"
fi
DEBUG_PREWARM_TIMEOUT_SECONDS="${LAB_DEBUG_PREWARM_TIMEOUT_SECONDS:-900}"
LOG_DIR="${LAB_BACKEND_LOG_DIR:-${ROOT}/_backend_logs}"
mkdir -p "${LOG_DIR}"
CHAT_LOG="${LOG_DIR}/chat_server.log"
API_LOG="${LOG_DIR}/api_server.log"
DEBUG_WORKER_LOG="${LOG_DIR}/debug_worker.log"
if [[ "${DEBUG_WORKER_ENABLED}" == "1" ]]; then
  export LAB_DEBUG_WORKER_URL="${LAB_DEBUG_WORKER_URL:-${DEBUG_WORKER_URL_DEFAULT}}"
else
  export LAB_DEBUG_WORKER_URL="${LAB_DEBUG_WORKER_URL:-}"
fi
export LAB_DEBUG_WORKER_TIMEOUT_SECONDS="${LAB_DEBUG_WORKER_TIMEOUT_SECONDS:-${DEBUG_PREWARM_TIMEOUT_SECONDS}}"
DOCKER_CMD=()
DOCKER_REQUIRES_HOST_SUDO=0
OLLAMA_CMD=()
OLLAMA_ON_HOST=0
OLLAMA_HOSTPORT="${OLLAMA_HOST}:${ACTIVE_OLLAMA_PORT}"
OLLAMA_BIN="ollama"

chat_pid=""
api_pid=""
debug_worker_pid=""
debug_worker_container_name=""

url_is_localhost() {
  # Detect whether a configured service URL points back to this machine.
  local url="${1:-}"
  case "${url}" in
    http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*|https://localhost:*|http://[::1]:*|https://[::1]:*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ollama_ready_url() {
  # Pick the correct readiness endpoint for either Ollama's native API base or its OpenAI-compatible base.
  local base_url="${1%/}"
  case "${base_url}" in
    */v1)
      printf '%s/models\n' "${base_url}"
      ;;
    */api)
      printf '%s/tags\n' "${base_url}"
      ;;
    *)
      printf '%s/api/tags\n' "${base_url}"
      ;;
  esac
}

run_privileged_cmd() {
  # Run a command with elevation, including from Flatpak dev shells where sudo needs a host PTY.
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return $?
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v sudo >/dev/null 2>&1'; then
    local quoted=""
    local arg
    for arg in "$@"; do
      quoted+=" $(printf '%q' "${arg}")"
    done
    if flatpak-spawn --host sh -lc 'command -v script >/dev/null 2>&1'; then
      flatpak-spawn --host script -qefc "sudo${quoted}" /dev/null
    else
      flatpak-spawn --host sudo "$@"
    fi
    return $?
  fi
  echo "Elevated access is required, but sudo is not available in this shell or on the host." >&2
  return 1
}

cleanup() {
  # Shut down background services in reverse startup order.
  local exit_code=$?
  if [[ -n "${api_pid}" ]] && kill -0 "${api_pid}" 2>/dev/null; then
    kill "${api_pid}" 2>/dev/null || true
    wait "${api_pid}" 2>/dev/null || true
  fi
  if [[ -n "${debug_worker_container_name}" ]] && ((${#DOCKER_CMD[@]} > 0)); then
    "${DOCKER_CMD[@]}" rm -f "${debug_worker_container_name}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${debug_worker_pid}" ]] && kill -0 "${debug_worker_pid}" 2>/dev/null; then
    kill "${debug_worker_pid}" 2>/dev/null || true
    wait "${debug_worker_pid}" 2>/dev/null || true
  fi
  if [[ -n "${chat_pid}" ]] && kill -0 "${chat_pid}" 2>/dev/null; then
    kill "${chat_pid}" 2>/dev/null || true
    wait "${chat_pid}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

init_docker_cmd() {
  # Prefer a local docker binary, but support Flatpak dev shells by hopping to the host.
  if command -v docker >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
    return 0
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v docker >/dev/null 2>&1'; then
    DOCKER_CMD=(flatpak-spawn --host docker)
    return 0
  fi
  echo "Docker CLI is not installed in this shell or on the host." >&2
  return 1
}

ensure_docker_access() {
  # If direct access fails, prompt for elevated access once so later background docker commands can reuse it.
  if "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo "Docker requires elevated access in this shell. You may be prompted for your password."
    sudo -v
    DOCKER_CMD=(sudo docker)
    return 0
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v sudo >/dev/null 2>&1'; then
    echo "Docker requires host elevated access. You may be prompted for your password."
    run_privileged_cmd -v
    DOCKER_CMD=(flatpak-spawn --host sudo docker)
    DOCKER_REQUIRES_HOST_SUDO=1
    return 0
  fi
  echo "Docker is installed but not accessible. Add your user to the docker group or use sudo on the host." >&2
  return 1
}

ensure_docker_ready() {
  # This runner depends on Docker because the local chat server is containerized.
  if "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
    return 0
  fi

  echo "Docker is not ready. Attempting to start the Docker daemon..."
  if command -v systemctl >/dev/null 2>&1; then
    run_privileged_cmd systemctl start docker || true
  elif command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v systemctl >/dev/null 2>&1'; then
    run_privileged_cmd systemctl start docker || true
  elif command -v service >/dev/null 2>&1; then
    run_privileged_cmd service docker start || true
  elif command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v service >/dev/null 2>&1'; then
    run_privileged_cmd service docker start || true
  fi

  for ((i=0; i<20; i++)); do
    if "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
      echo "Docker daemon is ready."
      return 0
    fi
    sleep 1
  done

  if [[ "${DOCKER_REQUIRES_HOST_SUDO}" == "1" ]]; then
    echo "Host Docker is installed, but this Flatpak shell cannot reliably reuse host sudo for repeated Docker commands." >&2
    echo "Add your host user to the docker group, then log out and back in:" >&2
    echo "  flatpak-spawn --host script -qefc \"sudo usermod -aG docker ${USER}\" /dev/null" >&2
  fi
  echo "Docker is still not ready after attempting to start it." >&2
  return 1
}

ensure_chat_image() {
  # Pull the image lazily so warm starts stay fast.
  if ! "${DOCKER_CMD[@]}" image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    echo "Pulling chat server image: ${VLLM_IMAGE}"
    "${DOCKER_CMD[@]}" pull "${VLLM_IMAGE}"
  fi
}

ensure_debug_worker_image() {
  # Build a CUDA-capable debug-worker image on top of a Jetson-native PyTorch container.
  local current_base=""
  if "${DOCKER_CMD[@]}" image inspect "${DEBUG_WORKER_CONTAINER_IMAGE}" >/dev/null 2>&1; then
    current_base="$("${DOCKER_CMD[@]}" image inspect --format '{{ index .Config.Labels "lab.debug_worker.base_image" }}' "${DEBUG_WORKER_CONTAINER_IMAGE}" 2>/dev/null || true)"
    if [[ "${current_base}" == "${DEBUG_WORKER_CONTAINER_BASE_IMAGE}" ]]; then
      return 0
    fi
    echo "Rebuilding debug worker image because the base image changed."
    echo "  current: ${current_base:-unknown}"
    echo "  desired: ${DEBUG_WORKER_CONTAINER_BASE_IMAGE}"
  fi
  echo "Building debug worker image: ${DEBUG_WORKER_CONTAINER_IMAGE}"
  if ! "${DOCKER_CMD[@]}" build \
    --build-arg "BASE_IMAGE=${DEBUG_WORKER_CONTAINER_BASE_IMAGE}" \
    -f "${ROOT}/docker/debug_worker.Dockerfile" \
    -t "${DEBUG_WORKER_CONTAINER_IMAGE}" \
    "${ROOT}"; then
    echo "Failed to build the debug worker image from ${DEBUG_WORKER_CONTAINER_BASE_IMAGE}." >&2
    echo "You can override the base image with LAB_DEBUG_WORKER_BASE_IMAGE if needed." >&2
    return 1
  fi
}

init_ollama_cmd() {
  # Prefer a local Ollama binary, but support Flatpak dev shells by hopping to the host.
  if command -v ollama >/dev/null 2>&1; then
    OLLAMA_BIN="$(command -v ollama)"
    OLLAMA_CMD=(env "OLLAMA_HOST=${OLLAMA_HOSTPORT}" "${OLLAMA_BIN}")
    return 0
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v ollama >/dev/null 2>&1'; then
    OLLAMA_BIN="$(flatpak-spawn --host sh -lc 'command -v ollama')"
    OLLAMA_CMD=(flatpak-spawn --host env "OLLAMA_HOST=${OLLAMA_HOSTPORT}" "${OLLAMA_BIN}")
    OLLAMA_ON_HOST=1
    return 0
  fi
  echo "Ollama CLI is not installed in this shell or on the host." >&2
  echo "Install it on the Jetson host: https://docs.ollama.com/linux" >&2
  return 1
}

start_ollama_service() {
  # Try the system service first because it works better from Flatpak shells.
  if command -v systemctl >/dev/null 2>&1; then
    run_privileged_cmd systemctl start ollama || true
    return 0
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v systemctl >/dev/null 2>&1'; then
    run_privileged_cmd systemctl start ollama || true
    return 0
  fi
  if command -v service >/dev/null 2>&1; then
    run_privileged_cmd service ollama start || true
    return 0
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v service >/dev/null 2>&1'; then
    run_privileged_cmd service ollama start || true
    return 0
  fi
  return 1
}

start_ollama_process() {
  # Fall back to a direct `ollama serve` when no service manager is available.
  local target_log="${1:-${CHAT_LOG}}"
  : >"${target_log}"
  local -a env_args=(
    "OLLAMA_HOST=${OLLAMA_HOSTPORT}"
    "OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}"
    "OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"
    "OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
    "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}"
  )
  if [[ "${OLLAMA_FORCE_CPU}" == "1" ]]; then
    env_args+=("OLLAMA_LLM_LIBRARY=cpu")
  fi
  if [[ "${OLLAMA_ON_HOST}" == "1" ]]; then
    local quoted_log
    local env_string=""
    local env_arg
    for env_arg in "${env_args[@]}"; do
      env_string+=" $(printf '%q' "${env_arg}")"
    done
    quoted_log="$(printf '%q' "${target_log}")"
    echo "Starting Ollama on the host without systemd..." | tee -a "${target_log}"
    flatpak-spawn --host sh -lc "nohup env${env_string} $(printf '%q' "${OLLAMA_BIN}") serve >> ${quoted_log} 2>&1 &"
    echo "Ollama was started as a host background process and will keep running after this script exits." >>"${target_log}"
    return 0
  fi
  echo "Starting Ollama in this shell..." | tee -a "${target_log}"
  env "${env_args[@]}" "${OLLAMA_BIN}" serve >>"${target_log}" 2>&1 &
  chat_pid=$!
}

ensure_ollama_ready() {
  local base_url="${1:-${LAB_CHAT_BASE_URL}}"
  local target_log="${2:-${CHAT_LOG}}"
  local role="${3:-Ollama server}"
  local ready_url
  ready_url="$(ollama_ready_url "${base_url}")"
  : >"${target_log}"
  echo "Backend: ollama" >>"${target_log}"
  echo "Base URL: ${base_url}" >>"${target_log}"
  echo "Ready URL: ${ready_url}" >>"${target_log}"

  if wait_for_http "${ready_url}" "${role}" 1 >/dev/null 2>&1; then
    echo "${role} is already reachable." >>"${target_log}"
    return 0
  fi

  echo "Ollama is not ready. Attempting to start ${role}..."
  echo "Attempting to start Ollama for ${base_url}" >>"${target_log}"
  if [[ "${OLLAMA_FORCE_CPU}" != "1" ]]; then
    start_ollama_service || true
  fi

  if wait_for_http "${ready_url}" "${role}" 30 >/dev/null 2>&1; then
    echo "${role} is ready."
    echo "Ollama is running via the host service. Use \`journalctl -u ollama\` on the host for service logs." >>"${target_log}"
    return 0
  fi

  start_ollama_process "${target_log}"
  wait_for_http "${ready_url}" "${role}" 120
  echo "${role} is ready."
}

ensure_ollama_model() {
  # Pull the requested model lazily so warm starts stay fast.
  local model_name="$1"
  if "${OLLAMA_CMD[@]}" list 2>/dev/null | awk 'NR > 1 { print $1 }' | grep -Fx -- "${model_name}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Pulling Ollama model: ${model_name}"
  "${OLLAMA_CMD[@]}" pull "${model_name}"
}

wait_for_http() {
  # Treat any HTTP response as readiness; downstream routes may legitimately return 4xx.
  local url="$1"
  local label="$2"
  local attempts="${3:-120}"
  for ((i=0; i<attempts; i++)); do
    if "${PY}" - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as resp:
    if 200 <= resp.status < 500:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  echo "${label} did not become ready in time: ${url}" >&2
  return 1
}

check_api_python_deps() {
  # Fail early when the API venv is missing lightweight host-side packages.
  local missing=""
  missing="$("${PY}" - <<'PY'
import importlib.util

required = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "requests": "requests",
    "numpy": "numpy",
    "joblib": "joblib",
}
missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
print(" ".join(missing))
PY
)"
  if [[ -z "${missing// }" ]]; then
    return 0
  fi

  echo "The API environment is missing required Python packages: ${missing}" >&2
  if ! "${PY}" -m pip --version >/dev/null 2>&1; then
    echo "This venv also has no pip. Bootstrap it with:" >&2
    echo "  ${PY} -m ensurepip --upgrade" >&2
  fi
  echo "Install the lightweight host-side API dependencies with:" >&2
  echo "  ${PY} -m pip install fastapi 'uvicorn[standard]' pydantic requests numpy joblib python-dotenv" >&2
  return 1
}

prewarm_debug_runtime() {
  local url="$1"
  local timeout_seconds="$2"
  "${PY}" - "$url" "$timeout_seconds" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
timeout_seconds = int(sys.argv[2])

req = urllib.request.Request(url, method="GET")
try:
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if not (200 <= resp.status < 300):
            raise SystemExit(f"Unexpected HTTP status {resp.status}: {body}")
        try:
            payload = json.loads(body or "{}")
        except json.JSONDecodeError:
            payload = {"raw": body}
        if payload.get("ok") is not True:
            raise SystemExit(f"Debug prewarm returned unexpected payload: {payload}")
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")
    raise SystemExit(f"HTTP {exc.code}: {detail}")
except Exception as exc:
    raise SystemExit(str(exc))
PY
}

start_debug_worker_host() {
  : >"${DEBUG_WORKER_LOG}"
  echo "Starting isolated debug worker on ${LAB_DEBUG_WORKER_URL}" >>"${DEBUG_WORKER_LOG}"
  LAB_DEBUG_WORKER_URL="" \
  CIRCUIT_DEBUG_DEVICE="${DEBUG_WORKER_DEVICE}" \
  CIRCUIT_DEBUG_MERGED_MODEL_DIR="${CIRCUIT_DEBUG_MERGED_MODEL_DIR:-}" \
  CIRCUIT_DEBUG_CPU_DTYPE="${CIRCUIT_DEBUG_CPU_DTYPE}" \
  CIRCUIT_DEBUG_LOCAL_FILES_ONLY="${CIRCUIT_DEBUG_LOCAL_FILES_ONLY}" \
  CIRCUIT_DEBUG_LOW_CPU_MEM_USAGE="${CIRCUIT_DEBUG_LOW_CPU_MEM_USAGE}" \
  CIRCUIT_DEBUG_USE_DEVICE_MAP="${CIRCUIT_DEBUG_USE_DEVICE_MAP}" \
  CIRCUIT_DEBUG_MAX_CPU_MEMORY="${CIRCUIT_DEBUG_MAX_CPU_MEMORY}" \
  CIRCUIT_DEBUG_MAX_GPU_MEMORY="${CIRCUIT_DEBUG_MAX_GPU_MEMORY}" \
  CIRCUIT_DEBUG_OFFLOAD_FOLDER="${CIRCUIT_DEBUG_OFFLOAD_FOLDER}" \
  "${PY}" -m uvicorn server:app --host "${DEBUG_WORKER_HOST}" --port "${DEBUG_WORKER_PORT}" >"${DEBUG_WORKER_LOG}" 2>&1 &
  debug_worker_pid=$!
}

start_debug_worker_container() {
  : >"${DEBUG_WORKER_LOG}"
  debug_worker_container_name="${DEBUG_WORKER_CONTAINER_NAME}"
  "${DOCKER_CMD[@]}" rm -f "${debug_worker_container_name}" >/dev/null 2>&1 || true

  local -a run_cmd=(
    "${DOCKER_CMD[@]}" run --rm
    --name "${debug_worker_container_name}"
    --runtime=nvidia
    --network host
    --ipc=host
    -w "${ROOT}"
    -v "${ROOT}:${ROOT}"
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e HOME=/root
    -e PYTHONUNBUFFERED=1
    -e LAB_DEBUG_WORKER_URL=
    -e LAB_CHAT_BACKEND="${REQUESTED_CHAT_BACKEND}"
    -e LAB_CHAT_BASE_URL="${LAB_CHAT_BASE_URL}"
    -e LAB_OLLAMA_BASE_URL="${LAB_OLLAMA_BASE_URL:-}"
    -e LAB_CHAT_MODEL="${LAB_CHAT_MODEL}"
    -e LAB_CHAT_API_KEY="${LAB_CHAT_API_KEY}"
    -e LAB_MANUAL_VERSION="${LAB_MANUAL_VERSION}"
    -e LAB_EMBED_MODEL="${OLLAMA_EMBED_MODEL}"
    -e LAB_DEBUG_WORKER_TIMEOUT_SECONDS="${LAB_DEBUG_WORKER_TIMEOUT_SECONDS}"
    -e CIRCUIT_DEBUG_DEVICE="${DEBUG_WORKER_DEVICE}"
    -e CIRCUIT_DEBUG_MERGED_MODEL_DIR="${CIRCUIT_DEBUG_MERGED_MODEL_DIR:-}"
    -e CIRCUIT_DEBUG_CPU_DTYPE="${CIRCUIT_DEBUG_CPU_DTYPE}"
    -e CIRCUIT_DEBUG_LOCAL_FILES_ONLY="${CIRCUIT_DEBUG_LOCAL_FILES_ONLY}"
    -e CIRCUIT_DEBUG_LOW_CPU_MEM_USAGE="${CIRCUIT_DEBUG_LOW_CPU_MEM_USAGE}"
    -e CIRCUIT_DEBUG_USE_DEVICE_MAP="${CIRCUIT_DEBUG_USE_DEVICE_MAP}"
    -e CIRCUIT_DEBUG_MAX_CPU_MEMORY="${CIRCUIT_DEBUG_MAX_CPU_MEMORY}"
    -e CIRCUIT_DEBUG_MAX_GPU_MEMORY="${CIRCUIT_DEBUG_MAX_GPU_MEMORY}"
    -e CIRCUIT_DEBUG_OFFLOAD_FOLDER="${CIRCUIT_DEBUG_OFFLOAD_FOLDER}"
  )
  if [[ -d "${HOME}/.cache/huggingface" ]]; then
    run_cmd+=(-v "${HOME}/.cache/huggingface:/root/.cache/huggingface")
  fi
  run_cmd+=(
    "${DEBUG_WORKER_CONTAINER_IMAGE}"
    /bin/sh -lc "python3 -c 'import torch; print(\"Container torch:\", torch.__version__); print(\"CUDA built:\", torch.backends.cuda.is_built()); print(\"CUDA available:\", torch.cuda.is_available()); print(\"CUDA device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"); print(\"CUDA arch list:\", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])' && exec python3 -m uvicorn server:app --host ${DEBUG_WORKER_HOST} --port ${DEBUG_WORKER_PORT}"
  )
  "${run_cmd[@]}" >"${DEBUG_WORKER_LOG}" 2>&1 &
  debug_worker_pid=$!
}

start_debug_worker() {
  if [[ "${DEBUG_WORKER_BACKEND}" == "container" ]]; then
    init_docker_cmd
    ensure_docker_access
    ensure_docker_ready
    ensure_debug_worker_image
    start_debug_worker_container
    return 0
  fi
  start_debug_worker_host
}

echo "Using Python: ${PY}"
echo "Chat backend: ${REQUESTED_CHAT_BACKEND}"
echo "Chat base URL: ${LAB_CHAT_BASE_URL}"
echo "Chat model: ${LAB_CHAT_MODEL}"
if [[ -n "${LAB_CHAT_API_KEY}" ]]; then
  echo "Chat API key: configured"
else
  echo "Chat API key: not configured"
fi
echo "Embed base URL: ${LAB_OLLAMA_BASE_URL}"
echo "Embed model: ${OLLAMA_EMBED_MODEL}"
echo "Lab manual version: ${LAB_MANUAL_VERSION}"
echo "Debug model device: ${CIRCUIT_DEBUG_DEVICE}"
if [[ -n "${CIRCUIT_DEBUG_MERGED_MODEL_DIR:-}" ]]; then
  echo "Debug merged model: ${CIRCUIT_DEBUG_MERGED_MODEL_DIR}"
fi
if [[ "${CIRCUIT_DEBUG_DEVICE}" == "cpu" ]]; then
  echo "Debug CPU dtype: ${CIRCUIT_DEBUG_CPU_DTYPE}"
fi
if [[ "${DEBUG_WORKER_ENABLED}" == "1" ]]; then
  echo "Debug worker: ${LAB_DEBUG_WORKER_URL} (${DEBUG_WORKER_DEVICE}, ${DEBUG_WORKER_BACKEND})"
  echo "Debug max CPU memory: ${CIRCUIT_DEBUG_MAX_CPU_MEMORY}"
  if [[ "${DEBUG_WORKER_DEVICE}" != "cpu" ]]; then
    echo "Debug max GPU memory: ${CIRCUIT_DEBUG_MAX_GPU_MEMORY}"
  fi
  echo "Debug offload folder: ${CIRCUIT_DEBUG_OFFLOAD_FOLDER}"
fi
if [[ "${REQUESTED_CHAT_BACKEND}" == "openai_compat" ]]; then
  echo "vLLM max model len: ${VLLM_MAX_MODEL_LEN}"
  echo "vLLM max num seqs: ${VLLM_MAX_NUM_SEQS}"
  echo "vLLM max num batched tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS}"
elif [[ "${REQUESTED_CHAT_BACKEND}" == "ollama" ]]; then
  echo "Ollama context length: ${OLLAMA_CONTEXT_LENGTH}"
  echo "Ollama keep alive: ${OLLAMA_KEEP_ALIVE}"
fi

check_api_python_deps

if [[ ! -f "${ROOT}/assets/model_bundle.joblib" ]]; then
  echo "Building runtime assets..."
  "${PY}" "${ROOT}/build_runtime_assets.py"
fi

if [[ ! -f "${ROOT}/assets_hybrid/hybrid_config.json" ]]; then
  echo "Building hybrid assets..."
  "${PY}" "${ROOT}/build_hybrid_assets.py"
fi

if [[ "${REQUESTED_CHAT_BACKEND}" == "openai_compat" ]] && [[ "${CHAT_MANAGE_SERVER}" == "0" ]] && [[ -z "${LAB_CHAT_API_KEY}" ]] && [[ "${LAB_CHAT_BASE_URL}" == https://ollama.com/* ]]; then
  echo "LAB_CHAT_API_KEY (or OLLAMA_API_KEY) is required for the default Ollama Cloud chat backend." >&2
  exit 1
fi

if [[ "${REQUESTED_CHAT_BACKEND}" != "ollama" ]] && url_is_localhost "${LAB_OLLAMA_BASE_URL}"; then
  echo "Ensuring local Ollama is ready for embeddings..."
  init_ollama_cmd
  ensure_ollama_ready "${LAB_OLLAMA_BASE_URL}" "${CHAT_LOG}" "Local Ollama embeddings server"
  ensure_ollama_model "${OLLAMA_EMBED_MODEL}"
fi

if [[ "${REQUESTED_CHAT_BACKEND}" == "openai_compat" ]]; then
  if [[ "${CHAT_MANAGE_SERVER}" == "0" ]]; then
    echo "Using an already-running OpenAI-compatible chat server."
    wait_for_http "${LAB_CHAT_BASE_URL}/models" "OpenAI-compatible chat server" "${CHAT_READY_ATTEMPTS}"
    echo "OpenAI-compatible chat server is ready."
  else
    init_docker_cmd
    ensure_docker_access
    ensure_docker_ready
    ensure_chat_image

    VLLM_CMD=(
      vllm serve "${LAB_CHAT_MODEL}"
      --host "${CHAT_HOST}"
      --port "${CHAT_PORT}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${VLLM_MAX_MODEL_LEN}"
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
      --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    )

    if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
      VLLM_CMD+=(--enforce-eager)
    fi

    if [[ "${VLLM_DISABLE_CASCADE_ATTN}" == "1" ]]; then
      VLLM_CMD+=(--disable-cascade-attn)
    fi

    echo "Starting local chat server..."
    "${DOCKER_CMD[@]}" run --rm \
      --runtime=nvidia \
      --network host \
      "${VLLM_IMAGE}" \
      "${VLLM_CMD[@]}" \
      >"${CHAT_LOG}" 2>&1 &
    chat_pid=$!

    wait_for_http "${LAB_CHAT_BASE_URL}/models" "Local chat server" "${CHAT_READY_ATTEMPTS}"
    echo "Local chat server is ready."
  fi
elif [[ "${REQUESTED_CHAT_BACKEND}" == "ollama" ]]; then
  : >"${CHAT_LOG}"
  echo "Backend: ollama" >>"${CHAT_LOG}"
  echo "Base URL: ${LAB_CHAT_BASE_URL}" >>"${CHAT_LOG}"
  if [[ "${CHAT_MANAGE_SERVER}" == "0" ]]; then
    echo "Using an already-running Ollama server."
    echo "Using an already-running Ollama server." >>"${CHAT_LOG}"
    wait_for_http "${LAB_CHAT_BASE_URL}/models" "Ollama server" "${CHAT_READY_ATTEMPTS}"
    echo "Ollama server is ready."
    echo "Ollama server is ready." >>"${CHAT_LOG}"
  else
    init_ollama_cmd
    ensure_ollama_ready
    ensure_ollama_model "${LAB_CHAT_MODEL}"
    ensure_ollama_model "${OLLAMA_EMBED_MODEL}"
  fi
else
  echo "Using in-process Transformers chat backend; no separate chat server will be started."
fi

if [[ "${DEBUG_WORKER_ENABLED}" == "1" ]]; then
  echo "Starting isolated debug worker..."
  start_debug_worker
  wait_for_http "${LAB_DEBUG_WORKER_URL}/health" "Debug worker" "${DEBUG_WORKER_READY_ATTEMPTS}"
  echo "Debug worker is ready."
  if [[ "${DEBUG_PREWARM}" == "1" ]]; then
    echo "Prewarming isolated debug worker..."
    prewarm_debug_runtime "${LAB_DEBUG_WORKER_URL}/debug/prewarm" "${DEBUG_PREWARM_TIMEOUT_SECONDS}"
    echo "Isolated debug worker is prewarmed."
  fi
fi

echo "Starting FastAPI..."
"${PY}" -m uvicorn server:app --host "${API_HOST}" --port "${API_PORT}" >"${API_LOG}" 2>&1 &
api_pid=$!

wait_for_http "http://${API_HOST}:${API_PORT}/health" "FastAPI server" "${API_READY_ATTEMPTS}"
echo "Backend ready:"
echo "  API:  http://${API_HOST}:${API_PORT}"
if [[ "${REQUESTED_CHAT_BACKEND}" == "transformers" ]]; then
  echo "  Chat: in-process (${LAB_CHAT_MODEL})"
else
  echo "  Chat: ${LAB_CHAT_BASE_URL}"
fi
if [[ "${DEBUG_WORKER_ENABLED}" == "1" ]]; then
  echo "  Debug: ${LAB_DEBUG_WORKER_URL}"
fi
echo "Logs:"
echo "  ${CHAT_LOG}"
echo "  ${API_LOG}"
if [[ "${DEBUG_WORKER_ENABLED}" == "1" ]]; then
  echo "  ${DEBUG_WORKER_LOG}"
fi

wait "${api_pid}"
