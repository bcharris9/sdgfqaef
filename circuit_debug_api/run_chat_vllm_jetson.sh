#!/usr/bin/env bash
set -euo pipefail

# Start a single local vLLM server on a Jetson and keep it attached to the terminal.
IMAGE="${LAB_CHAT_VLLM_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
MODEL="${LAB_CHAT_MODEL:-RedHatAI/Qwen2.5-3B-quantized.w4a16}"
HOST="${LAB_CHAT_HOST:-127.0.0.1}"
PORT="${LAB_CHAT_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${LAB_CHAT_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_MAX_MODEL_LEN="${LAB_CHAT_MAX_MODEL_LEN:-1024}"
VLLM_MAX_NUM_SEQS="${LAB_CHAT_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${LAB_CHAT_MAX_NUM_BATCHED_TOKENS:-128}"
VLLM_ENFORCE_EAGER="${LAB_CHAT_ENFORCE_EAGER:-1}"
VLLM_DISABLE_CASCADE_ATTN="${LAB_CHAT_DISABLE_CASCADE_ATTN:-1}"
DOCKER_CMD=()
DOCKER_REQUIRES_HOST_SUDO=0

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
  exit 1
}

init_docker_cmd() {
  # Support both host shells and Flatpak dev shells.
  if command -v docker >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
    return 0
  fi
  if command -v flatpak-spawn >/dev/null 2>&1 && flatpak-spawn --host sh -lc 'command -v docker >/dev/null 2>&1'; then
    DOCKER_CMD=(flatpak-spawn --host docker)
    return 0
  fi
  echo "Docker CLI is not installed in this shell or on the host." >&2
  exit 1
}

ensure_docker_access() {
  # Prompt once for elevated access when direct socket access is unavailable.
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
  exit 1
}

init_docker_cmd
ensure_docker_access

if ! "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
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
  for i in $(seq 1 20); do
    if "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
    if [[ "${DOCKER_REQUIRES_HOST_SUDO}" == "1" ]]; then
      echo "Host Docker is installed, but this Flatpak shell cannot reliably reuse host sudo for repeated Docker commands." >&2
      echo "Add your host user to the docker group, then log out and back in:" >&2
      echo "  flatpak-spawn --host script -qefc \"sudo usermod -aG docker ${USER}\" /dev/null" >&2
    fi
    echo "Docker is still not ready after attempting to start it." >&2
    exit 1
  fi
fi

if ! "${DOCKER_CMD[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
  # Pull lazily so repeat boots reuse the cached image.
  echo "Pulling chat server image: ${IMAGE}"
  "${DOCKER_CMD[@]}" pull "${IMAGE}"
fi

VLLM_CMD=(
  vllm serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
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

echo "Starting vLLM with:"
echo "  model: ${MODEL}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  max model len: ${VLLM_MAX_MODEL_LEN}"
echo "  max num seqs: ${VLLM_MAX_NUM_SEQS}"
echo "  max num batched tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  enforce eager: ${VLLM_ENFORCE_EAGER}"
echo "  disable cascade attn: ${VLLM_DISABLE_CASCADE_ATTN}"

exec "${DOCKER_CMD[@]}" run -it --rm \
  --runtime=nvidia \
  --network host \
  "${IMAGE}" \
  "${VLLM_CMD[@]}"
