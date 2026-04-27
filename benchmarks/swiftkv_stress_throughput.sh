#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Optimization #1 (SwiftKV / KV sharing): measure throughput under *stress*
# so KV-cache savings translate into observable tokens/sec — higher batch
# pressure, longer generations, and optionally a fixed KV budget copied from a
# prior engine log ("Current kv cache memory in use is N bytes" / suggested
# `--kv-cache-memory-bytes=N`).
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   export VLLM_CACHE_ROOT="$HOME/.cache/vllm"   # must be writable
#   ./benchmarks/swiftkv_stress_throughput.sh /path/to/swiftkv_checkpoint
#
# Environment (optional):
#   NUM_PROMPTS           default 1000
#   INPUT_LEN             default 512
#   OUTPUT_LEN            default 512   (stress decode; increase to saturate GPU)
#   SEED                  default 42
#   TENSOR_PARALLEL_SIZE  default 1
#   GPU_MEMORY_UTIL       default 0.90
#   MAX_MODEL_LEN         default 8192
#   KV_CACHE_MEMORY_BYTES unset = auto from gpu_memory_utilization; set to pin KV pool
#   VLLM_ATTENTION_BACKEND  unset = FlashAttention; FLASHINFER = experimental
#
# FlashInfer (pip: flashinfer-python, flashinfer-cubin): ensure writable caches.
# Override with FLASHINFER_CACHE_DIR / CUTE_DSL_CACHE_DIR / TMPDIR if needed.
#
# Attention backend (V1): default is unset → FlashAttention (`FLASH_ATTN_VLLM_V1`).
# Optional: `export VLLM_ATTENTION_BACKEND=FLASHINFER` for FlashInfer *attention*
# (requires compatible flashinfer-python + vLLM; Qwen3-8B + this fork currently
# fails kernel warmup on sm_scale / decode_wrapper state — use FA until fixed).
#
set -euo pipefail

MODEL="${1:?Usage: $0 <model_path_or_hf_id>}"

NUM_PROMPTS="${NUM_PROMPTS:-1000}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
SEED="${SEED:-42}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEM="${GPU_MEMORY_UTIL:-0.90}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

extra_args=()
if [[ -d "$MODEL" ]] || [[ "$MODEL" == /* ]]; then
  extra_args+=(--trust-remote-code)
fi

kv_args=()
if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
  kv_args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi

export VLLM_USE_V1="${VLLM_USE_V1:-1}"

_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME:-.}/.cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${_CACHE_ROOT}/flashinfer}"
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-${_CACHE_ROOT}/cutlass_dsl}"
mkdir -p "${FLASHINFER_CACHE_DIR}" "${CUTE_DSL_CACHE_DIR}"

echo "[swiftkv_stress_throughput] repo: ${REPO_ROOT}"
echo "[swiftkv_stress_throughput] model: ${MODEL}"
echo "[swiftkv_stress_throughput] attention_backend=${VLLM_ATTENTION_BACKEND:-<default FlashAttention>}"
echo "[swiftkv_stress_throughput] prompts=${NUM_PROMPTS} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN} tp=${TP} gpu_mem=${GPU_MEM}"
if [[ "${#kv_args[@]}" -gt 0 ]]; then
  echo "[swiftkv_stress_throughput] pinning KV cache: ${KV_CACHE_MEMORY_BYTES} bytes"
else
  echo "[swiftkv_stress_throughput] KV cache: auto (set KV_CACHE_MEMORY_BYTES to pin after reading engine logs)"
fi

cd "${REPO_ROOT}"
exec vllm bench throughput \
  --model "${MODEL}" \
  "${extra_args[@]}" \
  --backend vllm \
  --dataset-name random \
  --num-prompts "${NUM_PROMPTS}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --tensor-parallel-size "${TP}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_LEN}" \
  --seed "${SEED}" \
  "${kv_args[@]}" \
  "${@:2}"
