#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to a local DeepSeek-V4-Flash-DSpark snapshot}"
: "${CACHE_DIR:?set CACHE_DIR to a persistent, image-specific cache directory}"

IMAGE=${IMAGE:-ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:dspark}
DSPARK_TOKENS=${DSPARK_TOKENS:-5}
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-5}
DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-probabilistic}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1032192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.944}
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}

if ! [[ "$DSPARK_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DSPARK_TOKENS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$DSPARK_BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "DSPARK_BLOCK_SIZE must be a positive integer" >&2
  exit 2
fi
if (( DSPARK_TOKENS < DSPARK_BLOCK_SIZE )); then
  echo "DSPARK_TOKENS must be at least the checkpoint block size ($DSPARK_BLOCK_SIZE)" >&2
  exit 2
fi
if ! [[ "$MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]] || (( MAX_NUM_SEQS > 32 )); then
  echo "MAX_NUM_SEQS must be an integer from 1 through 32" >&2
  exit 2
fi
if [[ "$DRAFT_SAMPLE_METHOD" != "greedy" \
      && "$DRAFT_SAMPLE_METHOD" != "probabilistic" ]]; then
  echo "DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"

speculative_config=$(printf \
  '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' \
  "$DSPARK_TOKENS" "$DRAFT_SAMPLE_METHOD")

# DSpark's draft model schedules DSPARK_TOKENS query slots per request, while
# the target verification path schedules the current token plus those draft
# slots. Cover both batch * DSPARK_TOKENS and batch * (DSPARK_TOKENS + 1): a
# draft-only graph list leaves target verification eager at larger batches.
# Optimize the intended agentic surface (every C1-C8 point) and retain C16/C32
# guardrails without spending scarce DSpark memory on every intermediate shape.
logical_capture_sizes=(1 2 3 4 5 6 7 8 16 32)
draft_capture_factor=$DSPARK_TOKENS
target_capture_factor=$((DSPARK_TOKENS + 1))
capture_sizes=()
found_max=0
for batch in "${logical_capture_sizes[@]}"; do
  if (( batch <= MAX_NUM_SEQS )); then
    capture_sizes+=(
      "$((batch * draft_capture_factor))"
      "$((batch * target_capture_factor))"
    )
  fi
  if (( batch == MAX_NUM_SEQS )); then
    found_max=1
  fi
done
if (( found_max == 0 )); then
  capture_sizes+=(
    "$((MAX_NUM_SEQS * draft_capture_factor))"
    "$((MAX_NUM_SEQS * target_capture_factor))"
  )
fi
mapfile -t capture_sizes < <(
  printf '%s\n' "${capture_sizes[@]}" | sort -n -u
)
capture_sizes_csv=$(IFS=,; printf '%s' "${capture_sizes[*]}")
max_cudagraph_capture_size=$((MAX_NUM_SEQS * target_capture_factor))
compilation_config=$(printf \
  '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[%s]}' \
  "$capture_sizes_csv")

exec docker run --rm \
  --name dsv4-sm120-dspark \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --publish 8000:8000 \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$CACHE_DIR:/cache" \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  --env NCCL_PROTO=LL,LL128,Simple \
  --env NCCL_IB_DISABLE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env VLLM_USE_V2_MODEL_RUNNER=1 \
  --env VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  --env VLLM_ALLREDUCE_USE_FLASHINFER=1 \
  --env VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  --env VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel \
  --env VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  --env VLLM_CACHE_ROOT=/cache/vllm \
  --env TRITON_CACHE_DIR=/cache/triton \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  --env TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  --env TILELANG_CACHE_DIR=/cache/tilelang \
  --env TVM_CACHE_DIR=/cache/tvm \
  --env FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  --env CUDA_CACHE_PATH=/cache/jit/nv-compute \
  --env XDG_CACHE_HOME=/cache \
  "$IMAGE" \
  /model \
  --served-model-name DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format auto \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-cudagraph-capture-size "$max_cudagraph_capture_size" \
  --speculative-config "$speculative_config" \
  --compilation-config "$compilation_config" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --enable-request-id-headers \
  --enable-flashinfer-autotune \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kernel-config.moe_backend flashinfer_cutlass \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high
