#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to a local DeepSeek-V4-Flash snapshot}"
: "${CACHE_DIR:?set CACHE_DIR to a persistent, image-specific cache directory}"

IMAGE=${IMAGE:-ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:agentic-mtp0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1032192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.944}
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}

mkdir -p "$CACHE_DIR"

exec docker run --rm \
  --name dsv4-sm120-agentic-mtp0 \
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
  --max-cudagraph-capture-size 48 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,5,6,8,10,12,15,16,18,20,24,25,30,32,36,40,48]}' \
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
