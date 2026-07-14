#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PROFILE VLLM_SOURCE REPOSITORY OUTPUT_ENV" >&2
  exit 2
fi

profile=$1
vllm_source=$2
repository=$3
output_env=$4

vllm_patches=()
flashinfer_patches=()

case "$profile" in
  control)
    ;;
  agentic-mtp0)
    vllm_patches=(
      vllm-48303-mxfp4-flashinfer-cutlass.patch
      vllm-sm12x-flashinfer-allreduce-selector.patch
      vllm-48317-kv-capacity-reporting.patch
    )
    flashinfer_patches=(
      fi-3903-sm12x-allreduce.patch
      fi-3930-cuda-runtime-resolver.patch
    )
    ;;
  dspark-preview)
    vllm_patches=(
      vllm-48303-mxfp4-flashinfer-cutlass.patch
      vllm-48304-mtp-draft-rope.patch
      vllm-sm12x-flashinfer-allreduce-selector.patch
      vllm-48317-kv-capacity-reporting.patch
    )
    flashinfer_patches=(
      fi-3817-sm120-topk256-decode.patch
      fi-3834-sm120-topk256-prefill.patch
      fi-3903-sm12x-allreduce.patch
      fi-3930-cuda-runtime-resolver.patch
    )
    ;;
  *)
    echo "unknown build profile: $profile" >&2
    exit 2
    ;;
esac

mkdir -p "$vllm_source/patches-flashinfer"

vllm_manifest=()
for name in "${vllm_patches[@]}"; do
  path="$repository/patches/vllm/$name"
  [[ -f "$path" ]] || { echo "missing vLLM patch: $path" >&2; exit 1; }
  echo "Applying vLLM patch $name" >&2
  git -C "$vllm_source" apply --check "$path"
  git -C "$vllm_source" apply "$path"
  digest=$(sha256sum "$path" | awk '{print $1}')
  vllm_manifest+=("$name@sha256:$digest")
done

flashinfer_manifest=()
for name in "${flashinfer_patches[@]}"; do
  path="$repository/patches/flashinfer/$name"
  [[ -f "$path" ]] || { echo "missing FlashInfer patch: $path" >&2; exit 1; }
  echo "Staging FlashInfer patch $name" >&2
  cp "$path" "$vllm_source/patches-flashinfer/$name"
  digest=$(sha256sum "$path" | awk '{print $1}')
  flashinfer_manifest+=("$name@sha256:$digest")
done

join_manifest() {
  local joined
  if [[ $# -eq 0 ]]; then
    printf 'none'
    return
  fi
  joined=$(IFS=,; echo "$*")
  printf '%s' "$joined"
}

vllm_value=$(join_manifest "${vllm_manifest[@]}")
flashinfer_value=$(join_manifest "${flashinfer_manifest[@]}")
{
  printf 'BUILD_PROFILE=%s\n' "$profile"
  printf 'VLLM_PATCH_MANIFEST=%s\n' "$vllm_value"
  printf 'FLASHINFER_PATCH_MANIFEST=%s\n' "$flashinfer_value"
  printf 'PATCH_MANIFEST=vllm:%s;flashinfer:%s\n' \
    "$vllm_value" "$flashinfer_value"
} > "$output_env"
