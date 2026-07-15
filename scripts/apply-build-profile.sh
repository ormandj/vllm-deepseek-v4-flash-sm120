#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PROFILE VLLM_SOURCE REPOSITORY" >&2
  exit 2
fi

profile=$1
vllm_source=$2
repository=$3

vllm_patches=()
flashinfer_patches=()

case "$profile" in
  control)
    ;;
  mtp)
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
  dspark)
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

for name in "${vllm_patches[@]}"; do
  path="$repository/patches/vllm/$name"
  [[ -f "$path" ]] || { echo "missing vLLM patch: $path" >&2; exit 1; }
  echo "Applying vLLM patch $name" >&2
  git -C "$vllm_source" apply --check "$path"
  git -C "$vllm_source" apply "$path"
done

for name in "${flashinfer_patches[@]}"; do
  path="$repository/patches/flashinfer/$name"
  [[ -f "$path" ]] || { echo "missing FlashInfer patch: $path" >&2; exit 1; }
  echo "Staging FlashInfer patch $name" >&2
  cp "$path" "$vllm_source/patches-flashinfer/$name"
done
