#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PROFILE [IMAGE]" >&2
  exit 2
fi

profile=$1
image=${2:-ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:$profile}
container_engine=${CONTAINER_ENGINE:-podman}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vllm_commit=$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["vllm"]["base_commit"])')
integration_commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || printf unknown)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$work/vllm"
git -C "$work/vllm" checkout "$vllm_commit"
"$repo/scripts/apply-build-profile.sh" \
  "$profile" "$work/vllm" "$repo" "$work/profile.env"

patch_manifest=$(sed -n 's/^PATCH_MANIFEST=//p' "$work/profile.env")
[[ -n "$patch_manifest" ]] || { echo "missing PATCH_MANIFEST" >&2; exit 1; }

"$container_engine" build \
  --file "$repo/Containerfile" \
  --target vllm-openai \
  --build-arg max_jobs="${MAX_JOBS:-$(nproc)}" \
  --build-arg nvcc_threads="${NVCC_THREADS:-1}" \
  --build-arg SECURITY_REFRESH="$(date +%Y%m%d)" \
  --build-arg VLLM_BUILD_COMMIT="$vllm_commit" \
  --build-arg VLLM_BUILD_PIPELINE="${BUILD_PIPELINE:-local}" \
  --build-arg VLLM_BUILD_URL="${BUILD_URL:-https://github.com/ormandj/vllm-deepseek-v4-flash-sm120}" \
  --build-arg VLLM_IMAGE_TAG="$image" \
  --build-arg INTEGRATION_BUILD_PROFILE="$profile" \
  --build-arg INTEGRATION_PATCH_MANIFEST="$patch_manifest" \
  --build-arg INTEGRATION_BUILD_COMMIT="$integration_commit" \
  --tag "$image" \
  "$work/vllm"
