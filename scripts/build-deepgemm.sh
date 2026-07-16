#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${DEEPGEMM_TARGET:-deepgemm}
case "$target" in
  deepgemm | deepgemm-stack) ;;
  *)
    echo "unsupported DeepGEMM target: $target" >&2
    exit 2
    ;;
esac
if [[ $# -eq 0 ]]; then
  images=("ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:$target")
else
  images=("$@")
fi

read_lock() {
  python3 -c "import json; print(json.load(open('$repo/stack.lock.json'))$1)"
}

control_repository=$(read_lock '["control_image"]["repository"]')
control_digest=$(read_lock '["control_image"]["digest"]')
deepgemm_repository=$(read_lock '["deepgemm"]["repository"]')
deepgemm_commit=$(read_lock '["deepgemm"]["commit"]')
deepgemm_version=$(read_lock '["deepgemm"]["version"]')
deepgemm_cutlass_commit=$(read_lock '["deepgemm"]["cutlass_commit"]')
deepgemm_fmt_commit=$(read_lock '["deepgemm"]["fmt_commit"]')

container_engine=${CONTAINER_ENGINE:-podman}
build_command=("$container_engine" build)
output_args=()
if [[ -n "${BUILDX_BUILDER:-}" ]]; then
  [[ "$container_engine" == docker ]] || {
    echo "BUILDX_BUILDER requires CONTAINER_ENGINE=docker" >&2
    exit 2
  }
  build_command=(docker buildx build --builder "$BUILDX_BUILDER")
  if [[ "${BUILDX_PUSH:-0}" == 1 ]]; then
    output_args=(--push --provenance=false)
  else
    output_args=(--load)
  fi
fi

tag_args=()
for tag in "${images[@]}"; do
  tag_args+=(--tag "$tag")
done

"${build_command[@]}" \
  --platform linux/amd64 \
  --file "$repo/Containerfile.deepgemm" \
  --target "$target" \
  --build-arg CONTROL_IMAGE="$control_repository@$control_digest" \
  --build-arg DEEPGEMM_REPOSITORY="$deepgemm_repository" \
  --build-arg DEEPGEMM_COMMIT="$deepgemm_commit" \
  --build-arg DEEPGEMM_VERSION="$deepgemm_version" \
  --build-arg DEEPGEMM_CUTLASS_COMMIT="$deepgemm_cutlass_commit" \
  --build-arg DEEPGEMM_FMT_COMMIT="$deepgemm_fmt_commit" \
  "${tag_args[@]}" \
  "${output_args[@]}" \
  "$repo"
