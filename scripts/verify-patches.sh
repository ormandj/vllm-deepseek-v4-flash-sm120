#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vllm_commit=$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["vllm"]["base_commit"])')

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$work/vllm"
for profile in control mtp dspark; do
  git -C "$work/vllm" worktree add --detach "$work/$profile" "$vllm_commit"
  "$repo/scripts/apply-build-profile.sh" \
    "$profile" "$work/$profile" "$repo" "$work/$profile.env"
  git -C "$work/$profile" diff --check
done

python3 -m pip download \
  --dest "$work/wheel" \
  --index-url https://flashinfer.ai/whl \
  --platform manylinux_2_28_x86_64 \
  --python-version 312 \
  --only-binary=:all: \
  --no-deps \
  flashinfer-python==0.6.14 \
  flashinfer-cubin==0.6.14
python_wheels=("$work"/wheel/flashinfer_python-*.whl)
[[ ${#python_wheels[@]} -eq 1 ]]
python3 -m zipfile -e "${python_wheels[0]}" "$work/site"
for patch_file in "$repo"/patches/flashinfer/*.patch; do
  patch -p1 -d "$work/site" --forward --no-backup-if-mismatch < "$patch_file"
done

echo "all profiles apply cleanly to $vllm_commit and matched FlashInfer 0.6.14 packages"
