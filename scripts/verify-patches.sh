#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vllm_commit=$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["vllm"]["base_commit"])')
native_wheel_commit=$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["vllm"]["native_wheel_commit"])')
flashinfer_base=$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["flashinfer"]["package_base"])')
flashinfer_cubin=$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["flashinfer"]["package_cubin"])')

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$work/vllm"
git -C "$work/vllm" merge-base --is-ancestor "$native_wheel_commit" "$vllm_commit" || {
  echo "native wheel commit is not an ancestor of the vLLM source commit" >&2
  exit 1
}

native_changes=()
while IFS= read -r path; do
  case "$path" in
    csrc/*|cmake/*|rust/*|third_party/*|CMakeLists.txt|setup.py|pyproject.toml|build_rust.sh|rust-toolchain.toml|use_existing_torch.py|requirements/*|tools/build_*.py|tools/setup_deepgemm_pythons.sh|vllm/envs.py|.gitmodules)
      native_changes+=("$path")
      ;;
  esac
done < <(git -C "$work/vllm" diff --name-only "$native_wheel_commit..$vllm_commit")
if (( ${#native_changes[@]} > 0 )); then
  printf 'native build inputs changed after %s:\n' "$native_wheel_commit" >&2
  printf '  %s\n' "${native_changes[@]}" >&2
  exit 1
fi

for profile in control mtp dspark; do
  git -C "$work/vllm" worktree add --detach "$work/$profile" "$vllm_commit"
  "$repo/scripts/apply-build-profile.sh" "$profile" "$work/$profile" "$repo"
  git -C "$work/$profile" diff --check
done

python3 -m pip download \
  --dest "$work/wheel" \
  --index-url https://flashinfer.ai/whl \
  --platform manylinux_2_28_x86_64 \
  --python-version 312 \
  --only-binary=:all: \
  --no-deps \
  "$flashinfer_base" \
  "$flashinfer_cubin"
python_wheels=("$work"/wheel/flashinfer_python-*.whl)
[[ ${#python_wheels[@]} -eq 1 ]]
python3 -m zipfile -e "${python_wheels[0]}" "$work/site"
for patch_file in "$repo"/patches/flashinfer/*.patch; do
  patch -p1 -d "$work/site" --forward --no-backup-if-mismatch < "$patch_file"
done

echo "all profiles apply cleanly to $vllm_commit using native wheel $native_wheel_commit, $flashinfer_base, and $flashinfer_cubin"
