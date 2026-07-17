#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vllm_checkout=${1:-"$repo/../vllm"}
wheel_base_url=${VLLM_WHEEL_BASE_URL:-https://wheels.vllm.ai}

git -C "$vllm_checkout" fetch origin main

commit=
while IFS= read -r candidate; do
  metadata_url="$wheel_base_url/$candidate/cu130/vllm/metadata.json"
  if curl --fail --silent --show-error --output /dev/null "$metadata_url" 2>/dev/null; then
    commit=$candidate
    break
  fi
done < <(git -C "$vllm_checkout" rev-list --first-parent --max-count=100 origin/main)

if [[ -z "$commit" ]]; then
  echo "no published cu130 wheel found in the latest 100 upstream main commits" >&2
  exit 1
fi

python3 - "$repo/stack.lock.json" "$commit" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
commit = sys.argv[2]
lock = json.loads(path.read_text())
changed = (
    lock["vllm"]["base_commit"] != commit
    or lock["vllm"]["native_wheel_commit"] != commit
)
lock["vllm"]["base_commit"] = commit
lock["vllm"]["native_wheel_commit"] = commit
if changed:
    lock["control_image"]["digest"] = None
    lock["control_image"]["vllm_commit"] = None
    lock["control_image"]["flashinfer_base"] = None
    lock["control_image"]["flashinfer_cubin"] = None
path.write_text(json.dumps(lock, indent=2) + "\n")
PY

bash "$repo/scripts/verify-native-wheel.sh" "$commit"
echo "locked vLLM source and native wheel to $commit"
