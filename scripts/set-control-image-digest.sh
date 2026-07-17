#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 sha256:DIGEST" >&2
  exit 2
fi

digest=$1
if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "invalid control image digest: $digest" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$repo/stack.lock.json" "$digest" <<'PY'
import json
import pathlib
import sys

lock_path = pathlib.Path(sys.argv[1])
digest = sys.argv[2]
lock = json.loads(lock_path.read_text())
lock["control_image"]["digest"] = digest
lock["control_image"]["vllm_commit"] = lock["vllm"]["base_commit"]
lock["control_image"]["flashinfer_base"] = lock["flashinfer"]["package_base"]
lock["control_image"]["flashinfer_cubin"] = lock["flashinfer"]["package_cubin"]
lock_path.write_text(json.dumps(lock, indent=2) + "\n")
PY

echo "locked control image $digest"
