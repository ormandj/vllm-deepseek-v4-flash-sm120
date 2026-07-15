#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
commit=${1:-$(cd "$repo" && python3 -c 'import json; print(json.load(open("stack.lock.json"))["vllm"]["native_wheel_commit"])')}
base_url=${VLLM_WHEEL_BASE_URL:-https://wheels.vllm.ai}
metadata=$(mktemp)
trap 'rm -f "$metadata"' EXIT

for path in "$commit/cu130/vllm/metadata.json" "$commit/vllm/metadata.json"; do
  metadata_url="$base_url/$path"
  if ! curl --fail --silent --show-error --output "$metadata" "$metadata_url"; then
    continue
  fi
  if ! wheel_url=$(python3 -c '
import json
import sys
from urllib.parse import urljoin

metadata_url, path = sys.argv[1:]
entries = json.load(open(path))
wheel = next(
    entry for entry in entries
    if entry["platform_tag"] == "manylinux_2_28_x86_64"
)
print(urljoin(metadata_url, wheel["path"]))
' "$metadata_url" "$metadata"); then
    continue
  fi
  if curl --fail --silent --show-error --head --output /dev/null "$wheel_url"; then
    echo "verified native wheel: $wheel_url"
    exit 0
  fi
done

echo "no published native wheel metadata for vLLM commit $commit" >&2
exit 1
