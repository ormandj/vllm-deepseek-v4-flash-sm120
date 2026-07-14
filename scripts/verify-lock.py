#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent.parent
lock = json.loads((root / "stack.lock.json").read_text())
errors = []

vllm = lock["vllm"]
if vllm["native_wheel_commit"] != vllm["base_commit"]:
    errors.append(
        "native wheel commit must equal the pinned vLLM source commit: "
        f"{vllm['native_wheel_commit']} != {vllm['base_commit']}"
    )
expected_metadata = (
    f"https://wheels.vllm.ai/{vllm['native_wheel_commit']}/vllm/metadata.json"
)
if vllm["native_wheel_metadata"] != expected_metadata:
    errors.append(
        "native wheel metadata URL does not match native_wheel_commit: "
        f"{vllm['native_wheel_metadata']} != {expected_metadata}"
    )

for relative, expected in lock["patch_sha256"].items():
    path = root / relative
    if not path.is_file():
        errors.append(f"missing: {relative}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        errors.append(f"hash mismatch: {relative}: {actual} != {expected}")
    if relative.startswith("patches/vllm/"):
        for line in path.read_text().splitlines():
            if line.startswith("+++ b/") and not line[6:].endswith(".py"):
                errors.append(
                    "precompiled native wheel is unsafe with non-Python vLLM "
                    f"change: {relative}: {line[6:]}"
                )

if errors:
    raise SystemExit("\n".join(errors))

print(
    f"verified {len(lock['patch_sha256'])} patch hashes, exact native-wheel "
    "commit, and Python-only vLLM carries"
)
