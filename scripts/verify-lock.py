#!/usr/bin/env python3
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
for path in sorted((root / "patches/vllm").glob("*.patch")):
    relative = path.relative_to(root)
    for line in path.read_text().splitlines():
        if line.startswith("+++ b/") and not line[6:].endswith(".py"):
            errors.append(
                "precompiled native wheel is unsafe with non-Python vLLM "
                f"change: {relative}: {line[6:]}"
            )

if errors:
    raise SystemExit("\n".join(errors))

print("verified exact native-wheel commit and Python-only vLLM carries")
