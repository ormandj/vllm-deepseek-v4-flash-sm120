#!/usr/bin/env python3
import json
import re
from pathlib import Path


root = Path(__file__).resolve().parent.parent
lock = json.loads((root / "stack.lock.json").read_text())
errors = []

vllm = lock["vllm"]
for key in ("base_commit", "native_wheel_commit"):
    if not re.fullmatch(r"[0-9a-f]{40}", vllm[key]):
        errors.append(f"vllm {key} must be a full commit hash: {vllm[key]}")
for path in sorted((root / "patches/vllm").glob("*.patch")):
    relative = path.relative_to(root)
    for line in path.read_text().splitlines():
        if not line.startswith(("--- a/", "+++ b/")):
            continue
        changed_path = line[6:]
        safe_python = (
            changed_path.endswith(".py")
            and changed_path.startswith(("vllm/", "tests/"))
            and changed_path != "vllm/envs.py"
        )
        if not (safe_python or changed_path == "requirements/cuda.txt"):
            errors.append(
                "precompiled native wheel is unsafe with this vLLM "
                f"change: {relative}: {changed_path}"
            )

if errors:
    raise SystemExit("\n".join(errors))

print("verified locked vLLM commits and native-wheel-safe carries")
