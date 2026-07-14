#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent.parent
lock = json.loads((root / "stack.lock.json").read_text())
errors = []

for relative, expected in lock["patch_sha256"].items():
    path = root / relative
    if not path.is_file():
        errors.append(f"missing: {relative}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        errors.append(f"hash mismatch: {relative}: {actual} != {expected}")

if errors:
    raise SystemExit("\n".join(errors))

print(f"verified {len(lock['patch_sha256'])} patch hashes")
