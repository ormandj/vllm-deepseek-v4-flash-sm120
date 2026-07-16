#!/usr/bin/env python3
import re
import sys
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} PATH_PREFIX PATCH")

prefix = sys.argv[1].lstrip("/")
patch = Path(sys.argv[2]).read_text()
selected = []

for block in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
    match = re.match(r"diff --git a/(\S+) b/(\S+)\n", block)
    if match and match.group(2).startswith(prefix):
        selected.append(block)

if not selected:
    raise SystemExit(f"no paths under {prefix!r} in {sys.argv[2]}")

sys.stdout.write("".join(selected))
