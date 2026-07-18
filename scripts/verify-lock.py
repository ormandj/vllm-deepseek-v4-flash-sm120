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
if vllm["base_commit"] != vllm["native_wheel_commit"]:
    errors.append(
        "vLLM source and native wheel commits must match; run "
        "scripts/update-vllm-lock.sh"
    )

flashinfer = lock["flashinfer"]
if not re.fullmatch(r"[0-9a-f]{40}", flashinfer["release_commit"]):
    errors.append(
        "flashinfer release_commit must be a full commit hash: "
        f"{flashinfer['release_commit']}"
    )
nightly_match = re.fullmatch(
    r"nightly-v0\.6\.15-(\d{8})", flashinfer["release_tag"]
)
if not nightly_match:
    errors.append(
        "flashinfer release_tag must name an exact 0.6.15 nightly: "
        f"{flashinfer['release_tag']}"
    )
else:
    nightly_date = nightly_match.group(1)
    expected_base = f"flashinfer-python==0.6.15.dev{nightly_date}"
    expected_cubin = f"flashinfer-cubin==0.6.15.dev{nightly_date}"
    if flashinfer["package_base"] != expected_base:
        errors.append(
            "flashinfer package_base does not match release_tag: "
            f"{flashinfer['package_base']}"
        )
    if flashinfer["package_cubin"] != expected_cubin:
        errors.append(
            "flashinfer package_cubin does not match release_tag: "
            f"{flashinfer['package_cubin']}"
        )
if flashinfer["index_url"] != "https://flashinfer.ai/whl/nightly/":
    errors.append(
        "flashinfer index_url must be the official nightly index: "
        f"{flashinfer['index_url']}"
    )
requirements_patch = (
    root
    / "patches/vllm/vllm-flashinfer-0.6.15-nightly-requirements.patch"
).read_text()
for expected_line in (
    f"+--extra-index-url {flashinfer['index_url']}",
    f"+{flashinfer['package_base']}",
    f"+{flashinfer['package_cubin']}",
):
    if expected_line not in requirements_patch.splitlines():
        errors.append(
            "FlashInfer requirements patch does not match stack.lock.json: "
            f"{expected_line}"
        )

control_image = lock["control_image"]
control_digest = control_image["digest"]
control_vllm_commit = control_image.get("vllm_commit")
control_flashinfer_base = control_image.get("flashinfer_base")
control_flashinfer_cubin = control_image.get("flashinfer_cubin")
control_bindings = (
    control_digest,
    control_vllm_commit,
    control_flashinfer_base,
    control_flashinfer_cubin,
)
if any(value is None for value in control_bindings):
    if not all(value is None for value in control_bindings):
        errors.append(
            "control image digest and source bindings must all be set or all be null"
        )
elif not re.fullmatch(r"sha256:[0-9a-f]{64}", control_digest):
    errors.append(f"control image digest is invalid: {control_digest}")
elif control_vllm_commit != vllm["base_commit"]:
    errors.append(
        "control image was built from a different vLLM commit; publish the "
        "new control and run scripts/set-control-image-digest.sh"
    )
elif (
    control_flashinfer_base != lock["flashinfer"]["package_base"]
    or control_flashinfer_cubin != lock["flashinfer"]["package_cubin"]
):
    errors.append(
        "control image was built with different FlashInfer packages; publish "
        "the new control and run scripts/set-control-image-digest.sh"
    )

deepgemm = lock["deepgemm"]
for key in ("base_commit", "commit", "cutlass_commit", "fmt_commit"):
    if not re.fullmatch(r"[0-9a-f]{40}", deepgemm[key]):
        errors.append(f"deepgemm {key} must be a full commit hash: {deepgemm[key]}")
if not deepgemm["version"].endswith(f"+{deepgemm['commit'][:7]}"):
    errors.append("deepgemm version must contain the locked commit suffix")
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
