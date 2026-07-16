#!/usr/bin/env python3
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import deep_gemm
import torch
from vllm.utils.deep_gemm import _import_deep_gemm


expected_commit = os.environ["DEEPGEMM_EXPECTED_COMMIT"]
expected_version = os.environ["DEEPGEMM_EXPECTED_VERSION"]

assert sys.version_info[:2] == (3, 12)
assert torch.__version__.startswith("2.11.0")
assert torch.version.cuda == "13.0"
assert importlib.metadata.version("deep-gemm") == expected_version
assert expected_version.endswith(f"+{expected_commit[:7]}")
assert "cpython-312" in deep_gemm._C.__file__
assert _import_deep_gemm() is deep_gemm

required = {
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "set_mk_alignment_for_contiguous_layout",
}
missing = sorted(name for name in required if not hasattr(deep_gemm, name))
assert not missing, missing

package_root = Path(deep_gemm.__file__).parent
kernel = package_root / "include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh"
kernel_source = kernel.read_text()
assert "kACpAsync" in kernel_source
assert "cpasync_load_rows" in kernel_source

torch_library_path = Path(torch.__file__).parent / "lib"
ldd_environment = os.environ.copy()
ldd_environment["LD_LIBRARY_PATH"] = ":".join(
    filter(None, (str(torch_library_path), os.environ.get("LD_LIBRARY_PATH")))
)
linked_libraries = subprocess.check_output(
    ["ldd", deep_gemm._C.__file__],
    env=ldd_environment,
    text=True,
    stderr=subprocess.STDOUT,
)
assert "not found" not in linked_libraries, linked_libraries

print(f"deep_gemm {expected_version}: {deep_gemm.__file__}")
print(f"deep_gemm extension: {deep_gemm._C.__file__}")
