#!/usr/bin/env python3
import io
from pathlib import Path
from unittest.mock import patch

import flashinfer
import vllm
import deep_gemm
from flashinfer.comm.cuda_ipc import find_loaded_library
from vllm.utils.deep_gemm import _import_deep_gemm


site_packages = Path(vllm.__file__).parent.parent
assert _import_deep_gemm() is deep_gemm

allreduce_selector = (
    site_packages
    / "vllm/compilation/passes/fusion/allreduce_rms_fusion.py"
).read_text()
assert "120: {" in allreduce_selector
assert "121: {" in allreduce_selector

kv_cache_utils = (site_packages / "vllm/v1/core/kv_cache_utils.py").read_text()
assert "num_blocks_per_request = sum(" in kv_cache_utils

flashinfer_root = Path(flashinfer.__file__).parent
comm = (flashinfer_root / "jit/comm.py").read_text()
assert "supported_major_versions=[9, 10, 12]" in comm
trtllm_source = (flashinfer_root / "data/csrc/trtllm_allreduce.cu").read_text()
assert "i + MAX_RANKS_PER_NODE * 2" in trtllm_source

mxfp4 = (
    site_packages / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
).read_text()
assert "_swapped_weights_to_flashinfer_cutlass_kernel_format" not in mxfp4
attention = (site_packages / "vllm/models/deepseek_v4/attention.py").read_text()
assert "resolve_layer_compress_ratio" not in attention
sparse_mla = (flashinfer_root / "mla/_sparse_mla_sm120.py").read_text()
assert "(8, 256)" not in sparse_mla
sparse_prefill = (
    flashinfer_root / "data/csrc/sparse_mla_sm120_prefill.cu"
).read_text()
assert "else if (topk == 256)" not in sparse_prefill

maps = "".join(
    (
        "1000-2000 r-xp 0 00:00 0 /opt/tilelang/lib/libcudart_stub.so\n",
        "2000-3000 r-xp 0 00:00 0 /usr/local/cuda/lib64/libcudart.so.13\n",
    )
)
with patch("builtins.open", return_value=io.StringIO(maps)):
    assert find_loaded_library("libcudart") == "/usr/local/cuda/lib64/libcudart.so.13"

print("verified DeepGEMM stack runtime carries")
