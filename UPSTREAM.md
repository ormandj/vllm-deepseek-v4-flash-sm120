# Upstream merge map

The fixes are independently mergeable. They form one useful SM120
DeepSeek-V4-Flash stack, but they are not presented as an all-or-nothing patch
series.

| Path | Upstream work | Why it matters | Merge ask |
|---|---|---|---|
| DeepGEMM SM120 decode | [DeepGEMM #380](https://github.com/deepseek-ai/DeepGEMM/pull/380) | Reduces padded grouped-GEMM I/O and adds 32x64 decode tiles for the RTX PRO 6000 Blackwell path. | Keep experimental: its exact head loses the matched end-to-end serving matrix to FlashInfer CUTLASS and still errors on a valid small-N shape instead of falling back. |
| FlashInfer packaging | Merged [vLLM #47669](https://github.com/vllm-project/vllm/pull/47669) plus the exact July 18 nightly carry | #47669 supplies the official package index and matched 0.6.14 baseline. This integration advances to `nightly-v0.6.15-20260718`, gaining merged #3948 and #3970 while retaining the required sparse-MLA API. | Drop the version carry when vLLM pins this snapshot or a newer compatible FlashInfer release. |
| MXFP4 MoE | [vLLM #48303](https://github.com/vllm-project/vllm/pull/48303) | Enables the explicitly selected FlashInfer CUTLASS backend for DeepSeek-family MXFP4 and removes gpt-oss-specific activation constants from other models. | Review, `ready` label, full CI, merge. |
| DSpark draft semantics | [vLLM #48304](https://github.com/vllm-project/vllm/pull/48304) | Honors the checkpoint's in-range draft `compress_ratios=0` entry without violating KV arithmetic; preserves fallback behavior. | Review, `ready` label, full CI, merge. |
| KV reporting | [vLLM #48317](https://github.com/vllm-project/vllm/pull/48317) | Makes packed/hybrid capacity reporting agree across worker and scheduler shapes. Request execution is unchanged. | Review, `ready` label, full CI, merge. |
| DSpark TOPK=256 | [FlashInfer #3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) + [#3834](https://github.com/flashinfer-ai/flashinfer/pull/3834) | Adds the missing decode and prefill instantiations used by the DSpark draft. Both halves are required. | Authorized CI, review, merge both. |
| SM12x all-reduce | [FlashInfer #3903](https://github.com/flashinfer-ai/flashinfer/pull/3903) | Adds the SM120/121 TensorRT-LLM workspace and corrects the legacy Lamport pointer layout. | Authorized CI, review, merge. |
| CUDA runtime resolution | [FlashInfer #3930](https://github.com/flashinfer-ai/flashinfer/pull/3930) + [signed follow-up](https://github.com/ormandj/flashinfer/commit/bd6765dea271b23a579938132f8ca1b9cbf6a2a5) | Prevents a loaded `libcudart_stub.so` from being accepted as the CUDA runtime during workspace initialization. | Adopt the exact matcher/tests, run authorized CI, merge. |

## Dependency order

1. #47669 is merged; it is no longer carried as a patch.
2. #3817 and #3834 are independent functional kernel additions and can merge
   now.
3. #3903 can merge independently in FlashInfer.
4. #3930 can merge independently; its resolver is a practical initialization
   prerequisite in environments that load TileLang's CUDA stub.
5. The temporary vLLM SM12x all-reduce selector in this repository should be
   proposed only after vLLM can pin a FlashInfer release containing #3903 and
   guard the capability/version boundary.
6. The three vLLM PRs are otherwise independent of the FlashInfer merge order.

## Selected upstream snapshot

- vLLM source and cu130 native wheel are both
  `c233d90aa826df072872df47b201450059be8e71`. Merged #48660, #48641, and
  #48012 are therefore upstream inputs, not local carries.
- FlashInfer packages are the July 18 nightly at
  `82784bb112c32bb38e4d7ee171eab4855ad4f91a`. Merged #3948 and #3970 are
  included by the package snapshot.
- #48303, #48304, #48317, #3817, #3834, #3903, and the strengthened #3930
  resolver are not present in those selected inputs and remain explicit,
  removable profile carries.

## Objective evidence

- #48303 has isolated current-main MTP:0 attribution on identical hardware,
  source, model, server geometry, and client.
- #3903 has joint-stack evidence explicitly labeled as joint attribution; no
  isolated percentage is assigned to the FlashInfer PR alone.
- #3817/#3834 are supported by dispatch/reference correctness and successful
  DSpark serving, not an isolated throughput claim.
- #3930 is initialization evidence, not a performance claim.
- #48317 is reporting/accounting evidence, not a performance claim.
- #48304's acceptance figures are the same-configuration ablation reported on
  that PR.

The public images make the combined integration easy to reproduce. They do not
replace the narrower evidence already attached to each upstream review.
