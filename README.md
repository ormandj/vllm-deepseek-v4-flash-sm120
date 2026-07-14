# vLLM DeepSeek-V4-Flash on SM120

Reproducible vLLM images for **DeepSeek-V4-Flash on 2× NVIDIA RTX PRO 6000
Blackwell (96 GB, SM120)**.

The optimization target is a single user running agentic workloads: strong
C1/C2/C4/C8 decode, long-context prefill, shared-prefix reuse, and enough
headroom for moderate fan-out. C16/C32 are saturation checks, not the primary
objective.

This repository packages focused open vLLM and FlashInfer fixes while they are
under review upstream. Every source revision and patch digest is pinned in
[`stack.lock.json`](stack.lock.json). Model weights are not included.

## Images

| Tag | Purpose |
|---|---|
| `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:control` | Current pinned vLLM with the same CUDA/Python/FlashInfer toolchain and no source carries. |
| `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:agentic-mtp0` | Current recommended non-speculative stack. |
| `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:dspark-preview` | Adds the current DSpark prerequisites; preview until DSpark-vs-MTP tuning is complete. |

Use the immutable digest published in the release notes for repeatable tests.
The moving tags are convenience aliases.

## Run the current agentic profile

Requirements:

- Linux x86_64;
- 2× RTX PRO 6000 Blackwell with working peer access;
- NVIDIA Container Toolkit;
- a local DeepSeek-V4-Flash snapshot;
- substantial persistent cache space for first-start SM120 JIT artifacts.

```bash
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
CACHE_DIR=/models/cache/dsv4-sm120-agentic-mtp0 \
./examples/serve-agentic-mtp0.sh
```

The tested default exposes one 1,032,192-token request, admits up to 32
sequences, and uses a 4,096-token batch ceiling. Lower `MAX_MODEL_LEN` or
`GPU_MEMORY_UTILIZATION` if your cards have less free memory. The script uses
`NCCL_P2P_LEVEL=SYS`; verify peer access on your host first. If NCCL hangs at
the first collective, stop the server and diagnose P2P rather than benchmarking
the degraded state.

The first start compiles SM120 kernels and is intentionally slow. Reuse the
same cache only with the same image digest and profile.

## What is included

| Component | Role | Evidence type |
|---|---|---|
| [vLLM #48303](https://github.com/vllm-project/vllm/pull/48303) | DeepSeek-family MXFP4 → FlashInfer CUTLASS MoE wiring | Controlled serving A/B |
| [FlashInfer #3903](https://github.com/flashinfer-ai/flashinfer/pull/3903) + temporary vLLM selector | SM120/121 TensorRT-LLM all-reduce | Controlled joint-stack A/B |
| [FlashInfer #3930](https://github.com/flashinfer-ai/flashinfer/pull/3930) + [exact signed follow-up](https://github.com/ormandj/flashinfer/commit/bd6765dea271b23a579938132f8ca1b9cbf6a2a5) | Select the real CUDA runtime instead of TileLang's stub | Live failure/recovery + 12 CPU tests |
| [vLLM #48317](https://github.com/vllm-project/vllm/pull/48317) | Correct packed/hybrid KV-capacity reporting | Source inspection + focused tests; no performance claim |
| [vLLM #48304](https://github.com/vllm-project/vllm/pull/48304) | Honor the DSpark checkpoint's draft-layer rope semantics | Same-configuration acceptance result + focused tests |
| [FlashInfer #3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) and [#3834](https://github.com/flashinfer-ai/flashinfer/pull/3834) | TOPK=256 decode/prefill instantiations required by DSpark on SM120 | Dispatch/reference correctness + serving completion |

The local DeepSeek-V4 compile experiment is deliberately excluded: its
controlled marginal effect was mixed and within measurement noise. This image
is a merge preview for supported fixes, not a bag of every attempted tuning.

See [`UPSTREAM.md`](UPSTREAM.md) for the terse merge map and dependency story.
Live status is tracked in [issue #1](https://github.com/ormandj/vllm-deepseek-v4-flash-sm120/issues/1).

## Evidence already established

The sealed attribution matrix on the previous pinned vLLM main established:

- #48303: sustained MTP:0 decode changed by **+1.21% to +21.77%** across
  C1-C32; exact 8K/64K/128K cold prefill changed by **+1.79% to +3.92%**.
- The complete FlashInfer all-reduce integration stack: sustained MTP:0 decode
  changed by **+2.15% to +5.18%** across C1-C32; exact cold prefill changed by
  **+0.25% to +1.39%**.
- The compile carry did not establish a material independent or additive gain.

Those numbers remain tied to vLLM `fec64fea75103a1490e7fa0874c55a2292c110b1`.
This repository intentionally moved every profile together to the newer pin in
`stack.lock.json`; current-pin results will be published only after the matched
`control`/`agentic-mtp0` images pass the same benchmark contract.

## Build locally

The build is SM120-only and expensive. Podman is expected.

```bash
./scripts/verify-patches.sh
MAX_JOBS=48 ./scripts/build.sh agentic-mtp0 local/dsv4-sm120:agentic-mtp0
```

The image records the vLLM revision, repository revision, profile, and SHA256
of every selected patch in OCI labels and environment variables.

### FlashInfer package selection

The image installs matched `flashinfer-python==0.6.14` and
`flashinfer-cubin==0.6.14`; the cubin comes from FlashInfer's official index
because PyPI stops at 0.6.13. The official CUDA 13.0 index contains a 0.6.14
JIT-cache package, but a CUDA 13.3 index is not currently published. This CUDA
13.3 image therefore omits the JIT cache; keeping that omission across profiles
also makes control and source-patched images exercise the same runtime-JIT path.

## Scope

This is an independent integration project, not an official vLLM, FlashInfer,
NVIDIA, or DeepSeek image. Upstream PRs remain the source of truth. Carries are
removed as releases include them.
