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
| `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:mtp` | Current recommended stack for the standard checkpoint and runtime-selected MTP width. |
| `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:dspark` | MTP stack plus the DSpark checkpoint prerequisites. |

The publishing workflow keeps one current package version per published tag.
Record its digest when exact reproduction matters; superseded package versions
are removed.

## Run the recommended MTP profile

Requirements:

- Linux x86_64;
- 2× RTX PRO 6000 Blackwell with working peer access;
- NVIDIA Container Toolkit;
- a local DeepSeek-V4-Flash snapshot;
- substantial persistent cache space for first-start SM120 JIT artifacts.

```bash
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
CACHE_DIR=/models/cache/dsv4-sm120-mtp \
MTP_TOKENS=2 \
./examples/serve-mtp.sh
```

The image does not bake in the speculative width. Set `MTP_TOKENS` at runtime;
use the same `:mtp` image for MTP:2, MTP:3, or another supported width. The
`DRAFT_SAMPLE_METHOD` defaults to `probabilistic`; temperature-zero requests
still use vLLM's greedy fast path.

The launcher derives the CUDA graph shapes from `MTP_TOKENS` and
`MAX_NUM_SEQS`. With the tested MTP:2/C32 defaults it captures through 96;
using a graph cap of 48 causes a large C32 eager-fallback regression.

## Run the DSpark profile

Use the DSpark checkpoint with the `:dspark` image. The current checkpoint has
`dspark_block_size=5`, so 5 is the minimum valid width and the tested default.

```bash
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash-DSpark \
CACHE_DIR=/models/cache/dsv4-sm120-dspark \
DSPARK_TOKENS=5 \
./examples/serve-dspark.sh
```

The DSpark launcher uses `method=dspark`, probabilistic draft sampling, and
derives the parallel-draft graph set from `DSPARK_TOKENS`. DSpark:5/C32
captures through 160. Values below 5 are rejected because they produce an
unsupported draft layout for this checkpoint.

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
| [FlashInfer #3930](https://github.com/flashinfer-ai/flashinfer/pull/3930) + [exact signed follow-up](https://github.com/ormandj/flashinfer/commit/bd6765dea271b23a579938132f8ca1b9cbf6a2a5) | Reject the reproduced TileLang CUDA-stub look-alike during runtime resolution | Live failure/recovery + 12 CPU tests |
| [vLLM #48317](https://github.com/vllm-project/vllm/pull/48317) | Correct packed/hybrid KV-capacity reporting | Source inspection + focused tests; no performance claim |
| [vLLM #48304](https://github.com/vllm-project/vllm/pull/48304) | Honor the DSpark checkpoint's draft-layer rope semantics | Same-configuration acceptance result + focused tests |
| [FlashInfer #3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) and [#3834](https://github.com/flashinfer-ai/flashinfer/pull/3834) | TOPK=256 decode/prefill instantiations required by DSpark on SM120 | Dispatch/reference correctness + serving completion |

The local DeepSeek-V4 compile experiment is deliberately excluded: its
controlled marginal effect was mixed and within measurement noise. This image
is a focused integration of supported fixes, not a bag of every attempted tuning.

See [`UPSTREAM.md`](UPSTREAM.md) for the terse merge map and dependency story.
Live status is tracked in [issue #1](https://github.com/ormandj/vllm-deepseek-v4-flash-sm120/issues/1).

## Current matched benchmark

The following rows use the same public current-main images, standard
DeepSeek-V4-Flash checkpoint, and server geometry. Decode is a 30-second
sustained window at C1/C2/C4/C8/C16/C32 with temperature 0, an 8,192-token
request cap, and a 2,000,000-token launch budget. Prefill targets are exact
8K/64K/128K tokens against unique cold prompts. All speculative arms use
probabilistic draft sampling; temperature-zero requests take vLLM's greedy
request path.

Common server settings: TP=2, FP8 KV, `max_model_len=1032192`,
`max_num_seqs=32`, `max_num_batched_tokens=4096`,
`gpu_memory_utilization=0.944`, V2 model runner, asynchronous scheduling,
prefix caching, chunked prefill, and full/piecewise breakable CUDA graphs.

| Profile | C1 | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|
| `:control`, MTP:0 | 110.86 | 172.13 | 278.26 | 430.91 | 613.32 | 869.28 |
| `:mtp`, MTP:0 | 118.63 | 198.66 | 330.43 | 526.93 | 777.59 | 1,132.85 |
| `:mtp`, MTP:2, graph cap 96 | **203.81** | **330.30** | **503.62** | **723.94** | **1,045.06** | **1,558.09** |
| `:mtp`, MTP:3, graph cap 128 | 192.00 | 321.31 | 467.78 | 672.71 | 967.83 | 1,471.83 |

| Profile | 8K prefill | 64K prefill | 128K prefill | Reported KV tokens |
|---|---:|---:|---:|---:|
| `:control`, MTP:0 | 8,549 | 8,137 | 7,414 | 2,019,563 |
| `:mtp`, MTP:0 | 8,724 | 8,425 | 7,700 | 1,914,292 |
| `:mtp`, MTP:2 | 8,428 | 8,230 | 7,513 | 1,666,236 |
| `:mtp`, MTP:3 | 8,479 | 8,236 | 7,522 | 1,666,236 |

MTP:2 is the current recommendation. MTP:3 was 2.72%-7.39% slower at every
decode concurrency. MTP:2 accepted 65.05% of drafted tokens and averaged
1.301 accepted tokens per draft. MTP:3 accepted 48.41% overall; its third
position accepted only 17.44%, which did not repay its additional work.

### Benchmark command

The client is
[`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
at commit `e190b8ca5c52c1ef8db429a22b5c4d8daa56e82f` (v0.4.31). Before measurement,
wait 30 seconds after readiness, run a 3-second decode warmup at every measured
concurrency plus a 1-second exact-token prefill warmup, then wait another 30
seconds. The measured invocations are:

```bash
python3 llm_decode_bench.py \
  --host localhost --port 8000 --model DeepSeek-V4-Flash \
  --max-tokens 8192 --max-total-tokens 2000000 \
  --sustained-launch-stagger-seconds 0.25 --temperature 0 \
  --concurrency 1,2,4,8,16,32 --contexts 0 --duration 30 \
  --skip-prefill --no-hw-monitor --output decode.json

python3 llm_decode_bench.py \
  --host localhost --port 8000 --model DeepSeek-V4-Flash \
  --max-tokens 8192 --max-total-tokens 2000000 \
  --sustained-launch-stagger-seconds 0.25 \
  --concurrency 1 --contexts 0 --prefill-only --standalone-prefill \
  --prefill-contexts 8k,64k,128k --token-targeting exact \
  --prefill-duration 10 --no-hw-monitor --output prefill.json
```

## Upstream attribution evidence

The controlled benchmark matrix on the previous pinned vLLM main established:

- #48303: sustained MTP:0 decode changed by **+1.21% to +21.77%** across
  C1-C32; exact 8K/64K/128K cold prefill changed by **+1.79% to +3.92%**.
- The complete FlashInfer all-reduce integration stack: sustained MTP:0 decode
  changed by **+2.15% to +5.18%** across C1-C32; exact cold prefill changed by
  **+0.25% to +1.39%**.
- The compile carry did not establish a material independent or additive gain.

Those numbers remain tied to vLLM `fec64fea75103a1490e7fa0874c55a2292c110b1`.
They establish the marginal upstream-patch effects but are not mixed with the
newer current-pin performance rows above.

## Build locally

The build is SM120-only and expensive. Podman is expected.

```bash
./scripts/verify-patches.sh
MAX_JOBS=48 ./scripts/build.sh mtp local/dsv4-sm120:mtp
```

The image records the vLLM revision, repository revision, profile, and SHA256
of every selected patch in OCI labels and environment variables.

The vLLM carries in this repository change Python only. Builds therefore use
vLLM's official precompiled native wheel from the exact pinned vLLM commit and
package the patched Python source around those same-commit binaries. The native
wheel commit and metadata URL are pinned in `stack.lock.json` and recorded in
the image metadata. This avoids recompiling unchanged CUDA extensions on an
ephemeral hosted runner; it is not a cross-commit binary substitution.

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
