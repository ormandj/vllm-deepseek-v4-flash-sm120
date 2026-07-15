# vLLM DeepSeek-V4-Flash on SM120

Ready-to-run vLLM images and launch scripts for **DeepSeek-V4-Flash on 2×
NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120)**. The goal is fast single-user
agentic work with enough capacity for several simultaneous requests.

This repository is also a focused integration path toward upstream support,
not a permanent downstream fork. It makes the required vLLM and FlashInfer
fixes usable now, measures their effect, and provides concrete evidence for
merging them upstream. Carries are removed as upstream releases include them;
the intended end state is that stock mainline needs no patches. See
[What is included](#what-is-included) for the technical patch summary and
[`UPSTREAM.md`](UPSTREAM.md) for the merge/dependency map.

## Recommended setup

Use the `:mtp` image with the standard `deepseek-ai/DeepSeek-V4-Flash`
checkpoint and MTP:2. This is the fastest configuration tested so far at every
measured concurrency from 1 through 32.

### 1. Requirements

- Linux x86_64;
- 2× RTX PRO 6000 Blackwell with working GPU peer access;
- Docker and NVIDIA Container Toolkit;
- enough local storage for the model, image, and persistent compile cache.

Download the standard checkpoint if it is not already present:

```bash
python3 -m pip install --user huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "$HOME/models/DeepSeek-V4-Flash"
```

### 2. Start the server

```bash
git clone https://github.com/ormandj/vllm-deepseek-v4-flash-sm120.git
cd vllm-deepseek-v4-flash-sm120

MODEL_DIR="$HOME/models/DeepSeek-V4-Flash" \
CACHE_DIR="$HOME/.cache/vllm-deepseek-v4-flash-sm120/mtp" \
./examples/serve-mtp.sh
```

The script pulls `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:mtp` and starts
an OpenAI-compatible endpoint on port 8000. MTP:2 and probabilistic draft
sampling are already the defaults.

The first start on an empty cache took about 17 minutes on the tested system
because SM120 kernels and CUDA graphs must be compiled. Keep `CACHE_DIR`
persistent; later starts reuse it. The server is ready when the log says
`Application startup complete`.

### 3. Expected capacity

| Setting | Recommended value / measured result |
|---|---:|
| Image | `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:mtp` |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash` |
| GPUs | 2× RTX PRO 6000 Blackwell, TP=2 |
| Maximum context | 1,032,192 tokens |
| Maximum simultaneous sequences | 32 |
| Speculative decoding | MTP:2, probabilistic |
| CUDA graph maximum | 96 |
| Available KV memory | 11.55 GiB |
| Corrected KV capacity | 1,666,236 tokens |
| Max-context capacity | 1.61× a 1,032,192-token request |
| CUDA graph memory | 1.00 GiB |

### 4. Expected performance

Aggregate sustained decode throughput:

| Simultaneous requests | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Output tok/s | **203.81** | **330.30** | **503.62** | **723.94** | **1,045.06** | **1,558.09** |

Cold-prefill throughput:

| Prompt length | 8K | 64K | 128K |
|---|---:|---:|---:|
| Input tok/s | **8,428** | **8,230** | **7,513** |

These are aggregate server numbers. For example, C2 means two requests are
decoding simultaneously and producing about 330 tokens/second combined.

### 5. Verify it works

In another terminal:

```bash
curl --fail http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Return only the result of 17 * 25."}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Available images

| Tag | Use |
|---|---|
| `:mtp` | Recommended standard-checkpoint image. Set MTP width at runtime; MTP:2 is the current recommendation. |
| `:dspark` | DSpark-checkpoint image. Use the separate DSpark recipe below. |
| `:control` | Matched upstream control without the source carries; intended for comparison, not the recommended deployment. |

The publishing workflow keeps exactly one current package version for each
published tag and removes superseded versions.

## DSpark setup

DSpark uses a different checkpoint and launch method. The current
`DeepSeek-V4-Flash-DSpark` checkpoint has `dspark_block_size=5`, so 5 is the
minimum valid width.

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --local-dir "$HOME/models/DeepSeek-V4-Flash-DSpark"

MODEL_DIR="$HOME/models/DeepSeek-V4-Flash-DSpark" \
CACHE_DIR="$HOME/.cache/vllm-deepseek-v4-flash-sm120/dspark" \
./examples/serve-dspark.sh
```

The DSpark launcher defaults to DSpark:5 with probabilistic draft sampling and
captures CUDA graphs through the required C32 shape of 160. It rejects widths
below the checkpoint block size because those produce an unsupported draft
layout. Its measured KV capacity and performance will be added beside the MTP
results after the matched run completes.

## Why the MTP launcher captures through 96

MTP schedules one target token plus two draft slots for each request. At C32,
that is `32 × 3 = 96` scheduled query slots. A graph cap of 48 caused C32 to
fall from 1,558.09 to 474.33 tok/s; the launcher now derives the graph set from
`MTP_TOKENS` and `MAX_NUM_SEQS` automatically.

MTP:3 was also tested with full C32 graph coverage. Its third draft position
accepted only 17.44%, and it was 2.72%-7.39% slower than MTP:2 at every
measured concurrency.

## Full comparison

| Profile | C1 | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|
| `:control`, MTP:0 | 110.86 | 172.13 | 278.26 | 430.91 | 613.32 | 869.28 |
| `:mtp`, MTP:0 | 118.63 | 198.66 | 330.43 | 526.93 | 777.59 | 1,132.85 |
| `:mtp`, MTP:2, graph cap 96 | **203.81** | **330.30** | **503.62** | **723.94** | **1,045.06** | **1,558.09** |
| `:mtp`, MTP:3, graph cap 128 | 192.00 | 321.31 | 467.78 | 672.71 | 967.83 | 1,471.83 |

| Profile | 8K prefill | 64K prefill | 128K prefill | Available KV | Reported KV tokens |
|---|---:|---:|---:|---:|---:|
| `:control`, MTP:0 | 8,549 | 8,137 | 7,414 | 13.26 GiB | 2,019,563 |
| `:mtp`, MTP:0 | 8,724 | 8,425 | 7,700 | 13.26 GiB | 1,914,292 |
| `:mtp`, MTP:2 | 8,428 | 8,230 | 7,513 | 11.55 GiB | 1,666,236 |
| `:mtp`, MTP:3 | 8,479 | 8,236 | 7,522 | 11.55 GiB | 1,666,236 |

MTP:2 accepted 65.05% of drafted tokens and averaged 1.301 accepted tokens per
draft. MTP:3 accepted 48.41% overall and averaged 1.452 accepted tokens per
draft; the additional accepted token did not offset the cost of drafting a
third position.

## Benchmark protocol

The client is
[`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
at commit `e190b8ca5c52c1ef8db429a22b5c4d8daa56e82f` (v0.4.31).

All rows use TP=2, FP8 KV, `max_model_len=1032192`, `max_num_seqs=32`,
`max_num_batched_tokens=4096`, `gpu_memory_utilization=0.944`, V2 model runner,
asynchronous scheduling, prefix caching, chunked prefill, and full/piecewise
breakable CUDA graphs.

Before measurement, wait 30 seconds after readiness, run a 3-second decode
warmup at every measured concurrency plus a 1-second exact-token prefill
warmup, then wait another 30 seconds. The measured commands are:

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

`--max-tokens 8192` is the per-request output limit. The
`--max-total-tokens 2000000` value is only the benchmark client's aggregate
launch budget across all requests; it is **not** model context. The configured
server context is 1,032,192 tokens.

## What is included

This repository pins the source stack in [`stack.lock.json`](stack.lock.json)
and carries focused fixes while they are under upstream review:

| Component | Role |
|---|---|
| [vLLM #48303](https://github.com/vllm-project/vllm/pull/48303) | DeepSeek-family MXFP4 to FlashInfer CUTLASS MoE wiring |
| [FlashInfer #3903](https://github.com/flashinfer-ai/flashinfer/pull/3903) plus the temporary vLLM selector | SM120/121 TensorRT-LLM all-reduce |
| [FlashInfer #3930](https://github.com/flashinfer-ai/flashinfer/pull/3930) plus the [signed follow-up](https://github.com/ormandj/flashinfer/commit/bd6765dea271b23a579938132f8ca1b9cbf6a2a5) | Reject the reproduced CUDA-stub look-alike during runtime resolution |
| [vLLM #48317](https://github.com/vllm-project/vllm/pull/48317) | Correct packed/hybrid KV-capacity reporting |
| [vLLM #48304](https://github.com/vllm-project/vllm/pull/48304) | Honor DSpark draft-layer rope semantics |
| [FlashInfer #3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) and [#3834](https://github.com/flashinfer-ai/flashinfer/pull/3834) | DSpark sparse-MLA decode/prefill instantiations on SM120 |

See [`UPSTREAM.md`](UPSTREAM.md) for the merge/dependency map and
[issue #1](https://github.com/ormandj/vllm-deepseek-v4-flash-sm120/issues/1)
for live upstream status.

## Operational notes

- Reuse a cache directory only with the same image digest and profile.
- The scripts set `NCCL_P2P_LEVEL=SYS`; verify peer access before tuning.
- If NCCL hangs at the first collective, stop and fix GPU peer access instead
  of benchmarking a degraded path.
- Lower `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION` if the GPUs are shared with
  another process.
- The launch scripts run Docker in the foreground. Press Ctrl-C to stop; the
  container is removed automatically.

## Build locally

The build is SM120-only and expensive:

```bash
./scripts/verify-patches.sh
MAX_JOBS=48 ./scripts/build.sh mtp local/dsv4-sm120:mtp
```

The image uses vLLM's exact-commit official native wheel and packages patched
Python source around the same commit. It installs matched
`flashinfer-python==0.6.14` and `flashinfer-cubin==0.6.14`; the cubin is fetched
from FlashInfer's official index because PyPI stops at 0.6.13.

## Scope

This is an independent integration project, not an official vLLM, FlashInfer,
NVIDIA, or DeepSeek image. Model weights are not included. Upstream PRs remain
the source of truth, and carries are removed as releases include them.
