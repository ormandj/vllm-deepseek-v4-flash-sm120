# vLLM DeepSeek-V4-Flash on SM120

Ready-to-run vLLM images and launch scripts for **DeepSeek-V4-Flash on 2×
NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120)**. The goal is fast single-user
agentic work with enough capacity for several simultaneous requests.

This repository is also a focused integration path toward upstream support,
not a permanent downstream fork. It makes the required vLLM and FlashInfer
fixes usable now, measures their effect, and tracks their upstream review.
Carries are removed as upstream releases include them;
the intended end state is that stock mainline needs no patches. See
[What is included](#what-is-included) for the technical patch summary and
[`UPSTREAM.md`](UPSTREAM.md) for the merge/dependency map.

## Recommended setup

Use the `:dspark` image with the `deepseek-ai/DeepSeek-V4-Flash-DSpark`
checkpoint. The defaults are tuned for a workload dominated by one active
request, with C2-C8 as the normal multi-agent range and C16/C32 retained as
load guardrails.

Download the checkpoint once:

```bash
python3 -m pip install --user huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --local-dir "$HOME/models/DeepSeek-V4-Flash-DSpark"
```

Start the server:

```bash
git clone https://github.com/ormandj/vllm-deepseek-v4-flash-sm120.git
cd vllm-deepseek-v4-flash-sm120

MODEL_DIR="$HOME/models/DeepSeek-V4-Flash-DSpark" \
CACHE_DIR="$HOME/.cache/vllm-deepseek-v4-flash-sm120/dspark" \
./examples/serve-dspark.sh
```

This pulls `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:dspark` and starts an
OpenAI-compatible endpoint on port 8000. DSpark:5, probabilistic draft sampling,
TP=2, FP8 KV, and the measured CUDA-graph shapes are already set.

The first start on an empty cache took about 17 minutes on the tested system
because SM120 kernels and CUDA graphs must be compiled. Keep `CACHE_DIR`
persistent; later starts reuse it. The server is ready when the log says
`Application startup complete`.

### Expected capacity

| Setting | Recommended value / measured result |
|---|---:|
| Image | `ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120:dspark` |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| GPUs | 2× RTX PRO 6000 Blackwell, TP=2 |
| Maximum context | 1,032,192 tokens |
| Maximum simultaneous sequences | 32 |
| Speculative decoding | DSpark:5, probabilistic |
| Scheduler batch budget | 4,096 configured / 3,968 effective |
| CUDA graph maximum | 192 |
| Available KV memory | 7.70 GiB |
| Physical KV blocks | 7,676 GPU blocks |
| Max-context capacity | 1.04× a 1,032,192-token request |
| Max-context-equivalent capacity | 1,072,719 tokens |
| CUDA graph memory | 1.13 GiB |

`1,032,192` is the per-request model limit: the checkpoint's 1,048,576-token
window minus a 16,384-token operating reserve, aligned to the 256-token KV
block size. For this model's hybrid KV layout, vLLM's reported token capacity
means `maximum-context concurrency × configured maximum context`; it is not a
context-independent total-token pool. Compare physical KV bytes and GPU blocks
when evaluating profiles with different context limits.

### Expected performance

Aggregate sustained decode throughput:

| Simultaneous requests | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Output tok/s | **227.70** | **338.89** | **486.36** | **719.64** | **1,049.28** | **1,458.55** |

Cold-prefill throughput:

| Prompt length | 8K | 64K | 128K |
|---|---:|---:|---:|
| Input tok/s | **8,566** | **8,400** | **7,655** |

The five-run C1 coding median is **280.02 output tok/s**. Across the sustained
decode matrix, DSpark accepted **39.77%** of drafted tokens and averaged
**1.988 accepted tokens per draft**. These are aggregate server numbers: C2,
for example, means two simultaneous requests producing 338.89 tok/s combined.

### Verify it works

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

### Requirements

- Linux x86_64;
- 2× RTX PRO 6000 Blackwell with working GPU peer access;
- Docker and NVIDIA Container Toolkit;
- enough local storage for the 155 GiB checkpoint, image, and persistent cache.

## Available images

| Tag | Use |
|---|---|
| `:dspark` | Recommended image for the DSpark checkpoint; DSpark:5 is the native tested configuration. |
| `:mtp` | Standard-checkpoint alternative; MTP:2 is the best tested MTP width. |
| `:control` | Matched upstream control without the performance carries; intended for comparison, not the recommended deployment. |
| `:deepgemm` | Thin `:control` derivative carrying DeepGEMM PR #380 for isolated SM120 MoE evaluation. |

The publishing workflow keeps one current package version for each tag and
removes superseded versions. To use the standard checkpoint instead:

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "$HOME/models/DeepSeek-V4-Flash"

MODEL_DIR="$HOME/models/DeepSeek-V4-Flash" \
CACHE_DIR="$HOME/.cache/vllm-deepseek-v4-flash-sm120/mtp" \
./examples/serve-mtp.sh
```

The MTP launcher defaults to MTP:2 with probabilistic draft sampling.

## Why the MTP launcher captures through 96

MTP schedules one target token plus two draft slots for each request. At C32,
that is `32 × 3 = 96` scheduled query slots. A graph cap of 48 caused C32 to
fall from 1,558.09 to 474.33 tok/s; the launcher now derives the graph set from
`MTP_TOKENS` and `MAX_NUM_SEQS` automatically.

MTP:3 was also tested with full C32 graph coverage. Its third draft position
accepted only 17.44%, and it was 2.72%-7.39% slower than MTP:2 at every
measured concurrency.

## Full comparison

The current-main images and the named v10 image below were run on the same two
300 W Max-Q cards and PCIe 4.0 x16 host, with identical client flags and matched
server limits. The v10 comparison image was
`voipmonitor/vllm:fathomless-firmament-ds4-v10-vllmadf15ca-b12x90172a5-fi2cba2f7-cu132-20260712`.

| Profile | C1 | C2 | C4 | C8 | C16 | C32 | Coding C1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `:dspark`, DSpark:5 | **227.70** | **338.89** | 486.36 | 719.64 | **1,049.28** | 1,458.55 | **280.02** |
| `:mtp`, MTP:2 | 202.73 | 334.32 | **514.82** | **732.95** | 1,047.55 | **1,556.28** | 217.31 |
| v10, MTP:2 | 210.00 | 310.30 | 479.08 | 678.26 | 965.77 | 1,377.55 | 212.45 |
| v10, MTP:0 | 123.83 | 203.76 | 328.40 | 478.65 | 735.46 | 1,075.13 | 123.37 |

| Profile | 8K prefill | 64K prefill | 128K prefill | Available KV | Max-context-equivalent tokens |
|---|---:|---:|---:|---:|---:|
| `:dspark`, DSpark:5 | 8,566 | 8,400 | 7,655 | 7.70 GiB | 1,072,719 |
| `:mtp`, MTP:2 | **8,907** | **8,618** | **7,945** | 11.55 GiB | 1,666,236 |
| v10, MTP:2 | 6,048 | 7,069 | 6,609 | 11.55 GiB | 1,647,611 |
| v10, MTP:0 | 6,240 | 7,347 | 6,778 | 13.50 GiB | 1,925,828 |

DSpark is recommended because it improves the primary C1 sustained row by
12.32% and the coding median by 28.86% versus our MTP:2 image. MTP:2 remains
faster at C4-C8 and C32 and retains substantially more KV capacity, so it is a
reasonable alternative when concurrency or aggregate context capacity matters
more than single-agent speed.

## Additional tuning results

### Probabilistic versus greedy DSpark

Greedy draft sampling changed no KV setting and received two complete repeats.
It improved the C1/C2 means, but reduced coding throughput and did not improve
most of C4-C32, so the launcher retains probabilistic sampling.

| Profile | C1 | C2 | C4 | C8 | C16 | C32 | Coding C1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probabilistic | 227.70 | 338.89 | 486.36 | 719.64 | 1,049.28 | 1,458.55 | 280.02 |
| Greedy run 1 | 237.40 | 349.86 | 481.89 | 712.83 | 1,036.05 | 1,460.90 | 266.74 |
| Greedy run 2 | 228.82 | 346.15 | 491.75 | 713.74 | 1,031.43 | 1,450.75 | 273.90 |
| Greedy mean delta | +2.38% | +2.69% | +0.09% | -0.88% | -1.48% | -0.19% | -3.46% |

Greedy acceptance averaged 39.42% and 1.971 accepted tokens per draft versus
39.77% and 1.988 for probabilistic. Both modes exposed the same 7.70 GiB /
7,676-block KV pool and passed the sequential and C8 correctness checks.

### Asynchronous scheduling

Disabling asynchronous scheduling did not add physical KV: both modes exposed
7.70 GiB and 7,676 GPU blocks. It reduced throughput at every canonical
concurrency and reduced the coding median by 6.77%, so the launcher keeps
asynchronous scheduling enabled.

| C | Async on | Async off | Async-off delta |
|---:|---:|---:|---:|
| 1 | 227.70 | 195.66 | -14.07% |
| 2 | 338.89 | 308.76 | -8.89% |
| 4 | 486.36 | 456.56 | -6.13% |
| 8 | 719.64 | 653.91 | -9.13% |
| 16 | 1,049.28 | 1,018.04 | -2.98% |
| 32 | 1,458.55 | 1,419.81 | -2.66% |
| Coding median | 280.02 | 261.06 | -6.77% |

The max-context-equivalent token value increased from 1,072,719 to 1,384,674
when async scheduling was disabled despite identical physical KV. The hybrid
layout's per-request in-flight reservation changed; physical allocation did
not. This is another reason not to read that value as a universal KV-token
total.

### Scheduler batch budget

At the recommended 1,032,192-token maximum context, configuring 8,192 batched
tokens reduced available KV enough that the server could not admit one maximum-
length request. A 4,224 setting restored an effective 4,096-token scheduler
budget after DSpark's reservation, but reduced the physical KV pool from 7,676
to 7,650 blocks and produced no consistent throughput improvement on repeat.
The launcher therefore uses
4,096 configured / 3,968 effective tokens.

| Context limit | Configured batch | Effective batch | Available KV | GPU blocks | Max-context-equivalent tokens | Startup |
|---:|---:|---:|---:|---:|---:|---|
| 1,032,192 | 4,096 | 3,968 | 7.70 GiB | 7,676 | 1,072,719 | Pass |
| 1,032,192 | 4,224 | 4,096 | 7.67 GiB | 7,650 | 1,054,241 | Pass |
| 1,032,192 | 8,192 | 8,064 | 6.85 GiB | — | — | Fail |
| 262,144 | 8,192 | 8,064 | 7.98 GiB | 7,957 | 270,682 | Pass at `gpu_memory_utilization=0.946` |

| C | Batch 4,096 / context 1,032,192 | Batch 4,224 / context 1,032,192 | Batch 8,192 / context 262,144 |
|---:|---:|---:|---:|
| 1 | 227.70 | 228.43 | 223.33 |
| 2 | 338.89 | 349.92 | 341.56 |
| 4 | 486.36 | 502.62 | 496.93 |
| 8 | 719.64 | 719.25 | 721.21 |
| 16 | 1,049.28 | 1,039.50 | 1,046.51 |
| 32 | 1,458.55 | 1,455.78 | 1,431.85 |

The 262,144/8,192 arm did not materially improve sustained decode. It raised
the coding C1 median to 291.41 tok/s and cold-prefill throughput to
8,929/8,776/7,975 tok/s at 8K/64K/128K. It actually exposed a larger physical
KV pool: 7.98 GiB / 7,957 blocks versus 7.70 GiB / 7,676 blocks. Its smaller
`270,682` token value only expresses capacity at the smaller configured maximum
request. The recommended launcher keeps the larger-context 4,096-token profile
because the smaller-context arm did not improve sustained decode.

As a KV-capacity cross-check, the v10 v16 DSpark image
`voipmonitor/vllm:fathomless-firmament-v16-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714`
reported 7.92 GiB on the same hardware with the same 262,144-token context and
8,192-token scheduler budget. Our matching arm reported 7.98 GiB. Their
max-context-equivalent values were 263,176 and 270,682 respectively, so there
is no material v10 physical-KV advantage under the matched shape.

### Agentic workload checks

A 64,024-token shared prompt reused 61,440 cached tokens (95.96%) per measured
request and generated 128 tokens per request.

| Concurrent requests | Median TTFT | Aggregate output tok/s |
|---:|---:|---:|
| 1 | 0.59 s | 139.87 |
| 2 | 0.77 s | 173.01 |
| 4 | 1.36 s | 192.12 |
| 8 | 1.87 s | 236.29 |
| 16 | 3.41 s | 266.63 |
| 32 | 6.02 s | 290.01 |

The 180-second mixed test used fresh approximately 32K-token Rust source
prompts, natural EOS, and a 4,096-token output cap. All started requests drained
with zero failures.

| Concurrent requests | Input tok/s | Output tok/s | Started/drained | Steady median TTFT |
|---:|---:|---:|---:|---:|
| 8 | 4,094.85 | 369.59 | 23/23 | 4.18 s |
| 16 | 5,341.12 | 379.72 | 31/31 | 4.38 s |
| 32 | 5,875.05 | 384.79 | 34/34 | Insufficient steady turnover |

Two fresh 960,164-token requests reached first token in 274.79 and 275.64
seconds, corresponding to 3,494 and 3,483 input tok/s.

## Benchmark protocol

The client is
[`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
at commit `e190b8ca5c52c1ef8db429a22b5c4d8daa56e82f` (v0.4.31).

All current-image rows use TP=2, FP8 KV, `max_model_len=1032192`,
`max_num_seqs=32`, `max_num_batched_tokens=4096`,
`gpu_memory_utilization=0.944`, V2 model runner, asynchronous scheduling,
prefix caching, chunked prefill, and full/piecewise breakable CUDA graphs.

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

Every profile also receives five sequential C1 coding requests using
`Write a Python script that implements the Sieve of Eratosthenes.`, a
2,000-token output cap, and the server's default sampling behavior. The table
reports the median generation-only rate.

## What is included

This repository pins the source stack in [`stack.lock.json`](stack.lock.json)
and carries focused fixes while they are under upstream review:

| Component | Role |
|---|---|
| [vLLM #47669](https://github.com/vllm-project/vllm/pull/47669) requirements update | Keep the required FlashInfer 0.6.14 packages aligned with the rebuilt vLLM wheel metadata |
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
./scripts/update-vllm-lock.sh
./scripts/verify-patches.sh
MAX_JOBS=48 ./scripts/build.sh mtp local/dsv4-sm120:mtp
```

`update-vllm-lock.sh` fetches upstream `main`, finds its newest first-parent
commit with a published `cu130` wheel, and pins both the vLLM source and native
wheel to that exact commit. This is the control-update policy: do not wait for
newer wheel-less `main`, combine newer source with an older native wheel, or
compile vLLM's native extensions locally. A newer upstream commit enters the
control only after its `cu130` wheel is published. Pass a vLLM checkout as the
first argument when it is not available at `../vllm`.

The image packages the locked source and selected carries around that wheel.
It installs matched `flashinfer-python==0.6.14` and
`flashinfer-cubin==0.6.14`; the cubin is fetched from FlashInfer's official
index because PyPI stops at 0.6.13.

The DeepGEMM candidate reuses the immutable control image and adds only the
external package locked in `stack.lock.json`:

```bash
./scripts/build-deepgemm.sh local/dsv4-sm120:deepgemm
```

## Scope

This is an independent integration project, not an official vLLM, FlashInfer,
NVIDIA, or DeepSeek image. Model weights are not included. Upstream PRs remain
the source of truth, and carries are removed as releases include them.
