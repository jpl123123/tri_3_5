# vLLM-Ascend Integration

TriAttention can run through vLLM-Ascend by using the same runtime scheduler and
KV compaction path as the CUDA vLLM backend, with three Ascend-specific changes:

- `vllm_ascend.worker.worker.NPUWorker` is patched so the TriAttention model
  runner proxy is installed after the NPU model runner is created.
- vLLM-Ascend input preparation is patched so compressed KV length is reflected
  in NPU `seq_lens`, CPU `seq_lens_np`, and slot mappings. Without this,
  attention metadata can keep reading the original long context after KV has
  already been compacted, which commonly shows up as repeated tokens.
- NPU execution defaults to PyTorch/torch_npu scoring instead of the CUDA
  Triton scoring kernel.

The dense Ascend KV layout is supported in both forms:

- vLLM CUDA-style combined cache: `[2, num_blocks, block_size, num_kv_heads, head_dim]`
- vLLM-Ascend split cache: `(k_cache, v_cache)` where each tensor is
  `[num_blocks, block_size, num_kv_heads, head_dim]`

## Installation

Install vLLM and vLLM-Ascend first, then install this package in the same Python
environment:

```bash
pip install -e .
```

The vLLM plugin entry point activates automatically. You can disable it with:

```bash
export ENABLE_TRIATTENTION=0
```

## Server Example

```bash
export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/model_stats.pt
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_DIVIDE_LENGTH=128
export TRIATTN_RUNTIME_WINDOW_SIZE=128
export TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=1
export TRIATTN_RUNTIME_ENABLE_ASYNC_COMPRESSION_BOUNDARY=0
export TRIATTN_RUNTIME_EARLY_INSTALL_PROXY_ON_ASCEND=1
export TRIATTN_RUNTIME_PREINSTALL_INPUT_PATCH=1
export TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=1
export TRIATTN_RUNTIME_ENABLE_PACKED_POS_DELTA_ON_ASCEND=0

# auto = Triton on CUDA, PyTorch/torch_npu on NPU.
export TRIATTN_RUNTIME_SCORING_BACKEND=auto

vllm serve /path/to/model \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 1024
```

Recommended first-run settings:

- Use `--enforce-eager` while validating correctness and memory behavior.
- Disable prefix caching because compressed KV entries no longer match vLLM's
  original prefix-cache block hashes.
- Keep `--max-num-batched-tokens` modest so prefill chunks do not overshoot the
  KV budget before the compression boundary is reached.
- On Ascend, `TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=1` is the
  default. Compression is first applied after the full prompt prefill has
  completed, which is the most stable mode for long prompts on NPU attention
  backends. Set it to `0` only after validating streaming prefill compression on
  your vLLM-Ascend version.

## Scoring Backend

`TRIATTN_RUNTIME_SCORING_BACKEND` accepts:

| Value | Behavior |
|-------|----------|
| `auto` | Uses CUDA Triton on CUDA devices and PyTorch/torch_npu on NPU devices |
| `torch` / `pytorch` | Forces the PyTorch scoring path |
| `triton` | Forces the CUDA Triton scoring path |

On vLLM-Ascend, leave this as `auto` or set it to `torch`.

For correctness, the PyTorch/torch_npu scoring path explicitly promotes KV keys,
Q statistics, RoPE frequencies, and frequency scales to `float32` before scoring.
The KV cache itself remains in the model dtype; only the transient scoring chunks
are promoted.

## Expected Logs

At startup, look for these log lines:

```text
[TriAttention] Runtime (V2) plugin activated: patch_scheduler=True patch_worker=True
Installed TriAttention runtime worker patches for Ascend: vllm_ascend.worker.worker.NPUWorker
Installed TriAttention runtime input patches: ... vllm_ascend.worker.model_runner_v1.NPUModelRunner ...
```

Recent builds also include `build=ascend-prefix-only-v3-20260602` in the
plugin, scheduler, and worker logs. If that build id is missing, the running
container is still loading an older installed package or stale source path.

Compression events should report a status like `selector_status=enabled:torch:tp=1/2`
when the first compression boundary is reached on NPU. The `tp=rank/size`
suffix confirms that runtime scoring is using this worker's tensor-parallel
head shard. On vLLM-Ascend with `TRIATTN_RUNTIME_SCORING_BACKEND=auto`, the
status should say `enabled:torch`, not `enabled:triton`.

For a long prompt on Ascend, it is normal to see skipped compression events with
`reason=prefill_incomplete` during chunked prefill. The first real compression
should happen once the prompt has finished prefill and decode starts.

If proxy injection is visible but no compression line appears, the runtime now
backfills request state from the NPU runner and allows the first decode step to
trigger compression even when vLLM-Ascend's scheduler counters still lag behind
the full prompt length.

## Performance Tuning

On Ascend, the default path keeps sparse-stat per-head selection and all-layer
scoring so the selected KV set matches the reference sparse path. The runtime
still avoids the expensive full `[kept + dropped]` KV rewrite when tail blocks
are physically reclaimed. For latency-sensitive serving after validating output
quality, try:

```bash
export TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND=8
export TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND=8
export TRIATTN_RUNTIME_SPARSE_NORMALIZE_SCORES=0
export TRIATTN_RUNTIME_PERF_PROFILE=1
export TRIATTN_RUNTIME_PERF_LOG_EVERY=50
```

`TRIATTN_RUNTIME_SCORE_MAX_LAYERS_ON_ASCEND` defaults to `0`, which means score
all layers. It is used only when `TRIATTN_RUNTIME_SCORE_MAX_LAYERS=0`. Set
`SCORE_MAX_LAYERS` explicitly to force a value for every backend. Use `8` as
the conservative first latency setting, then try `4` if quality is stable. The
runtime log will include
`selector_status=enabled:torch:tp=...:score_layers=max8,stride1`.

`TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND` prevents very small compactions
such as `2175 -> 2048` from running on NPU. With `--block-size 128`, the default
`8` means compression waits until it can reclaim about 1024 KV tokens, which
better amortizes scoring, KV movement, and scheduler/worker synchronization.

For maximum TTFT improvement on very long prompts, validate prefill compression
after confirming the build id above:

```bash
export TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=0
```

This allows compression before later prefill chunks, so the remaining prompt
prefill also benefits from the shorter KV. Keep it disabled if your target
vLLM-Ascend build shows any quality regression.

To isolate selector overhead from the NPU attention speedup, run one benchmark
with the recency-only selector:

```bash
export TRIATTN_RUNTIME_FAST_RECENCY_ONLY=1
```

This skips sparse-stat scoring and keeps the newest `KV_BUDGET` tokens. When
`KV_BUDGET` is a multiple of `--block-size`, vLLM-Ascend uses a zero-copy tail
block remap by default instead of copying KV tensors; the expected compression
reason is `kv_compacted:zero_copy_tail`. To compare against the older copy path,
set `TRIATTN_RUNTIME_ENABLE_ZERO_COPY_RECENCY=0`.

For correctness on long prompts, `TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD`
defaults to `1`: when `TRIATTN_RUNTIME_SPARSE_STATS_PATH` is set, sparse-stat
TriAttention selection is used instead of pure recency, even if
`TRIATTN_RUNTIME_FAST_RECENCY_ONLY=1` is left in the environment. Pure recency
is a performance diagnostic and can degrade or repeat on 20k+ prompts.

The async compression boundary is disabled by default because it can repeatedly
block vLLM's batch-queue lookahead during generation. Re-enable it only for
debugging with `TRIATTN_RUNTIME_ENABLE_ASYNC_COMPRESSION_BOUNDARY=1`.

On Ascend, the runner proxy and input patches are installed during worker
initialization by default. This avoids first-request patch installation and ACL
graph replay in the measured request path. To compare against the older lazy
behavior, set `TRIATTN_RUNTIME_EARLY_INSTALL_PROXY_ON_ASCEND=0` and
`TRIATTN_RUNTIME_PREINSTALL_INPUT_PATCH=0`.

`TRIATTN_RUNTIME_ENABLE_PACKED_POS_DELTA_ON_ASCEND` is disabled by default. It
is an experimental slot-mapping micro-optimization and should only be enabled
after validating output quality for the target vLLM-Ascend build.

## Calibration Stats

TriAttention still requires a model-specific statistics file produced by
`scripts/calibrate.py`. The stats should match the model architecture and RoPE
layout used for serving. See [Calibration Guide](calibration.md).

## Current Limits

- Dense attention KV caches are the primary supported Ascend path.
- Tensor parallel serving is supported by slicing calibration statistics to the
  local TP head shard before TopK selection.
- Ascend sparse attention or MLA layouts may attach extra tensors after
  `(k_cache, v_cache)`. The current compaction path moves the dense K/V tensors
  and should be validated before using those model families in production.
- `NPUWorker310` and `XliteWorker` init hooks are patched on a best-effort basis,
  but the main validated target is `vllm_ascend.worker.worker.NPUWorker`.
