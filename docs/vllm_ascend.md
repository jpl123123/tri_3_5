# vLLM-Ascend Integration

TriAttention can run through vLLM-Ascend by using the same runtime scheduler and
KV compaction path as the CUDA vLLM backend, with two Ascend-specific changes:

- `vllm_ascend.worker.worker.NPUWorker` is patched so the TriAttention model
  runner proxy is installed after the NPU model runner is created.
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
```

Compression events should report `selector_status=enabled:torch` when the first
compression boundary is reached on NPU.

## Calibration Stats

TriAttention still requires a model-specific statistics file produced by
`scripts/calibrate.py`. The stats should match the model architecture and RoPE
layout used for serving. See [Calibration Guide](calibration.md).

## Current Limits

- Dense attention KV caches are the primary supported Ascend path.
- Ascend sparse attention or MLA layouts may attach extra tensors after
  `(k_cache, v_cache)`. The current compaction path moves the dense K/V tensors
  and should be validated before using those model families in production.
- `NPUWorker310` and `XliteWorker` init hooks are patched on a best-effort basis,
  but the main validated target is `vllm_ascend.worker.worker.NPUWorker`.
