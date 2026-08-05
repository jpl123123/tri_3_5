#!/usr/bin/env python3
"""Calibrate frequency-domain statistics for TriAttention (Qwen3.5 hybrid).

Runs a single forward pass on plain text, hooks the *full-attention* layers of
a Qwen3.5-style hybrid model, captures the **pre-RoPE** query states directly
(``q_proj -> drop gate -> q_norm``), and computes per-head frequency statistics
over the rotary sub-block only (partial rotary).

Why this differs from the generic single-head-rotary calibrator:
- Qwen3.5 is a hybrid model: only ``layer_types[i] == "full_attention"`` layers
  own a ``self_attn`` with a KV cache. Linear-attention layers are skipped.
- ``q_proj`` emits ``num_heads * head_dim * 2`` and is chunked into (query, gate);
  the gate is applied post-attention and must be dropped for calibration.
- Partial rotary: only the first ``rotary_dim = int(head_dim * partial_rotary_factor)``
  dimensions are rotated (``rotary_dim = 64`` for Qwen3.5-27B, head_dim=256).
  Frequency statistics are computed over the ``freq_count = rotary_dim // 2 = 32``
  rotary complex pairs. The remaining pass-through dims carry no distance
  preference and are intentionally excluded from scoring (method A).

Because we capture Q *before* RoPE, there is no RoPE inversion and no MRoPE
reconstruction to get wrong.

The output ``.pt`` uses the R-KV layout consumed by
``triattention.vllm.core.utils.load_frequency_stats``:
    payload["stats"]["layer{L:02d}_head{H:02d}"] = {q_mean_real, q_mean_imag, q_abs_mean}
with absolute model layer indices (e.g. 3, 7, ..., 63) so the runtime
``kv_group_resolver`` maps them straight onto the model's full-attention layers.
TP head-shard slicing is handled at serving time, so one full ``.pt`` suffices.

Usage
-----
    python scripts/calibrate.py \
        --model /path/to/Qwen3.5-27B \
        --input calib.txt \
        --output stats.pt \
        --max-length 8192 \
        --device npu \
        --attn-implementation eager
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Model-structure discovery (robust to Qwen3.5 hybrid + nested text config)
# ---------------------------------------------------------------------------

def _maybe_import_torch_npu(device: str) -> None:
    """Import torch_npu so that ``npu`` devices resolve. No-op elsewhere."""
    if str(device).lower().startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "device='npu' requested but torch_npu could not be imported."
            ) from exc


def _text_config(config) -> object:
    """Return the text sub-config for nested multimodal configs."""
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        return text_cfg
    # Some builds expose get_text_config()
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    return config


def _find_text_backbone(model: nn.Module) -> nn.Module:
    """Locate the text decoder backbone.

    The text backbone is the unique submodule owning both a ``layers``
    ``ModuleList`` and a ``rotary_emb``. The vision tower uses ``blocks`` and a
    differently named rotary, so it is naturally excluded. If several match,
    pick the one with the most layers.
    """
    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and hasattr(module, "rotary_emb"):
            candidates.append((name, module))
    if not candidates:
        # Fallback: any module with a `layers` ModuleList.
        for name, module in model.named_modules():
            layers = getattr(module, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                candidates.append((name, module))
    if not candidates:
        raise RuntimeError(
            "Could not locate the text backbone: no submodule exposes a "
            "`layers` ModuleList. Is this a supported decoder model?"
        )
    _, backbone = max(candidates, key=lambda item: len(item[1].layers))
    return backbone


def _iter_full_attention_attns(backbone: nn.Module) -> List[Tuple[int, nn.Module]]:
    """Return (absolute_layer_idx, self_attn_module) for full-attention layers."""
    result: List[Tuple[int, nn.Module]] = []
    for pos, layer in enumerate(backbone.layers):
        layer_type = getattr(layer, "layer_type", None)
        attn = getattr(layer, "self_attn", None)
        is_full = (layer_type == "full_attention") or (attn is not None)
        if not is_full or attn is None:
            continue
        # Prefer the module's own layer_idx (authoritative); fall back to loop pos.
        layer_idx = int(getattr(attn, "layer_idx", pos))
        result.append((layer_idx, attn))
    if not result:
        raise RuntimeError(
            "No full-attention layers with `self_attn` found. This model may "
            "not use a hybrid linear/full-attention layout as expected."
        )
    return result


def _rotary_freq_count(backbone: nn.Module, head_dim: int) -> Tuple[int, torch.Tensor]:
    """Return (freq_count, inv_freq) from the model's own rotary embedding.

    Using the live ``rotary_emb.inv_freq`` guarantees the calibration frequency
    basis matches the model exactly (partial rotary, theta, any scaling).
    """
    rotary = getattr(backbone, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None) if rotary is not None else None
    if not isinstance(inv_freq, torch.Tensor) or inv_freq.numel() == 0:
        raise RuntimeError(
            "Could not read rotary_emb.inv_freq from the text backbone; cannot "
            "determine the rotary frequency basis."
        )
    inv_freq = inv_freq.detach().to(device="cpu", dtype=torch.float32).contiguous()
    freq_count = int(inv_freq.numel())
    if freq_count * 2 > head_dim:
        raise RuntimeError(
            f"Derived rotary_dim ({freq_count * 2}) exceeds head_dim ({head_dim})."
        )
    return freq_count, inv_freq


# ---------------------------------------------------------------------------
# Main calibration logic
# ---------------------------------------------------------------------------

class _HeadAccumulator:
    """Per-layer running sums of pre-RoPE Q complex-pair statistics (CPU fp64)."""

    def __init__(self, num_heads: int, freq_count: int):
        self.sum_real = torch.zeros(num_heads, freq_count, dtype=torch.float64)
        self.sum_imag = torch.zeros(num_heads, freq_count, dtype=torch.float64)
        self.sum_abs = torch.zeros(num_heads, freq_count, dtype=torch.float64)
        self.count = 0

    def update(self, real: torch.Tensor, imag: torch.Tensor) -> None:
        # real/imag: [q_len, num_heads, freq_count] (fp32, any device)
        absval = torch.sqrt(real * real + imag * imag)
        self.sum_real += real.sum(dim=0).to(device="cpu", dtype=torch.float64)
        self.sum_imag += imag.sum(dim=0).to(device="cpu", dtype=torch.float64)
        self.sum_abs += absval.sum(dim=0).to(device="cpu", dtype=torch.float64)
        self.count += int(real.shape[0])


def calibrate(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    max_length: int = 8192,
    device: str = "npu",
    attn_implementation: str = "eager",
    device_map: str = "auto",
) -> None:
    _maybe_import_torch_npu(device)
    dtype = torch.bfloat16

    # --- Load config, tokenizer, model (sharded across all visible cards) ---
    print(f"Loading model: {model_name_or_path}", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # A single-card 27B does not fit; shard the weights across every visible
    # NPU with device_map="auto". The forward pass dispatches across shards.
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()

    text_cfg = _text_config(config)
    num_heads = int(getattr(text_cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_heads))
    head_dim = int(
        getattr(text_cfg, "head_dim", None)
        or (int(text_cfg.hidden_size) // num_heads)
    )

    backbone = _find_text_backbone(model)
    freq_count, inv_freq = _rotary_freq_count(backbone, head_dim)
    rotary_dim = freq_count * 2
    print(
        f"head_dim={head_dim} rotary_dim={rotary_dim} freq_count={freq_count} "
        f"num_heads={num_heads} num_kv_heads={num_kv_heads}",
        file=sys.stderr,
    )

    full_attn = _iter_full_attention_attns(backbone)
    full_attn_layers = sorted(idx for idx, _ in full_attn)
    print(
        f"Full-attention layers ({len(full_attn_layers)}): {full_attn_layers}",
        file=sys.stderr,
    )

    # --- Read and tokenize input ---
    print(f"Reading input: {input_path}", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    embed_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(embed_device)
    seq_len = int(input_ids.shape[1])
    print(f"Tokenized length: {seq_len}", file=sys.stderr)

    # --- Register pre-hooks to capture pre-RoPE Q on full-attention layers ---
    accumulators: Dict[int, _HeadAccumulator] = {}
    handles = []

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            hd = int(module.head_dim)
            bsz, q_len, _ = hidden_states.shape
            # Replicate the model's pre-RoPE query construction exactly:
            #   q_proj -> view(..., -1, head_dim*2) -> chunk(2) -> drop gate -> q_norm
            qkv = module.q_proj(hidden_states)
            qkv = qkv.view(bsz, q_len, -1, hd * 2)
            query, _gate = torch.chunk(qkv, 2, dim=-1)  # [bsz, q_len, nheads, hd]
            query = module.q_norm(query)  # RMSNorm over head_dim, still pre-RoPE
            query = query[0].to(torch.float32)  # [q_len, nheads, hd] (bsz=1)
            # Rotary sub-block only; half-layout complex pairing (j, j+freq_count).
            q_rot = query[..., :rotary_dim]
            real = q_rot[..., :freq_count].contiguous()
            imag = q_rot[..., freq_count:rotary_dim].contiguous()
            acc = accumulators.get(layer_idx)
            if acc is None:
                acc = _HeadAccumulator(real.shape[1], freq_count)
                accumulators[layer_idx] = acc
            acc.update(real, imag)

        return hook_fn

    for layer_idx, attn in full_attn:
        handles.append(
            attn.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True)
        )

    # --- Forward pass (batch_size=1, no labels needed) ---
    print("Running forward pass...", file=sys.stderr)
    with torch.inference_mode():
        model(input_ids)
    print("Forward pass complete.", file=sys.stderr)

    for handle in handles:
        handle.remove()

    # --- Reduce accumulators into per-head stats ---
    print("Computing frequency statistics...", file=sys.stderr)
    stats_dict: Dict[str, Dict[str, torch.Tensor]] = {}
    sampled_heads: List[Tuple[int, int]] = []
    for layer_idx in sorted(accumulators.keys()):
        acc = accumulators[layer_idx]
        if acc.count == 0:
            print(f"  [warn] No tokens captured for layer {layer_idx}", file=sys.stderr)
            continue
        mean_real = (acc.sum_real / acc.count).to(torch.float32)
        mean_imag = (acc.sum_imag / acc.count).to(torch.float32)
        mean_abs = (acc.sum_abs / acc.count).to(torch.float32)
        for head_idx in range(mean_real.shape[0]):
            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            stats_dict[key] = {
                "q_mean_real": mean_real[head_idx].clone(),
                "q_mean_imag": mean_imag[head_idx].clone(),
                "q_abs_mean": mean_abs[head_idx].clone(),
            }
            sampled_heads.append((layer_idx, head_idx))

    # --- RoPE metadata ---
    rope_params = getattr(text_cfg, "rope_parameters", None)
    if isinstance(rope_params, dict):
        rope_theta = rope_params.get("rope_theta", getattr(text_cfg, "rope_theta", 10000.0))
        rope_type = rope_params.get("rope_type") or "default"
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", getattr(text_cfg, "partial_rotary_factor", 1.0)
        )
    else:
        rope_theta = getattr(text_cfg, "rope_theta", 10000.0)
        rope_type = getattr(text_cfg, "rope_type", "default") or "default"
        partial_rotary_factor = getattr(text_cfg, "partial_rotary_factor", 1.0)

    metadata = {
        "num_traces": 1,
        "head_dim": head_dim,
        # Rotary basis: authoritative freq_count/inv_freq for partial rotary.
        "rotary_dim": rotary_dim,
        "freq_count": freq_count,
        "partial_rotary_factor": float(partial_rotary_factor),
        "inv_freq": inv_freq.tolist(),
        "num_attention_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "dtype": str(dtype).replace("torch.", ""),
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": attn_implementation,
        "rope_style": "half",
        "rope_type": rope_type,
        "rope_theta": float(rope_theta),
        "full_attention_layers": full_attn_layers,
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
    }

    payload = {"metadata": metadata, "stats": stats_dict}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(
        f"Saved stats to {out} "
        f"({len(sampled_heads)} heads across {len(full_attn_layers)} layers, "
        f"freq_count={freq_count})",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate TriAttention frequency statistics (Qwen3.5 hybrid)."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument("--input", required=True, help="Plain text calibration file.")
    parser.add_argument("--output", required=True, help="Output .pt path for stats.")
    parser.add_argument(
        "--max-length", type=int, default=8192, help="Max token length (default: 8192)."
    )
    parser.add_argument(
        "--device", default="npu", help="Target device family (default: npu)."
    )
    parser.add_argument(
        "--attn-implementation", default="eager",
        help="Attention implementation (default: eager).",
    )
    parser.add_argument(
        "--device-map", default="auto",
        help="HF device_map for sharded loading across cards (default: auto).",
    )
    args = parser.parse_args()
    calibrate(
        model_name_or_path=args.model,
        input_path=args.input,
        output_path=args.output,
        max_length=args.max_length,
        device=args.device,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
