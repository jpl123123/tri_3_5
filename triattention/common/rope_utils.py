"""Rotary position embedding utilities for TriAttention."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoConfig

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
except ImportError:
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as Qwen3RotaryEmbedding
    except ImportError:
        Qwen3RotaryEmbedding = None  # type: ignore[assignment]
try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
except ImportError:
    LlamaRotaryEmbedding = None  # type: ignore[assignment]


def determine_rope_style(config: AutoConfig) -> str:
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        return "half"
    return "half"  # front/back pairing (Qwen)


def _normalize_rope_scaling(rope_scaling: Optional[dict]) -> dict:
    """Normalize a model config's ``rope_scaling`` dict for transformers compatibility.

    Some checkpoints (e.g. Qwen3-32B) store ``rope_scaling["factor"]`` as an
    ``int`` (e.g. ``3``). Older transformers releases (v4.45) emit a warning
    ``rope_scaling's factor field must be a float >= 1, got 3`` and, depending
    on the code path, may skip applying the YaRN/linear scaling factor when the
    value is not a ``float``. TriAttention derives ``inv_freq`` and
    ``freq_scale_sq`` from this rotary embedding, so a silently-dropped scaling
    factor corrupts the scoring formula and causes accuracy regressions.

    This helper coerces the numeric fields (``factor``, ``attention_factor``,
    ``beta_fast``, ``beta_slow``, ``low_freq_factor``, ``high_freq_factor``)
    to ``float`` and migrates legacy keys (``attn_factor`` -> ``attention_factor``,
    ``type`` -> ``rope_type``) before the rotary embedding is constructed.
    """
    scaling = dict(rope_scaling or {})
    if "attn_factor" in scaling and "attention_factor" not in scaling:
        scaling["attention_factor"] = scaling["attn_factor"]
    scaling.pop("attn_factor", None)
    if "rope_type" not in scaling:
        scaling["rope_type"] = scaling.get("type", "default")
    scaling.pop("type", None)

    # Coerce numeric fields to float so transformers (v4.45 and similar) does
    # not reject integer factors. This keeps YaRN/linear scaling intact and
    # silences the warning that confused accuracy debugging on Qwen3.
    _float_keys = (
        "factor",
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "low_freq_factor",
        "high_freq_factor",
    )
    for key in _float_keys:
        value = scaling.get(key)
        if isinstance(value, bool):
            # bool is a subclass of int; never coerce booleans.
            continue
        if isinstance(value, int):
            scaling[key] = float(value)
    return scaling


def build_rotary(
    cache_device: torch.device,
    model_path: Path,
    dtype: torch.dtype,
    config: Optional[AutoConfig] = None,
) -> object:
    if config is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    rope_style = determine_rope_style(config)
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        if LlamaRotaryEmbedding is None:
            raise ImportError("Llama rotary embedding is unavailable in the current transformers build.")

        config.rope_scaling = _normalize_rope_scaling(getattr(config, "rope_scaling", None))
        rotary = LlamaRotaryEmbedding(config=config, device=cache_device)
        rotary.to(dtype=dtype, device=cache_device)
        rotary._rope_style = rope_style  # type: ignore[attr-defined]
        return rotary

    if Qwen3RotaryEmbedding is None:
        raise ImportError(
            "Neither Qwen3 nor Qwen2 rotary embeddings are available in the installed transformers package."
        )

    config.rope_scaling = _normalize_rope_scaling(getattr(config, "rope_scaling", None))
    rotary = Qwen3RotaryEmbedding(config=config, device=cache_device)
    rotary.to(dtype=dtype)
    rotary._rope_style = rope_style  # type: ignore[attr-defined]
    return rotary


def compute_frequency_scaling(
    rotary: Any,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]
    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    return scale.to(device=device, dtype=torch.float32)
