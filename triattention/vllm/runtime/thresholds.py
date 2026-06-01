"""Compression threshold helpers shared by scheduler and worker paths."""

from __future__ import annotations

import os
import sys
from typing import Any

from .config import TriAttentionRuntimeConfig


_ASCEND_ENV_CACHE = False


def is_ascend_environment_available() -> bool:
    """Best-effort detect vLLM-Ascend when scheduler classes stay upstream."""
    global _ASCEND_ENV_CACHE
    if _ASCEND_ENV_CACHE:
        return True
    if any(
        name == "vllm_ascend" or name.startswith("vllm_ascend.")
        for name in sys.modules
    ):
        _ASCEND_ENV_CACHE = True
        return True
    for env_name in (
        "ASCEND_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "VLLM_TARGET_DEVICE",
        "DEVICE_TARGET",
    ):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        value = raw.strip().lower()
        if not value or value in {"-1", "none", "cpu"}:
            continue
        if "ascend" in value or "npu" in value:
            _ASCEND_ENV_CACHE = True
            return True
        if "ascend" in env_name.lower() or "npu" in env_name.lower():
            _ASCEND_ENV_CACHE = True
            return True
    return False


def is_ascend_runtime(obj: Any) -> bool:
    module_name = getattr(type(obj), "__module__", "")
    if isinstance(module_name, str) and module_name.startswith("vllm_ascend."):
        return True
    if "vllm_ascend" in repr(type(obj)).lower():
        return True
    for candidate in (
        getattr(obj, "device_config", None),
        getattr(getattr(obj, "vllm_config", None), "device_config", None),
    ):
        for attr_name in ("device", "device_type"):
            raw = getattr(candidate, attr_name, None)
            if raw is None:
                continue
            value = str(raw).lower()
            if "npu" in value or "ascend" in value:
                return True
    return is_ascend_environment_available()


def compression_reclaim_interval_tokens(
    config: TriAttentionRuntimeConfig,
    *,
    block_size: int,
    is_ascend: bool,
) -> int:
    block_size_i = max(1, int(block_size or 1))
    min_reclaim_blocks = max(0, int(getattr(config, "min_reclaim_blocks", 0) or 0))
    if is_ascend:
        min_reclaim_blocks = max(
            min_reclaim_blocks,
            max(0, int(getattr(config, "min_reclaim_blocks_on_ascend", 0) or 0)),
        )
    min_reclaim_tokens = min_reclaim_blocks * block_size_i
    return max(1, int(config.divide_length), min_reclaim_tokens)


def compression_length_threshold(
    config: TriAttentionRuntimeConfig,
    *,
    prefill_len: int,
    block_size: int,
    is_ascend: bool,
) -> int:
    threshold = int(config.kv_budget) + compression_reclaim_interval_tokens(
        config,
        block_size=block_size,
        is_ascend=is_ascend,
    )
    if config.protect_prefill and not config.include_prefill_in_budget:
        threshold += max(0, int(prefill_len))
    return threshold
