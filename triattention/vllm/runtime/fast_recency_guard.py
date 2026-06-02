"""Fast-recency safety guards."""

from __future__ import annotations

from .config import TriAttentionRuntimeConfig


def should_guard_fast_recency_long_context(
    *,
    config: TriAttentionRuntimeConfig,
    effective_tokens: int,
    prefill_len: int,
) -> bool:
    if not bool(getattr(config, "fast_recency_only", False)):
        return False
    if not bool(getattr(config, "fast_recency_long_context_guard", True)):
        return False
    if getattr(config, "sparse_stats_path", None) is None:
        return False
    threshold = int(getattr(config, "fast_recency_long_context_guard_tokens", 0) or 0)
    if threshold <= 0:
        return False
    context_len = max(int(effective_tokens), int(prefill_len))
    return context_len >= threshold
