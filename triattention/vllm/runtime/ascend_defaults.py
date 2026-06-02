"""Ascend-specific runtime defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import TriAttentionRuntimeConfig

_AUTO_FAST_RECENCY_MIN_RECLAIM_BLOCKS_ON_ASCEND = 1


def apply_ascend_fast_recency_defaults(
    config: TriAttentionRuntimeConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    if not bool(getattr(config, "auto_fast_recency_on_ascend", True)):
        return
    env_map = os.environ if env is None else env
    if env_map.get("TRIATTN_RUNTIME_FAST_RECENCY_ONLY") is None:
        config.fast_recency_only = True
    if bool(getattr(config, "fast_recency_only", False)):
        # The Ascend auto-fast path is intentionally a recency-only path.  A
        # stale accuracy-guard export silently routes it back to the expensive
        # stats/scoring path and hurts decode throughput, so auto mode owns this
        # default unless the caller disables auto or fast-recency entirely.
        config.fast_recency_accuracy_guard = False
        # Zero-copy recency compaction is cheap on Ascend, while letting decode
        # drift by 8 blocks dilutes most of the attention-length reduction for
        # 10k/1k latency tests. Auto mode owns this perf-critical default even
        # when a stale reclaim env is present; disable auto-fast-recency to keep
        # a custom interval.
        config.min_reclaim_blocks_on_ascend = (
            _AUTO_FAST_RECENCY_MIN_RECLAIM_BLOCKS_ON_ASCEND
        )
