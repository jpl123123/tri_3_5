"""Ascend-specific runtime defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import TriAttentionRuntimeConfig


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
