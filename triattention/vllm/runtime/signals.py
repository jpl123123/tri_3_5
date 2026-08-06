"""Signal schema exchanged between scheduler and model runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TriggerReason = Literal["none", "length_threshold", "kv_usage_threshold"]


@dataclass(frozen=True)
class CompressionSignal:
    """Per-request compression signal for one scheduler step."""

    req_id: str
    should_compress: bool
    reason: TriggerReason
    estimated_cache_len: int
    step: int
    kv_usage: float | None
    protect_prefill: bool
    prefill_len: int
    # Number of tokens scheduled for this request in the current scheduler step.
    # This can be >1 for chunked prefill or speculative decode validation.
    scheduled_tokens: int = 1
    # Worker-local hard boundary triggers cannot be delayed without risking a
    # slot write past the current block-table capacity.
    force: bool = False
    # Per-request KV budget. 0 means "not set"; consumers fall back to the
    # global config.kv_budget. When dynamic_kv_budget is enabled, the scheduler
    # fills this from resolve_dynamic_kv_budget(prefill_len) so each request is
    # compressed to its own budget instead of the global default.
    kv_budget: int = 0
