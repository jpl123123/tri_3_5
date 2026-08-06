"""Dynamic per-request KV budget based on input (prefill) length.

When ``TRIATTN_RUNTIME_DYNAMIC_KV_BUDGET=1`` is set, the runtime ignores the
global ``TRIATTN_RUNTIME_KV_BUDGET`` env value for per-request compression
decisions and instead derives the KV budget from each request's prefill length
(total prompt token count) using the bucket mapping below.

Bucket boundaries use 1k = 1024 tokens, left-closed / right-open except the last:

    prefill_len < 16384          -> None  (compression disabled for this request)
    [16384, 40960)               -> 12288
    [40960, 65536)               -> 32256
    [65536, 98304)               -> 84480
    [98304, 131072)              -> 115200
    [131072, 147456)             -> 130560
    [147456, 163840)             -> 145920
    [163840, 196608)             -> 176640
    prefill_len >= 196608        -> 176640

Returning ``None`` means "do not engage TriAttention for this request at all":
the scheduler emits no compression signal, so the worker hook (and thus the
scoring kernel) is never invoked for it. This matches the behavior of setting
``ENABLE_TRIATTENTION=0`` but scoped to a single short request rather than
globally.

All budget values and all bucket boundaries are multiples of 128 (the typical
vLLM paged-attention block size), so the Ascend zero-copy recency fast path
(which requires ``budget % block_size == 0``) stays usable.
"""

from __future__ import annotations

DYNAMIC_KV_BUDGET_UPPER_BOUND: int = 176640
"""Largest budget the mapping can ever return.

Used as the selector's baked-in ``kv_budget`` (sizes the trig cache table via
``max(offset_max_length, kv_budget + divide_length)``) and as the worst-case
headroom basis for ``_compute_max_chunk_for_compression`` when dynamic mode is
on. Keeping this as a constant separate from the mapping makes the upper-bound
contract explicit even if the mapping is edited later.
"""

_BUCKETS: tuple[tuple[int, int, int], ...] = (
    (16384, 40960, 12288),
    (40960, 65536, 32256),
    (65536, 98304, 84480),
    (98304, 131072, 115200),
    (131072, 147456, 130560),
    (147456, 163840, 145920),
    (163840, 196608, 176640),
)
"""(low_inclusive, high_exclusive, budget) tuples, ordered by low ascending."""

_DISABLED_THRESHOLD: int = 16384
"""prefill_len strictly below this value disables compression for the request."""


def resolve_dynamic_kv_budget(prefill_len: int) -> int | None:
    """Resolve a per-request KV budget from the input (prefill) length.

    Args:
        prefill_len: Total prompt token count for the request.

    Returns:
        The KV budget as an int, or ``None`` when compression must be disabled
        for this request (prefill_len < 16384). No interpolation is performed:
        the first matching bucket wins.
    """
    n = int(prefill_len or 0)
    if n < _DISABLED_THRESHOLD:
        return None
    if n >= 196608:
        return 176640
    for low, high, budget in _BUCKETS:
        if low <= n < high:
            return budget
    return 176640


def dynamic_kv_budget_upper_bound() -> int:
    """Return the maximum budget the dynamic mapping can produce."""
    return DYNAMIC_KV_BUDGET_UPPER_BOUND
