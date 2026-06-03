from pathlib import Path

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.fast_recency_guard import (
    should_guard_fast_recency_long_context,
)


def _config(**overrides):
    defaults = {
        "fast_recency_only": True,
        "fast_recency_accuracy_guard": False,
        "fast_recency_long_context_guard": True,
        "fast_recency_long_context_guard_tokens": 16384,
        "sparse_stats_path": Path("/tmp/triattention-stats.pt"),
    }
    defaults.update(overrides)
    return TriAttentionRuntimeConfig(**defaults)


def test_fast_recency_guard_allows_10k_benchmark_context():
    assert not should_guard_fast_recency_long_context(
        config=_config(),
        effective_tokens=10000,
        prefill_len=10000,
    )


def test_fast_recency_guard_boundary():
    assert not should_guard_fast_recency_long_context(
        config=_config(),
        effective_tokens=16383,
        prefill_len=10000,
    )
    assert should_guard_fast_recency_long_context(
        config=_config(),
        effective_tokens=16384,
        prefill_len=10000,
    )


def test_fast_recency_guard_blocks_20k_accuracy_risk():
    assert should_guard_fast_recency_long_context(
        config=_config(),
        effective_tokens=19789,
        prefill_len=19789,
    )


def test_fast_recency_guard_allows_sparse_accuracy_guard():
    assert not should_guard_fast_recency_long_context(
        config=_config(fast_recency_accuracy_guard=True),
        effective_tokens=19789,
        prefill_len=19789,
    )


def test_fast_recency_guard_blocks_20k_even_without_sparse_stats():
    assert should_guard_fast_recency_long_context(
        config=_config(sparse_stats_path=None),
        effective_tokens=19789,
        prefill_len=19789,
    )
