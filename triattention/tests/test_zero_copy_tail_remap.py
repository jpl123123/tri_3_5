import sys
import types

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig

if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)

from triattention.vllm.runtime.hook_group_pipeline import (  # noqa: E402
    try_build_recency_tail_block_remap,
)


def _config():
    return TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        enable_zero_copy_recency=True,
        enable_experimental_block_reclaim=True,
    )


def test_zero_copy_tail_remap_preserves_decode_trailing_block():
    outcome = try_build_recency_tail_block_remap(
        config=_config(),
        mutable_block_ids_by_group=[list(range(80))],
        effective_tokens=10000,
        budget_total=2048,
        block_size=128,
    )

    assert outcome is not None
    assert outcome.cache_len_after == 1936
    assert outcome.mutable_block_ids_by_group == [list(range(63, 80))]
    assert outcome.block_reclaim_groups[0].block_ids_removed == list(range(63))


def test_zero_copy_tail_remap_still_handles_exact_block_table():
    outcome = try_build_recency_tail_block_remap(
        config=_config(),
        mutable_block_ids_by_group=[list(range(79))],
        effective_tokens=10000,
        budget_total=2048,
        block_size=128,
    )

    assert outcome is not None
    assert outcome.cache_len_after == 1936
    assert outcome.mutable_block_ids_by_group == [list(range(63, 79))]
    assert outcome.block_reclaim_groups[0].block_ids_removed == list(range(63))
