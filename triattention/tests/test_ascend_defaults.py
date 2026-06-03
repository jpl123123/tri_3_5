from triattention.vllm.runtime.ascend_defaults import (
    apply_ascend_fast_recency_defaults,
)
from triattention.vllm.runtime.config import TriAttentionRuntimeConfig


def test_auto_fast_recency_enables_safe_ascend_zero_copy_defaults():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(config, env={})

    assert config.fast_recency_only
    assert not config.fast_recency_accuracy_guard


def test_auto_fast_recency_respects_explicit_user_mode():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=False,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_FAST_RECENCY_ONLY": "0"},
    )

    assert not config.fast_recency_only
    assert config.fast_recency_accuracy_guard


def test_auto_fast_recency_overrides_stale_accuracy_guard():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD": "1"},
    )

    assert config.fast_recency_only
    assert not config.fast_recency_accuracy_guard


def test_auto_fast_recency_keeps_documented_decode_reclaim_interval():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        auto_fast_recency_on_ascend=True,
    )

    apply_ascend_fast_recency_defaults(config, env={})

    assert config.min_reclaim_blocks_on_ascend == 8


def test_auto_fast_recency_overrides_stale_reclaim_interval():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=False,
        auto_fast_recency_on_ascend=True,
        min_reclaim_blocks_on_ascend=8,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_MIN_RECLAIM_BLOCKS_ON_ASCEND": "8"},
    )

    assert config.min_reclaim_blocks_on_ascend == 8


def test_auto_fast_recency_can_be_disabled_to_keep_accuracy_guard():
    config = TriAttentionRuntimeConfig(
        fast_recency_only=True,
        fast_recency_accuracy_guard=True,
        auto_fast_recency_on_ascend=False,
    )

    apply_ascend_fast_recency_defaults(
        config,
        env={"TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD": "1"},
    )

    assert config.fast_recency_only
    assert config.fast_recency_accuracy_guard


def test_early_install_proxy_on_ascend_defaults_to_lazy():
    assert not TriAttentionRuntimeConfig().early_install_proxy_on_ascend
