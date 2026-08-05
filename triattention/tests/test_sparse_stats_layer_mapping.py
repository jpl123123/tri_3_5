from types import SimpleNamespace

import torch

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.selector_hf import build_triattention_selector
from triattention.vllm.core.utils import load_frequency_stats


def test_rkv_sparse_sampled_layers_preserve_max_layer_index(tmp_path):
    stats_path = tmp_path / "qwen35_sparse_layers.pt"
    freq_count = 4
    stats = {}
    for layer_idx in (0, 7, 15, 19):
        for head_idx in range(2):
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.ones(freq_count),
                "q_mean_imag": torch.zeros(freq_count),
                "q_abs_mean": torch.ones(freq_count),
            }
    torch.save(
        {
            "metadata": {
                "head_dim": freq_count * 2,
                "rope_style": "half",
                "sampled_heads": [[layer_idx, 0] for layer_idx in (0, 7, 15, 19)],
            },
            "stats": stats,
        },
        stats_path,
    )

    metadata, head_stats = load_frequency_stats(
        stats_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=2,
    )

    assert metadata["num_layers"] == 20
    assert sorted(head_stats) == [0, 7, 15, 19]
    assert head_stats[19]["freq_scale_sq"].shape == (2, freq_count)


def test_selector_falls_back_when_stats_freq_mismatches_runtime_kv(tmp_path):
    stats_path = tmp_path / "freq4_stats.pt"
    freq_count = 4
    layer_stats = {
        0: {
            "q_mean_complex": torch.zeros(2, freq_count, 2),
            "q_abs_mean": torch.ones(2, freq_count),
            "freq_scale_sq": torch.ones(2, freq_count),
        },
    }
    torch.save(
        {
            "metadata": {
                "num_attention_heads": 2,
                "num_kv_heads": 2,
                "head_dim": freq_count * 2,
                "num_layers": 1,
                "rope_style": "half",
            },
            "layer_stats": layer_stats,
        },
        stats_path,
    )
    config = TriAttentionRuntimeConfig(
        kv_budget=4,
        window_size=1,
        sparse_stats_path=stats_path,
        enable_experimental_kv_compaction=True,
        require_triton_scoring=True,
        scoring_backend="torch",
        sparse_normalize_scores=False,
        log_execution_path=False,
    )

    selector, _group_selector, status = build_triattention_selector(
        config,
        base_runner=SimpleNamespace(device=torch.device("cpu")),
    )

    assert selector is not None
    assert status.startswith("enabled:torch")
    # Runtime head_dim=16 means runtime freq_count=8, while stats only have 4.
    result = selector(
        keys_dense=torch.zeros(1, 2, 8, 16),
        total_tokens=8,
        prefill_len=0,
        protect_prefill=False,
        layer_idx=0,
        round_start=8,
        budget_total=4,
        req_id="req",
        gid=0,
    )

    assert result["mode"] == "shared"
    assert result["semantic"] == "recency_fallback_stats_incompatible"
    assert result["indices"] == [4, 5, 6, 7]
