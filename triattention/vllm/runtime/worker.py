"""TriAttention v2 worker integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vllm.logger import logger

try:
    from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker
except Exception:  # pragma: no cover - vLLM-Ascend may not import CUDA worker
    VLLMGPUWorker = object  # type: ignore[assignment]

from .config import TriAttentionRuntimeConfig
from .hook_impl import install_runner_compression_hook
from .runner import TriAttentionModelRunner
from .thresholds import compression_length_threshold, is_ascend_runtime


def _debug_early_install_proxy_enabled() -> bool:
    return os.environ.get("TRIATTN_DEBUG_EARLY_INSTALL_PROXY", "0") == "1"


def _maybe_backfill_model_path(worker: Any, config: TriAttentionRuntimeConfig) -> None:
    if config.model_path is not None:
        return
    model_config = getattr(worker, "model_config", None)
    model_path = getattr(model_config, "model", None)
    if isinstance(model_path, str) and model_path.strip():
        config.model_path = Path(model_path.strip())


def _get_block_size_from_model_runner(model_runner: Any) -> int | None:
    cache_config = getattr(model_runner, "cache_config", None)
    block_size = int(getattr(cache_config, "block_size", 0) or 0)
    return block_size if block_size > 0 else None


def _get_actual_kv_from_model_runner(model_runner: Any, req_id: str) -> int | None:
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is None:
        return None
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        return None
    req_index = req_id_to_index.get(req_id)
    if not isinstance(req_index, int):
        return None
    block_table_obj = getattr(input_batch, "block_table", None)
    if block_table_obj is None:
        return None
    inner_tables = getattr(block_table_obj, "block_tables", None)
    first_table = (
        inner_tables[0]
        if isinstance(inner_tables, list) and inner_tables
        else block_table_obj
    )
    num_blocks_per_row = getattr(first_table, "num_blocks_per_row", None)
    if num_blocks_per_row is None:
        return None
    block_size = _get_block_size_from_model_runner(model_runner)
    if block_size is None:
        return None
    return int(num_blocks_per_row[req_index]) * block_size


def _compression_threshold_for_signal(
    config: TriAttentionRuntimeConfig,
    signal: Any,
    *,
    block_size: int,
    is_ascend: bool,
) -> int:
    return compression_length_threshold(
        config,
        prefill_len=max(0, int(getattr(signal, "prefill_len", 0) or 0)),
        block_size=block_size,
        is_ascend=is_ascend,
    )


def should_install_triattention_runner_proxy(
    worker: Any,
    scheduler_output: Any,
) -> bool:
    """Avoid installing the proxy for stale scheduler triggers below budget."""
    signals = getattr(scheduler_output, "triattention_signals", None)
    if not signals:
        return False
    if getattr(worker, "_triattention_runner_proxy_installed", False):
        return True

    model_runner = getattr(worker, "model_runner", None)
    config = getattr(worker, "_triattention_runtime_config", None)
    if config is None:
        config = TriAttentionRuntimeConfig.from_env()
        worker._triattention_runtime_config = config

    saw_trigger_without_worker_length = False
    block_size = _get_block_size_from_model_runner(model_runner)
    for req_id, signal in signals.items():
        if not bool(getattr(signal, "should_compress", False)):
            continue
        actual_kv = _get_actual_kv_from_model_runner(model_runner, str(req_id))
        if actual_kv is None or block_size is None:
            saw_trigger_without_worker_length = True
            continue
        if actual_kv >= _compression_threshold_for_signal(
            config,
            signal,
            block_size=block_size,
            is_ascend=is_ascend_runtime(worker) or is_ascend_runtime(model_runner),
        ):
            return True
    return saw_trigger_without_worker_length


class TriAttentionWorker(VLLMGPUWorker):
    """GPU worker that injects TriAttention model-runner proxy."""

    def init_device(self):
        super_init_device = getattr(super(), "init_device", None)
        if not callable(super_init_device):
            raise RuntimeError("vLLM GPU Worker is unavailable")
        super_init_device()
        if isinstance(self.model_runner, TriAttentionModelRunner):
            return

        # Keep native vLLM GPUModelRunner untouched during warmup/graph-capture and
        # pre-trigger decode. We lazily wrap on the first step that carries a
        # TriAttention signal (trigger/compressed-request update), which minimizes
        # impact on the common no-compression path.
        self._triattention_runtime_config = TriAttentionRuntimeConfig.from_env()
        _maybe_backfill_model_path(self, self._triattention_runtime_config)
        self._triattention_runner_proxy_installed = False
        if _debug_early_install_proxy_enabled():
            self._ensure_triattention_runner_proxy()
            logger.debug("TriAttentionWorker: eagerly installed runner proxy during init_device")

    def _ensure_triattention_runner_proxy(self) -> None:
        if getattr(self, "_triattention_runner_proxy_installed", False):
            return
        if isinstance(self.model_runner, TriAttentionModelRunner):
            self._triattention_runner_proxy_installed = True
            return
        config = getattr(self, "_triattention_runtime_config", None) or TriAttentionRuntimeConfig.from_env()
        _maybe_backfill_model_path(self, config)
        base_runner = self.model_runner
        install_runner_compression_hook(base_runner=base_runner, config=config)
        self.model_runner = TriAttentionModelRunner(
            base_runner=base_runner,
            config=config,
        )
        self._triattention_runner_proxy_installed = True
        logger.info(
            "TriAttentionWorker lazily injected runner proxy: budget=%d divide_length=%d "
            "seq_len_override_patch=%s stats_path=%s model_path=%s protect_prefill=%s "
            "window_size=%s",
            config.kv_budget,
            config.divide_length,
            "deferred",
            str(config.sparse_stats_path) if config.sparse_stats_path is not None else None,
            str(config.model_path) if config.model_path is not None else None,
            config.protect_prefill,
            config.window_size,
        )

    def execute_model(self, scheduler_output):  # type: ignore[override]
        # Sparse scheduler signals are empty in the common pre-trigger path.
        # Install the proxy only when TriAttention behavior is actually needed.
        if should_install_triattention_runner_proxy(self, scheduler_output):
            self._ensure_triattention_runner_proxy()
        super_execute_model = getattr(super(), "execute_model", None)
        if not callable(super_execute_model):
            raise RuntimeError("vLLM GPU Worker is unavailable")
        return super_execute_model(scheduler_output)
