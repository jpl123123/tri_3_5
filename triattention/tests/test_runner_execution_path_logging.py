import sys
import types


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


if "vllm" not in sys.modules:
    sys.modules["vllm"] = types.SimpleNamespace()
if "vllm.logger" not in sys.modules:
    sys.modules["vllm.logger"] = types.SimpleNamespace(logger=_Logger())
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(
        Tensor=object,
        is_tensor=lambda value: False,
    )
if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.SimpleNamespace()

from triattention.vllm.runtime.runner import TriAttentionModelRunner
from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.signals import CompressionSignal


def test_runner_trigger_guard_marks_pre_core_skip():
    logger = _Logger()
    runner = object.__new__(TriAttentionModelRunner)
    runner._log_execution_path = True
    runner._logged_execution_path_trigger_guards = set()
    runner._last_step = 11
    runner._logger = logger

    runner._log_execution_path_trigger_guard(
        req_id="req-1",
        reason="fast_recency_long_context_guard",
        hint="set_sparse_stats_path_or_disable_long_context_guard",
        prefill_len=19789,
    )
    runner._log_execution_path_trigger_guard(
        req_id="req-1",
        reason="fast_recency_long_context_guard",
        hint="set_sparse_stats_path_or_disable_long_context_guard",
        prefill_len=19789,
    )

    assert len(logger.lines) == 1
    line = logger.lines[0]
    assert "TRIATTN_EXEC_PATH runner_trigger_guard" in line
    assert "reason=fast_recency_long_context_guard" in line
    assert "core_entered=False" in line
    assert "hint=set_sparse_stats_path_or_disable_long_context_guard" in line


class _AscendRunner:
    pass


_AscendRunner.__module__ = "vllm_ascend.test"


class _StateStore:
    def __init__(self):
        self.state = types.SimpleNamespace(
            prefill_len=9863,
            compression_count=0,
            current_cache_len=0,
        )
        self.skipped = None

    def get(self, req_id):
        return self.state

    def mark_compression_skipped(self, **kwargs):
        self.skipped = kwargs


def test_runner_drops_existing_signal_during_initial_decode_grace():
    logger = _Logger()
    base_runner = _AscendRunner()
    base_runner.cache_config = types.SimpleNamespace(block_size=128)
    base_runner.requests = {
        "req-1": types.SimpleNamespace(num_computed_tokens=9863),
    }
    state_store = _StateStore()
    runner = object.__new__(TriAttentionModelRunner)
    runner.config = TriAttentionRuntimeConfig(
        defer_prefill_compression_on_ascend=False,
        min_decode_tokens_before_compress_on_ascend=2048,
        log_decisions=False,
    )
    runner._base_runner = base_runner
    runner.state_store = state_store
    runner._last_step = 6
    runner._logger = logger
    runner._log_execution_path = True
    runner._logged_execution_path_trigger_guards = set()
    runner._get_actual_kv_from_block_table = lambda req_id: 9864

    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=9864,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=9863,
        scheduled_tokens=1,
    )

    signals = runner._supplement_worker_self_triggers(
        types.SimpleNamespace(num_scheduled_tokens={"req-1": 1}),
        {"req-1": signal},
    )

    assert signals == {}
    assert state_store.skipped["reason"] == "initial_decode_grace"
    assert len(logger.lines) == 1
    assert "reason=initial_decode_grace" in logger.lines[0]
    assert "scheduler_had_signal=True" in logger.lines[0]
