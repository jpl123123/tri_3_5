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
