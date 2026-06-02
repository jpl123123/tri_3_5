from triattention.vllm.runtime.executor import CompressionExecutionResult
from triattention.vllm.runtime.runner_compression_actions import (
    execute_runner_compression_actions,
)
from triattention.vllm.runtime.signals import CompressionSignal


class _Executor:
    def __init__(self, reason="zero_copy_recency_not_ready"):
        self.reason = reason

    def execute(self, *, req_id, signal, scheduler_output):
        return CompressionExecutionResult(
            applied=False,
            reason=self.reason,
            cache_len_after=4096,
        )


class _StateStore:
    def mark_compression_skipped(self, **kwargs):
        self.skipped = kwargs


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def test_zero_copy_recency_not_ready_is_allowed_in_strict_mode():
    state_store = _StateStore()
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=4096,
        step=4,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=1,
    )

    events = execute_runner_compression_actions(
        executor=_Executor(),
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"zero_copy_recency_not_ready"},
        logger=_Logger(),
        log_decisions=False,
    )

    assert events == [
        {
            "req_id": "req-1",
            "step": 4,
            "status": "skipped",
            "reason": "zero_copy_recency_not_ready",
            "cache_len_after": 4096,
            "details": {},
            "scheduled_tokens": 1,
            "estimated_cache_len": 4096,
            "prefill_len": 10000,
        }
    ]
    assert state_store.skipped["reason"] == "zero_copy_recency_not_ready"


def test_prefill_compression_limit_is_allowed_in_strict_mode():
    state_store = _StateStore()
    signal = CompressionSignal(
        req_id="req-1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=4096,
        step=6,
        kv_usage=None,
        protect_prefill=False,
        prefill_len=10000,
        scheduled_tokens=2048,
    )

    events = execute_runner_compression_actions(
        executor=_Executor(reason="prefill_compression_limit"),
        state_store=state_store,
        scheduler_output=object(),
        signals={"req-1": signal},
        strict_no_downgrade=True,
        allowed_strict_skip_reasons={"prefill_compression_limit"},
        logger=_Logger(),
        log_decisions=False,
    )

    assert events == [
        {
            "req_id": "req-1",
            "step": 6,
            "status": "skipped",
            "reason": "prefill_compression_limit",
            "cache_len_after": 4096,
            "details": {},
            "scheduled_tokens": 2048,
            "estimated_cache_len": 4096,
            "prefill_len": 10000,
        }
    ]
    assert state_store.skipped["reason"] == "prefill_compression_limit"
