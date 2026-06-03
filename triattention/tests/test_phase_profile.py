from triattention.vllm.runtime.phase_profile import (
    TriAttentionPhaseProfile,
    make_timed_wrapper,
    phase_profile_enabled,
    reset_phase_profile_for_tests,
)


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


def test_phase_profile_logs_top_phases_and_details():
    logger = _Logger()
    profile = TriAttentionPhaseProfile(
        logger=logger,
        enabled=True,
        log_every_calls=2,
    )

    profile.record_phase(
        "base_runner_execute_model",
        10.0,
        {"num_reqs": 16, "total_tokens": 16, "overrides": 1},
    )
    profile.record_phase(
        "ascend_v2_build_attn_metadata",
        2.0,
        {"max_seq_in": 40960, "max_seq_out": 2048},
    )

    line = logger.lines[-1]
    assert "TRIATTN_PHASE_PERF calls=2" in line
    assert "base_runner_execute_model:calls=1,avg=10.00" in line
    assert "ascend_v2_build_attn_metadata:calls=1,avg=2.00" in line
    assert "max_seq_in=40960" in line
    assert "max_seq_out=2048" in line


def test_phase_profile_has_dedicated_env_gate(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_PERF_PROFILE", "1")
    monkeypatch.delenv("TRIATTN_RUNTIME_PHASE_PROFILE", raising=False)
    reset_phase_profile_for_tests()

    assert not phase_profile_enabled()

    monkeypatch.setenv("TRIATTN_RUNTIME_PHASE_PROFILE", "1")
    reset_phase_profile_for_tests()

    assert phase_profile_enabled()
    reset_phase_profile_for_tests()


def test_timed_wrapper_preserves_fast_path_when_phase_profile_disabled(monkeypatch):
    calls = []

    def original(value):
        calls.append(value)
        return value + 1

    monkeypatch.delenv("TRIATTN_RUNTIME_PHASE_PROFILE", raising=False)
    reset_phase_profile_for_tests()

    wrapped = make_timed_wrapper("sample_phase", original)

    assert wrapped(41) == 42
    assert calls == [41]
    reset_phase_profile_for_tests()
