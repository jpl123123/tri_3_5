from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.perf_profile import TriAttentionPerfProfile
from triattention.vllm.runtime.phase_profile import (
    phase_profile_enabled,
    reset_phase_profile_for_tests,
)


class _Logger:
    def info(self, fmt, *args):
        raise AssertionError("profile logging should be disabled")


def test_runtime_logging_defaults_keep_existing_decision_logs(monkeypatch):
    monkeypatch.delenv("TRIATTN_RUNTIME_LOGGING", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_DECISIONS", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", raising=False)
    monkeypatch.delenv("TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS", raising=False)

    config = TriAttentionRuntimeConfig.from_env()

    assert config.logging_enabled
    assert config.log_decisions
    assert config.log_execution_path
    assert not config.log_all_worker_events


def test_runtime_logging_master_overrides_verbose_subswitches(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_DECISIONS", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_ALL_WORKER_EVENTS", "1")

    config = TriAttentionRuntimeConfig.from_env()

    assert not config.logging_enabled
    assert not config.log_decisions
    assert not config.log_execution_path
    assert not config.log_all_worker_events


def test_runtime_execution_path_log_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_LOG_EXECUTION_PATH", "0")

    config = TriAttentionRuntimeConfig.from_env()

    assert config.logging_enabled
    assert not config.log_execution_path


def test_runtime_logging_master_disables_perf_profiles(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_PERF_PROFILE", "1")
    monkeypatch.setenv("TRIATTN_RUNTIME_E2E_PROFILE", "1")

    profile = TriAttentionPerfProfile.from_env(_Logger())

    assert not profile.enabled
    assert not profile.e2e_enabled


def test_runtime_logging_master_disables_phase_profile(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_LOGGING", "0")
    monkeypatch.setenv("TRIATTN_RUNTIME_PHASE_PROFILE", "1")
    reset_phase_profile_for_tests()

    assert not phase_profile_enabled()

    reset_phase_profile_for_tests()
