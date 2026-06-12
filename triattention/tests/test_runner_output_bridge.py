import sys
import types
from types import SimpleNamespace


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


if "vllm" not in sys.modules:
    sys.modules["vllm"] = types.SimpleNamespace()
if "vllm.logger" not in sys.modules:
    sys.modules["vllm.logger"] = types.SimpleNamespace(logger=_Logger())
try:
    import torch  # noqa: F401
except Exception:
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.SimpleNamespace(
            Tensor=object,
            is_tensor=lambda value: False,
        )

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig
from triattention.vllm.runtime.input_adapter import (
    EffectiveInputOverrides,
    prepare_effective_input_overrides,
)
from triattention.vllm.runtime import runner_output_bridge as bridge
from triattention.vllm.runtime.state import RequestStateStore


class _AscendRunner:
    pass


_AscendRunner.__module__ = "vllm_ascend.test"


def _overrides():
    return EffectiveInputOverrides(
        seq_base_map={0: 2048},
        pos_delta_map={0: -30000},
        single_seq_base=2048,
        single_pos_delta=-30000,
    )


def test_ascend_multi_req_effective_overrides_enable_graph_guard():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})

    assert bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(),
    )


def test_single_req_effective_overrides_keep_graph_mode_eligible():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1})

    assert not bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(),
    )


def test_graph_guard_respects_config_opt_out():
    base_runner = _AscendRunner()
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})

    assert not bridge._should_guard_ascend_multi_req_effective_overrides(
        base_runner=base_runner,
        scheduler_output=scheduler_output,
        overrides=_overrides(),
        config=TriAttentionRuntimeConfig(
            force_eager_multi_req_on_ascend_effective_overrides=False,
        ),
    )


def test_temporary_enforce_eager_restores_original_value():
    base_runner = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))

    with bridge._temporary_model_config_enforce_eager(base_runner, enabled=True) as active:
        assert active
        assert base_runner.model_config.enforce_eager

    assert not base_runner.model_config.enforce_eager


def test_effective_overrides_prefer_active_input_batch_rows():
    state_store = RequestStateStore()
    state_store.ensure("req-1", prefill_len=4096, protect_prefill=False)
    state_store.mark_compressed(
        "req-1",
        step=7,
        cache_len=2048,
        scheduled_tokens=1,
        scheduler_nct=4096,
    )
    base_runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"req-1": 5}),
        input_batch=SimpleNamespace(req_id_to_index={"req-1": 0}),
        requests={"req-1": SimpleNamespace(num_computed_tokens=4096)},
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-1": 1},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens=[4096],
        ),
    )

    overrides = prepare_effective_input_overrides(
        base_runner=base_runner,
        state_store=state_store,
        scheduler_output=scheduler_output,
        config=TriAttentionRuntimeConfig(),
    )

    assert overrides.seq_base_map == {0: 2048}
    assert overrides.pos_delta_map == {0: -2048}
    assert overrides.expected_req_row_indices == (0,)
