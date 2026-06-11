import sys
import types
from types import SimpleNamespace

import numpy as np


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

from triattention.vllm.runtime import input_patch_state
from triattention.vllm.runtime.input_patch_ascend_backend import (
    make_patched_ascend_v2_build_attn_metadata,
    make_patched_ascend_v2_update_seq_lens_cpu,
)
from triattention.vllm.runtime.input_patch_installer import (
    make_patched_ascend_v1_block_table_get_device_tensor,
)


def test_ascend_v2_seq_override_updates_seq_lens_np():
    def _original(self, scheduler_output, req_ids):
        del scheduler_output
        for row, req_id in enumerate(req_ids):
            req_idx = self.req_states.req_id_to_index[req_id]
            self.input_buffers.seq_lens_np[row] = (
                self.req_states.num_computed_tokens_cpu[req_idx] + 1
            )

    runner = SimpleNamespace(
        input_buffers=SimpleNamespace(seq_lens_np=np.array([5001], dtype=np.int32)),
        req_states=SimpleNamespace(
            req_id_to_index={"req-1": 3},
            num_computed_tokens_cpu=np.array([0, 0, 0, 5000], dtype=np.int32),
        ),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 1})
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={3: 2048},
        effective_pos_delta_by_req_idx={3: -2952},
    )

    try:
        patched = make_patched_ascend_v2_update_seq_lens_cpu(_original)
        patched(runner, scheduler_output, ["req-1"])
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert runner.input_buffers.seq_lens_np[0] == 2049
    assert runner.req_states.num_computed_tokens_cpu[3] == 5000


def test_ascend_v2_metadata_uses_effective_max_seq_len():
    def _original_build_attn_metadata(*args, **kwargs):
        del args
        return kwargs["max_seq_len"]

    patched = make_patched_ascend_v2_build_attn_metadata(
        _original_build_attn_metadata
    )
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_sparse_overrides(
        effective_base_by_req_idx={3: 2048},
        effective_pos_delta_by_req_idx={3: -2952},
    )

    try:
        max_seq_len = patched(
            num_reqs=3,
            seq_lens_np=np.array([2049, 2305, 0], dtype=np.int32),
            max_seq_len=40960,
        )
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_sparse_overrides(
            effective_base_by_req_idx=None,
            effective_pos_delta_by_req_idx=None,
        )

    assert max_seq_len == 2305


def test_ascend_v1_block_table_trim_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE", raising=False)
    monkeypatch.delenv("TRIATTN_DEBUG_DISABLE_ASCEND_BLOCK_TABLE_TRIM", raising=False)

    tensor = np.arange(20, dtype=np.int32).reshape(2, 10)
    patched = make_patched_ascend_v1_block_table_get_device_tensor(
        lambda self: tensor
    )
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_max_seq_len(256)
    input_patch_state.set_active_block_table_trim_observation(
        block_size=None,
        original_cols=None,
        effective_cols=None,
    )

    try:
        out = patched(SimpleNamespace(block_size=128))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_max_seq_len(None)

    assert out is tensor
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS is None


def test_ascend_v1_block_table_trim_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE", "1")
    monkeypatch.delenv("TRIATTN_DEBUG_DISABLE_ASCEND_BLOCK_TABLE_TRIM", raising=False)

    tensor = np.arange(20, dtype=np.int32).reshape(2, 10)
    patched = make_patched_ascend_v1_block_table_get_device_tensor(
        lambda self: tensor
    )
    input_patch_state.set_active_effective_overrides_enabled(True)
    input_patch_state.set_active_effective_max_seq_len(256)
    input_patch_state.set_active_block_table_trim_observation(
        block_size=None,
        original_cols=None,
        effective_cols=None,
    )

    try:
        out = patched(SimpleNamespace(block_size=128))
    finally:
        input_patch_state.set_active_effective_overrides_enabled(False)
        input_patch_state.set_active_effective_max_seq_len(None)

    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, tensor[:, :2])
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_BLOCK_SIZE == 128
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_ORIGINAL_COLS == 10
    assert input_patch_state.ACTIVE_BLOCK_TABLE_TRIM_EFFECTIVE_COLS == 2
