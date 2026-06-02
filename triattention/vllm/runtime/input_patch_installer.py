"""Installer for vLLM runtime input patch hooks used by TriAttention runtime."""
from __future__ import annotations

import os
from typing import Any, Callable

from vllm.logger import logger

from .input_patch_ascend_backend import (
    make_patched_ascend_v2_build_attn_metadata,
    make_patched_ascend_v2_compute_slot_mappings,
    make_patched_ascend_v2_update_seq_lens_cpu,
)
from .input_patch_vllm_backend import (
    make_patched_compute_slot_mappings,
    make_patched_prepare_pos_seq_lens,
)
from .input_patch_vllm_v1_backend import make_patched_v1_prepare_inputs

_PATCH_INSTALLED = False
_ORIGINAL_PREPARE_POS_SEQ_LENS: Callable[..., Any] | None = None
_ORIGINAL_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None
_ORIGINAL_V1_PREPARE_INPUTS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V1_PREPARE_INPUTS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA: Callable[..., Any] | None = None
_ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA: Callable[..., Any] | None = None


def _debug_disable_v1_override_path() -> bool:
    return os.environ.get("TRIATTN_DEBUG_DISABLE_V1_OVERRIDE_PATH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_triattention_patched(func: Any) -> bool:
    return bool(getattr(func, "_triattention_patched", False))


def install_runtime_input_patch_hooks() -> bool:
    """Patch vLLM GPU input prep once.

    Returns True when the patch is active (including repeated calls).
    """
    global _PATCH_INSTALLED, _ORIGINAL_PREPARE_POS_SEQ_LENS, _ORIGINAL_COMPUTE_SLOT_MAPPINGS
    global _ORIGINAL_V1_PREPARE_INPUTS, _ORIGINAL_ASCEND_V1_PREPARE_INPUTS
    global _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS
    global _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU
    global _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS
    global _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA
    global _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA
    patched_any = False
    patched_targets: list[str] = []

    try:
        import vllm.v1.worker.gpu.block_table as gpu_block_table
        import vllm.v1.worker.gpu.model_runner as gpu_model_runner
    except Exception:
        gpu_block_table = None
        gpu_model_runner = None

    if gpu_block_table is not None and gpu_model_runner is not None:
        original = getattr(gpu_model_runner, "prepare_pos_seq_lens", None)
        compute_slot_mappings = getattr(gpu_block_table.BlockTables, "compute_slot_mappings", None)
        if (
            original is not None
            and compute_slot_mappings is not None
            and _ORIGINAL_PREPARE_POS_SEQ_LENS is None
            and _ORIGINAL_COMPUTE_SLOT_MAPPINGS is None
        ):
            _ORIGINAL_PREPARE_POS_SEQ_LENS = original
            _ORIGINAL_COMPUTE_SLOT_MAPPINGS = compute_slot_mappings
            gpu_model_runner.prepare_pos_seq_lens = make_patched_prepare_pos_seq_lens(
                _ORIGINAL_PREPARE_POS_SEQ_LENS
            )
            gpu_block_table.BlockTables.compute_slot_mappings = make_patched_compute_slot_mappings(
                _ORIGINAL_COMPUTE_SLOT_MAPPINGS
            )
            patched_any = True
            patched_targets.append("vllm.v1.worker.gpu")

    if not _debug_disable_v1_override_path():
        try:
            import vllm.v1.worker.gpu_model_runner as gpu_model_runner_v1
        except Exception:
            gpu_model_runner_v1 = None
        if gpu_model_runner_v1 is not None:
            original_v1_prepare_inputs = getattr(gpu_model_runner_v1.GPUModelRunner, "_prepare_inputs", None)
            if (
                original_v1_prepare_inputs is not None
                and _ORIGINAL_V1_PREPARE_INPUTS is None
            ):
                _ORIGINAL_V1_PREPARE_INPUTS = original_v1_prepare_inputs
                gpu_model_runner_v1.GPUModelRunner._prepare_inputs = make_patched_v1_prepare_inputs(
                    _ORIGINAL_V1_PREPARE_INPUTS
                )
                patched_any = True
                patched_targets.append("vllm.v1.worker.gpu_model_runner.GPUModelRunner")

        try:
            import vllm_ascend.worker.model_runner_v1 as ascend_model_runner_v1
        except Exception:
            ascend_model_runner_v1 = None
        if ascend_model_runner_v1 is not None:
            original_ascend_v1_prepare_inputs = getattr(
                ascend_model_runner_v1.NPUModelRunner,
                "_prepare_inputs",
                None,
            )
            if original_ascend_v1_prepare_inputs is not None:
                if _ORIGINAL_ASCEND_V1_PREPARE_INPUTS is None:
                    _ORIGINAL_ASCEND_V1_PREPARE_INPUTS = original_ascend_v1_prepare_inputs
                    ascend_model_runner_v1.NPUModelRunner._prepare_inputs = make_patched_v1_prepare_inputs(
                        _ORIGINAL_ASCEND_V1_PREPARE_INPUTS
                    )
                    patched_any = True
                    patched_targets.append("vllm_ascend.worker.model_runner_v1.NPUModelRunner")

    try:
        import vllm_ascend.worker.v2.model_runner as ascend_model_runner_v2
    except Exception:
        ascend_model_runner_v2 = None
    if ascend_model_runner_v2 is not None:
        original_prepare_pos_seq_lens = getattr(
            ascend_model_runner_v2,
            "prepare_pos_seq_lens",
            None,
        )
        if (
            original_prepare_pos_seq_lens is not None
            and _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS is None
        ):
            _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS = original_prepare_pos_seq_lens
            ascend_model_runner_v2.prepare_pos_seq_lens = make_patched_prepare_pos_seq_lens(
                _ORIGINAL_ASCEND_V2_PREPARE_POS_SEQ_LENS
            )
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens")

        original_update_seq_lens_cpu = getattr(
            ascend_model_runner_v2.NPUModelRunner,
            "_update_seq_lens_cpu",
            None,
        )
        if (
            original_update_seq_lens_cpu is not None
            and _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU is None
        ):
            _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU = original_update_seq_lens_cpu
            ascend_model_runner_v2.NPUModelRunner._update_seq_lens_cpu = (
                make_patched_ascend_v2_update_seq_lens_cpu(
                    _ORIGINAL_ASCEND_V2_UPDATE_SEQ_LENS_CPU
                )
            )
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.model_runner.NPUModelRunner")

    try:
        import vllm_ascend.worker.v2.attn_utils as ascend_attn_utils_v2
    except Exception:
        ascend_attn_utils_v2 = None
    patched_ascend_build_attn_metadata = None
    if ascend_attn_utils_v2 is not None:
        original_build_attn_metadata = getattr(
            ascend_attn_utils_v2,
            "build_attn_metadata",
            None,
        )
        if _is_triattention_patched(original_build_attn_metadata):
            patched_ascend_build_attn_metadata = original_build_attn_metadata
        elif (
            original_build_attn_metadata is not None
            and _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA is None
        ):
            _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA = original_build_attn_metadata
            patched_ascend_build_attn_metadata = (
                make_patched_ascend_v2_build_attn_metadata(
                    _ORIGINAL_ASCEND_V2_BUILD_ATTN_METADATA
                )
            )
            ascend_attn_utils_v2.build_attn_metadata = patched_ascend_build_attn_metadata
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.attn_utils.build_attn_metadata")

    try:
        import vllm_ascend.worker.v2.model_states.default as ascend_default_state_v2
    except Exception:
        ascend_default_state_v2 = None
    if ascend_default_state_v2 is not None:
        original_default_build_attn_metadata = getattr(
            ascend_default_state_v2,
            "build_attn_metadata",
            None,
        )
        if _is_triattention_patched(original_default_build_attn_metadata):
            if patched_ascend_build_attn_metadata is None:
                patched_ascend_build_attn_metadata = original_default_build_attn_metadata
        elif (
            original_default_build_attn_metadata is not None
            and _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA is None
        ):
            _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA = (
                original_default_build_attn_metadata
            )
            if patched_ascend_build_attn_metadata is None:
                patched_ascend_build_attn_metadata = (
                    make_patched_ascend_v2_build_attn_metadata(
                        _ORIGINAL_ASCEND_V2_DEFAULT_BUILD_ATTN_METADATA
                    )
                )
            ascend_default_state_v2.build_attn_metadata = patched_ascend_build_attn_metadata
            patched_any = True
            patched_targets.append(
                "vllm_ascend.worker.v2.model_states.default.build_attn_metadata"
            )

    try:
        import vllm_ascend.worker.v2.block_table as ascend_block_table_v2
    except Exception:
        ascend_block_table_v2 = None
    if ascend_block_table_v2 is not None:
        original_ascend_compute_slot_mappings = getattr(
            ascend_block_table_v2.AscendBlockTables,
            "compute_slot_mappings",
            None,
        )
        if (
            original_ascend_compute_slot_mappings is not None
            and _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS is None
        ):
            _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS = (
                original_ascend_compute_slot_mappings
            )
            ascend_block_table_v2.AscendBlockTables.compute_slot_mappings = (
                make_patched_ascend_v2_compute_slot_mappings(
                    _ORIGINAL_ASCEND_V2_COMPUTE_SLOT_MAPPINGS
                )
            )
            patched_any = True
            patched_targets.append("vllm_ascend.worker.v2.block_table.AscendBlockTables")

    _PATCH_INSTALLED = _PATCH_INSTALLED or patched_any
    if patched_targets:
        logger.info(
            "Installed TriAttention runtime input patches: %s",
            ", ".join(patched_targets),
        )
    return _PATCH_INSTALLED
