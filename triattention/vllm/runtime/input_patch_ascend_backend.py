"""vLLM-Ascend input patch helpers for TriAttention runtime overrides."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch

from . import input_patch_state as _patch_state
from .input_patch_ops import shift_positions_from_sparse_deltas


def _debug_drop_pos_delta() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_POS_DELTA", "0") == "1"


def _debug_drop_seq_base() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_SEQ_BASE", "0") == "1"


def _debug_disable_max_seq_len_override() -> bool:
    return (
        os.environ.get("TRIATTN_DEBUG_DISABLE_ASCEND_MAX_SEQ_LEN_OVERRIDE", "0")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _build_effective_slot_positions_tensor(
    *,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if _debug_drop_pos_delta():
        return None
    if int(idx_mapping.shape[0]) <= 0 or int(positions.numel()) <= 0:
        return None
    if (
        int(idx_mapping.shape[0]) == 1
        and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0
    ):
        out = positions.clone()
        out.add_(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
        return out
    packed_deltas = _patch_state.get_active_packed_pos_deltas(positions.device)
    if packed_deltas is not None:
        total_query_tokens = int(packed_deltas.numel())
        if total_query_tokens > 0 and total_query_tokens <= int(positions.numel()):
            out = positions.clone()
            out[:total_query_tokens].add_(
                packed_deltas.to(device=positions.device, dtype=out.dtype)
            )
            return out
    sparse_pos_deltas = _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
    if not sparse_pos_deltas:
        return None
    lookup = _patch_state.get_active_effective_pos_delta_lookup_tensors(
        positions.device
    )
    shifted = shift_positions_from_sparse_deltas(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        pos_delta_by_req_idx=sparse_pos_deltas,
        pos_delta_lookup_tensors=lookup,
    )
    if shifted is None and int(idx_mapping.shape[0]) > 0:
        raise RuntimeError(
            "TRIATTN_ASCEND_SLOT_MAPPING_SPARSE_SHIFT_FAILED:"
            "sparse_pos_delta_present_but_shift_positions_failed"
        )
    return shifted


def _apply_sparse_seq_len_overrides_cpu(
    *,
    seq_lens_np: np.ndarray,
    req_ids: list[str],
    req_id_to_index: dict[str, int],
    num_scheduled_tokens_by_req_id: dict[str, int],
) -> bool:
    if _debug_drop_seq_base():
        return False
    num_reqs = len(req_ids)
    if num_reqs <= 0:
        return False
    if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        req_id = req_ids[0]
        seq_lens_np[0] = (
            int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE)
            + int(num_scheduled_tokens_by_req_id[req_id])
        )
        return True

    sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
    if not sparse_bases:
        return False

    matched = 0
    for batch_idx, req_id in enumerate(req_ids):
        req_idx = req_id_to_index.get(req_id)
        if not isinstance(req_idx, int):
            continue
        effective_base = sparse_bases.get(req_idx)
        if effective_base is None:
            continue
        seq_lens_np[batch_idx] = int(effective_base) + int(
            num_scheduled_tokens_by_req_id[req_id]
        )
        matched += 1
    if matched != len(sparse_bases):
        raise RuntimeError(
            "TRIATTN_ASCEND_SEQ_LENS_SPARSE_BASE_APPLY_FAILED:"
            f"matched={matched}:expected={len(sparse_bases)}"
        )
    return matched > 0


def _effective_max_seq_len_from_np(
    *,
    seq_lens_np: Any,
    num_reqs: Any,
) -> int | None:
    if _debug_disable_max_seq_len_override():
        return None
    if not isinstance(seq_lens_np, np.ndarray):
        return None
    try:
        num_reqs_i = int(num_reqs)
    except (TypeError, ValueError):
        return None
    if num_reqs_i <= 0:
        return None
    active = seq_lens_np[:num_reqs_i]
    if active.size <= 0:
        return None
    try:
        max_seq_len = int(active.max(initial=0))
    except TypeError:
        max_seq_len = int(active.max())
    return max(1, max_seq_len)


def make_patched_ascend_v2_update_seq_lens_cpu(
    original_update_seq_lens_cpu: Callable[..., Any],
) -> Callable[..., Any]:
    """Patch Ascend V2 CPU seq_lens, used by NPU attention metadata."""

    def _patched_update_seq_lens_cpu(self, scheduler_output, req_ids):
        out = original_update_seq_lens_cpu(self, scheduler_output, req_ids)
        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return out
        _patch_state.mark_active_effective_overrides_consumed()
        input_buffers = getattr(self, "input_buffers", None)
        seq_lens_np = getattr(input_buffers, "seq_lens_np", None)
        req_states = getattr(self, "req_states", None)
        req_id_to_index = getattr(req_states, "req_id_to_index", None)
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if (
            not isinstance(seq_lens_np, np.ndarray)
            or not isinstance(req_id_to_index, dict)
            or not isinstance(num_scheduled, dict)
        ):
            return out
        _apply_sparse_seq_len_overrides_cpu(
            seq_lens_np=seq_lens_np,
            req_ids=list(req_ids),
            req_id_to_index=req_id_to_index,
            num_scheduled_tokens_by_req_id=num_scheduled,
        )
        return out

    return _patched_update_seq_lens_cpu


def make_patched_ascend_v2_build_attn_metadata(
    original_build_attn_metadata: Callable[..., Any],
) -> Callable[..., Any]:
    """Patch Ascend V2 attention metadata to use effective max seq length."""

    def _patched_build_attn_metadata(*args, **kwargs):
        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return original_build_attn_metadata(*args, **kwargs)
        effective_max_seq_len = _effective_max_seq_len_from_np(
            seq_lens_np=kwargs.get("seq_lens_np"),
            num_reqs=kwargs.get("num_reqs"),
        )
        if effective_max_seq_len is None:
            return original_build_attn_metadata(*args, **kwargs)
        original_max = kwargs.get("max_seq_len")
        try:
            original_max_i = int(original_max)
        except (TypeError, ValueError):
            original_max_i = effective_max_seq_len
        kwargs["max_seq_len"] = min(original_max_i, effective_max_seq_len)
        return original_build_attn_metadata(*args, **kwargs)

    setattr(_patched_build_attn_metadata, "_triattention_patched", True)
    setattr(_patched_build_attn_metadata, "_triattention_original", original_build_attn_metadata)
    return _patched_build_attn_metadata


def make_patched_ascend_v2_compute_slot_mappings(
    original_compute_slot_mappings: Callable[..., Any],
) -> Callable[..., Any]:
    """Patch Ascend V2 slot mapping to write new KV after compressed history."""

    def _patched_compute_slot_mappings(
        self,
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded,
        *args,
        **kwargs,
    ):
        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return original_compute_slot_mappings(
                self,
                idx_mapping,
                query_start_loc,
                positions,
                num_tokens_padded,
                *args,
                **kwargs,
            )
        _patch_state.mark_active_effective_overrides_consumed()
        effective_positions = _build_effective_slot_positions_tensor(
            idx_mapping=idx_mapping,
            query_start_loc=query_start_loc,
            positions=positions,
        )
        if effective_positions is None:
            effective_positions = positions
        return original_compute_slot_mappings(
            self,
            idx_mapping,
            query_start_loc,
            effective_positions,
            num_tokens_padded,
            *args,
            **kwargs,
        )

    return _patched_compute_slot_mappings
