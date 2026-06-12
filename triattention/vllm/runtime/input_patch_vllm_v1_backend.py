"""V1 ModelRunner input patch helpers.

This module provides the compatibility layer used by legacy/default vLLM V1
GPUModelRunner and vLLM-Ascend NPUModelRunner paths.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from . import input_patch_state as _patch_state
from .phase_profile import (
    phase_elapsed_ms,
    phase_now,
    phase_profile_enabled,
    record_phase,
)


def _debug_drop_pos_delta() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_POS_DELTA", "0") == "1"


def _debug_drop_seq_base() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_SEQ_BASE", "0") == "1"


def _debug_preserve_rope_positions() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_PRESERVE_ROPE_POSITIONS", "0") == "1"


def _validate_expected_v1_batch_mapping(
    *,
    req_indices: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
) -> None:
    expected_rows = _patch_state.ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    if expected_rows is None:
        return
    expected_rows_np = expected_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    if expected_rows_np.size == 0:
        return
    row_mask = (expected_rows_np >= 0) & (expected_rows_np < int(num_reqs))
    if not np.all(row_mask):
        raise RuntimeError(
            "TRIATTN_V1_IDX_MAPPING_MISMATCH:"
            f"num_reqs={num_reqs}:expected={expected_rows_np.tolist()}"
        )

    expected_q_lens = _patch_state.ACTIVE_EXPECTED_QUERY_LENS_CPU
    if expected_q_lens is not None:
        expected_q_lens_np = (
            expected_q_lens.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        if expected_q_lens_np.shape != expected_rows_np.shape:
            raise RuntimeError(
                "TRIATTN_V1_QUERY_LENS_COUNT_MISMATCH:"
                f"rows={expected_rows_np.tolist()}:qlens={expected_q_lens_np.tolist()}"
            )
        actual_q_lens = np.asarray(
            num_scheduled_tokens[expected_rows_np],
            dtype=np.int64,
        )
        if not np.array_equal(actual_q_lens, expected_q_lens_np):
            raise RuntimeError(
                "TRIATTN_V1_QUERY_LENS_MISMATCH:"
                f"actual={actual_q_lens.tolist()}:expected={expected_q_lens_np.tolist()}"
            )

    req_indices_i64 = req_indices.astype(np.int64, copy=False)
    present_rows = set(int(v) for v in np.unique(req_indices_i64).tolist())
    missing_rows = [int(row) for row in expected_rows_np.tolist() if int(row) not in present_rows]
    if missing_rows:
        raise RuntimeError(
            "TRIATTN_V1_TOKEN_ROW_MAPPING_MISMATCH:"
            f"missing={missing_rows}:actual={req_indices_i64.tolist()}"
        )


def _block_table_inner_tables(block_table_obj: Any) -> list[Any]:
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, (list, tuple)) and inner_tables:
        return list(inner_tables)
    return [block_table_obj]


def _table_block_size(table: Any) -> int | None:
    for attr_name in ("block_size", "logical_block_size", "physical_block_size"):
        try:
            value = int(getattr(table, attr_name))
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _table_row_block_count(table: Any, row_idx: int) -> int | None:
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if num_blocks_per_row is None:
        return None
    try:
        return int(num_blocks_per_row[row_idx])
    except Exception:
        return None


def _active_override_rows(*, req_indices: np.ndarray, num_reqs: int) -> list[int]:
    rows: set[int] = set()
    expected_rows = _patch_state.ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    if expected_rows is not None:
        try:
            expected_np = expected_rows.detach().cpu().numpy().astype(np.int64, copy=False)
            rows.update(
                int(row)
                for row in expected_np.tolist()
                if 0 <= int(row) < int(num_reqs)
            )
        except Exception:
            pass

    if _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None and num_reqs == 1:
        rows.add(0)
    if _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0 and num_reqs == 1:
        rows.add(0)

    for mapping in (
        _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX,
        _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX,
    ):
        if not mapping:
            continue
        for row in mapping.keys():
            try:
                row_i = int(row)
            except Exception:
                continue
            if 0 <= row_i < int(num_reqs):
                rows.add(row_i)

    if not rows and req_indices.size:
        rows.update(
            int(row)
            for row in np.unique(req_indices.astype(np.int64, copy=False)).tolist()
            if 0 <= int(row) < int(num_reqs)
        )
    return sorted(rows)


def _validate_v1_block_table_bounds(
    *,
    block_table: Any,
    seq_lens_np: np.ndarray,
    req_indices: np.ndarray,
    slot_positions_np: np.ndarray | None,
    num_reqs: int,
    validate_seq_lens: bool,
) -> None:
    if block_table is None or num_reqs <= 0:
        return
    tables = _block_table_inner_tables(block_table)
    rows = _active_override_rows(req_indices=req_indices, num_reqs=num_reqs)
    if not tables or not rows:
        return

    if slot_positions_np is not None and np.any(slot_positions_np < 0):
        raise RuntimeError(
            "TRIATTN_ASCEND_V1_SLOT_POSITION_NEGATIVE:"
            f"positions={slot_positions_np.astype(np.int64, copy=False).tolist()}"
        )

    req_indices_i64 = req_indices.astype(np.int64, copy=False)
    slot_positions_i64 = (
        slot_positions_np.astype(np.int64, copy=False)
        if slot_positions_np is not None
        else None
    )
    for row in rows:
        row_mask = req_indices_i64 == int(row)
        max_slot_position: int | None = None
        if slot_positions_i64 is not None and bool(row_mask.any()):
            row_positions = slot_positions_i64[row_mask]
            max_slot_position = int(row_positions.max(initial=-1))
        seq_len = int(seq_lens_np[row]) if validate_seq_lens else None

        for gid, table in enumerate(tables):
            block_size = _table_block_size(table)
            current_blocks = _table_row_block_count(table, row)
            if block_size is None or current_blocks is None:
                continue
            capacity = max(0, int(block_size) * int(current_blocks))
            if validate_seq_lens and seq_len is not None and seq_len > capacity:
                raise RuntimeError(
                    "TRIATTN_ASCEND_V1_BLOCK_TABLE_CAPACITY_MISMATCH:"
                    f"row={row}:gid={gid}:seq_len={seq_len}:"
                    f"blocks={current_blocks}:block_size={block_size}:"
                    f"capacity={capacity}"
                )
            if max_slot_position is not None and max_slot_position >= capacity:
                raise RuntimeError(
                    "TRIATTN_ASCEND_V1_SLOT_POSITION_OOB:"
                    f"row={row}:gid={gid}:max_slot_position={max_slot_position}:"
                    f"blocks={current_blocks}:block_size={block_size}:"
                    f"capacity={capacity}"
                )


def _build_effective_slot_positions(
    *,
    positions_np: np.ndarray,
    req_indices: np.ndarray,
) -> np.ndarray | None:
    if _debug_drop_pos_delta():
        return None
    if (
        int(req_indices.size) == 0
        or int(positions_np.size) == 0
    ):
        return None

    # Slot positions may follow the compacted KV layout, but decode-time
    # RoPE positions must stay in the original logical sequence space.
    out = positions_np.copy()

    if int(req_indices.max(initial=-1)) + 1 == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0:
        out += int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
        return out

    sparse_pos_deltas = _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
    if not sparse_pos_deltas:
        return None

    row_deltas = np.zeros(int(req_indices.max()) + 1, dtype=positions_np.dtype)
    for req_idx, delta in sparse_pos_deltas.items():
        if 0 <= int(req_idx) < row_deltas.shape[0]:
            row_deltas[int(req_idx)] = int(delta)
    out += row_deltas[req_indices]
    return out


def _apply_sparse_seq_len_overrides_in_place(
    *,
    seq_lens_np: np.ndarray,
    num_computed_tokens_cpu: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
) -> bool:
    if _debug_drop_seq_base():
        return False
    if num_reqs <= 0:
        return False

    applied = False
    if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        seq_lens_np[0] = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE) + int(num_scheduled_tokens[0])
        return True

    sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
    if not sparse_bases:
        return False

    seq_lens_np[:num_reqs] = num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens[:num_reqs]
    for req_idx, effective_base in sparse_bases.items():
        idx = int(req_idx)
        if 0 <= idx < num_reqs:
            seq_lens_np[idx] = int(effective_base) + int(num_scheduled_tokens[idx])
            applied = True
    return applied


def _record_active_effective_max_seq_len(
    *,
    seq_lens_np: np.ndarray,
    num_reqs: int,
) -> int | None:
    if num_reqs <= 0:
        _patch_state.set_active_effective_max_seq_len(None)
        return None
    try:
        active = seq_lens_np[:num_reqs]
        max_seq_len = int(active.max(initial=0))
    except TypeError:
        try:
            max_seq_len = int(seq_lens_np[:num_reqs].max())
        except Exception:
            _patch_state.set_active_effective_max_seq_len(None)
            return None
    except Exception:
        _patch_state.set_active_effective_max_seq_len(None)
        return None
    _patch_state.set_active_effective_max_seq_len(max_seq_len)
    return max_seq_len


def make_patched_v1_prepare_inputs(
    original_prepare_inputs: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        profile_enabled = phase_profile_enabled()
        t0 = phase_now() if profile_enabled else 0.0
        overrides_enabled = bool(_patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED)
        seq_applied = False
        slot_applied = False
        effective_max_seq_len = None
        total_num_scheduled_tokens = 0
        num_reqs = 0
        slot_positions_np: np.ndarray | None = None
        try:
            out = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

            if not overrides_enabled:
                return out

            _patch_state.mark_active_effective_overrides_consumed()

            total_num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
            num_reqs = int(getattr(self.input_batch, "num_reqs", 0))
            if total_num_scheduled_tokens <= 0 or num_reqs <= 0:
                return out

            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            _validate_expected_v1_batch_mapping(
                req_indices=req_indices,
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=num_reqs,
            )

            slot_positions_np = _build_effective_slot_positions(
                positions_np=positions_np,
                req_indices=req_indices,
            )
            if slot_positions_np is not None:
                _validate_v1_block_table_bounds(
                    block_table=self.input_batch.block_table,
                    seq_lens_np=self.seq_lens.np,
                    req_indices=req_indices,
                    slot_positions_np=slot_positions_np,
                    num_reqs=num_reqs,
                    validate_seq_lens=False,
                )
                self.input_batch.block_table.compute_slot_mapping(req_indices, slot_positions_np)
                self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
                slot_applied = True
            seq_applied = _apply_sparse_seq_len_overrides_in_place(
                seq_lens_np=self.seq_lens.np,
                num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=num_reqs,
            )
            if seq_applied:
                self.seq_lens.np[num_reqs:].fill(0)
                self.seq_lens.copy_to_gpu()
                effective_max_seq_len = _record_active_effective_max_seq_len(
                    seq_lens_np=self.seq_lens.np,
                    num_reqs=num_reqs,
                )
            else:
                _patch_state.set_active_effective_max_seq_len(None)

            if seq_applied:
                _validate_v1_block_table_bounds(
                    block_table=self.input_batch.block_table,
                    seq_lens_np=self.seq_lens.np,
                    req_indices=req_indices,
                    slot_positions_np=None,
                    num_reqs=num_reqs,
                    validate_seq_lens=seq_applied,
                )

            return out
        finally:
            if profile_enabled:
                try:
                    max_sched = int(num_scheduled_tokens.max(initial=0))
                except TypeError:
                    max_sched = int(num_scheduled_tokens.max())
                except Exception:
                    max_sched = None
                record_phase(
                    "ascend_v1_prepare_inputs",
                    phase_elapsed_ms(t0),
                    {
                        "num_reqs": num_reqs,
                        "total_tokens": total_num_scheduled_tokens,
                        "max_scheduled": max_sched,
                        "overrides": int(overrides_enabled),
                        "seq_override": int(seq_applied),
                        "slot_override": int(slot_applied),
                        "effective_max_seq_len": effective_max_seq_len,
                    },
                )

    return _patched_prepare_inputs
