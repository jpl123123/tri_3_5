"""Runtime monkey-patch for vLLM ``KVConnectorOutput`` / ``KVOutputAggregator``.

This module ports the hand-edits that used to live inside the installed
``vllm`` tree (see ``vllm-git-diff.txt`` at the repo root) into a runtime
patch applied by TriAttention when its vLLM plugin activates.

Why this exists
---------------
TriAttention needs to ferry compression events from worker processes back to
the scheduler process.  Upstream vLLM only declares ``kv_cache_events`` on
``KVConnectorOutput`` for that purpose, and that field is owned by the native
KV-cache-events subsystem.  Reusing it conflicts with native events and is
fragile across pickling / aggregation boundaries.

The clean fix (originally a hard edit to the installed vLLM) is to give
``KVConnectorOutput`` its own ``triattention_compression_events`` field and
teach ``KVOutputAggregator.aggregate`` and ``KVConnectorOutput.merge`` to
combine it.  This module applies that fix at runtime so the installed vLLM
tree no longer needs to be modified by hand.

What gets patched
-----------------
1. ``vllm.v1.outputs.KVConnectorOutput``
   - add ``triattention_compression_events`` class-level default (``None``)
   - replace ``is_empty`` so the new field is considered
   - replace ``merge`` so the new field is combined across workers
2. ``vllm.distributed.kv_transfer.kv_connector.utils.KVOutputAggregator.aggregate``
   - combine ``triattention_compression_events`` from every worker the same
     way ``kv_cache_events`` is combined.
3. ``vllm.v1.metrics.loggers.LoggingStatLogger.log``
   - tag the "External prefix cache hit rate" label with a trailing ``-``
   so it is easy to confirm at runtime that the TriAttention patch is live.

The patch is idempotent: calling :func:`install_vllm_output_patches` more
than once is a no-op after the first successful install.
"""

from __future__ import annotations

from typing import Any, Callable

from vllm.logger import logger

from .logging_control import runtime_logging_enabled

_PATCHED_OUTPUT = False
_ORIG_KV_CONNECTOR_OUTPUT_IS_EMPTY: Callable[..., Any] | None = None
_ORIG_KV_CONNECTOR_OUTPUT_MERGE: Callable[..., Any] | None = None
_ORIG_KV_OUTPUT_AGGREGATOR_AGGREGATE: Callable[..., Any] | None = None
_ORIG_LOGGING_STAT_LOGGER_LOG: Callable[..., Any] | None = None


def _combine_triattention_events(
    acc: Any | None,
    incoming: Any | None,
) -> Any | None:
    """Merge two ``triattention_compression_events`` carriers.

    The carrier is expected to be a ``_TriattentionEventBag`` (or any object
    exposing an ``events`` list).  ``None`` shortcuts are handled so callers
    can use this for both first-touch and subsequent accumulation.
    """
    if incoming is None:
        return acc
    if acc is None:
        return incoming
    # Prefer a real ``merge`` method when the carrier provides one.
    merge_fn = getattr(acc, "merge", None)
    if callable(merge_fn):
        try:
            return merge_fn(incoming)
        except Exception:
            # Fall through to the list-extend fallback below.
            pass
    acc_events = getattr(acc, "events", None)
    incoming_events = getattr(incoming, "events", None)
    if isinstance(acc_events, list) and isinstance(incoming_events, list):
        acc_events.extend(incoming_events)
        return acc
    return acc


def _patched_kv_connector_output_is_empty(self) -> bool:
    """``is_empty`` that also checks ``triattention_compression_events``."""
    assert _ORIG_KV_CONNECTOR_OUTPUT_IS_EMPTY is not None
    if not _ORIG_KV_CONNECTOR_OUTPUT_IS_EMPTY(self):
        return False
    return not getattr(self, "triattention_compression_events", None)


def _patched_kv_connector_output_merge(cls, *outputs: "KVConnectorOutput"):  # type: ignore[name-defined]
    """``merge`` classmethod that also combines ``triattention_compression_events``."""
    assert _ORIG_KV_CONNECTOR_OUTPUT_MERGE is not None
    # ``_ORIG_KV_CONNECTOR_OUTPUT_MERGE`` was captured from
    # ``KVConnectorOutput.merge`` which is a ``@classmethod``.  In Python 3,
    # accessing a classmethod through the class yields a bound method with
    # ``cls`` already bound, so we must NOT pass ``cls`` again.
    merged = _ORIG_KV_CONNECTOR_OUTPUT_MERGE(*outputs)
    if merged is None:
        return merged
    triattention_events = None
    for output in outputs:
        triattention_events = _combine_triattention_events(
            triattention_events,
            getattr(output, "triattention_compression_events", None),
        )
    # Only attach when we actually have something; keeps ``is_empty`` truthful.
    if triattention_events is not None:
        setattr(merged, "triattention_compression_events", triattention_events)
    return merged


def _patched_kv_output_aggregator_aggregate(
    self,
    outputs: list[Any] | None,
    output_rank: int = 0,
) -> Any | None:
    """``KVOutputAggregator.aggregate`` that also combines triattention events."""
    assert _ORIG_KV_OUTPUT_AGGREGATOR_AGGREGATE is not None
    # Collect triattention events BEFORE delegating to the original aggregate.
    # The original aggregate replaces ``outputs[output_rank].kv_connector_output``
    # with a fresh ``KVConnectorOutput``, so reading afterwards would lose the
    # events that lived on the original per-worker ``kv_connector_output``.
    combined_triattention_events = None
    for model_runner_output in outputs or ():
        if model_runner_output is None:
            continue
        kv_output = getattr(model_runner_output, "kv_connector_output", None)
        if not kv_output:
            continue
        combined_triattention_events = _combine_triattention_events(
            combined_triattention_events,
            getattr(kv_output, "triattention_compression_events", None),
        )

    output = _ORIG_KV_OUTPUT_AGGREGATOR_AGGREGATE(self, outputs, output_rank)
    if output is None:
        return None
    kv_connector_output = getattr(output, "kv_connector_output", None)
    if kv_connector_output is None:
        return output

    if combined_triattention_events is not None:
        setattr(
            kv_connector_output,
            "triattention_compression_events",
            combined_triattention_events,
        )
    return output


def _patched_logging_stat_logger_log(self, *args, **kwargs) -> None:
    """``LoggingStatLogger.log`` wrapper that tags the external-prefix label.

    Upstream appends the literal ``"External prefix cache hit rate: %.1f%%"``
    to a local ``log_parts`` list and joins it before logging.  We cannot see
    that local list from the outside, so we wrap ``log`` to temporarily
    intercept the logging call (``logger.info``/``logger.debug`` chosen by
    ``engine_is_idle``) and rewrite the label fragment in the already-joined
    message.  This is portable across vLLM versions and avoids duplicating
    the upstream ``log`` body.
    """
    assert _ORIG_LOGGING_STAT_LOGGER_LOG is not None
    # The original method picks ``log_fn = logger.debug if self.engine_is_idle
    # else logger.info`` locally.  We intercept by temporarily wrapping the
    # two candidate loggers so the joined message can be rewritten.
    marker_from = "External prefix cache hit rate:"
    marker_to = "External prefix cache hit rate-:"

    # Capture the real logger callables the original method will select from.
    info_fn = logger.info
    debug_fn = logger.debug

    def _make_wrapped(real_fn):
        def _wrapped(msg, *margs, **mkwargs):
            if isinstance(msg, str) and marker_from in msg:
                msg = msg.replace(marker_from, marker_to)
            return real_fn(msg, *margs, **mkwargs)
        return _wrapped

    logger.info = _make_wrapped(info_fn)  # type: ignore[method-assign]
    logger.debug = _make_wrapped(debug_fn)  # type: ignore[method-assign]
    try:
        _ORIG_LOGGING_STAT_LOGGER_LOG(self, *args, **kwargs)
    finally:
        logger.info = info_fn  # type: ignore[method-assign]
        logger.debug = debug_fn  # type: ignore[method-assign]


def _install_kv_connector_output_field() -> None:
    """Ensure ``KVConnectorOutput`` carries ``triattention_compression_events``.

    ``KVConnectorOutput`` is a ``@dataclass`` whose generated ``__init__`` does
    not know about our extra field.  We expose it as a class-level default so
    every instance reads as ``None`` until TriAttention populates it via
    ``setattr``.  ``is_empty`` and ``merge`` are wrapped so the field
    participates in emptiness checks and cross-worker combination.
    """
    global _ORIG_KV_CONNECTOR_OUTPUT_IS_EMPTY, _ORIG_KV_CONNECTOR_OUTPUT_MERGE

    from vllm.v1.outputs import KVConnectorOutput

    # Idempotent: skip if the class already exposes our field as data.
    if "triattention_compression_events" in getattr(
        KVConnectorOutput, "_triattention_patched_fields", set()
    ):
        return

    # Class-level default.  Instances that never set this attribute will read
    # ``None`` from the class, matching the diff's dataclass field default.
    if not hasattr(KVConnectorOutput, "triattention_compression_events"):
        KVConnectorOutput.triattention_compression_events = None

    _ORIG_KV_CONNECTOR_OUTPUT_IS_EMPTY = KVConnectorOutput.is_empty
    KVConnectorOutput.is_empty = _patched_kv_connector_output_is_empty

    _ORIG_KV_CONNECTOR_OUTPUT_MERGE = KVConnectorOutput.merge
    KVConnectorOutput.merge = classmethod(_patched_kv_connector_output_merge)

    patched = set(getattr(KVConnectorOutput, "_triattention_patched_fields", set()))
    patched.add("triattention_compression_events")
    KVConnectorOutput._triattention_patched_fields = patched  # type: ignore[attr-defined]


def _install_kv_output_aggregator_patch() -> None:
    """Patch ``KVOutputAggregator.aggregate`` to combine triattention events."""
    global _ORIG_KV_OUTPUT_AGGREGATOR_AGGREGATE

    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

    if getattr(KVOutputAggregator, "_triattention_aggregate_patched", False):
        return

    _ORIG_KV_OUTPUT_AGGREGATOR_AGGREGATE = KVOutputAggregator.aggregate
    KVOutputAggregator.aggregate = _patched_kv_output_aggregator_aggregate
    KVOutputAggregator._triattention_aggregate_patched = True  # type: ignore[attr-defined]


def _install_logging_marker_patch() -> None:
    """Wrap ``LoggingStatLogger.log`` to tag the external-prefix label."""
    global _ORIG_LOGGING_STAT_LOGGER_LOG

    try:
        from vllm.v1.metrics.loggers import LoggingStatLogger
    except Exception:
        # Metrics logger layout may differ across vLLM builds; the marker is
        # cosmetic, so silently skip when unavailable.
        return

    if getattr(LoggingStatLogger, "_triattention_marker_patched", False):
        return

    _ORIG_LOGGING_STAT_LOGGER_LOG = LoggingStatLogger.log
    LoggingStatLogger.log = _patched_logging_stat_logger_log
    LoggingStatLogger._triattention_marker_patched = True  # type: ignore[attr-defined]


def install_vllm_output_patches(*, force: bool = False) -> bool:
    """Apply all TriAttention ``KVConnectorOutput`` patches to vLLM.

    Returns ``True`` when a fresh patch was applied, ``False`` when the patch
    was already installed (no-op).  Pass ``force=True`` to re-apply even if
    the patch marker is set -- this is mainly for tests.
    """
    global _PATCHED_OUTPUT
    if _PATCHED_OUTPUT and not force:
        return False

    try:
        _install_kv_connector_output_field()
    except Exception:
        logger.warning(
            "[TriAttention] Failed to patch KVConnectorOutput field.",
            exc_info=True,
        )
        return False

    try:
        _install_kv_output_aggregator_patch()
    except Exception:
        logger.warning(
            "[TriAttention] Failed to patch KVOutputAggregator.aggregate.",
            exc_info=True,
        )

    try:
        _install_logging_marker_patch()
    except Exception:
        logger.warning(
            "[TriAttention] Failed to apply logging marker patch.",
            exc_info=True,
        )

    _PATCHED_OUTPUT = True
    if runtime_logging_enabled():
        logger.info(
            "[TriAttention] vLLM output patches installed: "
            "KVConnectorOutput.triattention_compression_events field, "
            "KVOutputAggregator.aggregate combine, logging marker."
        )
    return True


def is_vllm_output_patched() -> bool:
    """Return whether :func:`install_vllm_output_patches` has run."""
    return _PATCHED_OUTPUT
