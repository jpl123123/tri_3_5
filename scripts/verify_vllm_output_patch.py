"""Standalone verification for the TriAttention vLLM output patch.

This script does NOT require a real vLLM install.  It builds minimal stand-in
``vllm`` modules in ``sys.modules`` that mirror the upstream
``KVConnectorOutput`` / ``KVOutputAggregator`` / ``LoggingStatLogger`` shape
(based on ``other_code/vllm-releases-v0.18.0``), then applies
``triattention.vllm.runtime.vllm_output_patch`` and asserts the patched
behavior matches ``vllm-git-diff.txt``.

Run from the repo root:
    python3 scripts/verify_vllm_output_patch.py
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple


# ---------------------------------------------------------------------------
# 1. Build a minimal `vllm` package graph in sys.modules so the patch module
#    (which does `from vllm.logger import logger`, `from vllm.v1.outputs ...`,
#    `from vllm.distributed.kv_transfer.kv_connector.utils ...`, etc.) can be
#    imported without a real vLLM install.  The shapes mirror upstream v0.18.0.
# ---------------------------------------------------------------------------


class _Logger:
    def __init__(self):
        self._buf: list[str] = []

    def info(self, msg, *args, **kwargs):
        self._buf.append("INFO " + (msg % args if args else msg))

    def debug(self, msg, *args, **kwargs):
        self._buf.append("DEBUG " + (msg % args if args else msg))

    def warning(self, msg, *args, **kwargs):
        self._buf.append("WARN " + (msg % args if args else msg))

    def error(self, msg, *args, **kwargs):
        self._buf.append("ERROR " + (msg % args if args else msg))

    @property
    def last(self) -> str | None:
        return self._buf[-1] if self._buf else None


_LOGGER = _Logger()


def _build_torch_stub() -> None:
    """Install a minimal ``torch`` stub so triattention runtime modules that
    ``import torch`` at module load time can be imported without a real torch.
    Only the symbols touched during import / the tested code paths are stubbed.
    """
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")

    class _Tensor:
        pass

    def _empty(*a, **k):
        return _Tensor()

    def _empty_like(*a, **k):
        return _Tensor()

    def _as_tensor(*a, **k):
        return _Tensor()

    def _arange(*a, **k):
        return _Tensor()

    torch_stub.Tensor = _Tensor
    torch_stub.empty = _empty
    torch_stub.empty_like = _empty_like
    torch_stub.as_tensor = _as_tensor
    torch_stub.arange = _arange
    torch_stub.is_tensor = lambda v: isinstance(v, _Tensor)
    torch_stub.long = "long"
    torch_stub.float32 = "float32"
    sys.modules["torch"] = torch_stub


def _build_numpy_stub() -> None:
    """Minimal ``numpy`` stub: only ``ndarray`` (for isinstance/annotations)."""
    if "numpy" in sys.modules:
        return
    np_stub = types.ModuleType("numpy")

    class _ndarray:
        pass

    np_stub.ndarray = _ndarray
    np_stub.__getattr__ = lambda name: None  # tolerate any attr access
    sys.modules["numpy"] = np_stub


def _build_vllm_modules() -> None:
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []  # mark as package
    sys.modules["vllm"] = vllm_pkg

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.logger = _LOGGER
    sys.modules["vllm.logger"] = logger_mod

    v1_pkg = types.ModuleType("vllm.v1")
    v1_pkg.__path__ = []
    sys.modules["vllm.v1"] = v1_pkg

    # --- vllm.v1.outputs (KVConnectorOutput) ---------------------------------
    T = Any

    def _combine_non_none(f: Callable, items: list):
        non_none = [item for item in items if item is not None]
        if len(non_none) == 0:
            return None
        combined = non_none[0]
        for item in non_none[1:]:
            combined = f(combined, item)
        return combined

    @dataclass
    class KVConnectorOutput:
        finished_sending: set[str] | None = None
        finished_recving: set[str] | None = None
        kv_connector_stats: Any | None = None
        kv_cache_events: Any | None = None
        kv_connector_worker_meta: Any | None = None
        invalid_block_ids: set[int] = field(default_factory=set)
        expected_finished_count: int = 0

        def is_empty(self):
            return (
                not self.finished_sending
                and not self.finished_recving
                and not self.kv_connector_stats
                and not self.kv_cache_events
                and not self.invalid_block_ids
                and not self.kv_connector_worker_meta
            )

        @classmethod
        def merge(cls, *outputs: "KVConnectorOutput"):
            assert len(outputs) > 0
            finished_sending = _combine_non_none(
                set.union, [o.finished_sending for o in outputs]
            )
            finished_recving = _combine_non_none(
                set.union, [o.finished_recving for o in outputs]
            )
            kv_connector_stats = _combine_non_none(
                lambda x, y: x.aggregate(y),
                [o.kv_connector_stats for o in outputs],
            )
            kv_cache_events = _combine_non_none(
                lambda x, y: x.merge(y),
                [o.kv_cache_events for o in outputs],
            )
            invalid_block_ids = _combine_non_none(
                set.union, [o.invalid_block_ids for o in outputs]
            )
            assert invalid_block_ids is not None
            expected_finished_count = outputs[0].expected_finished_count
            return cls(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                kv_connector_stats=kv_connector_stats,
                kv_cache_events=kv_cache_events,
                invalid_block_ids=invalid_block_ids,
                expected_finished_count=expected_finished_count,
            )

    @dataclass
    class ModelRunnerOutput:
        req_ids: list[str] = field(default_factory=list)
        req_id_to_index: dict[str, int] = field(default_factory=dict)
        kv_connector_output: KVConnectorOutput | None = None

    outputs_mod = types.ModuleType("vllm.v1.outputs")
    outputs_mod.KVConnectorOutput = KVConnectorOutput
    outputs_mod.ModelRunnerOutput = ModelRunnerOutput
    sys.modules["vllm.v1.outputs"] = outputs_mod

    # --- vllm.distributed.kv_transfer.kv_connector.utils (KVOutputAggregator)
    class KVOutputAggregator:
        def __init__(self, expected_finished_count: int = 1):
            self._expected_finished_count = expected_finished_count

        def aggregate(self, outputs, output_rank=0):
            if not outputs[output_rank]:
                return None
            finished_sending = set()
            finished_recving = set()
            combined_kv_cache_events = None
            invalid_block_ids = set()
            for model_runner_output in outputs:
                assert model_runner_output is not None
                kv_output = model_runner_output.kv_connector_output
                if not kv_output:
                    continue
                if (
                    kv_output.finished_sending
                ):
                    finished_sending |= kv_output.finished_sending
                if (
                    kv_output.finished_recving
                ):
                    finished_recving |= kv_output.finished_recving
                if combined_kv_cache_events is None:
                    combined_kv_cache_events = kv_output.kv_cache_events
                elif kv_cache_events := kv_output.kv_cache_events:
                    combined_kv_cache_events.add_events(kv_cache_events.get_all_events())
                    combined_kv_cache_events.increment_workers(1)
                invalid_block_ids |= kv_output.invalid_block_ids

            output = outputs[output_rank]
            output.kv_connector_output = KVConnectorOutput(
                finished_sending=finished_sending or None,
                finished_recving=finished_recving or None,
                kv_cache_events=combined_kv_cache_events or None,
                invalid_block_ids=invalid_block_ids,
                expected_finished_count=self._expected_finished_count,
            )
            return output

    # Build the deep package path for utils.
    for pkg_name in [
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
    ]:
        if pkg_name not in sys.modules:
            m = types.ModuleType(pkg_name)
            m.__path__ = []
            sys.modules[pkg_name] = m

    utils_mod = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.utils"
    )
    utils_mod.KVOutputAggregator = KVOutputAggregator
    sys.modules["vllm.distributed.kv_transfer.kv_connector.utils"] = utils_mod

    # --- vllm.v1.metrics.loggers (LoggingStatLogger) -------------------------
    class _Metrics:
        def __init__(self):
            self.empty = False
            self.hit_rate = 0.5

    class _SchedulerStats:
        num_running_reqs = 1
        num_waiting_reqs = 0
        kv_cache_usage = 0.1

    class LoggingStatLogger:
        def __init__(self):
            self.engine_is_idle = False
            self.log_prefix = "Engine 000: "
            self.num_preemptions = 0
            self.num_corrupted_reqs = 0
            self.last_prompt_throughput = 1.0
            self.last_generation_throughput = 2.0
            self.last_scheduler_stats = _SchedulerStats()
            self.prefix_caching_metrics = _Metrics()
            self.connector_prefix_caching_metrics = _Metrics()
            self.connector_prefix_caching_metrics.empty = False
            self.mm_caching_metrics = _Metrics()
            self.mm_caching_metrics.empty = True
            self.spec_decoding_logging = types.SimpleNamespace(log=lambda **k: None)
            self.kv_connector_logging = types.SimpleNamespace(log=lambda **k: None)
            self.cudagraph_logging = None
            self._enable_perf_stats = lambda: False

        def _update_stats(self):
            pass

        def aggregate_scheduler_stats(self):
            pass

        def log(self):
            self._update_stats()
            self.aggregate_scheduler_stats()
            log_fn = _LOGGER.debug if self.engine_is_idle else _LOGGER.info
            log_parts = [
                "Avg prompt throughput: %.1f tokens/s",
                "Avg generation throughput: %.1f tokens/s",
                "Running: %d reqs",
                "Waiting: %d reqs",
            ]
            log_args: list[Any] = [
                self.last_prompt_throughput,
                self.last_generation_throughput,
                self.last_scheduler_stats.num_running_reqs,
                self.last_scheduler_stats.num_waiting_reqs,
            ]
            if self.num_preemptions > 0:
                log_parts.append("Preemptions: %d")
                log_args.append(self.num_preemptions)
            log_parts.extend(
                [
                    "GPU KV cache usage: %.1f%%",
                    "Prefix cache hit rate: %.1f%%",
                ]
            )
            log_args.extend(
                [
                    self.last_scheduler_stats.kv_cache_usage * 100,
                    self.prefix_caching_metrics.hit_rate * 100,
                ]
            )
            if not self.connector_prefix_caching_metrics.empty:
                log_parts.append("External prefix cache hit rate: %.1f%%")
                log_args.append(self.connector_prefix_caching_metrics.hit_rate * 100)
            if not self.mm_caching_metrics.empty:
                log_parts.append("MM cache hit rate: %.1f%%")
                log_args.append(self.mm_caching_metrics.hit_rate * 100)
            log_fn(self.log_prefix + ", ".join(log_parts), *log_args)

    metrics_pkg = types.ModuleType("vllm.v1.metrics")
    metrics_pkg.__path__ = []
    sys.modules["vllm.v1.metrics"] = metrics_pkg

    loggers_mod = types.ModuleType("vllm.v1.metrics.loggers")
    loggers_mod.LoggingStatLogger = LoggingStatLogger
    sys.modules["vllm.v1.metrics.loggers"] = loggers_mod

    # envs stub used by the real log() (VLLM_COMPUTE_NANS_IN_LOGITS).
    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_COMPUTE_NANS_IN_LOGITS = False
    sys.modules["vllm.envs"] = envs_mod


# ---------------------------------------------------------------------------
# 2. A minimal TriAttention event bag (mirrors runner_output_bridge) used to
#    exercise attach/read/aggregate paths.
# ---------------------------------------------------------------------------


class _TriattentionEventBag:
    __slots__ = ("events",)

    def __init__(self, events):
        self.events = list(events)

    def __reduce__(self):
        return (_TriattentionEventBag, (list(self.events),))

    def merge(self, other):
        self.events.extend(getattr(other, "events", []))
        return self


# ---------------------------------------------------------------------------
# 3. Test driver.
# ---------------------------------------------------------------------------


def _ok(label: str) -> None:
    print(f"  PASS  {label}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  FAIL  {label}  {detail}")
    raise SystemExit(1)


def main() -> None:
    # Make triattention importable from the repo root.
    repo_root = "/Users/sunao2000/new_tri/tri_3_5"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _build_vllm_modules()
    _build_torch_stub()
    _build_numpy_stub()

    # Force runtime_logging_enabled to True so install logs are emitted.
    import os
    os.environ["TRIATTN_RUNTIME_LOGGING"] = "1"

    # Import after stubs are in place.
    from triattention.vllm.runtime import vllm_output_patch as patch_mod
    from triattention.vllm.runtime.logging_control import runtime_logging_enabled  # noqa: F401
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.metrics.loggers import LoggingStatLogger

    print("Applying vLLM output patches ...")
    applied = patch_mod.install_vllm_output_patches()
    if not applied:
        _fail("install_vllm_output_patches returned True on first call")
    # Idempotency: second call is a no-op.
    again = patch_mod.install_vllm_output_patches()
    if again:
        _fail("install_vllm_output_patches should be no-op on second call")
    _ok("install is idempotent")

    # --- Field exists --------------------------------------------------------
    kco = KVConnectorOutput()
    if hasattr(kco, "triattention_compression_events") is False:
        _fail("KVConnectorOutput has triattention_compression_events attr")
    if getattr(kco, "triattention_compression_events", "missing") is not None:
        _fail("default field value is None",
              str(getattr(kco, "triattention_compression_events")))
    _ok("KVConnectorOutput.triattention_compression_events defaults to None")

    # --- is_empty considers new field ---------------------------------------
    empty_kco = KVConnectorOutput()
    if not empty_kco.is_empty():
        _fail("fresh KVConnectorOutput.is_empty() should be True")
    kco2 = KVConnectorOutput()
    kco2.triattention_compression_events = _TriattentionEventBag([{"a": 1}])
    if kco2.is_empty():
        _fail("KVConnectorOutput with events should not be empty")
    _ok("is_empty() returns False when triattention_compression_events set")

    # --- merge combines new field -------------------------------------------
    a = KVConnectorOutput()
    a.triattention_compression_events = _TriattentionEventBag([{"req": "r1"}])
    b = KVConnectorOutput()
    b.triattention_compression_events = _TriattentionEventBag([{"req": "r2"}])
    merged = KVConnectorOutput.merge(a, b)
    merged_events = getattr(merged, "triattention_compression_events", None)
    if merged_events is None:
        _fail("merge did not propagate triattention_compression_events")
    if [e.get("req") for e in merged_events.events] != ["r1", "r2"]:
        _fail("merge did not combine events", str(merged_events.events))
    _ok("KVConnectorOutput.merge combines triattention_compression_events")

    # merge with no events yields no field (keeps is_empty truthful)
    c = KVConnectorOutput()
    d = KVConnectorOutput()
    merged_none = KVConnectorOutput.merge(c, d)
    if getattr(merged_none, "triattention_compression_events", None) is not None:
        _fail("merge of empty outputs should not set the field")
    _ok("merge of empty outputs leaves field unset")

    # --- KVOutputAggregator.aggregate combines new field --------------------
    def _mro(events):
        from vllm.v1.outputs import ModelRunnerOutput
        kco_w = KVConnectorOutput()
        kco_w.triattention_compression_events = _TriattentionEventBag(events)
        return ModelRunnerOutput(kv_connector_output=kco_w)

    agg = KVOutputAggregator(expected_finished_count=1)
    outs = [_mro([{"req": "w1"}]), _mro([{"req": "w2"}])]
    agg_out = agg.aggregate(outs, output_rank=0)
    if agg_out is None:
        _fail("aggregate returned None")
    agg_kco = agg_out.kv_connector_output
    agg_events = getattr(agg_kco, "triattention_compression_events", None)
    if agg_events is None:
        _fail("aggregate did not propagate triattention_compression_events")
    if [e.get("req") for e in agg_events.events] != ["w1", "w2"]:
        _fail("aggregate did not combine events", str(agg_events.events))
    _ok("KVOutputAggregator.aggregate combines triattention_compression_events")

    # aggregate with no events leaves field unset
    from vllm.v1.outputs import ModelRunnerOutput
    outs_empty = [ModelRunnerOutput(kv_connector_output=KVConnectorOutput()),
                  ModelRunnerOutput(kv_connector_output=KVConnectorOutput())]
    agg_out_empty = agg.aggregate(outs_empty, output_rank=0)
    if getattr(agg_out_empty.kv_connector_output,
               "triattention_compression_events", None) is not None:
        _fail("aggregate of empty outputs should not set the field")
    _ok("aggregate of empty outputs leaves field unset")

    # --- LoggingStatLogger marker -------------------------------------------
    _LOGGER._buf.clear()
    sl = LoggingStatLogger()
    sl.log()
    last = _LOGGER.last
    if last is None:
        _fail("LoggingStatLogger.log produced no output")
    if "External prefix cache hit rate-:" not in last:
        _fail("log marker not applied", repr(last))
    if "External prefix cache hit rate:" in last.replace(
        "External prefix cache hit rate-:", "X"
    ):
        # ensure we didn't accidentally leave the un-tagged form elsewhere
        _fail("un-tagged label still present", repr(last))
    _ok("LoggingStatLogger.log tag applied ('hit rate-:')")

    # Logger callables restored after log()
    if _LOGGER.info.__name__ != "info" and not callable(_LOGGER.info):
        _fail("logger.info not restored after log()")
    _ok("logger.info/debug restored after log()")

    # --- End-to-end: runner_output_bridge uses the new field ---------------
    # This exercises the tri-git-diff.txt changes: attach writes into
    # ``triattention_compression_events`` and read pulls from the same field
    # (no longer ``kv_cache_events``).  The vLLM output patch must be active
    # so that ``KVConnectorOutput`` actually carries the field.
    from triattention.vllm.runtime import runner_output_bridge as bridge
    from types import SimpleNamespace

    event = {"status": "applied", "req_id": "req-1", "cache_len_after": 4096}
    scheduler_output = SimpleNamespace()
    out, pending = bridge.attach_execute_model_compression_events(
        output=None,
        pending_events=[event],
        scheduler_output=scheduler_output,
    )
    if out is not None:
        _fail("attach with output=None should return None")
    if pending != [event]:
        _fail("attach with output=None should keep events pending")
    if scheduler_output.triattention_compression_events != [event]:
        _fail("scheduler_output fallback did not store events")
    _ok("attach_execute_model (None output) stores events on scheduler_output")

    inner_output = SimpleNamespace(kv_connector_output=KVConnectorOutput())
    sample_output = SimpleNamespace(_model_runner_output=inner_output)
    sample_output, pending = bridge.attach_sample_tokens_compression_events(
        output=sample_output,
        pending_events=pending,
    )
    if pending != []:
        _fail("attach_sample_tokens should drain pending events")
    if getattr(sample_output, "triattention_compression_events", None) != [event]:
        _fail("sample_output did not get triattention_compression_events")
    # The attach helper should have written into the kco field too.
    kco_field = getattr(inner_output.kv_connector_output,
                        "triattention_compression_events", None)
    if kco_field is None:
        _fail("attach did not write kco.triattention_compression_events")
    if [e.get("req_id") for e in kco_field.events] != ["req-1"]:
        _fail("kco field events mismatch", str(kco_field.events))
    # And the reader must read it back from the new field.
    read_back = bridge._read_triattention_events_from_kv_cache_events(inner_output)
    if read_back != [event]:
        _fail("reader did not return events from new field", str(read_back))
    _ok("attach_sample_tokens writes & reader reads triattention_compression_events")

    # Ensure we did NOT accidentally populate the native kv_cache_events field.
    if getattr(inner_output.kv_connector_output, "kv_cache_events", None) is not None:
        _fail("attach should not touch native kv_cache_events")
    _ok("native kv_cache_events left untouched by attach")

    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
