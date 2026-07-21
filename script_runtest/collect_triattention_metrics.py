#!/usr/bin/env python3
"""从 vLLM 服务日志和 ais_bench 评测日志中提取关键指标。

用法:
    python3 scripts/collect_triattention_metrics.py \
        --service-log test_logs/vllm_service_xxx.log \
        --bench-log   test_logs/ais_bench_xxx.log \
        --summary     test_logs/summary_xxx.txt

提取的指标:
  服务日志:
    - TriAttention 初始化参数(budget, divide_length, stats path 等)
    - 压缩信号触发次数和请求列表
    - 压缩事件统计(applied/skipped, cache_len_after, block reclaim)
    - 末尾吞吐指标(generation throughput, prompt throughput, KV cache usage)
    - Spec decoding 接受率
    - 错误和异常(Traceback, RuntimeError, ValueError 等)
  评测日志:
    - 整体精度分数
    - 逐任务/逐数据集结果
    - 样本数
    - 评测耗时
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def extract_service_metrics(log_path: str) -> dict:
    """从 vLLM 服务日志提取关键指标。"""
    metrics: dict = {
        "triattention_init": {},
        "compression_signals": [],
        "compression_events": {"applied": 0, "skipped": 0, "reasons": {}},
        "block_reclaim": {"count": 0, "total_freed_blocks": 0},
        "throughput": {"last_generation_tps": None, "last_prompt_tps": None, "last_kv_usage": None},
        "spec_decoding": {"last_acceptance_rate": None, "last_mean_acceptance_length": None},
        "errors": [],
        "fatal_errors": [],
    }

    try:
        content = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        metrics["read_error"] = str(e)
        return metrics

    lines = content.split("\n")

    # --- TriAttention 初始化参数 ---
    # 匹配 "TriAttentionScheduler initialized" 或 "TriAttention monkeypatched Scheduler initialized"
    init_patterns = [
        r"TriAttentionScheduler initialized:.*budget=(\d+).*divide_length=(\d+)",
        r"TriAttention monkeypatched Scheduler initialized:.*budget=(\d+).*divide_length=(\d+)",
        r"TriAttentionWorker.*injected runner proxy:.*budget=(\d+).*divide_length=(\d+)",
    ]
    for line in lines:
        for pat in init_patterns:
            m = re.search(pat, line)
            if m:
                metrics["triattention_init"]["kv_budget"] = int(m.group(1))
                metrics["triattention_init"]["divide_length"] = int(m.group(2))
                # 提取更多参数
                for field in [
                    "stats_path", "model_path", "protect_prefill",
                    "window_size", "score_max_layers",
                    "fast_recency_only", "fast_recency_accuracy_guard",
                    "defer_prefill_on_ascend",
                    "min_reclaim_blocks_on_ascend",
                ]:
                    fm = re.search(rf"{field}=([^\s,]+)", line)
                    if fm:
                        val = fm.group(1).rstrip(",")
                        if val.lower() in ("true", "false"):
                            val = val.lower() == "true"
                        elif val.isdigit():
                            val = int(val)
                        metrics["triattention_init"][field] = val
                break

    # --- 压缩信号触发 ---
    for line in lines:
        m = re.search(r"TriAttention signal triggered req=(\S+).*step=(\d+).*reason=(\S+)", line)
        if m:
            metrics["compression_signals"].append({
                "req_id": m.group(1),
                "step": int(m.group(2)),
                "reason": m.group(3),
            })

    # --- 压缩事件统计 ---
    for line in lines:
        # "TriAttention update_from_output: received N events (M applied) via ..."
        m = re.search(r"received (\d+) events \((\d+) applied\)", line)
        if m:
            metrics["compression_events"]["applied"] += int(m.group(2))

        # skip reasons
        for reason_field in ["under_budget", "prefill_incomplete", "defer_recompress",
                             "zero_copy_recency_not_ready", "fast_recency_long_context_guard",
                             "initial_decode_grace", "prefill_exceeds_budget",
                             "batch_queue_dedup", "req_state_not_found"]:
            if f'reason="{reason_field}"' in line or f"reason={reason_field}" in line:
                metrics["compression_events"]["skipped"] += 1
                metrics["compression_events"]["reasons"][reason_field] = \
                    metrics["compression_events"]["reasons"].get(reason_field, 0) + 1

    # --- block reclaim ---
    for line in lines:
        m = re.search(r"FREE_BLOCKS:.*req=(\S+).*gid=(\d+).*freed=(\d+)", line)
        if m:
            metrics["block_reclaim"]["count"] += 1
            metrics["block_reclaim"]["total_freed_blocks"] += int(m.group(4))

    # --- worker self-trigger ---
    worker_triggers = 0
    for line in lines:
        if "TriAttention worker self-trigger:" in line:
            worker_triggers += 1
    metrics["compression_events"]["worker_self_triggers"] = worker_triggers

    # --- 吞吐指标(取最后几条) ---
    last_gen_tps = None
    last_prompt_tps = None
    last_kv_usage = None
    for line in lines:
        m = re.search(r"Avg generation throughput: ([\d.]+) tokens/s", line)
        if m:
            last_gen_tps = float(m.group(1))
        m = re.search(r"Avg prompt throughput: ([\d.]+) tokens/s", line)
        if m:
            last_prompt_tps = float(m.group(1))
        m = re.search(r"GPU KV cache usage: ([\d.]+)%", line)
        if m:
            last_kv_usage = float(m.group(1))
    metrics["throughput"]["last_generation_tps"] = last_gen_tps
    metrics["throughput"]["last_prompt_tps"] = last_prompt_tps
    metrics["throughput"]["last_kv_usage"] = last_kv_usage

    # 收集吞吐序列(用于看趋势)
    gen_tps_series = []
    for line in lines:
        m = re.search(r"Avg generation throughput: ([\d.]+) tokens/s", line)
        if m:
            gen_tps_series.append(float(m.group(1)))
    metrics["throughput"]["generation_tps_samples"] = gen_tps_series[-20:]  # 最后20条

    # --- Spec decoding ---
    for line in lines:
        m = re.search(r"Avg Draft acceptance rate: ([\d.]+)%", line)
        if m:
            metrics["spec_decoding"]["last_acceptance_rate"] = float(m.group(1))
        m = re.search(r"Mean acceptance length: ([\d.]+)", line)
        if m:
            metrics["spec_decoding"]["last_mean_acceptance_length"] = float(m.group(1))

    # --- 错误和异常 ---
    current_traceback: list[str] = []
    for line in lines:
        if "Traceback (most recent call last)" in line:
            if current_traceback:
                metrics["errors"].append("\n".join(current_traceback[:5]))
            current_traceback = [line]
        elif current_traceback:
            current_traceback.append(line)
            if len(current_traceback) > 20:
                current_traceback = current_traceback[:20]
            # 检查是否是 fatal
            if "TRIATTN_FATAL" in line or "EngineCore encountered a fatal error" in line:
                metrics["fatal_errors"].append("\n".join(current_traceback[-10:]))
        # 单行错误
        for err_marker in ["ERROR", "ValueError:", "RuntimeError:", "EngineDeadError"]:
            if err_marker in line and "Traceback" not in line:
                if line.strip() and line not in str(metrics["errors"]):
                    metrics["errors"].append(line.strip()[:200])
                    break
    if current_traceback:
        metrics["errors"].append("\n".join(current_traceback[:5]))

    # 去重
    metrics["errors"] = list(dict.fromkeys(metrics["errors"]))[:20]
    metrics["fatal_errors"] = list(dict.fromkeys(metrics["fatal_errors"]))[:5]

    # --- 服务是否崩溃 ---
    metrics["service_crashed"] = bool(metrics["fatal_errors"]) or \
        "EngineDeadError" in content

    return metrics


def extract_bench_metrics(log_path: str) -> dict:
    """从 ais_bench 评测日志提取关键指标。"""
    metrics: dict = {
        "overall_score": None,
        "task_scores": {},
        "num_samples": 0,
        "num_correct": 0,
        "errors": [],
        "raw_tail": [],
    }

    if not log_path:
        return metrics

    try:
        content = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        metrics["read_error"] = str(e)
        return metrics

    lines = content.split("\n")

    # LongBench-v2 通常输出 accuracy 或 score
    for line in lines:
        # 匹配 "accuracy: 0.xx" 或 "score: 0.xx" 或 "Accuracy: xx%"
        m = re.search(r"[Aa]ccuracy[:\s]+([\d.]+)%?", line)
        if m and metrics["overall_score"] is None:
            metrics["overall_score"] = float(m.group(1))
        m = re.search(r"[Oo]verall.*[Ss]core[:\s]+([\d.]+)", line)
        if m and metrics["overall_score"] is None:
            metrics["overall_score"] = float(m.group(1))
        # 匹配 "task_name: score"
        m = re.search(r"(\w+[\w-]*)\s*[:\s]\s*([\d.]+)\s*%?", line)
        if m and "accuracy" in line.lower():
            task_name = m.group(1)
            score = float(m.group(2))
            metrics["task_scores"][task_name] = score

    # 匹配 "correct: N / total: M" 类格式
    for line in lines:
        m = re.search(r"correct[:\s]+(\d+)", line, re.IGNORECASE)
        if m:
            metrics["num_correct"] = max(metrics["num_correct"], int(m.group(1)))
        m = re.search(r"total[:\s]+(\d+)", line, re.IGNORECASE)
        if m:
            metrics["num_samples"] = max(metrics["num_samples"], int(m.group(1)))
        m = re.search(r"(\d+)\s*samples", line, re.IGNORECASE)
        if m:
            metrics["num_samples"] = max(metrics["num_samples"], int(m.group(1)))

    # 如果有 num_samples 和 num_correct 但没有 overall_score,计算
    if metrics["overall_score"] is None and metrics["num_samples"] > 0:
        metrics["overall_score"] = round(
            metrics["num_correct"] / metrics["num_samples"] * 100, 2
        )

    # 错误
    for line in lines:
        if "ERROR" in line or "error" in line.lower() or "Traceback" in line:
            if line.strip():
                metrics["errors"].append(line.strip()[:200])
    metrics["errors"] = list(dict.fromkeys(metrics["errors"]))[:10]

    # 末尾 20 行(用于看评测是否正常结束)
    metrics["raw_tail"] = [l for l in lines[-20:] if l.strip()]

    return metrics


def extract_perf_metrics(log_path: str) -> dict:
    """从 aisbench_test.py 性能测试日志提取关键指标。

    日志格式(ais_bench perf 模式表格输出):
        │ TTFT                     │ total   │ 92967.3165 ms  │ ...
        │ TPOT                     │ total   │ 13.684 ms      │ ...
        │ Output Token Throughput  │ total   │ 9.5731 token/s │
        │ Benchmark Duration       │ total   │ 427864.8989 ms │
    """
    metrics: dict = {
        "output_token_throughput": None,   # Output Token Throughput (token/s)
        "ttft_avg_ms": None,               # TTFT 平均 (ms)
        "tpot_avg_ms": None,               # TPOT 平均 (ms)
        "benchmark_duration_ms": None,     # Benchmark Duration (ms)
        "prefill_throughput": None,         # Prefill Token Throughput
        "total_token_throughput": None,    # Total Token Throughput
        "input_len": None,
        "output_len": None,
        "data_num": None,
        "concurrency": None,
        "raw_tail": [],
        "errors": [],
    }

    if not log_path:
        return metrics

    try:
        content = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        metrics["read_error"] = str(e)
        return metrics

    lines = content.split("\n")

    # 提取测试参数
    for line in lines:
        m = re.search(r"input token length: (\d+)", line)
        if m:
            metrics["input_len"] = int(m.group(1))
        m = re.search(r"output token length: (\S+)", line)
        if m:
            metrics["output_len"] = m.group(1)
        m = re.search(r"number of dataset: (\d+)", line)
        if m:
            metrics["data_num"] = int(m.group(1))
        m = re.search(r"concurrency: (\S+)", line)
        if m:
            metrics["concurrency"] = m.group(1)

    # 提取核心性能指标
    # 格式: │ <metric_name> │ total │ <value> <unit> │ ...
    # 第一个数值是 average/total
    for line in lines:
        # TTFT: │ TTFT  │ total │ 92967.3165 ms │ ...
        m = re.search(r"│\s*TTFT\s*│\s*total\s*│\s*([\d.]+)\s*ms", line)
        if m and metrics["ttft_avg_ms"] is None:
            metrics["ttft_avg_ms"] = float(m.group(1))

        # TPOT: │ TPOT  │ total │ 13.684 ms │ ...
        m = re.search(r"│\s*TPOT\s*│\s*total\s*│\s*([\d.]+)\s*ms", line)
        if m and metrics["tpot_avg_ms"] is None:
            metrics["tpot_avg_ms"] = float(m.group(1))

        # Output Token Throughput: │ Output Token Throughput │ total │ 9.5731 token/s │
        m = re.search(r"│\s*Output Token Throughput\s*│\s*total\s*│\s*([\d.]+)\s*token/s", line)
        if m and metrics["output_token_throughput"] is None:
            metrics["output_token_throughput"] = float(m.group(1))

        # Benchmark Duration: │ Benchmark Duration │ total │ 427864.8989 ms │
        m = re.search(r"│\s*Benchmark Duration\s*│\s*total\s*│\s*([\d.]+)\s*ms", line)
        if m and metrics["benchmark_duration_ms"] is None:
            metrics["benchmark_duration_ms"] = float(m.group(1))

        # Prefill Token Throughput
        m = re.search(r"│\s*Prefill Token Throughput\s*│\s*total\s*│\s*([\d.]+)\s*token/s", line)
        if m and metrics["prefill_throughput"] is None:
            metrics["prefill_throughput"] = float(m.group(1))

        # Total Token Throughput
        m = re.search(r"│\s*Total Token Throughput\s*│\s*total\s*│\s*([\d.]+)\s*token/s", line)
        if m and metrics["total_token_throughput"] is None:
            metrics["total_token_throughput"] = float(m.group(1))

    # 从 aisbench_result.csv 补充提取(如果有)
    import csv
    for csv_path in [
        Path(log_path).parent / "aisbench_result.csv",
        Path(log_path).parent.parent / "aisbench_result.csv",
    ]:
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for k, v in row.items():
                            k_lower = k.lower().strip()
                            if "ttft" in k_lower and "avg" in k_lower and metrics["ttft_avg_ms"] is None:
                                try: metrics["ttft_avg_ms"] = float(v)
                                except (ValueError, TypeError): pass
                            if "tpot" in k_lower and "avg" in k_lower and metrics["tpot_avg_ms"] is None:
                                try: metrics["tpot_avg_ms"] = float(v)
                                except (ValueError, TypeError): pass
                            if "e2e" in k_lower and "time" in k_lower and metrics["benchmark_duration_ms"] is None:
                                try: metrics["benchmark_duration_ms"] = float(v)
                                except (ValueError, TypeError): pass
                            if "output" in k_lower and "throughput" in k_lower and metrics["output_token_throughput"] is None:
                                try: metrics["output_token_throughput"] = float(v)
                                except (ValueError, TypeError): pass
            except Exception:
                pass
            break

    # 错误
    for line in lines:
        if "ERROR" in line or "error" in line.lower() or "Traceback" in line:
            if line.strip():
                metrics["errors"].append(line.strip()[:200])
    metrics["errors"] = list(dict.fromkeys(metrics["errors"]))[:10]

    # 末尾 15 行
    metrics["raw_tail"] = [l for l in lines[-15:] if l.strip()]

    return metrics


def format_metrics(service: dict, bench: dict, perf: dict) -> str:
    """格式化指标为可读文本。"""
    lines: list[str] = []

    # ===== 性能测试结果 =====
    if perf:
        lines.append("----- Performance Test Results -----")
        lines.append(f"  Input length:           {perf.get('input_len')}")
        lines.append(f"  Output length:          {perf.get('output_len')}")
        lines.append(f"  Data num:               {perf.get('data_num')}")
        lines.append(f"  Concurrency:            {perf.get('concurrency')}")
        lines.append(f"  Output Token Throughput: {perf.get('output_token_throughput')} token/s")
        lines.append(f"  TTFT (avg):             {perf.get('ttft_avg_ms')} ms")
        lines.append(f"  TPOT (avg):             {perf.get('tpot_avg_ms')} ms")
        lines.append(f"  Benchmark Duration:     {perf.get('benchmark_duration_ms')} ms")
        if perf.get('prefill_throughput') is not None:
            lines.append(f"  Prefill Throughput:      {perf.get('prefill_throughput')} token/s")
        if perf.get('total_token_throughput') is not None:
            lines.append(f"  Total Token Throughput:  {perf.get('total_token_throughput')} token/s")
        perf_tail = perf.get("raw_tail", [])
        if perf_tail:
            lines.append("  Perf log tail (last 15 non-empty lines):")
            for l in perf_tail[:10]:
                lines.append(f"    {l}")
        perf_errors = perf.get("errors", [])
        if perf_errors:
            lines.append(f"  Perf errors ({len(perf_errors)}):")
            for err in perf_errors[:5]:
                lines.append(f"    {err}")
        lines.append("")

    # ===== TriAttention 配置 =====
    init = service.get("triattention_init", {})
    lines.append("----- TriAttention Configuration -----")
    if init:
        for k, v in init.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  (not found in log)")

    # ===== 压缩统计 =====
    ce = service.get("compression_events", {})
    signals = service.get("compression_signals", [])
    lines.append("")
    lines.append("----- Compression Statistics -----")
    lines.append(f"  Signal triggers:     {len(signals)}")
    lines.append(f"  Applied events:      {ce.get('applied', 0)}")
    lines.append(f"  Skipped events:       {ce.get('skipped', 0)}")
    lines.append(f"  Worker self-triggers: {ce.get('worker_self_triggers', 0)}")
    if ce.get("reasons"):
        lines.append(f"  Skip reasons:")
        for reason, count in sorted(ce["reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"    {reason}: {count}")
    br = service.get("block_reclaim", {})
    lines.append(f"  Block reclaim events: {br.get('count', 0)}")
    lines.append(f"  Total freed blocks:   {br.get('total_freed_blocks', 0)}")

    # ===== 吞吐 =====
    tp = service.get("throughput", {})
    lines.append("")
    lines.append("----- Throughput (last reported) -----")
    lines.append(f"  Generation throughput: {tp.get('last_generation_tps')} tokens/s")
    lines.append(f"  Prompt throughput:     {tp.get('last_prompt_tps')} tokens/s")
    lines.append(f"  KV cache usage:        {tp.get('last_kv_usage')}%")
    gen_series = tp.get("generation_tps_samples", [])
    if gen_series:
        lines.append(f"  Gen TPS series (last {len(gen_series)}): " +
                     ", ".join(f"{v:.1f}" for v in gen_series))

    # ===== Spec decoding =====
    sd = service.get("spec_decoding", {})
    lines.append("")
    lines.append("----- Speculative Decoding -----")
    lines.append(f"  Mean acceptance length:  {sd.get('last_mean_acceptance_length')}")
    lines.append(f"  Avg draft acceptance:    {sd.get('last_acceptance_rate')}%")

    # ===== 评测精度 =====
    lines.append("")
    lines.append("----- Benchmark Accuracy (longbenchv2_gen) -----")
    if bench.get("overall_score") is not None:
        lines.append(f"  Overall score:   {bench['overall_score']}")
    else:
        lines.append("  Overall score:   (not found in log)")
    lines.append(f"  Num samples:     {bench.get('num_samples', 0)}")
    lines.append(f"  Num correct:     {bench.get('num_correct', 0)}")
    if bench.get("task_scores"):
        lines.append("  Per-task scores:")
        for task, score in bench["task_scores"].items():
            lines.append(f"    {task}: {score}")

    # ===== 错误 =====
    lines.append("")
    lines.append("----- Errors / Exceptions -----")
    fatals = service.get("fatal_errors", [])
    if fatals:
        lines.append(f"  *** FATAL ERRORS ({len(fatals)}): ***")
        for fe in fatals[:3]:
            for fe_line in fe.split("\n")[:5]:
                lines.append(f"    {fe_line}")
    else:
        lines.append("  No fatal errors detected in service log.")

    svc_errors = service.get("errors", [])
    if svc_errors:
        lines.append(f"  Service errors/warnings ({len(svc_errors)} unique):")
        for err in svc_errors[:5]:
            lines.append(f"    {err}")

    bench_errors = bench.get("errors", [])
    if bench_errors:
        lines.append(f"  Benchmark errors ({len(bench_errors)}):")
        for err in bench_errors[:5]:
            lines.append(f"    {err}")

    if service.get("service_crashed"):
        lines.append("")
        lines.append("  *** SERVICE CRASHED ***")
    else:
        lines.append("")
        lines.append("  Service did not crash (no fatal errors).")

    # ===== 评测日志末尾 =====
    raw_tail = bench.get("raw_tail", [])
    if raw_tail:
        lines.append("")
        lines.append("----- Benchmark log tail (last 20 non-empty lines) -----")
        for l in raw_tail[:15]:
            lines.append(f"  {l}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract TriAttention + benchmark metrics from logs"
    )
    parser.add_argument("--service-log", required=True, help="vLLM service log path")
    parser.add_argument("--perf-log", default="", help="Performance test (aisbench_test.py) log path")
    parser.add_argument("--bench-log", default="", help="ais_bench accuracy log path")
    parser.add_argument("--summary", default="", help="Existing summary file (appended to)")
    args = parser.parse_args()

    service_metrics = extract_service_metrics(args.service_log)
    perf_metrics = extract_perf_metrics(args.perf_log) if args.perf_log else {}
    bench_metrics = extract_bench_metrics(args.bench_log) if args.bench_log else {}

    output = format_metrics(service_metrics, bench_metrics, perf_metrics)
    print(output)

    # 如果有 summary 文件,追加
    if args.summary and Path(args.summary).exists():
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
