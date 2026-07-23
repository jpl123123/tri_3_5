#!/usr/bin/env bash
#
# run_qwen35_test.sh — Qwen3.5-27B + TriAttention 自动化测试脚本
#
# 支持任意时长的性能/精度测试,不受调用方超时限制。
# 核心机制:用 setsid 脱离终端进程组,分阶段写 marker,可轮询进度。
#
# 用法:
#   bash script_runtest/run_qwen35_test.sh [--start]   # 启动测试(默认只跑性能)
#   RUN_ACCURACY=1 bash script_runtest/run_qwen35_test.sh  # 性能+精度都跑
#   bash script_runtest/run_qwen35_test.sh --status    # 查看当前进度
#   bash script_runtest/run_qwen35_test.sh --collect   # 从最新日志提取指标
#   bash script_runtest/run_qwen35_test.sh --stop      # 停止当前运行的服务和测试
#
# 可覆盖的环境变量:
#   KV_BUDGET=6144              TriAttention KV 预算
#   SPARSE_STATS_PATH=...       sparse stats 文件路径
#   PORT=5555                   服务端口
#   KEEP_SERVICE=1              测试后保留服务
#   RUN_ACCURACY=1              同时跑精度评测
#   REPEAT=3                    重复跑3次性能和精度(默认1次)
#   PERF_INPUT_LEN=260096       性能测试输入长度
#   PERF_OUTPUT_LEN=1024        性能测试输出长度
#   PERF_DATA_NUM=4             性能测试数据量
#   PERF_CONCURRENCY=1           性能测试并发数
#
# 日志保存在:  script_runtest/test_logs/
# ---------------------------------------------------------------------------

set -euo pipefail

# ========== 固定路径 ==========
REPO_DIR="/cache/z00872399/qwen3.5_tri/triattention-main"
MODEL_PATH="/cache/Qwen3.5-27B-w8a8-mtp"
BENCH_DIR="/cache/z00872399/benchmark_copy2"
PERF_BENCH_DIR="/cache/z00872399/aisbench_auto_tools_prefix_copy"
LOG_DIR="${REPO_DIR}/script_runtest/test_logs"
STATUS_DIR="${LOG_DIR}/status"
METRICS_SCRIPT="${REPO_DIR}/script_runtest/collect_triattention_metrics.py"

# ========== 可配置参数 ==========
PORT="${PORT:-5555}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5}"
KV_BUDGET="${KV_BUDGET:-6144}"
SPARSE_STATS_PATH="${SPARSE_STATS_PATH:-/cache/qwen35_stats.pt}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-12,13}"
TP_SIZE="${TP_SIZE:-2}"
PERF_INPUT_LEN="${PERF_INPUT_LEN:-260096}"
PERF_OUTPUT_LEN="${PERF_OUTPUT_LEN:-1024}"
PERF_DATA_NUM="${PERF_DATA_NUM:-4}"
PERF_CONCURRENCY="${PERF_CONCURRENCY:-1}"
KEEP_SERVICE="${KEEP_SERVICE:-0}"
RUN_ACCURACY="${RUN_ACCURACY:-0}"
REPEAT="${REPEAT:-1}"
DEFER_PREFILL_ON_ASCEND="${DEFER_PREFILL_ON_ASCEND:-true}"
SERVICE_READY_TIMEOUT="${SERVICE_READY_TIMEOUT:-1800}"

# ========== 内部:编排模式标识 ==========
# 当 _TRIATTN_ORCHESTRATOR=1 时,本脚本运行实际的工作流(由 --start 用 setsid 拉起)
_ORCH_MODE="${_TRIATTN_ORCHESTRATOR:-0}"

# =====================================================================
# 状态文件读写
# =====================================================================
mkdir -p "${STATUS_DIR}"

write_status() { echo "$1" > "${STATUS_DIR}/phase"; }
read_status()  { cat "${STATUS_DIR}/phase" 2>/dev/null || echo "idle"; }
write_info()   { echo "$1" >> "${STATUS_DIR}/run_info.txt"; }

# =====================================================================
# NPU 进程清理:杀掉所有持有 davinci 设备句柄的进程
# 这是解决 NPU 内存泄漏的关键:setsid 启动的 vllm 服务在被杀后,
# worker 子进程仍存活并持有 /dev/davinci* 设备句柄,
# 导致 NPU HBM 内存不释放,下次启动报 "Free memory" 错误。
# =====================================================================
kill_npu_processes() {
    local dev_list="${VISIBLE_DEVICES:-12,13}"
    dev_list="${dev_list//,/ }"
    local killed_any=0

    # 1. 杀掉所有持有 /dev/davinci{dev} 句柄的进程
    for dev in $dev_list; do
        local pids=$(find /proc/*/fd -lname "*davinci${dev}" 2>/dev/null | sed 's|/proc/||;s|/fd.*||' | sort -u 2>/dev/null)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                if [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then
                    echo "  Killing PID=$pid (holds /dev/davinci${dev})"
                    kill -9 "$pid" 2>/dev/null || true
                    killed_any=1
                fi
            done
        fi
    done

    # 2. 杀掉 vllm 相关进程(EngineCore, Worker 等)
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "aisbench_test\|ais_bench" 2>/dev/null || true
    pkill -9 -f "multiprocessing.spawn\|multiprocessing.forkserver\|resource_tracker" 2>/dev/null || true

    # 3. 等待 NPU 内存释放
    if [ "$killed_any" = "1" ]; then
        echo "  Waiting for NPU memory release..."
        for i in $(seq 1 12); do
            sleep 5
            local all_free=1
            for dev in $dev_list; do
                local pids=$(find /proc/*/fd -lname "*davinci${dev}" 2>/dev/null | sed 's|/proc/||;s|/fd.*||' | sort -u 2>/dev/null)
                [ -n "$pids" ] && all_free=0
            done
            [ "$all_free" = "1" ] && break
        done
    fi
    echo "  NPU cleanup done."
}

# =====================================================================
# --status: 查看当前进度
# =====================================================================
if [ "${1:-}" = "--status" ]; then
    PHASE=$(read_status)
    echo "========== TriAttention Test Status =========="
    echo "Phase: ${PHASE}"
    if [ -f "${STATUS_DIR}/run_progress" ]; then
        echo "Run progress: $(cat "${STATUS_DIR}/run_progress" 2>/dev/null)"
    fi
    echo ""

    # 读取 run_info
    if [ -f "${STATUS_DIR}/run_info.txt" ]; then
        echo "--- Run Info ---"
        cat "${STATUS_DIR}/run_info.txt"
        echo ""
    fi

    # 服务状态
    SVC_PID=""
    [ -f "${STATUS_DIR}/service.pid" ] && SVC_PID=$(cat "${STATUS_DIR}/service.pid" 2>/dev/null)
    if [ -n "${SVC_PID}" ] && kill -0 "${SVC_PID}" 2>/dev/null; then
        echo "Service: PID=${SVC_PID} ALIVE"
    elif [ -n "${SVC_PID}" ]; then
        echo "Service: PID=${SVC_PID} DEAD"
    else
        echo "Service: not started"
    fi

    # 测试进程状态
    for proc_name in "perf" "bench"; do
        PID_FILE="${STATUS_DIR}/${proc_name}.pid"
        if [ -f "${PID_FILE}" ]; then
            TPID=$(cat "${PID_FILE}" 2>/dev/null)
            if [ -n "${TPID}" ] && kill -0 "${TPID}" 2>/dev/null; then
                echo "${proc_name}: PID=${TPID} RUNNING"
            else
                echo "${proc_name}: PID=${TPID} FINISHED/DEAD"
            fi
        fi
    done

    echo ""
    # 最新日志 tail
    SVC_LOG=""
    [ -f "${STATUS_DIR}/service.log" ] && SVC_LOG=$(cat "${STATUS_DIR}/service.log" 2>/dev/null)
    PERF_LOG=""
    [ -f "${STATUS_DIR}/perf.log" ] && PERF_LOG=$(cat "${STATUS_DIR}/perf.log" 2>/dev/null)
    BENCH_LOG=""
    [ -f "${STATUS_DIR}/bench.log" ] && BENCH_LOG=$(cat "${STATUS_DIR}/bench.log" 2>/dev/null)

    echo "--- Service log tail ---"
    tail -3 "${SVC_LOG}" 2>/dev/null || echo "(no service log)"
    echo ""
    if [ -n "${PERF_LOG}" ] && [ -f "${PERF_LOG}" ]; then
        echo "--- Perf log tail ---"
        tail -3 "${PERF_LOG}" 2>/dev/null
        echo ""
    fi
    if [ -n "${BENCH_LOG}" ] && [ -f "${BENCH_LOG}" ]; then
        echo "--- Bench log tail ---"
        tail -3 "${BENCH_LOG}" 2>/dev/null
        echo ""
    fi

    echo "=============================================="
    exit 0
fi

# =====================================================================
# --stop: 停止所有运行中的进程
# =====================================================================
if [ "${1:-}" = "--stop" ]; then
    echo "Stopping all TriAttention test processes..."
    # 杀编排器和测试进程
    for pid_file in "${STATUS_DIR}/orchestrator.pid" "${STATUS_DIR}/service.pid" "${STATUS_DIR}/perf.pid" "${STATUS_DIR}/bench.pid"; do
        if [ -f "${pid_file}" ]; then
            PID=$(cat "${pid_file}" 2>/dev/null)
            if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
                echo "  Killing PID=${PID} ($(basename "${pid_file}" .pid))"
                kill -9 "${PID}" 2>/dev/null || true
            fi
            rm -f "${pid_file}"
        fi
    done
    # 彻底清理所有 NPU 进程(包括 setsid 派生的 worker 子进程)
    kill_npu_processes
    write_status "stopped"
    echo "Done."
    exit 0
fi

# =====================================================================
# --collect: 从最新日志提取指标
# =====================================================================
if [ "${1:-}" = "--collect" ]; then
    set +e
    LATEST_SUMMARY=$(ls -t "${LOG_DIR}"/summary_*.txt 2>/dev/null | head -1)
    LATEST_SERVICE=$(ls -t "${LOG_DIR}"/vllm_service_*.log 2>/dev/null | head -1)
    LATEST_PERF=$(ls -t "${LOG_DIR}"/perf_bench_*.log 2>/dev/null | head -1)
    LATEST_BENCH=$(ls -t "${LOG_DIR}"/ais_bench_*.log 2>/dev/null | head -1)
    set -e
    if [ -z "${LATEST_SERVICE}" ]; then
        echo "No service log found in ${LOG_DIR}/"
        exit 1
    fi
    echo "Parsing latest logs:"
    echo "  service: ${LATEST_SERVICE}"
    # 查找同一次 run 的所有 perf/bench 日志
    PERF_RUNS=$(ls -t "${LOG_DIR}"/perf_bench_*_run*.log 2>/dev/null | head -20)
    BENCH_RUNS=$(ls -t "${LOG_DIR}"/ais_bench_*_run*.log 2>/dev/null | head -20)
    if [ -z "${PERF_RUNS}" ]; then
        # 没有 run 后缀的,用单次日志
        PERF_RUNS="${LATEST_PERF}"
        BENCH_RUNS="${LATEST_BENCH}"
    fi
    echo "  perf logs: ${PERF_RUNS}"
    echo "  bench logs: ${BENCH_RUNS:-<none>}"
    # 对每个 perf log 调用一次 collector
    for perf_log in ${PERF_RUNS}; do
        BENCH_ARG=""
        for bench_log in ${BENCH_RUNS}; do
            perf_run=$(echo "${perf_log}" | grep -oP '_run\K\d+' || echo "")
            bench_run=$(echo "${bench_log}" | grep -oP '_run\K\d+' || echo "")
            if [ "${perf_run}" = "${bench_run}" ] || [ -z "${perf_run}" ]; then
                BENCH_ARG="--bench-log ${bench_log}"
                break
            fi
        done
        python3 "${METRICS_SCRIPT}" \
            --service-log "${LATEST_SERVICE}" \
            --perf-log "${perf_log}" \
            ${BENCH_ARG} \
            --summary "${LATEST_SUMMARY:-/dev/null}" \
            2>&1 || true
    done
    exit 0
fi

# =====================================================================
# --start (默认): 启动后台编排器
# =====================================================================
if [ "${_ORCH_MODE}" = "0" ]; then
    # --- 清理旧状态和残留 NPU 进程 ---
    echo "Cleaning up previous processes..."
    kill_npu_processes

    # 清除旧状态文件
    rm -rf "${STATUS_DIR}"
    mkdir -p "${STATUS_DIR}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SERVICE_LOG="${LOG_DIR}/vllm_service_${TIMESTAMP}.log"
    PERF_LOG="${LOG_DIR}/perf_bench_${TIMESTAMP}.log"
    BENCH_LOG="${LOG_DIR}/ais_bench_${TIMESTAMP}.log"
    SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.txt"
    BENCH_WORK_DIR="${LOG_DIR}/bench_workdir_${TIMESTAMP}"

    # 写 run_info
    : > "${STATUS_DIR}/run_info.txt"
    write_info "timestamp=${TIMESTAMP}"
    write_info "service_log=${SERVICE_LOG}"
    write_info "perf_log=${PERF_LOG}"
    write_info "bench_log=${BENCH_LOG}"
    write_info "summary=${SUMMARY_FILE}"
    write_info "kv_budget=${KV_BUDGET}"
    write_info "run_accuracy=${RUN_ACCURACY}"
    write_info "repeat=${REPEAT}"
    write_info "defer_prefill_on_ascend=${DEFER_PREFILL_ON_ASCEND}"
    write_info "perf_params=input_len=${PERF_INPUT_LEN},output_len=${PERF_OUTPUT_LEN},data_num=${PERF_DATA_NUM},concurrency=${PERF_CONCURRENCY}"

    echo "${SERVICE_LOG}" > "${STATUS_DIR}/service.log"
    echo "${PERF_LOG}" > "${STATUS_DIR}/perf.log"
    echo "${BENCH_LOG}" > "${STATUS_DIR}/bench.log"
    echo "${SUMMARY_FILE}" > "${STATUS_DIR}/summary.txt"
    echo "${BENCH_WORK_DIR}" > "${STATUS_DIR}/bench_workdir.txt"

    write_status "launching"

    echo "Starting TriAttention test orchestrator in background..."
    echo "  KV_BUDGET=${KV_BUDGET}, RUN_ACCURACY=${RUN_ACCURACY}, REPEAT=${REPEAT}, DEFER_PREFILL=${DEFER_PREFILL_ON_ASCEND}"
    echo "  Service log: ${SERVICE_LOG}"
    echo "  Perf log:    ${PERF_LOG}"
    [ "${RUN_ACCURACY}" = "1" ] && echo "  Bench log:   ${BENCH_LOG}"
    echo ""
    echo "Poll with:  bash script_runtest/run_qwen35_test.sh --status"
    echo "Collect:    bash script_runtest/run_qwen35_test.sh --collect"
    echo "Stop:       bash script_runtest/run_qwen35_test.sh --stop"

    # 用 setsid 启动编排器,完全脱离当前进程组
    # 这样即使调用方(bash 工具)超时被杀,编排器和所有子进程都能继续运行
    setsid env \
        _TRIATTN_ORCHESTRATOR=1 \
        REPO_DIR="${REPO_DIR}" \
        MODEL_PATH="${MODEL_PATH}" \
        BENCH_DIR="${BENCH_DIR}" \
        PERF_BENCH_DIR="${PERF_BENCH_DIR}" \
        LOG_DIR="${LOG_DIR}" \
        STATUS_DIR="${STATUS_DIR}" \
        PORT="${PORT}" \
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
        KV_BUDGET="${KV_BUDGET}" \
        SPARSE_STATS_PATH="${SPARSE_STATS_PATH}" \
        VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
        TP_SIZE="${TP_SIZE}" \
        PERF_INPUT_LEN="${PERF_INPUT_LEN}" \
        PERF_OUTPUT_LEN="${PERF_OUTPUT_LEN}" \
        PERF_DATA_NUM="${PERF_DATA_NUM}" \
        PERF_CONCURRENCY="${PERF_CONCURRENCY}" \
        KEEP_SERVICE="${KEEP_SERVICE}" \
        RUN_ACCURACY="${RUN_ACCURACY}" \
        REPEAT="${REPEAT}" \
        DEFER_PREFILL_ON_ASCEND="${DEFER_PREFILL_ON_ASCEND}" \
        SERVICE_READY_TIMEOUT="${SERVICE_READY_TIMEOUT}" \
        SERVICE_LOG="${SERVICE_LOG}" \
        PERF_LOG="${PERF_LOG}" \
        BENCH_LOG="${BENCH_LOG}" \
        SUMMARY_FILE="${SUMMARY_FILE}" \
        BENCH_WORK_DIR="${BENCH_WORK_DIR}" \
        METRICS_SCRIPT="${METRICS_SCRIPT}" \
        bash "${BASH_SOURCE[0]}" --start \
        > "${LOG_DIR}/orchestrator_${TIMESTAMP}.log" 2>&1 &

    ORCH_PID=$!
    echo "${ORCH_PID}" > "${STATUS_DIR}/orchestrator.pid"
    echo "Orchestrator PID=${ORCH_PID}"
    sleep 2
    kill -0 "${ORCH_PID}" 2>/dev/null && echo "Orchestrator is running." || echo "WARNING: Orchestrator may have failed to start."
    exit 0
fi

# =====================================================================
# 编排器模式:实际执行工作流
# =====================================================================
# 所有环境变量已通过 setsid env 传入,直接使用

: > "${SUMMARY_FILE}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${SUMMARY_FILE}"
}

die() {
    log "FATAL: $*"
    write_status "failed"
    log "=== Service log tail (last 80 lines) ==="
    tail -80 "${SERVICE_LOG}" 2>/dev/null >> "${SUMMARY_FILE}" || true
    exit 1
}

# ========== Pre-flight 检查 ==========
write_status "preflight"
log "================ Qwen3.5-27B TriAttention Test ================"
log "KV_BUDGET=${KV_BUDGET}, RUN_ACCURACY=${RUN_ACCURACY}"
log "Perf: input_len=${PERF_INPUT_LEN} output_len=${PERF_OUTPUT_LEN} data_num=${PERF_DATA_NUM} concurrency=${PERF_CONCURRENCY}"

[ ! -d "${MODEL_PATH}" ] && die "Model path not found: ${MODEL_PATH}"
[ ! -f "${SPARSE_STATS_PATH}" ] && die "Sparse stats not found: ${SPARSE_STATS_PATH}"
[ ! -d "${BENCH_DIR}" ] && die "Benchmark dir not found: ${BENCH_DIR}"
[ ! -d "${PERF_BENCH_DIR}" ] && die "Perf bench dir not found: ${PERF_BENCH_DIR}"
log "Pre-flight checks: OK"

# ========== 环境变量 ==========
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export TASK_QUEUE_ENABLE=1
export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_KV_BUDGET="${KV_BUDGET}"
export TRIATTN_RUNTIME_SPARSE_STATS_PATH="${SPARSE_STATS_PATH}"
export TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND="${DEFER_PREFILL_ON_ASCEND}"
export TRIATTN_RUNTIME_LOGGING=true
export TRIATTN_RUNTIME_LOG_DECISIONS=true
export TRIATTN_RUNTIME_LOG_EXECUTION_PATH=true

# ========== 启动 vLLM 服务 ==========
write_status "service_starting"
log "Starting vLLM serve (port=${PORT}, tp=${TP_SIZE})..."

cd "${REPO_DIR}"

# 用 setsid 启动 vllm,使其完全独立于编排器
# 这样即使编排器被杀,服务也能继续运行(配合 KEEP_SERVICE)
setsid vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --data-parallel-size 1 \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --quantization ascend \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true}' \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
    > "${SERVICE_LOG}" 2>&1 &

SERVICE_PID=$!
echo "${SERVICE_PID}" > "${STATUS_DIR}/service.pid"
log "vLLM service PID=${SERVICE_PID}, log=${SERVICE_LOG}"

sleep 5
if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    die "vLLM process died immediately after startup"
fi

# ========== 等待服务就绪 ==========
log "Waiting for service ready (timeout: ${SERVICE_READY_TIMEOUT}s)..."
READY=0
for i in $(seq 1 $((SERVICE_READY_TIMEOUT / 5))); do
    if grep -q "Application startup complete" "${SERVICE_LOG}" 2>/dev/null; then
        READY=1
        log "Service is ready!"
        break
    fi
    if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
        die "Service process died during startup"
    fi
    if [ $((i % 12)) -eq 0 ]; then
        LAST_LINE=$(tail -1 "${SERVICE_LOG}" 2>/dev/null || echo "<empty>")
        log "  waiting... (${i}*5s) ${LAST_LINE}"
    fi
    sleep 5
done
[ "${READY}" = "1" ] || die "Service not ready within ${SERVICE_READY_TIMEOUT}s"

write_status "service_ready"
sleep 10
log "Service confirmed ready."

# ========== 运行性能测试(可重复多次) ==========
PERF_LOGS=""
for run_idx in $(seq 1 "${REPEAT}"); do
    write_status "perf_running"
    echo "${run_idx}/${REPEAT}" > "${STATUS_DIR}/run_progress"
    RUN_PERF_LOG="${PERF_LOG%.log}_run${run_idx}.log"
    log "Starting performance test (run ${run_idx}/${REPEAT})..."
    log "  input_len=${PERF_INPUT_LEN} output_len=${PERF_OUTPUT_LEN} data_num=${PERF_DATA_NUM} concurrency=${PERF_CONCURRENCY}"

    cd "${PERF_BENCH_DIR}"

    rm -f "${STATUS_DIR}/perf.exitcode"
    setsid bash -c "
        cd '${PERF_BENCH_DIR}'
        python3 aisbench_test.py \
            --input_len ${PERF_INPUT_LEN} \
            --output_len ${PERF_OUTPUT_LEN} \
            --data_num ${PERF_DATA_NUM} \
            --concurrency ${PERF_CONCURRENCY}
        echo \$? > '${STATUS_DIR}/perf.exitcode'
    " > "${RUN_PERF_LOG}" 2>&1 &
    PERF_PID=$!
    echo "${PERF_PID}" > "${STATUS_DIR}/perf.pid"

    log "Waiting for perf test run ${run_idx}/${REPEAT} to complete..."
    while kill -0 "${PERF_PID}" 2>/dev/null; do
        sleep 10
    done
    PERF_EXIT=""
    [ -f "${STATUS_DIR}/perf.exitcode" ] && PERF_EXIT=$(cat "${STATUS_DIR}/perf.exitcode" 2>/dev/null)
    log "Perf test run ${run_idx}/${REPEAT} finished (exit=${PERF_EXIT:-unknown})"
    PERF_LOGS="${PERF_LOGS} ${RUN_PERF_LOG}"
done
write_status "perf_done"

# ========== 运行精度评测(可选,可重复多次) ==========
BENCH_LOGS=""
if [ "${RUN_ACCURACY}" = "1" ]; then
    for run_idx in $(seq 1 "${REPEAT}"); do
        write_status "bench_running"
        echo "${run_idx}/${REPEAT}" > "${STATUS_DIR}/run_progress"
        RUN_BENCH_LOG="${BENCH_LOG%.log}_run${run_idx}.log"
        RUN_BENCH_WD="${BENCH_WORK_DIR}_run${run_idx}"
        log "Starting accuracy benchmark run ${run_idx}/${REPEAT} (longbenchv2_gen)..."

        cd "${BENCH_DIR}"

        rm -f "${STATUS_DIR}/bench.exitcode"
        setsid bash -c "
            cd '${BENCH_DIR}'
            ais_bench --models vllm_api_general_chat --datasets longbenchv2_gen --debug --dump-eval-details -w '${RUN_BENCH_WD}'
            echo \$? > '${STATUS_DIR}/bench.exitcode'
        " > "${RUN_BENCH_LOG}" 2>&1 &
        BENCH_PID=$!
        echo "${BENCH_PID}" > "${STATUS_DIR}/bench.pid"

        log "Waiting for accuracy benchmark run ${run_idx}/${REPEAT} to complete..."
        while kill -0 "${BENCH_PID}" 2>/dev/null; do
            sleep 10
        done
        BENCH_EXIT=""
        [ -f "${STATUS_DIR}/bench.exitcode" ] && BENCH_EXIT=$(cat "${STATUS_DIR}/bench.exitcode" 2>/dev/null)
        log "Accuracy benchmark run ${run_idx}/${REPEAT} finished (exit=${BENCH_EXIT:-unknown})"
        BENCH_LOGS="${BENCH_LOGS} ${RUN_BENCH_LOG}"
    done
    write_status "bench_done"
fi

# ========== 收集指标(每次 run 分别提取) ==========
write_status "collecting"
log ""
log "==================== METRICS SUMMARY ===================="
log ""

if [ -f "${METRICS_SCRIPT}" ]; then
    for perf_log in ${PERF_LOGS}; do
        RUN_TAG=""
        echo "${perf_log}" | grep -q "_run" && RUN_TAG=$(echo "${perf_log}" | grep -oP '_run\K\d+')
        [ -n "${RUN_TAG}" ] && log "" && log "--- Perf Run ${RUN_TAG}/${REPEAT} ---"
        BENCH_ARGS=""
        if [ -n "${BENCH_LOGS}" ]; then
            for bench_log in ${BENCH_LOGS}; do
                BENCH_RUN_TAG=""
                echo "${bench_log}" | grep -q "_run" && BENCH_RUN_TAG=$(echo "${bench_log}" | grep -oP '_run\K\d+')
                if [ "${BENCH_RUN_TAG}" = "${RUN_TAG}" ]; then
                    BENCH_ARGS="--bench-log ${bench_log}"
                    break
                fi
            done
        fi
        python3 "${METRICS_SCRIPT}" \
            --service-log "${SERVICE_LOG}" \
            --perf-log "${perf_log}" \
            ${BENCH_ARGS} \
            --summary "${SUMMARY_FILE}" \
            >> "${SUMMARY_FILE}" 2>&1 || log "(metrics collection encountered errors for ${perf_log})"
    done
fi

# ========== 完成 ==========
write_status "all_done"
log ""
log "==================== END OF TEST ===================="
log "Service log:  ${SERVICE_LOG}"
log "Perf log:     ${PERF_LOG}"
[ "${RUN_ACCURACY}" = "1" ] && log "Bench log:    ${BENCH_LOG}"
log "Summary:      ${SUMMARY_FILE}"

# ========== 服务清理 ==========
if [ "${KEEP_SERVICE}" != "1" ]; then
    log "Cleaning up service and NPU processes..."
    kill_npu_processes
else
    log "KEEP_SERVICE=1, service PID=${SERVICE_PID} left running."
fi

log "ALL DONE."
