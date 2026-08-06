#!/bin/bash
set -eo pipefail

# ===================== 全局配置 =====================
MODEL_PATH="/softwarePlatform/z00872399/Qwen/Qwen3-32B"
LOG_ROOT="/workspace/llm_deploy_logs"

# ===================== 【主机级】强制清理昇腾NPU =====================
force_clean_npu() {
    echo -e "\n🧹 【主机】强制彻底清理 NPU 所有残留资源..."
    fuser -k /dev/davinci*  2>/dev/null || true
    fuser -k /dev/drv0*     2>/dev/null || true
    fuser -k /dev/ascend*   2>/dev/null || true
    rm -rf /dev/shm/* /dev/shm/.*torch* /dev/shm/.*vllm* /dev/shm/.*ascend* 2>/dev/null || true
    sync
    sleep 15
    echo -e "✅ 【主机】NPU 环境已完全重置，可以安全启动新容器\n"
}

# ===================== vLLM 启动函数定义 =====================
# 基础版 vLLM 启动命令（无 H2O）
vllmbase() {
    local TP="$1"
    local PP="$2"
    local LOG_FILE="$3"

    nohup vllm serve "$MODEL_PATH" \
        --max-model-len 40960 \
        --served-model-name Qwen3-32B \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9 \
        --block-size 128 \
        --distributed-executor-backend mp \
        --trust-remote-code \
        --port 8000 \
        --no-enable-prefix-caching \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}' \
        > "$LOG_FILE" 2>&1 &
}

# H2O 增强版 vLLM 启动命令
vllmh2o() {
    local TP="$1"
    local PP="$2"
    local LOG_FILE="$3"
    local KV_BUDGET="${4:-4096}"  # 【修改点】新增第4个参数，默认值4096兼容旧调用
    
    echo -e "************开始扫描KV_BUDGET为$KV_BUDGET *****************"

    export ENABLE_TRIATTENTION=1
    export TRIATTN_RUNTIME_KV_BUDGET="$KV_BUDGET"  # 【修改点】使用动态传入的 KV_BUDGET
    export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/softwarePlatform/z00872399/Longcontext_inference_acceleration/triattention/triattention/triattention/vllm/stats/qwen3_32b_int4_stats.pt
    # ========================================================

    # ==================== 模型信息 ============================

    nohup vllm serve "$MODEL_PATH" \
        --max-model-len 40960 \
        --served-model-name Qwen3-32B \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9 \
        --block-size 128 \
        --distributed-executor-backend mp \
        --trust-remote-code \
        --port 8000 \
        --no-enable-prefix-caching \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}' \
        > "$LOG_FILE" 2>&1 &
}

# ===================== 单个Docker测试函数 =====================
run_docker_test() {
    local DOCKER_NAME="$1"
    local TP="$2"
    local PP="$3"
    local PORT="$4"
    local VLLM_FUNC="$5"
    local BENCH_DIR="$6"
    local BATCH_SIZE="$7"
    local MACHINE_NUM="$8"

    # 参数校验：设置默认值
    if [ -z "$BATCH_SIZE" ]; then
        BATCH_SIZE=16
        echo -e "\n⚠️  未指定Batch Size，自动使用默认值：16"
    fi

    if [ -z "$MACHINE_NUM" ]; then
        MACHINE_NUM=93
        echo -e "\n⚠️  未指定机器编号，自动使用默认值：93"
    fi

    # 【核心】每次启动前，必须彻底清理NPU
    force_clean_npu

    echo -e "\n\033[1;32m=============================================="
    echo "  启动测试容器：$DOCKER_NAME"
    echo "  机器编号：$MACHINE_NUM"
    echo "  TP=$TP PP=$PP PORT=8000"
    echo "  使用配置：$VLLM_FUNC"
    echo "  Batch Size：$BATCH_SIZE"
    echo "  Benchmark目录：$BENCH_DIR"
    echo -e "==============================================\033[0m\n"

    docker exec -i "$DOCKER_NAME" /bin/bash << EOF
        set -eo pipefail

        # 🔧 关键修复：在容器内部显式定义所有全局变量
        MODEL_PATH="$MODEL_PATH"
        BENCH_DIR="$BENCH_DIR"
        LOG_ROOT="$LOG_ROOT"
        TP="$TP"
        PP="$PP"
        VLLM_FUNC="$VLLM_FUNC"
        BATCH_SIZE="$BATCH_SIZE"
        MACHINE_NUM="$MACHINE_NUM"

        # 【修改点】判断是否为 vllmh2o，决定遍历的 Budget 列表
        if [ "\$VLLM_FUNC" = "vllmh2o" ]; then
            # BUDGETS="2048 4096 6144 8192 12288"
            BUDGETS="6144 8192"
        else
            BUDGETS="default"
        fi

        # 注入两个启动函数到容器内部
        $(declare -f vllmbase)
        $(declare -f vllmh2o)

        # 【修改点】针对不同的 Budget 进行循环测试
        for BUDGET in \$BUDGETS; do
            if [ "\$BUDGET" != "default" ]; then
                echo -e "\n\033[1;33m========================================================"
                echo "  🚀 开始测试 TRIATTN_RUNTIME_KV_BUDGET = \$BUDGET"
                echo -e "========================================================\033[0m\n"
                # 为每个 budget 生成独立的日志目录，避免覆盖
                LOG_DIR="\$LOG_ROOT/\${DOCKER_NAME}_node\${MACHINE_NUM}_tp${TP}_pp${PP}_bs${BATCH_SIZE}_\${VLLM_FUNC}_budget\${BUDGET}_\$(date +%Y%m%d_%H%M%S)"
            else
                LOG_DIR="\$LOG_ROOT/\${DOCKER_NAME}_node\${MACHINE_NUM}_tp${TP}_pp${PP}_bs${BATCH_SIZE}_\${VLLM_FUNC}_\$(date +%Y%m%d_%H%M%S)"
            fi
            
            mkdir -p "\$LOG_DIR"
            VLLM_LOG="\$LOG_DIR/vllm.log"

            cd "\$BENCH_DIR"

            # 启动 vLLM（根据传入的函数名调用）
            echo "[Docker] 启动 vLLM (\$VLLM_FUNC)..."

            # 调用指定的启动函数，如果是 vllmh2o，传入 BUDGET 作为第4个参数
            if [ "\$VLLM_FUNC" = "vllmh2o" ]; then
                \$VLLM_FUNC "\$TP" "\$PP" "\$VLLM_LOG" "\$BUDGET"
            else
                \$VLLM_FUNC "\$TP" "\$PP" "\$VLLM_LOG"
            fi

            VLLM_PID=\$!
            echo "[Docker] vLLM PID: \$VLLM_PID"

            # 等待启动
            echo "[Docker] 等待 vLLM 启动..."
            START=\$(date +%s)
            TIMEOUT=1200

            while true; do
                if ! kill -0 \$VLLM_PID 2>/dev/null; then
                    echo -e "\n❌ vLLM 进程已崩溃"
                    tail -100 "\$VLLM_LOG"
                    exit 1
                fi

                if grep -q "Application startup complete" "\$VLLM_LOG" 2>/dev/null; then
                    echo -e "✅ vLLM 启动成功！"
                    break
                fi

                NOW=\$(date +%s)
                if [ \$((NOW-START)) -gt \$TIMEOUT ]; then
                    echo -e "❌ 启动超时"
                    tail -100 "\$VLLM_LOG"
                    exit 1
                fi

                sleep 3
            done

            # ================== 性能测试 ==================
            echo -e "\n📊 运行性能测试..."
            cd "\$BENCH_DIR"

            PERF_SCRIPT="/softwarePlatform/s00968471/testing/ais/\${MACHINE_NUM}/vllm_api_stream_chat_perf_bs\${BATCH_SIZE}.py"
            echo "[Docker] 使用性能测试脚本：\$PERF_SCRIPT"
            cp "\$PERF_SCRIPT" ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

            run_perf() {
                local f=\$1
                local ctx_len=\$(basename "\$f" .jsonl | grep -oE '[0-9]+k')
                echo -e "\n\033[1;34m=============================================="
                echo "  开始性能测试"
                echo "  机器：node\$MACHINE_NUM, 配置：TP=\$TP, PP=\$PP, BS=\$BATCH_SIZE, 启动方式=\$VLLM_FUNC"
                if [ "\$BUDGET" != "default" ]; then
                    echo "  KV_BUDGET：\$BUDGET"
                fi
                echo "  上下文长度：\$ctx_len"
                echo "  数据集：\$f"
                echo -e "==============================================\033[0m\n"

                rm -rf ~/.cache/huggingface/datasets
                cp -f "\$f" ais_bench/datasets/LongBench/data/multifieldqa_zh.jsonl
                for i in 1 2 3; do
                    echo "  性能测试第 \$i 次"
                    ais_bench --models vllm_api_stream_chat --datasets longbench --debug --mode perf --num-prompts 120
                    sleep 2
                done
            }

             
            run_perf "/softwarePlatform/s00968471/testing/data/multifieldqa_zh_16k.jsonl"
            run_perf "/softwarePlatform/s00968471/testing/data/multifieldqa_zh_32k.jsonl"
            # run_perf "/softwarePlatform/s00968471/testing/data/multifieldqa_zh_64k.jsonl"

            # ================== 精度测试 ==================
            echo -e "\n\033[1;34m=============================================="
            echo "  开始精度测试"
            echo "  机器：node\$MACHINE_NUM, 配置：TP=\$TP, PP=\$PP, BS=\$BATCH_SIZE, 启动方式=\$VLLM_FUNC"
            if [ "\$BUDGET" != "default" ]; then
                echo "  KV_BUDGET：\$BUDGET"
            fi
            echo -e "==============================================\033[0m\n"

            cd "\$BENCH_DIR"
            ACC_SCRIPT="/softwarePlatform/s00968471/testing/ais/\${MACHINE_NUM}/vllm_api_stream_chat_acc.py"
            echo "[Docker] 使用精度测试脚本：\$ACC_SCRIPT"
            cp "\$ACC_SCRIPT" ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

            rm -rf ~/.cache/huggingface/datasets
            cp /softwarePlatform/s00968471/testing/data/multifieldqa_zh.jsonl ais_bench/datasets/LongBench/data/multifieldqa_zh.jsonl

            for i in 1 2 3 4 5; do
                echo "  精度测试第 \$i 次"
                ais_bench --models vllm_api_general_chat --datasets longbenchv2_gen --debug           
                sleep 5
            done

            # 【修改点】单个 Budget 测试完成后的清理工作，确保下一个 Budget 启动环境干净
            echo -e "\n🧹 清理当前 Budget (\$BUDGET) 的残留资源..."
            kill -2 \$VLLM_PID 2>/dev/null || true
            sleep 2
            fuser -k /dev/davinci* 2>/dev/null || true
            sleep 2
            
            if [ "\$BUDGET" != "default" ]; then
                echo -e "✅ TRIATTN_RUNTIME_KV_BUDGET = \$BUDGET 测试全部完成！\n"
            fi
        done

        echo -e "\n✅ 容器 $DOCKER_NAME 所有配置测试全部完成！"
        exit
EOF

    echo -e "\n\033[1;32m=============================================="
    echo " ✅ 容器 $DOCKER_NAME 主机侧调度完成 "
    echo -e "==============================================\033[0m\n"
}

# ===================== 执行 =====================
# 调用格式：run_docker_test 容器ID TP PP 端口 启动函数名 Benchmark目录 BatchSize 当前机器编号
# 示例：node91机器，BS=16，H2O版本 (脚本会自动遍历 2048, 4096, 6144, 8192, 12288)
run_docker_test "c3de9f20ab6e" 8 1 8000 vllmh2o "/workspace/benchmark-3.1-20260119-master" 16 93
run_docker_test "c3de9f20ab6e" 8 1 8000 vllmh2o "/workspace/benchmark-3.1-20260119-master" 32 93
run_docker_test "c3de9f20ab6e" 4 1 8000 vllmh2o "/workspace/benchmark-3.1-20260119-master" 32 93
run_docker_test "c3de9f20ab6e" 4 1 8000 vllmh2o "/workspace/benchmark-3.1-20260119-master" 16 93
run_docker_test "d17a2fc69d32" 8 1 8000 vllmbase "/workspace/benchmark-3.1-20260119-master" 16 93
run_docker_test "d17a2fc69d32" 8 1 8000 vllmbase "/workspace/benchmark-3.1-20260119-master" 32 93
run_docker_test "d17a2fc69d32" 4 1 8000 vllmbase "/workspace/benchmark-3.1-20260119-master" 16 93
run_docker_test "d17a2fc69d32" 4 1 8000 vllmbase "/workspace/benchmark-3.1-20260119-master" 32 93

# 示例2：如果想跑基础版本（不遍历），保持原样即可，脚本会自动识别并只跑一次

