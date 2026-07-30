export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_KV_BUDGET=6144
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/softwarePlatform/z00872399/Longcontext_inference_acceleration/triattention/triattention/triattention/vllm/stats/qwen3_32b_int4_stats.pt
# ========================================================

# ==================== 模型信息 ============================
export Model_PATH=/softwarePlatform/z00872399/Qwen

MODEL_NAME="Qwen3-32B"
BASE_ARGS=(
    serve "${Model_PATH}/${MODEL_NAME}"
    --max-model-len 40960
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size 4
    --data-parallel-size 1
    --block_size 128
    --distributed-executor-backend mp
    --trust-remote-code
    --port 8001
    --gpu_memory_utilization 0.9
    --no-enable-prefix-caching
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,12,16,32,48,64], "cudagraph_mode":"FULL_DECODE_ONLY"}'
)

# 启动服务
vllm "${BASE_ARGS[@]}"
