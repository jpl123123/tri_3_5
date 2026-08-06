# 一、triattn offload 安装指南

## 0. 配置代理
```bash
echo 'export http_proxy=http://p_atlas:proxy%40123@proxy.huawei.com:8080' >> /etc/bash.bashrc
echo 'export https_proxy=http://p_atlas:proxy%40123@proxy.huawei.com:8080' >> /etc/bash.bashrc
echo 'export no_proxy=localhost,127.0.0.1,.huawei.com' >> /etc/bash.bashrc
source /etc/bash.bashrc

pip config set global.index-url [https://mirrors.tools.huawei.com/pypi/simple](https://mirrors.tools.huawei.com/pypi/simple)
pip config set global.trusted-host mirrors.tools.huawei.com
pip config set global.timeout 120
```

## 1. 安装代码仓

### 1.1 memfabric_hybrid
```bash
git clone [https://gitcode.com/Ascend/memfabric_hybrid.git](https://gitcode.com/Ascend/memfabric_hybrid.git)
cd memfabric_hybrid

# git checkout release/1.1  # 这里以release/1.1为例 
# 默认的master分支就行。版本要求1.2.0

git clean -xdf
git reset --hard

bash script/build_and_pack_run.sh

cd output
bash memfabric_hybrid-*_*_*.run  # optional: --install-path=${your path}
# 注意替换run路径
# 一定要先装fabric
```

### 1.2 memcache
```bash
git clone [https://gitcode.com/chris_waity/memcache.git](https://gitcode.com/chris_waity/memcache.git)
cd memcache

# Initialize and update only the required submodules (excluding test dependencies)
git submodule update --init 3rdparty/

# Update memfabric_hybrid to the latest version of a specific branch (e.g., master)
# Replace 'master' with your target branch name
git -c submodule.3rdparty/memfabric_hybrid.branch=master submodule update --remote 3rdparty/memfabric_hybrid

bash script/build_and_pack_run.sh --build_mode RELEASE --build_test OFF

cd output
bash memcache_hybrid-*_linux_aarch64.run # 请修改为实际路径和文件名
```

### 1.3 benchmark
```bash
cp -r /softwarePlatform/s00968471/benchmark-3.1-20260119-master/ ./
cd benchmark-3.1-20260119-master
pip3 install -e ./ --use-pep517
```

### 1.4 triattention
```bash
git clone [https://github.com/jpl123123/tri_3_5.git](https://github.com/jpl123123/tri_3_5.git)
git switch yout_branch #这里用pc_offload_qwen3-0.18.0
pip install -e . --no-deps
```

## 2. 各种脚本与配置

* **最新的启动 vllm 的脚本** (`start_vllm_latest.sh`)
  ```bash
  cp /softwarePlatform/images_directory/tri/start_vllm_latest.sh ./
  ```
* **meta 启动脚本** (`start_meta.py` - 不会手动启动，但必须和 vllm 启动脚本同级)
  ```bash
  cp /softwarePlatform/z50058184/start_meta.py ./
  ```
* **ais_bench 启动脚本** (`run_perf.py`)
  ```bash
  cp /softwarePlatform/z50058184/run_perf.sh ./
  ```
  *(注：在这里改端口)*

**配置文件路径：**
* `/usr/local/memcache_hybrid/latest/config/mmc-local.conf`
* `/usr/local/memcache_hybrid/latest/config/mmc-meta.conf`

**查看 meta 日志路径：**
* `/var/log/memcache_hybrid/logs/mmc-meta.log`

---

# 二、triattn offload 使用指南

*(安装指南见：triattn offload安装指南)*

## 1. 核心运行脚本

说明：此脚本在容器中运行，目录为 `/workspace/run_vllm.sh`；
实现拉起 vLLM 服务，跑 aisbench 测评，kill 掉所有（由当前容器拉起的）进程，自动化全流程测评/单流程补测的功能。

```bash
#!/bin/bash
set -eo pipefail

# ====================================================================
# 模块 1：全局配置与路径定义 (Global Config)
# ====================================================================
export MODEL_PATH="/softwarePlatform/z00872399/Qwen/Qwen3-32B"
export LOG_ROOT="/workspace/llm_deploy_logs"
export BENCH_DIR="/workspace/benchmark-3.1-20260119-master"

# 数据集绝对路径
DATASET_PERF_16K="/softwarePlatform/z50058184/longbench_v1/multifieldqa_zh_16k_repeated.jsonl"
DATASET_PERF_32K="/softwarePlatform/datasets_directory/longbench-repeat50/multifieldqa-f_32k.jsonl"
DATASET_ACC="/softwarePlatform/z50058184/longbench_v2/filtered_below_40k.json"

# ====================================================================
# 模块 2：环境初始化与清理 (Env & Cleanup)
# ====================================================================
check_and_install_fuser() {
    if ! command -v fuser &> /dev/null; then
        echo "🔧 [系统检查] 容器内未检测到 fuser，正在极速安装..."
        yum install -y psmisc --setopt=sslverify=false -q
    fi
}

clean_npu() {
    echo -n "🧹 [容器级清场] 正在彻底清理残留进程与 NPU 资源..."
    
    local process=$(ps aux | grep -v $$ | grep -v sshd | grep -v bash | grep -v PID | awk '{print $2}')
    if [ -n "$process" ]; then
        kill -9 $process >/dev/null 2>&1 || true
    fi
    pkill -9 -f python3 >/dev/null 2>&1 || true
    
    fuser -k -9 /dev/davinci* >/dev/null 2>&1 || true
    
    rm -f /dev/shm/psm_* /dev/shm/sem.loky-* /dev/shm/sem.mp-* >/dev/null 2>&1 || true
    rm -rf /dev/shm/*vllm* /dev/shm/*torch* >/dev/null 2>&1 || true
    rm -rf "/home/kvcache"
    mkdir -p "/home/kvcache"
    
    echo " ✅ 清理完毕！"
    sleep 3
}

# ====================================================================
# 模块 3：服务拉起与探活 (Service Start & Probe)
# ====================================================================
wait_for_vllm_ready() {
    local PID="$1"
    local LOG_FILE="$2"
    local TIMEOUT="${3:-1200}"
    local START=$(date +%s)
    
    echo -n "⏳ 正在等待 vLLM 服务启动就绪 (探测 PID: $PID)..."
    while true; do
        if ! kill -0 "$PID" 2>/dev/null; then 
            echo -e "\n❌ 错误: vLLM 进程意外退出！请排查日志: $LOG_FILE" >&2
            exit 1
        fi
        if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then 
            echo -e "\n✅ vLLM 启动成功！"
            break
        fi
        if [ $(( $(date +%s) - START )) -gt "$TIMEOUT" ]; then 
            echo -e "\n❌ 错误: vLLM 启动超时 (${TIMEOUT}s)！请排查日志: $LOG_FILE" >&2
            exit 1
        fi
        sleep 3
    done
}

setup_memcache() {
    echo "⚙️  [Offload 组件] 挂载大页内存并启动 Memcache Meta 服务..."
    mount -o remount,size=512G /dev/shm
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    source /usr/local/memfabric_hybrid/set_env.sh
    source /usr/local/memcache_hybrid/set_env.sh

    local CONFIG_DIR=/usr/local/memcache_hybrid/latest/config
    export MMC_META_CONFIG_PATH=${CONFIG_DIR}/mmc-meta.conf
    export MMC_LOCAL_CONFIG_PATH=${CONFIG_DIR}/mmc-local.conf
    export MMC_DIR_STORE_CONFIG_PATH="${CONFIG_DIR}/dir_store.conf"

    local MEM_PER_NPU="32"

    sed -i "s#ock.mmc.local_service.dram.size.*#ock.mmc.local_service.dram.size = ${MEM_PER_NPU}GB#g" "${MMC_LOCAL_CONFIG_PATH}"
    sed -i "s#ock.mmc.local_service.storage.size.*#ock.mmc.local_service.storage.size = 512GB#g" "${MMC_LOCAL_CONFIG_PATH}"
    sed -i "s#ock.mmc.local_service.protocol.*#ock.mmc.local_service.protocol = host_shm#g" "${MMC_LOCAL_CONFIG_PATH}"
    sed -i "s#ock.mmc.log_level.*#ock.mmc.log_level = error#g" "${MMC_META_CONFIG_PATH}"
    sed -i "s#ock.mmc.evict_threshold_high.*#ock.mmc.evict_threshold_high = 80#g" "${MMC_META_CONFIG_PATH}"
    sed -i "s#ock.mmc.evict_threshold_low.*#ock.mmc.evict_threshold_low = 70#g" "${MMC_META_CONFIG_PATH}"

    python /workspace/start_meta.py &
    sleep 3 
}

vllm_tri_no_pc_offload() {
    local TP="$1"
    local PP="$2"
    local LOG_FILE="$3"
    local KV_BUDGET="${4:-8192}" # 接收 Budget，默认 8192

    cd /workspace
    setup_memcache
    export ENABLE_TRIATTENTION=1
    export TRIATTN_RUNTIME_KV_BUDGET="$KV_BUDGET"
    export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/softwarePlatform/z00872399/Longcontext_inference_acceleration/triattention/triattention/triattention/vllm/stats/qwen3_32b_int4_stats.pt

    local BASE_ARGS=(
        serve "${MODEL_PATH}"
        --max-model-len 40960
        --served-model-name "Qwen3-32B"
        --tensor-parallel-size "$TP"
        --data-parallel-size 1
        --block-size 128
        --distributed-executor-backend mp
        --trust-remote-code
        --port 8000
        --gpu-memory-utilization 0.9
        --no-enable-prefix-caching 
        --disable-log-stats
        --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,12,16,32,48,64], "cudagraph_mode":"FULL_DECODE_ONLY"}'
        --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"0","load_async":"true"}}'
    )
    echo "🚀 [启动 vLLM] 模式: TriAttn + No_PC + Offload (TP=$TP, PP=$PP, BUDGET=$KV_BUDGET)"
    nohup vllm "${BASE_ARGS[@]}" > "$LOG_FILE" 2>&1 &
}

vllm_base_pc_no_offload() {
    local TP="$1"
    local PP="$2"
    local LOG_FILE="$3"
    # 如果是 default，就给个 8192 防止环境变量报错，虽然 base 模式下它不起作用
    local KV_BUDGET="${4:-8192}"
    if [ "$KV_BUDGET" = "default" ]; then KV_BUDGET="8192"; fi

    cd /workspace
    export ENABLE_TRIATTENTION=0
    export TRIATTN_RUNTIME_KV_BUDGET="$KV_BUDGET"
    export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/softwarePlatform/z00872399/Longcontext_inference_acceleration/triattention/triattention/triattention/vllm/stats/qwen3_32b_int4_stats.pt

    local BASE_ARGS=(
        serve "${MODEL_PATH}"
        --max-model-len 40960
        --served-model-name "Qwen3-32B"
        --tensor-parallel-size "$TP"
        --data-parallel-size 1
        --block-size 128
        --distributed-executor-backend mp
        --trust-remote-code
        --port 8000
        --gpu-memory-utilization 0.9
        --enable-prefix-caching
        --disable-log-stats
        --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,12,16,32,48,64], "cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    echo "🚀 [启动 vLLM] 模式: Base + PC + No_Offload (TP=$TP, PP=$PP, BUDGET 忽略)"
    nohup vllm "${BASE_ARGS[@]}" > "$LOG_FILE" 2>&1 &
}

# ====================================================================
# 模块 4：核心执行引擎 (Core Execution Engine)
# ====================================================================
run_perf() {
    local vllm_func="$1"
    local DS_PATH="$2"
    local BATCH_SIZE="$3"
    local TP="$4"
    local PP="$5"
    local KV_BUDGET="$6"
    
    local filename=$(basename "$DS_PATH")
    local ctx_len="${filename%.*}"

    for i in 1 2 3; do
        echo -e "\n\033[1;34m=============================================="
        echo "  [PERF - $i/3] 模式: $vllm_func | Budget: $KV_BUDGET | 数据集: $ctx_len | BS: $BATCH_SIZE"
        echo -e "==============================================\033[0m\n"

        # 日志目录带上 budget 信息，方便后续分析
        local LOG_DIR="$LOG_ROOT/perf_${vllm_func}_budget${KV_BUDGET}_bs${BATCH_SIZE}_${ctx_len}_run${i}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$LOG_DIR"
        local VLLM_LOG="$LOG_DIR/vllm.log"
        echo -e "📁 专属 vLLM 日志将写入: \033[1;32m$LOG_DIR\033[0m"

        $vllm_func "$TP" "$PP" "$VLLM_LOG" "$KV_BUDGET"
        VLLM_PID=$!
        wait_for_vllm_ready "$VLLM_PID" "$VLLM_LOG"

        echo -e "\n📊 开始运行 ais_bench 性能测试打流..."
        local PERF_SCRIPT="/softwarePlatform/z50058184/vllm_config/vllm_api_stream_chat_perf_bs${BATCH_SIZE}.py"
        cd "$BENCH_DIR"
        cp "$PERF_SCRIPT" ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
        rm -rf ~/.cache/huggingface/datasets
        cp -f "$DS_PATH" ais_bench/datasets/LongBench/data/multifieldqa_zh.jsonl
        
        ais_bench --models vllm_api_stream_chat --datasets longbench --debug --mode perf --num-prompts 120
        sleep 2

        echo -e "\n🧹 结束本次服务进程..."
        kill -2 $VLLM_PID 2>/dev/null || true
        sleep 2
        clean_npu
    done
}

run_acc() {
    local vllm_func="$1"
    local DS_PATH="$2"
    local BATCH_SIZE="$3" 
    local TP="$4"
    local PP="$5"
    local KV_BUDGET="$6"
    
    local filename=$(basename "$DS_PATH")
    local ctx_len="${filename%.*}"

    echo -e "\n\033[1;36m=============================================="
    echo "  [ACC 精度测试] 模式: $vllm_func | Budget: $KV_BUDGET | 数据集: $ctx_len | BS: $BATCH_SIZE"
    echo -e "==============================================\033[0m\n"

    # 日志目录带上 budget 信息
    local LOG_DIR="$LOG_ROOT/acc_${vllm_func}_budget${KV_BUDGET}_bs${BATCH_SIZE}_${ctx_len}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    local VLLM_LOG="$LOG_DIR/vllm.log"
    echo -e "📁 专属 vLLM 日志将写入: \033[1;32m$LOG_DIR\033[0m"

    $vllm_func "$TP" "$PP" "$VLLM_LOG" "$KV_BUDGET"
    VLLM_PID=$!
    wait_for_vllm_ready "$VLLM_PID" "$VLLM_LOG"

    echo -e "\n🎯 准备执行连续 5 次的精度测试..."
    local ACC_SCRIPT="/softwarePlatform/z50058184/vllm_config/vllm_api_stream_chat_acc.py"
    cd "$BENCH_DIR"
    cp "$ACC_SCRIPT" ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
    
    rm -rf ~/.cache/huggingface/datasets
    mkdir -p ais_bench/datasets/LongBench-v2
    cp -f "$DS_PATH" ais_bench/datasets/LongBench-v2/data.json

    for j in {1..5}; do
        echo "  >>> 正在运行精度测试，轮次: $j/5"
        # 备注：你的打流命令是 general_chat，请确保你的脚本支持
        ais_bench --models vllm_api_general_chat --datasets longbenchv2_gen --debug
        sleep 2
    done

    echo -e "\n🧹 结束本次服务进程..."
    kill -2 $VLLM_PID 2>/dev/null || true
    sleep 2
    clean_npu
}

# ====================================================================
# 模块 5：业务流水线组装 (Pipelines)
# ====================================================================
pipeline_perf() {
    local TP="$1"
    local PP="$2"
    local modes=("vllm_tri_no_pc_offload" "vllm_base_pc_no_offload")
    local datasets=("$DATASET_PERF_16K" "$DATASET_PERF_32K")
    local batch_sizes=(16 32)

    for mode in "${modes[@]}"; do
        # 智能判定：如果是 base 模式，由于不开启 TriAttn，没必要测多次 Budget 浪费时间
        local current_budgets=("6144" "8192")
        if [[ "$mode" == *"base"* ]]; then
            current_budgets=("default")
        fi

        for budget in "${current_budgets[@]}"; do
            for ds in "${datasets[@]}"; do
                for bs in "${batch_sizes[@]}"; do
                    run_perf "$mode" "$ds" "$bs" "$TP" "$PP" "$budget"
                done
            done
        done
    done
}

pipeline_acc() {
    local TP="$1"
    local PP="$2"
    local modes=("vllm_tri_no_pc_offload" "vllm_base_pc_no_offload")
    local datasets=("$DATASET_ACC")
    local batch_sizes=(16 32) 

    for mode in "${modes[@]}"; do
        # 同样智能判定，避免浪费时间
        local current_budgets=("6144" "8192")
        if [[ "$mode" == *"base"* ]]; then
            current_budgets=("default")
        fi

        for budget in "${current_budgets[@]}"; do
            for ds in "${datasets[@]}"; do
                for bs in "${batch_sizes[@]}"; do
                    run_acc "$mode" "$ds" "$bs" "$TP" "$PP" "$budget"
                done
            done
        done
    done
}

# ====================================================================
# 模块 6：脚本入口路由 (Entrypoint)
# ====================================================================

# 测试环境保障（全局执行 1 次）
check_and_install_fuser
clean_npu

# 提取命令参数
COMMAND="$1"

case "$COMMAND" in
    "all")
        pipeline_perf 4 1
        pipeline_acc 4 1
        ;;
    "perf")
        pipeline_perf 4 1
        ;;
    "acc")
        pipeline_acc 4 1
        ;;
    "debug_perf")
        # 手动跑单条: mode, ds_path, bs, TP, PP, KV_BUDGET
        run_perf "$2" "$3" "$4" "${5:-8}" "${6:-1}" "${7:-8192}"
        ;;
    "debug_acc")
        run_acc "$2" "$3" "$4" "${5:-8}" "${6:-1}" "${7:-8192}"
        ;;
    *)
        echo "=========================================================="
        echo "启动方式指南 (Usage): "
        echo "  bash $0 all          -> 全量测试 (自动在底层应用硬编码的 TP/PP)"
        echo "  bash $0 perf         -> 仅运行 Perf 性能测试全矩阵"
        echo "  bash $0 acc          -> 仅运行 Acc 精度测试全矩阵"
        echo "  bash $0 debug_perf <mode> <ds_path> <bs> [TP] [PP] [KV_BUDGET]"
        echo "=========================================================="
        exit 1
        ;;
esac
```

### 1.1 脚本运行
表示在后台运行你的整个服务：
```bash
nohup bash run_vllm.sh > your_log.log 2>&1 &
```

## 2. 脚本踩坑指南

### 2.1 ais_bench 测评 config 端口
比如这里指定 `general_chat`，则需要去以下路径修改：
`/workspace/benchmark-3.1-20260119-master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`

*(注意：这里以 92 为例子，一定要指定 ip 为 `135.82.26.92`，不要写成 `localhost`，不然会报错。)*

### 2.2 vLLM 拉起时关闭日志 debug
否则会遇到 tri attn 和 vLLM 不兼容，导致计数到负值，直接中断 vLLM 进程。
**解决办法**：启动时一定要指定 `--disable-log-stats`

### 2.3 端口问题

#### 2.3.1 vLLM 端口和 ais_bench 端口
这两个端口必须保持一致。

#### 2.3.2 memcache_hybrid 端口
检查以下两个配置文件：
* `/usr/local/memcache_hybrid/latest/config/mmc-local.conf`
* `/usr/local/memcache_hybrid/latest/config/mmc-meta.conf`

这两个 conf 里的端口要保持一致，但**绝对不能**和 vLLM 端口冲突。
