# vllm‑ascend Qwen 启动与分支说明
## 一、服务启动脚本
```bash
# 拉起 Qwen‑3.5 推理服务
bash start_vllm_qwen3.5.sh

# 拉起 Qwen3 推理服务
bash start_vllm_qwen3.sh
```

## 二、各分支用途说明
| 分支名称 | 适配版本 & 使用场景 |
|---|---|
| `main-0.18.0` | vllm‑ascend‑0.18.0，内置 Tri‑Attention，同时兼容 Qwen3、Qwen3.5 |
| `pc_offload_qwen3-0.18.0` | vllm‑ascend‑0.18.0，专门用于 Qwen3 KV‑Cache 卸载相关功能 |
| `fix/partial-rope-qwen35-v0.23.0rc1` | vllm‑ascend‑0.23.0rc1，修复适配 Qwen3.5 partial‑rope 兼容性问题 |

## 三、使用指引
1. 运行 Tri‑Attention 推理（Qwen3 / Qwen3.5 0.18.0）：切换至 `main-0.18.0`
2. 需要开启 Qwen3 KV Cache 卸载：切换至 `pc_offload_qwen3-0.18.0`
3. 基于 vllm‑ascend 0.23.0rc1 部署 Qwen3.5：使用 `fix/partial-rope-qwen35-v0.23.0rc1`

> 切换分支之后，再执行对应 shell 脚本拉起推理实例。
