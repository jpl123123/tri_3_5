# vllm‑ascend Qwen 启动与分支说明
## 一、服务启动脚本
```bash
# 拉起 Qwen‑3.5 推理服务
bash start_vllm_qwen3.5.sh

# 拉起 Qwen3 推理服务
bash start_vllm_qwen3.sh
```

## 二、自动化精度性能调优脚本
```bash
# Qwen‑3 基于 vllm‑ascend 全自动精度、性能参数寻优
bash auto_pref_acc.sh

# 清理寻优脚本输出的原始日志，规整日志数据
python clean_log.py
```

## 三、各分支用途说明
| 分支名称 | 适配版本 & 使用场景 |
|---|---|
| `main-0.18.0` | vllm‑ascend‑0.18.0，内置 Tri‑Attention，同时兼容 Qwen3、Qwen3.5，可运行自动精度性能寻优脚本 |
| `pc_offload_qwen3-0.18.0` | vllm‑ascend‑0.18.0，专门用于 Qwen3 KV‑Cache 卸载相关功能 |
| `fix/partial-rope-qwen35-v0.23.0rc1` | vllm‑ascend‑0.23.0rc1，修复适配 Qwen3.5 partial‑rope 兼容性问题 |

## 四、完整使用指引
1. 运行 Tri‑Attention 推理（Qwen3 / Qwen3.5‑0.18.0、执行全自动寻优）：切换至 `main-0.18.0`
2. 需要开启 Qwen3 KV‑Cache 缓存卸载：切换至 `pc_offload_qwen3-0.18.0`
3. 基于 vllm‑ascend 0.23.0rc1 部署 Qwen3.5：使用 `fix/partial-rope-qwen35-v0.23.0rc1`

> 切换分支完成环境编译部署后，再执行对应 shell 脚本拉起推理实例；
> 跑完 `auto_pref_acc.sh` 参数遍历任务之后，运行 `clean_log.py` 处理输出日志。
