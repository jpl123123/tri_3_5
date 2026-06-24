# TriAttention vLLM-Ascend Codex Project Handoff

Last updated: 2026-06-24

This file preserves the working context for another Codex user/agent. Read it before making changes.

## 1. Current Repository State

- Workspace: `/Users/zhangxiangjun/我的项目/triattention-vllm-ascend/triattention-main`
- Remote: `git@github.com:xj-zhang2018/triattention-main.git`
- Active branch: `tri_zxj_version0615`
- Main development branch to avoid: `codex/vllm-ascend-triattention`
- Important baseline:
  - Initial project commit: `a55a4c4 初识项目代码`
  - Main TriAttention Ascend line before ZXJ branch doc work: `312a132 Port kv cache eviction fix to zxj branch`
  - Latest committed state before this handoff file: `802d85f Use development wording in TriAttention summary`

Current untracked local directories at handoff time:

- `.tmp/`
- `other_code_sa/`

Do not delete or clean these without user approval. `other_code_sa/tri_xj` contains an older TriAttention source snapshot and the important file `other_code_sa/tri_xj/triattention_xj_actual_fixes.md`, which was used to port the KV eviction fix.

## 2. User Collaboration Rules

The user has repeatedly emphasized these rules:

- If the user says "先不要改代码", do not edit runtime/source code. Analysis and documentation are acceptable only when clearly requested.
- When code is changed, report modified file names and absolute paths.
- Keep work on `tri_zxj_version0615`; avoid polluting `codex/vllm-ascend-triattention`.
- For vllm-ascend source reference, check the local `other_code` directory if available.
- Issue monitoring is not automatic. Fetch issue comments only when the user explicitly asks.
- Earlier accidental feedback in issue `#8` should be ignored if it comes up again.
- The user wanted a new issue instead of issue `#7`, because `#7` was being used by someone else.
- After future code fixes, commit the fix and summarize it to the active issue when the workflow is issue-driven.

## 3. Documents To Read First

Read these before planning new work:

- `docs/triattention_ascend_development_summary.md`
  - High-level development summary for课题汇报.
  - Contains a simplified execution flow diagram.
  - Uses the current "开发" wording requested by the user.
- `docs/kv_eviction_strategy_toggle_analysis.md`
  - Explains current KV eviction behavior and why config switches did not fully disable observed low KV usage.
- `docs/vllm_ascend.md`
  - Deployment, logging, runtime flags, and expected vLLM-Ascend behavior.
- `other_code_sa/tri_xj/triattention_xj_actual_fixes.md`
  - Local-only reference explaining the old version's actual KV eviction bug fixes.

## 4. What Has Been Developed

The project started from upstream TriAttention and was developed into a vLLM-Ascend runtime implementation.

Core development areas:

- vLLM-Ascend runtime injection:
  - Scheduler and worker patches.
  - NPUWorker / NPUModelRunner proxy installation.
  - Runtime config and logging controls.
- Ascend input metadata handling:
  - Effective `seq_lens`, `seq_lens_np`, slot mapping, and block table view updates.
  - Boundary clamp for slot positions and seq lengths.
- TriAttention sparse scoring on Ascend:
  - CUDA uses Triton path.
  - Ascend uses PyTorch/torch_npu scoring path.
  - Supports tensor-parallel stats slicing and GQA head reduction.
- KV compaction and real block reclaim:
  - Supports combined vLLM KV cache and Ascend split K/V cache.
  - Moves kept K/V to a compact effective prefix.
  - Releases reclaimable tail blocks.
- Concurrency stability:
  - Per-request effective KV views.
  - Batch row remap handling.
  - Decode slack and block growth preservation.
  - Compression-per-step limits.
- KV eviction bug fix:
  - Cross-process compression event transfer through `kv_connector_output.kv_cache_events`.
  - Engine-core event reading fallback order.
  - Worker block-table capacity clamp at decode boundaries.
- Observability:
  - Execution path logs.
  - Core trace and selector debug options.
  - Phase/E2E/perf profiling.
- Tests:
  - Ascend defaults.
  - Input patching.
  - Runner output bridge.
  - Worker reclaim sync.
  - Zero-copy tail remap.
  - Runtime logging and profiling.

Approximate development size from initial commit to `802d85f`:

- Core `triattention/vllm/runtime/` net new code: about `7.18K` lines.
- Production `triattention/vllm/` net new code: about `7.20K` lines.
- Total including tests, docs, scripts: about `11.88K` lines.

## 5. Key Files

Runtime core:

- `triattention/vllm/runtime/config.py`
- `triattention/vllm/runtime/worker.py`
- `triattention/vllm/runtime/runner.py`
- `triattention/vllm/runtime/scheduler.py`
- `triattention/vllm/runtime/selector_hf.py`
- `triattention/vllm/runtime/hook_group_pipeline.py`
- `triattention/vllm/runtime/hook_impl.py`
- `triattention/vllm/runtime/kv_compaction.py`
- `triattention/vllm/runtime/layout_engine.py`
- `triattention/vllm/runtime/runner_output_bridge.py`
- `triattention/vllm/runtime/input_patch_vllm_v1_backend.py`
- `triattention/vllm/runtime/input_patch_ascend_backend.py`
- `triattention/vllm/runtime/integration_monkeypatch.py`
- `triattention/vllm/runtime/worker_reclaim_sync.py`

Scoring core:

- `triattention/vllm/core/scoring.py`
- `triattention/vllm/core/compressor.py`
- `triattention/vllm/core/utils.py`
- `triattention/common/rope_utils.py`

Tests:

- `triattention/tests/test_ascend_input_patch.py`
- `triattention/tests/test_runner_output_bridge.py`
- `triattention/tests/test_worker_reclaim_sync.py`
- `triattention/tests/test_zero_copy_tail_remap.py`
- `triattention/tests/test_runner_execution_path_logging.py`

## 6. Algorithm Notes

TriAttention acceleration comes from reducing effective KV length during decode:

```text
Full context KV length -> KV_BUDGET-sized effective KV view
```

TPOT improves because each decode token reads and attends over fewer K/V entries. The offline stats do not directly create speed; they make the eviction decision smarter so accuracy is less likely to drop.

Sparse scoring path:

- Offline stats provide RoPE-pre Q/K frequency features such as `q_mean_complex`, `q_abs_mean`, and `freq_scale_sq`.
- Online scoring still reads the real current request K cache.
- The score estimates which historical K/V tokens are likely to be useful for future queries.
- Top-K by score is kept according to `KV_BUDGET`.
- Recent `WINDOW_SIZE` tokens are protected to preserve local generation continuity.

Important hyperparameters:

- `KV_BUDGET`: total retained KV budget after compression. Smaller means more speed/memory benefit but higher accuracy risk.
- `WINDOW_SIZE`: recent token protection window, default around `128`; this protects local continuity and formatting.
- `DIVIDE_LENGTH`: compression/check granularity; often defaults near `128`.
- `SCORE_MAX_LAYERS_ON_ASCEND`: Ascend scoring layer cap; used to reduce PyTorch/torch_npu eager scoring overhead.

Known analysis conclusions:

- Larger `KV_BUDGET` usually improves accuracy but reduces speedup.
- For 32K context, `KV_BUDGET=4096` can work for some needle tests, while `6144/8192` is safer for accuracy-sensitive runs.
- Qwen3.5-27B should not be considered verified until model-specific sparse stats are generated and vLLM-Ascend baseline works.
- Do not blindly reuse Qwen3-32B stats for a different model if accuracy matters.
- TP8 may show lower TriAttention relative gain than TP4 because communication/synchronization/fixed overhead becomes a larger part of TPOT.

## 7. Current Model Support Understanding

Verified or packaged stats in this repo include:

- `triattention/vllm/stats/qwen3_32b_int4_stats.pt`
- `triattention/vllm/stats/gpt_oss_120b_stats.pt`

Current runtime auto-fallback maps generic `qwen` model hints to the Qwen3-32B packaged stats if no explicit stats path is provided. That is convenient but can be unsafe for new models.

For a new model:

1. Confirm vLLM-Ascend baseline inference works without TriAttention.
2. Generate model-specific sparse stats.
3. Set both:
   - `TRIATTN_RUNTIME_MODEL_PATH`
   - `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
4. Validate with long-context correctness tests before performance claims.

## 8. Issue Workflow

There used to be an issue-comment workflow. The automatic monitor was stopped by user request.

When the user says there is a new comment:

1. Use the `gh-issue-comment-monitor` skill if available.
2. Fetch only the latest relevant comment.
3. Ignore stale or accidental issue `#8` feedback if it appears.
4. Analyze before coding if the user says "先不要改代码".
5. After a code fix, commit and summarize the fix to the issue if requested.

## 9. Suggested New Codex Startup Prompt

The incoming colleague can paste this into Codex:

```text
请先读取 AGENTS.md 和 .codex/PROJECT_HANDOFF.md，然后读取 docs/triattention_ascend_development_summary.md、docs/kv_eviction_strategy_toggle_analysis.md、docs/vllm_ascend.md。当前分支应该是 tri_zxj_version0615。请不要修改 codex/vllm-ascend-triattention 主开发分支。后续如修改代码，请反馈修改文件名和绝对路径，并在完成后提交。
```

## 10. Practical Commands For New Agent

```bash
git status --short --branch
git log --oneline --decorate -n 30
git diff --stat main..HEAD
rg -n "TRIATTN_RUNTIME|score_max_layers|kv_budget|window_size|kv_cache_events|effective_base" triattention/vllm/runtime triattention/vllm/core
```

For tests after future code changes:

```bash
python -m pytest triattention/tests
```

For targeted KV eviction bridge work:

```bash
python -m pytest triattention/tests/test_runner_output_bridge.py triattention/tests/test_ascend_input_patch.py
```
