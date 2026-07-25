# Codex Handoff Entry

This repository has an active Codex handoff file. Before changing code, read:

- `.codex/PROJECT_HANDOFF.md`
- `docs/triattention_ascend_development_summary.md`
- `docs/kv_eviction_strategy_toggle_analysis.md`
- `docs/vllm_ascend.md`

Current collaboration rules:

- Work on branch `tri_zxj_version0615`; do not modify `codex/vllm-ascend-triattention` unless the user explicitly asks.
- When changing code, report every modified file name and absolute path in the final response.
- After fixes, commit the code and summarize the change back to the active GitHub issue if the user is working through issue comments.
- Do not run automatic issue monitoring. Fetch latest issue comments only when the user asks.
- If vllm-ascend source is needed, first check the local `other_code` directory when available.

