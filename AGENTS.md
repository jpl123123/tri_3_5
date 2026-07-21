# Codex Handoff Entry

This repository has an active Codex handoff file. Before changing code, read:

- `.codex/PROJECT_HANDOFF.md`
- `docs/triattention_ascend_development_summary.md`
- `docs/kv_eviction_strategy_toggle_analysis.md`
- `docs/vllm_ascend.md`

Current collaboration rules:

- Base working branch is `tri_zxj_version0709_qwen3.5` (user's current working branch, replaces the older `tri_zxj_version0615`). When asked to commit, create a new feature branch based on `tri_zxj_version0709_qwen3.5` (e.g. `tri_zxj_version0709_<feature>`) and push it to `origin`. Do not modify `codex/vllm-ascend-triattention` unless the user explicitly asks.
- When changing code, report every modified file name and absolute path in the final response.
- After fixes, commit the code and summarize the change back to the active GitHub issue if the user is working through issue comments.
- Do not run automatic issue monitoring. Fetch latest issue comments only when the user asks.
- If vllm-ascend source is needed, first check the local `other_code` directory when available.

## opencode git / push configuration (verified working)

- Remote `origin` = `https://gitee.com/xj_zhang2024/triattention-main.git` (HTTPS; SSH port 22 is blocked in this environment).
- Commit identity (local to this repo): `user.name=xj_zhang2024`, `user.email=zhangxiangjun1@huawei.com`.
- Auth to Gitee: HTTPS Personal Access Token via `credential.helper=store`; token stored in `~/.git-credentials` (chmod 600). Do not echo/print the token.
- Push link verified with `git ls-remote origin` (all remote branches listed).
- Workflow when the user asks to commit: `git checkout -b tri_zxj_version0709_<feature>` from the current `tri_zxj_version0709_qwen3.5` tip → `git add` intended files → `git commit` → `git push -u origin <feature-branch>`. Do not commit on `codex/vllm-ascend-triattention`. Never commit secrets/keys.

