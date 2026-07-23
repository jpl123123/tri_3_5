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
- PR 描述文字一律用中文表述(commit message 标题与正文仍可用英文以匹配仓库历史风格,但 PR body / 描述 / 面向人的说明文字用中文).
- PR 创建走 Gitee API v5: `POST https://gitee.com/api/v5/repos/xj_zhang2024/triattention-main/pulls`, `access_token` 取自 `~/.git-credentials`;`base` 默认用 `tri_zxj_version0709_qwen3.5`,不要用 `codex/vllm-ascend-triattention`.

## opencode email configuration (verified working)

- 用户要求发邮件("发邮件/把报告发到邮箱/把结果发给我"等)时,默认使用 `send-email` skill。
- skill 位置:`~/.config/opencode/skills/send-email/`(SKILL.md + scripts/send_email.py)。
- 默认发件邮箱:`xj_zhang2017@163.com`(SMTP `smtp.163.com:465` SSL)。授权码存于 `~/.config/opencode/skills/send-email/.smtp.env`(chmod 600,不回显/不打印)。
- 发送命令:`python3 ~/.config/opencode/skills/send-email/scripts/send_email.py --to 收件人 --subject "主题" --body "正文"`(支持 `--html`/`--attach`/`--cc`,可多次)。
- 不要用 QQ 个人邮箱 `1205975093@qq.com` 发件:机房 IP 会触发 QQ 账号保护,返回 `535 Login fail. Account is abnormal`(实测 163 不触发,可用)。
- 邮件正文用中文(遵循 PR 描述同一规则)。
- 重启 opencode 后 `send-email` 出现在 `available_skills`,可用 `skill` 工具加载;不重启则直接 bash 调 `send_email.py`。

