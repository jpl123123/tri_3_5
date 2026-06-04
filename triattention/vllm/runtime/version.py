"""Runtime build fingerprint for deployment/log verification."""

from __future__ import annotations

from pathlib import Path

RUNTIME_BUILD_ID = "remove-initial-decode-bypass-v25-20260604"


def runtime_build_info() -> str:
    return f"{RUNTIME_BUILD_ID} source={Path(__file__).resolve()}"
