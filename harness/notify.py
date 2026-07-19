"""Fail-loud plumbing: guard kills must reach the human and leave an
in-band artifact the viewer renders — a log banner alone is silent
(housekeeping ops rule, 2026-07-11)."""

import time
from pathlib import Path

import httpx

ENV = Path.home() / "code" / "housekeeping" / ".env"


def _env(key):
    try:
        for line in ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def telegram(text):
    tok, uid = _env("TELEGRAM_BOT_TOKEN"), _env("TELEGRAM_USER_ID")
    if not (tok and uid):
        print(f"[notify] no telegram credentials; NOT delivered: {text}")
        return False
    try:
        httpx.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": uid, "text": text},
            timeout=30,
        ).raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 — never let notification kill the abort path
        print(f"[notify] telegram send failed ({e}); message was: {text}")
        return False


def abort_run(run_dir, log, reason):
    """Record the abort in-band (event + ABORTED.md) and Telegram the human.

    Caller is responsible for exiting non-zero afterwards.
    """
    log("aborted", {"reason": reason})
    (Path(run_dir) / "ABORTED.md").write_text(
        f"# RUN ABORTED\n\n{time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n{reason}\n"
    )
    telegram(f"schema-qwen run ABORTED ({Path(run_dir).name}): {reason}")
