"""Run-local cloud quota circuit breaker, shared by spawned task workers."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

STOP_PATH_ENV = "BENCHMARK_PROVIDER_STOP_PATH"


class ProviderRunStopped(RuntimeError):
    """A cloud account needs attention; preserve the run and resume later."""


def quota_reason(code, detail: str) -> str | None:
    """Separate exhausted credit/daily quotas from ordinary RPM throttling."""
    text = str(detail).lower()
    try:
        code = int(code)
    except (ValueError, TypeError):
        match = re.search(r"\bhttp\s+(\d{3})\b", text)
        code = int(match[1]) if match else None
    if code == 402:
        return "payment_required"
    if any(token in text for token in (
        "insufficient_quota", "billing_hard_limit", "credit_balance_too_low",
        "insufficient balance", "insufficient credit", "balance exhausted",
        "credit balance", "credits exhausted", "credits have been exhausted",
        "prepaid balance", "余额不足", "余额已耗尽", "欠费",
    )):
        return "credit_exhausted"
    if code == 429 and any(token in text for token in ("perday", "per_day", "daily")):
        return "daily_quota_exhausted"
    return None


def stop_record():
    path = os.environ.get(STOP_PATH_ENV)
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def check_provider_stop(provider=None):
    record = stop_record()
    if record and (provider is None or record["provider"] == provider):
        raise ProviderRunStopped(
            f"{record['provider']} {record['reason']}; restore quota and resume the same run"
        )


def stop_on_quota(provider, error=None, *, code=None, detail=None):
    if provider not in {"openai", "gemini"}:
        return
    if isinstance(error, ProviderRunStopped):
        raise error
    if error is not None:
        code = getattr(error, "code", getattr(error, "status_code", code))
        detail = str(error) if detail is None else detail
    reason = quota_reason(code, detail or "")
    if reason is None:
        return
    path = os.environ.get(STOP_PATH_ENV)
    if path:
        path = Path(path)
        # Store only classified metadata, never server bodies or credentials.
        record = {"provider": provider, "reason": reason,
                  "detected_utc": datetime.now(timezone.utc).isoformat()}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, path)
    raise ProviderRunStopped(f"{provider} {reason}; restore quota and resume the same run")


@contextmanager
def provider_run_guard(output: Path, *, resume: bool):
    """Enter only after acquiring the experiment's exclusive writer lock."""
    path = output / "provider_stop.json"
    if path.exists():
        if not resume:
            raise ProviderRunStopped("Existing quota stop requires --resume")
        history = output / "provider_interruptions"
        history.mkdir(exist_ok=True)
        path.replace(history / f"{uuid4().hex}.json")
    previous = os.environ.get(STOP_PATH_ENV)
    os.environ[STOP_PATH_ENV] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(STOP_PATH_ENV, None)
        else:
            os.environ[STOP_PATH_ENV] = previous
