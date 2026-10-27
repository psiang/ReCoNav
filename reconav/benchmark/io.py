"""Validate experiment directory names used by runtime and result storage."""
from __future__ import annotations

import re

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid benchmark run id: {run_id!r}")
    return value
