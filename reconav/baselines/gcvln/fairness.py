"""Validate GC-VLN motion against the step and trajectory limits."""

from __future__ import annotations

from typing import Any


class TeleportGuard:
    def __init__(self, recorder: Any) -> None:
        self.recorder = recorder

    def __call__(self, pos: Any, rotation: Any) -> None:
        del pos, rotation
        self.recorder.teleport_blocked()
        raise RuntimeError("teleport/recovery is physically disabled for scored GC-VLN")
