"""Integration-side visibility for InternNav System 2 inference failures."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class S2InferenceInstrumentation:
    """Observe InternNav's existing initial-attempt/retry/fallback behavior."""

    def __init__(self, output: Path | None = None) -> None:
        self.output = Path(output) if output is not None else None
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._retry_pending = False
        self._inference_index = 0
        self._fallback_stops_pending = 0
        self._genuine_stops_pending = 0

    def instrument_policy(self, policy: Any) -> None:
        original = policy.s2_step

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self.call(original, *args, **kwargs)

        policy.s2_step = wrapped

    def call(self, s2_step: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            is_retry = self._retry_pending
            if not is_retry:
                self._inference_index += 1
            inference_index = self._inference_index
            event = {
                "inference_index": inference_index,
                "attempt_index": 2 if is_retry else 1,
                "success": False,
                "error_type": None,
                "stop_origin": None,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.events.append(event)
            self._write_locked()
        try:
            output = s2_step(*args, **kwargs)
        except Exception as exc:
            with self._lock:
                event["error_type"] = type(exc).__name__
                if is_retry:
                    event["stop_origin"] = "s2_failure_fallback"
                    self._retry_pending = False
                    self._fallback_stops_pending += 1
                else:
                    self._retry_pending = True
                self._write_locked()
            raise

        output_actions = getattr(output, "output_action", None)
        with self._lock:
            event["success"] = True
            self._retry_pending = False
            if isinstance(output_actions, (list, tuple)):
                stop_count = sum(int(action) == 0 for action in output_actions)
                self._genuine_stops_pending += stop_count
                if stop_count:
                    event["stop_origin"] = "genuine_model_stop"
            self._write_locked()
        return output

    def classify_returned_action(self, action_code: int) -> str | None:
        if int(action_code) != 0:
            return None
        with self._lock:
            if self._fallback_stops_pending:
                self._fallback_stops_pending -= 1
                origin = "s2_failure_fallback"
            else:
                if self._genuine_stops_pending:
                    self._genuine_stops_pending -= 1
                origin = "genuine_model_stop"
            self._write_locked()
            return origin

    def counters(self) -> dict[str, int]:
        return {
            "s2_inference_calls": sum(event["attempt_index"] == 1 for event in self.events),
            "s2_exception_count": sum(event["error_type"] is not None for event in self.events),
            "s2_retry_count": sum(event["attempt_index"] == 2 for event in self.events),
            "s2_second_failure_count": sum(
                event["attempt_index"] == 2 and event["error_type"] is not None
                for event in self.events
            ),
            "s2_fallback_stop_count": sum(
                event["stop_origin"] == "s2_failure_fallback" for event in self.events
            ),
        }

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return {"counters": self.counters(), "events": [dict(event) for event in self.events]}

    def write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        if self.output is None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"counters": self.counters(), "events": self.events}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.output)
