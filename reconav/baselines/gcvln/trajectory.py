"""Canonical actual-primitive trajectory recorder shared with external scoring."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from reconav.baselines.common.trajectory_normalization import (
    POSITION_EQUALITY_EPSILON_M,
    normalize_consecutive_duplicate_positions,
)


class ActualPrimitiveTrajectory:
    def __init__(
        self,
        output: Path,
        task_id: str,
        *,
        run_kind: str = "adapted_official_prompt_deterministic_lowering",
        valid_for_scoring: bool = True,
    ) -> None:
        self.output = Path(output)
        self.task_id = task_id
        self.run_kind = run_kind
        self._eligible_for_scoring = bool(valid_for_scoring)
        self.records: list[dict[str, Any]] = []
        self.teleport_attempts = 0
        self.termination = "not_terminated"
        self.path_length_m = 0.0
        self.recording_seconds = 0.0
        self.snapshot_count = 0
        self._journal = None
        self._dirty = False

    def reset(self, position: list[float], rotation: list[float], heading: float) -> None:
        started = time.perf_counter()
        if self._journal is not None:
            self._journal.close()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._journal = self.output.with_suffix('.jsonl').open('w', encoding='utf-8')
        self.records = []
        self.teleport_attempts = 0
        self.termination = "not_terminated"
        self.path_length_m = 0.0
        self.recording_seconds = time.perf_counter() - started
        self.snapshot_count = 0
        self._append("RESET", position, rotation, heading, False)

    def primitive(
        self,
        action: int | str,
        position: list[float],
        rotation: list[float],
        heading: float,
        *,
        done: bool,
    ) -> None:
        self._append(action, position, rotation, heading, done)
        if done:
            self.termination = "native_stop" if action == 0 else "native_done"
            self.write()

    def teleport_blocked(self) -> None:
        self.teleport_attempts += 1
        self.termination = "invalid_teleport_attempt_blocked"
        self._dirty = True
        self.write()

    def _append(
        self,
        action: int | str,
        position: list[float],
        rotation: list[float],
        heading: float,
        done: bool,
    ) -> None:
        started = time.perf_counter()
        if self.records:
            previous = self.records[-1]['position']
            self.path_length_m += math.dist((previous['x'], previous['y'], previous['z']), position)
        self.records.append(
            {
                "step": len(self.records),
                "action": action,
                "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
                "rotation_xyzw": [float(value) for value in rotation],
                "heading": float(heading),
                "done": bool(done),
            }
        )
        self._journal.write(json.dumps(self.records[-1], ensure_ascii=False, separators=(',', ':')) + '\n')
        if len(self.records) % 16 == 0:
            self._journal.flush()
        self._dirty = True
        self.recording_seconds += time.perf_counter() - started

    def payload(self) -> dict[str, Any]:
        waypoints = [dict(record["position"]) for record in self.records]
        path_length = self.path_length_m
        actions = max(len(self.records) - 1, 0)
        moved = path_length > POSITION_EQUALITY_EPSILON_M
        evaluator_waypoints = normalize_consecutive_duplicate_positions(waypoints) if moved else []
        return {
            "task_id": self.task_id,
            "method": "GC-VLN" if os.environ.get('GCVLN_BENCHMARK_MODEL') else "GC-VLN-GPT5.6-adapted",
            "language_model": os.environ.get('GCVLN_BENCHMARK_MODEL'),
            "run_kind": self.run_kind,
            "valid_for_scoring": self._eligible_for_scoring and self.teleport_attempts == 0,
            "coordinate_frame": "Habitat world xyz",
            "trajectory_source": "actual simulator pose after every primitive Habitat action",
            "termination": self.termination,
            "executed_actions": actions,
            "recorded_poses": len(self.records),
            "pose_action_invariant": len(self.records) == actions + 1,
            "path_length_m": path_length,
            "teleflag": False,
            "teleport_attempts_blocked": self.teleport_attempts,
            "prediction_empty": not moved,
            "route": {"waypoints": evaluator_waypoints},
            "raw_route": {"waypoints": waypoints},
            "action_trace": self.records,
        }

    def write(self) -> None:
        if not self._dirty:
            return
        started = time.perf_counter()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.output)
        self._journal.flush()
        self.snapshot_count += 1
        self._dirty = False
        self.recording_seconds += time.perf_counter() - started

    def close(self) -> None:
        if self._journal is None:
            return
        self.write()
        started = time.perf_counter()
        self._journal.close()
        self._journal = None
        self.recording_seconds += time.perf_counter() - started
        self.output.with_name('trajectory_recording.json').write_text(json.dumps({
            'format': 'buffered-jsonl-final-json-v1',
            'seconds': self.recording_seconds,
            'snapshot_count': self.snapshot_count,
            'recorded_poses': len(self.records),
            'journal_flush_interval': 16,
        }, indent=2) + '\n', encoding='utf-8')
