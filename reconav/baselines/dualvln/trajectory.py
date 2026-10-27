"""Record only DualVLN actions actually executed by Habitat-Sim."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from reconav.baselines.common.trajectory_normalization import normalize_consecutive_duplicate_positions


def _position(state: Any) -> dict[str, float]:
    p = state.position
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _rotation(state: Any) -> list[float]:
    q = state.rotation
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


class DualVLNTrajectory:
    def __init__(self, output: Path, task_id: str) -> None:
        self.output = Path(output)
        self.task_id = task_id
        self.action_trace: list[dict[str, Any]] = []
        self.route_positions: list[dict[str, float]] = []
        self.path_length_m = 0.0
        self.moved_primitive_count = 0
        self.termination = "not_terminated"
        self.runtime_exception: dict[str, str] | None = None

    def reset(self, state: Any) -> None:
        pose = _position(state)
        self.route_positions = [pose]
        self.action_trace = [{
            "step": 0,
            "action": "RESET",
            "position": pose,
            "rotation_xyzw": _rotation(state),
            "moved_m": 0.0,
        }]
        self.write()

    def action(self, action: str, before: Any, after: Any, *, native_stop: bool = False) -> None:
        first = tuple(_position(before).values())
        second_pose = _position(after)
        second = tuple(second_pose.values())
        distance = math.dist(first, second)
        self.path_length_m += distance
        if distance > 1e-9:
            self.moved_primitive_count += 1
        self.action_trace.append({
            "step": len(self.action_trace),
            "action": action,
            "position": second_pose,
            "rotation_xyzw": _rotation(after),
            "moved_m": distance,
            "native_stop": bool(native_stop),
        })
        self.route_positions.append(second_pose)
        if native_stop:
            self.termination = "native_stop"
        self.write()

    def fail(self, exc: BaseException) -> None:
        self.termination = "runtime_exception"
        self.runtime_exception = {"type": type(exc).__name__, "message": str(exc)}
        self.write()

    def finish_max_steps(self) -> None:
        self.termination = "max_primitive_actions"
        self.write()

    def payload(self) -> dict[str, Any]:
        moved = self.moved_primitive_count > 0
        executed = max(len(self.action_trace) - 1, 0)
        raw_route = [dict(position) for position in self.route_positions]
        evaluator_route = normalize_consecutive_duplicate_positions(raw_route) if moved else []
        return {
            "task_id": self.task_id,
            "method": "DualVLN",
            "run_kind": "released_internnav_dualvln_goal_free_hm3d_adapter",
            "valid_for_scoring": True,
            "coordinate_frame": "Habitat world xyz",
            "trajectory_source": "actual Habitat-Sim primitive action states only",
            "termination": self.termination,
            "executed_actions": executed,
            "movement_primitive_count": self.moved_primitive_count,
            "recorded_poses": len(self.action_trace),
            "pose_action_invariant": len(self.action_trace) == executed + 1,
            "path_length_m": self.path_length_m,
            "prediction_empty": not moved,
            "route": {"waypoints": evaluator_route},
            "raw_route": {"waypoints": raw_route},
            "action_trace": self.action_trace,
            "runtime_exception": self.runtime_exception,
        }

    def write(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.output)
