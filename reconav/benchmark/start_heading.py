"""Canonical, task-independent benchmark start-heading validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


HEADING_RANGE = "[0, 2*pi)"
POSITION_DECIMALS = 6


def generate_start_heading(scene_id: str, start_position: Sequence[float]) -> float:
    """Return a stable Habitat/world yaw determined only by scene and position.

    ``start_position`` is ordered as world ``[x, y, z]``. Coordinates are
    serialized at micrometre precision so the hash contract is stable across
    processes and independent of Python's randomized builtin ``hash``.
    """

    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a non-empty string")
    if len(start_position) != 3:
        raise ValueError("start_position must contain exactly [x, y, z]")
    coordinates = tuple(float(value) for value in start_position)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("start_position coordinates must be finite")
    rounded = [round(value, POSITION_DECIMALS) for value in coordinates]
    rounded = [0.0 if value == 0.0 else value for value in rounded]
    quantized = [f"{value:.{POSITION_DECIMALS}f}" for value in rounded]
    key = json.dumps([scene_id, quantized], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    unit_interval = int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64
    return math.tau * unit_interval


def task_start_position(task: Mapping[str, Any]) -> tuple[float, float, float]:
    """Read the benchmark robot position in Habitat/world ``x, y, z`` order."""

    robot = task.get("robot_position")
    if not isinstance(robot, Mapping):
        raise ValueError("task has no robot_position mapping")
    try:
        return float(robot["x"]), float(robot["floor_height"]), float(robot["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("robot_position must contain numeric x, floor_height, and y") from exc




def validate_start_heading(task: Mapping[str, Any]) -> float:
    """Validate the canonical heading stored on a generated task."""

    actual = task.get("start_heading")
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError("task has no numeric start_heading")
    heading = float(actual)
    if not math.isfinite(heading) or not 0.0 <= heading < math.tau:
        raise ValueError(f"start_heading must be finite and in {HEADING_RANGE}")
    expected = generate_start_heading(task.get("scene_id"), task_start_position(task))
    if heading != expected:
        raise ValueError("start_heading is not canonical for scene_id + start_position")
    return heading
