from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def trajectory_path_length(trajectory: dict[str, Any]) -> float:
    """Recompute path length from authoritative recorded simulator positions."""
    candidates = (trajectory.get('raw_route', {}).get('waypoints'), [record.get('position') for record in trajectory.get('action_trace', [])], trajectory.get('route', {}).get('waypoints'))
    waypoints = next((items for items in candidates if items), [])
    positions = [(float(point['x']), float(point['y']), float(point['z'])) for point in waypoints if isinstance(point, dict) and all((axis in point for axis in ('x', 'y', 'z')))]
    return sum((math.dist(first, second) for first, second in zip(positions, positions[1:])))

def write_zero_motion_trajectory(cell: Path, task: dict[str, Any], baseline: str, reason: str) -> None:
    start = task['robot_position']
    point = {'x': float(start['x']), 'y': float(start['floor_height']), 'z': float(start['y'])}
    write_json(cell / 'trajectory.json', {'task_id': task['task_id'], 'method': baseline, 'run_kind': 'failed_attempt_zero_motion', 'valid_for_scoring': True, 'coordinate_frame': 'Habitat world xyz', 'trajectory_source': 'configured reset pose; failure occurred before a simulator action', 'termination': reason, 'executed_actions': 0, 'recorded_poses': 1, 'pose_action_invariant': True, 'path_length_m': 0.0, 'prediction_empty': True, 'route': {'waypoints': []}, 'raw_route': {'waypoints': [point]}, 'action_trace': [{'step': 0, 'action': 'RESET', 'position': point}], 'runtime_exception': reason})
