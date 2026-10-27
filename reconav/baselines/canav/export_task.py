"""Export the legal benchmark inputs as one CA-Nav VLN-CE episode."""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reconav.benchmark.start_heading import validate_start_heading


CA_NAV_EPISODE_ID = 1


@dataclass(frozen=True)
class PolicyInput:
    benchmark_task_id: str
    scene_id: str
    instruction: str
    start_position: tuple[float, float, float]
    start_heading: float


def load_policy_input(task_path: Path) -> PolicyInput:
    """Read only policy-visible fields from a benchmark task."""

    task = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError("benchmark task must be a JSON object")
    heading = validate_start_heading(task)
    robot = task["robot_position"]
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task has no natural-language instruction")
    return PolicyInput(
        benchmark_task_id=str(task["task_id"]),
        scene_id=str(task["scene_id"]),
        instruction=instruction,
        start_position=(
            float(robot["x"]),
            float(robot["floor_height"]),
            float(robot["y"]),
        ),
        start_heading=heading,
    )


def heading_to_habitat_xyzw(heading: float) -> list[float]:
    """Convert Habitat/world yaw radians to Habitat dataset quaternion xyzw."""

    value = float(heading)
    if not math.isfinite(value) or not 0.0 <= value < math.tau:
        raise ValueError("start_heading must be finite and in [0, 2*pi)")
    return [0.0, math.sin(value / 2.0), 0.0, math.cos(value / 2.0)]


def export_episode(
    policy_input: PolicyInput,
    *,
    scene_path: Path,
    scene_root: Path,
    dataset_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    scene_path = scene_path.resolve()
    scene_root = scene_root.resolve()
    try:
        relative_scene = scene_path.relative_to(scene_root).as_posix()
    except ValueError as exc:
        raise ValueError("scene_path must be below scene_root") from exc
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    rotation = heading_to_habitat_xyzw(policy_input.start_heading)
    tokens = re.findall(r"[A-Za-z0-9']+", policy_input.instruction.lower())
    dataset = {
        "instruction_vocab": {"word_list": sorted(set(tokens))},
        "episodes": [
            {
                "episode_id": CA_NAV_EPISODE_ID,
                "trajectory_id": CA_NAV_EPISODE_ID,
                "scene_id": relative_scene,
                "start_position": list(policy_input.start_position),
                "start_rotation": rotation,
                "goals": [],
                "reference_path": [],
                "instruction": {
                    "instruction_text": policy_input.instruction,
                    "instruction_tokens": [],
                },
            }
        ],
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dataset_path, "wt", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False)

    manifest = {
        "policy_input": asdict(policy_input),
        "ca_nav_episode_id": CA_NAV_EPISODE_ID,
        "scene_path": str(scene_path),
        "scene_root": str(scene_root),
        "start_rotation_xyzw": rotation,
        "goals": {
            "value": [],
            "reason": "goal-free episode; goal-dependent CA-Nav/R2R measures are disabled",
        },
        "runtime_observations": ["online RGB", "online depth", "native relative pose"],
        "forbidden_benchmark_fields_exported": [],
        "explicitly_absent": [
            "GT goal coordinate",
            "GT route",
            "shortest path",
            "GT ASG",
            "GT contract",
            "GT grounding",
            "semantic GT segmentation",
            "oracle viewpoint",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
