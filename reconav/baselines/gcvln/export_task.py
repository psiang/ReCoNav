"""Export only policy-legal fields as one goal-free GC-VLN episode."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reconav.baselines.canav.export_task import PolicyInput, heading_to_habitat_xyzw


EPISODE_ID = 1


def write_preflight_manifest(
    policy_input: PolicyInput,
    *,
    scene_path: Path,
    manifest_path: Path,
    preprocessing_status: str,
) -> dict[str, Any]:
    manifest = {
        "policy_input": asdict(policy_input),
        "scene_path": str(scene_path.resolve()),
        "method_owned_preprocessing": {"status": preprocessing_status, "runtime_dag": None},
        "runtime_observations": ["online RGB panorama", "online depth panorama", "GPS", "compass"],
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


def export_episode(
    policy_input: PolicyInput,
    *,
    scene_path: Path,
    scene_root: Path,
    runtime_dag_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    llm_model: str = "gpt-5.6-sol",
) -> dict[str, Any]:
    scene_path = scene_path.resolve()
    scene_root = scene_root.resolve()
    runtime_dag_path = runtime_dag_path.resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not runtime_dag_path.is_file():
        raise FileNotFoundError(runtime_dag_path)
    relative_scene = scene_path.relative_to(scene_root).as_posix()
    runtime_dag = json.loads(runtime_dag_path.read_text(encoding="utf-8"))
    runtime_dag_string = json.dumps(runtime_dag, ensure_ascii=False, separators=(",", ":"))
    rotation = heading_to_habitat_xyzw(policy_input.start_heading)
    tokens = re.findall(r"[A-Za-z0-9']+", policy_input.instruction.lower())
    episode = {
        "episode_id": EPISODE_ID,
        "trajectory_id": EPISODE_ID,
        "scene_id": relative_scene,
        "start_position": list(policy_input.start_position),
        "start_rotation": rotation,
        "instruction": {
            "instruction_text": policy_input.instruction,
            # The released visualization reads text_English even for R2R.
            # Leaving it None triggers a swallowed exception -> fake STOP.
            "instruction_text_English": policy_input.instruction,
            "instruction_tokens": [],
            "instruction_DAG": runtime_dag_string,
            "LLM_model": llm_model,
        },
    }
    dataset = {
        "instruction_vocab": {"word_list": sorted(set(tokens))},
        "episodes": [episode],
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dataset_path, "wt", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False)
    manifest = {
        "policy_input": asdict(policy_input),
        "gcvln_episode_id": EPISODE_ID,
        "scene_path": str(scene_path),
        "scene_root": str(scene_root),
        "start_rotation_xyzw": rotation,
        "method_owned_preprocessing": {
            "runtime_dag_path": str(runtime_dag_path),
            "runtime_dag_sha256": hashlib.sha256(runtime_dag_path.read_bytes()).hexdigest(),
        },
        "runtime_observations": ["online RGB panorama", "online depth panorama", "GPS", "compass"],
        "forbidden_benchmark_fields_exported": [],
        "explicitly_absent": [
            "goals",
            "reference_path",
            "shortest_path",
            "GT ASG",
            "GT contract",
            "GT grounding",
            "GT semantic segmentation",
            "oracle viewpoint",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
