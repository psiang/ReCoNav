"""Released DualVLN configuration and model-facing input firewall."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reconav.baselines.canav.export_task import PolicyInput, heading_to_habitat_xyzw


@dataclass(frozen=True)
class SanitizedObservation:
    """The exact goal-free fields passed to the released InternNav agent."""

    instruction: str
    rgb: np.ndarray
    depth: np.ndarray

    def as_upstream_input(self) -> list[dict[str, Any]]:
        return [{"instruction": self.instruction, "rgb": self.rgb, "depth": self.depth}]


def released_dualvln_config(model_path: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    return {
        "model_path": str(model_path.resolve()),
        "policy_name": "InternVLAN1_Policy",
        "state_encoder": None,
        "checkpoint_variant": "InternVLA-N1-DualVLN",
        "checkpoint_system1": "nextdit_async",
        "width": 640,
        "height": 480,
        "hfov": 79,
        "resize_w": 384,
        "resize_h": 384,
        "num_frames": 32,
        "num_history": 8,
        "num_future_steps": 4,
        "predict_step_nums": 32,
        "continuous_traj": True,
        "infer_mode": "partial_async",
        "sys2_max_forward_step": 8,
        "device": device,
        "vis_debug": False,
        "vis_debug_path": "unused",
        "camera": {
            "resolution": [640, 480],
            "hfov": 79,
            "height_m": 1.25,
            "depth_min_m": 0.0,
            "depth_max_m": 10.0,
            "render_near_m": 0.01,
        },
        "actions": {
            "forward_step_m": 0.25,
            "turn_angle_deg": 15.0,
            "tilt_angle_deg": 15.0,
        },
    }


def write_policy_manifest(
    policy: PolicyInput,
    *,
    scene_path: Path,
    model_path: Path,
    output: Path,
) -> dict[str, Any]:
    adapter_input = asdict(policy)
    manifest = {
        "adapter_input": adapter_input,
        "scene_path": str(scene_path.resolve()),
        "start_rotation_xyzw": heading_to_habitat_xyzw(policy.start_heading),
        "model_facing_policy_object": ["instruction", "rgb", "depth"],
        "native_policy_inputs": [
            "natural-language instruction",
            "current RGB and the policy's native internally sampled RGB history",
            "native look-down RGB when requested",
            "normalized native depth consumed only by released nextdit_async System 1",
        ],
        "model_facing_absolute_pose": False,
        "model_facing_gps": False,
        "forbidden_benchmark_fields_exported": [],
        "explicitly_absent": [
            "GT goal coordinate",
            "GT target object ID",
            "GT semantic segmentation",
            "GT room/object annotations",
            "GT ASG",
            "GT grounding",
            "GT contract",
            "reference trajectory",
            "shortest path",
            "oracle waypoint",
            "oracle progress",
        ],
        "released_checkpoint": str(model_path.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
