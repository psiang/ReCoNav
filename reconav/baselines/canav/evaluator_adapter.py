"""Thin output/evaluation adapter around the official CA-Nav evaluator."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from reconav.baselines.canav.compat import repair_legacy_gelu_modules
from reconav.baselines.common.trajectory_normalization import (
    POSITION_EQUALITY_EPSILON_M,
    normalize_consecutive_duplicate_positions,
)
from vlnce_baselines.ZS_Evaluator_mp import ZeroShotVlnEvaluatorMP


def _plain_action(action: Any) -> int | str | None:
    value = action.get("action") if isinstance(action, dict) else action
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


class ActualPoseRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.termination = "not_terminated"

    def observe(self, observation: dict[str, Any], action: Any, *, done: bool) -> None:
        position = np.asarray(observation["position"], dtype=float).tolist()
        heading_raw = np.asarray(observation.get("heading", [np.nan]), dtype=float).reshape(-1)
        action_value = _plain_action(action)
        self.records.append(
            {
                "step": len(self.records),
                "action": action_value,
                "position": {"x": position[0], "y": position[1], "z": position[2]},
                "heading": float(heading_raw[0]) if heading_raw.size else None,
                "done": bool(done),
            }
        )
        if done:
            self.termination = "stop" if action_value == 0 else "native_max_steps"

    def route(self) -> dict[str, Any]:
        raw = self.raw_route()["waypoints"]
        normalized = normalize_consecutive_duplicate_positions(raw)
        moved = any(
            np.linalg.norm(
                np.asarray(tuple(first.values()), dtype=float)
                - np.asarray(tuple(second.values()), dtype=float)
            ) > POSITION_EQUALITY_EPSILON_M
            for first, second in zip(raw, raw[1:])
        )
        return {"waypoints": normalized if moved else []}

    def raw_route(self) -> dict[str, Any]:
        return {"waypoints": [dict(record["position"]) for record in self.records]}


class RecordingVectorEnv:
    def __init__(self, wrapped: Any, recorder: ActualPoseRecorder) -> None:
        self._wrapped = wrapped
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def reset(self) -> Any:
        observations = self._wrapped.reset()
        expected = np.asarray(self._wrapped.current_episodes()[0].start_position, dtype=float)
        actual = np.asarray(observations[0]['position'], dtype=float)
        if np.linalg.norm(expected-actual) > 1e-5:
            raise RuntimeError('CA-Nav changed the exact benchmark start pose')
        self._recorder.observe(observations[0], "RESET", done=False)
        return observations

    def step(self, actions: Any) -> Any:
        outputs = self._wrapped.step(actions)
        first_action = actions[0] if isinstance(actions, (list, tuple)) else actions
        observation, _, done, _ = outputs[0]
        self._recorder.observe(observation, first_action, done=bool(done))
        return outputs


class CANavHM3DSingleEvaluator(ZeroShotVlnEvaluatorMP):
    """Preserve CA-Nav policy logic while replacing R2R scoring/output hooks."""

    def __init__(
        self,
        config: Any,
        *,
        task_path: Path,
        output_dir: Path,
        method: str,
        run_kind: str,
        valid_for_scoring: bool,
        habitat_sensor_spec_compat: bool,
    ) -> None:
        super().__init__(config)
        self._benchmark_task_path = Path(task_path)
        self._output_dir = Path(output_dir)
        self._method = method
        self._run_kind = run_kind
        self._valid_for_scoring = bool(valid_for_scoring)
        self._habitat_sensor_spec_compat = bool(habitat_sensor_spec_compat)
        self._recorder = ActualPoseRecorder()
        self._legacy_gelu_repairs: dict[str, int] = {}

    def _init_envs(self) -> None:
        super()._init_envs()
        self.envs = RecordingVectorEnv(self.envs, self._recorder)

    def eval(self) -> None:
        # The frozen evaluator calls time.time() but omits ``import time``.
        # Supply that missing module global externally without changing rollout logic.
        sys.modules[ZeroShotVlnEvaluatorMP.__module__].time = time
        super().eval()

    def _collect_val_traj(self) -> None:
        """R2R GT path is intentionally unavailable to the policy process."""

    def _initialize_policy(self) -> None:
        super()._initialize_policy()
        self._legacy_gelu_repairs = {
            "blip2": repair_legacy_gelu_modules(self.value_map_module.model),
            "vqa": repair_legacy_gelu_modules(self.constraints_monitor.model),
        }

    def _calculate_metric(self, infos: list[Any]) -> None:
        del infos
        task = json.loads(self._benchmark_task_path.read_text(encoding="utf-8"))
        route = self._recorder.route()
        raw_route = self._recorder.raw_route()
        trajectory = {
            "task_id": task["task_id"],
            "method": self._method,
            "run_kind": self._run_kind,
            "valid_for_scoring": self._valid_for_scoring,
            "coordinate_frame": "Habitat world xyz",
            "trajectory_source": "actual simulator pose after every primitive action",
            "termination": self._recorder.termination,
            "executed_actions": max(len(self._recorder.records) - 1, 0),
            "recorded_poses": len(self._recorder.records),
            "route": route,
            "raw_route": raw_route,
            "prediction_empty": not bool(route["waypoints"]),
            "action_trace": self._recorder.records,
        }
        trajectory["compatibility"] = {
            "legacy_gelu_fields_repaired": self._legacy_gelu_repairs,
            "habitat_lab_017_sensor_spec_bridge": self._habitat_sensor_spec_compat,
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "trajectory.json").write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        episode_id = self.envs.current_episodes()[0].episode_id
        self.state_eps[episode_id] = {
            "trajectory": str(self._output_dir / "trajectory.json"),
            "executed_actions": trajectory["executed_actions"],
            "termination": trajectory["termination"],
        }
