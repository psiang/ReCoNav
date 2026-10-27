"""Score GC-VLN's recorded physical route with the existing unified evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reconav.baselines.canav.score_trajectory import extract_external_baseline_metrics
from reconav.baselines.common.trajectory_normalization import normalize_evaluator_route
from reconav.benchmark.metrics import evaluate_prediction
from reconav.scene.io import load_asg


def score_recorded_trajectory(task: dict, trajectory: dict, asg: object) -> dict:
    if not trajectory.get("valid_for_scoring", False):
        raise ValueError("trajectory is explicitly ineligible for scoring")
    if trajectory.get("run_kind") == "UNSCORED_RUNTIME_SMOKE_FIXTURE":
        raise ValueError("UNSCORED_RUNTIME_SMOKE_FIXTURE must never enter the evaluator")
    route = normalize_evaluator_route(
        trajectory["route"], prediction_empty=bool(trajectory.get("prediction_empty", False))
    )
    prediction = {"task_id": task["task_id"], "route": route}
    canonical = evaluate_prediction(task, prediction, asg)
    result = extract_external_baseline_metrics(
        task_id=task["task_id"],
        method=trajectory.get('method', 'GC-VLN-GPT5.6-adapted'),
        trajectory=trajectory,
        canonical_metrics=canonical,
    )
    result["path_length_m"] = trajectory.get("path_length_m")
    result["teleport_attempts_blocked"] = trajectory.get("teleport_attempts_blocked")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--asg", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    result = score_recorded_trajectory(task, trajectory, load_asg(args.asg))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
