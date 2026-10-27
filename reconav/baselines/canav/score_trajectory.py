"""Score a recorded CA-Nav trajectory with the project's unified evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reconav.benchmark.metrics import evaluate_prediction
from reconav.baselines.common.trajectory_normalization import normalize_evaluator_route
from reconav.scene.io import load_asg


def extract_external_baseline_metrics(
    *,
    task_id: str,
    method: str,
    trajectory: dict,
    canonical_metrics: dict,
) -> dict:
    """Return only canonical route/system metrics applicable to an external baseline."""

    diagnostics = canonical_metrics.get("diagnostics", {})
    result = {
        "task_id": task_id,
        "method": method,
        "route_found": canonical_metrics.get("route_found"),
        "route_replay_valid": diagnostics.get("route_replay_valid"),
        "route_errors": diagnostics.get("route_errors", []),
        "hard_ns": canonical_metrics.get("hard_ns"),
        "preference_optimality": canonical_metrics.get("preference_optimality"),
        "full_is": canonical_metrics.get("full_is"),
        "hard_components": canonical_metrics.get("hard_components"),
        "failure_category": diagnostics.get("failure_category"),
        "termination": trajectory.get("termination"),
        "executed_actions": trajectory.get("executed_actions"),
        "recorded_poses": trajectory.get("recorded_poses"),
    }
    if "preference_satisfaction" in canonical_metrics:
        result.update({key: canonical_metrics.get(key) for key in (
            "preference_satisfaction", "preference_violations", "preference_satisfied", "preference_total")})
        result["prefer"] = canonical_metrics["preference_satisfaction"]
        result["preference_task"] = bool(diagnostics.get("preference_task", False))
    return result


def score_recorded_trajectory(task: dict, trajectory: dict, asg: object) -> dict:
    """Score a saved trajectory canonically, then expose the external-baseline view."""

    route = normalize_evaluator_route(
        trajectory["route"], prediction_empty=bool(trajectory.get("prediction_empty", False))
    )
    prediction = {"task_id": task["task_id"], "route": route}
    canonical_metrics = evaluate_prediction(task, prediction, asg)
    return extract_external_baseline_metrics(
        task_id=task["task_id"],
        method=str(trajectory.get("method", "CA-Nav")),
        trajectory=trajectory,
        canonical_metrics=canonical_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--asg", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    output = score_recorded_trajectory(task, trajectory, load_asg(args.asg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
