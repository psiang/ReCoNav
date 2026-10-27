"""Score DualVLN's actual physical route with the shared evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reconav.baselines.canav.score_trajectory import score_recorded_trajectory
from reconav.scene.io import load_asg


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
    result["path_length_m"] = trajectory.get("path_length_m")
    result["movement_primitive_count"] = trajectory.get("movement_primitive_count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

