"""Generate and replay route witnesses for tasks authored against released ASGs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Timer

from reconav import paths


def construct_task(task_path: Path, *, data_root: Path | None = None, solver_seconds: float = 90.0) -> dict:
    data = paths.configure(data_root)
    from reconav.benchmark.start_heading import generate_start_heading, task_start_position
    from reconav.benchmark.metrics import grounded_gold, replayable_route, stored_dsl_witness_report, stored_route_replay_input
    from reconav.benchmark.route_evaluator import evaluate_grounded_route
    from reconav.scene.io import load_asg
    from reconav.scene.navigation import load_or_build_geometry_graph
    from reconav.solver.cancellation import CancellationToken
    from reconav.solver.dsl_parser import parse_dsl, solve_program
    from reconav.solver.query_evidence import MIN_PENALTY

    if solver_seconds <= 0:
        raise ValueError("solver_seconds must be positive")
    task_path = Path(task_path)
    task = json.loads(task_path.read_text())
    task["task_id"] = task_path.stem
    task["start_heading"] = generate_start_heading(task["scene_id"], task_start_position(task))
    scene = data / "benchmark/scenes/hm3dsem" / task["scene_id"]
    asg = load_asg(scene / "asg")
    load_or_build_geometry_graph(data_source="hm3dsem", asg=asg)
    token = CancellationToken()
    timer = Timer(solver_seconds, token.cancel)
    timer.daemon = True
    timer.start()
    try:
        result = solve_program(parse_dsl(task["grounded_dsl"]), asg, request=MIN_PENALTY, cancel_token=token)
    finally:
        timer.cancel()
    if result.status != "SAT":
        raise ValueError(f"{task_path}: {result.status}: {result.status_reason}")
    task["selected_route"] = {
        "waypoints": [[p.x, p.y, p.z] for p in result.waypoints],
        "traversed_region_ids": list(result.traversed_region_ids),
        "portal_ids": list(result.portal_indices),
        "milestone_hits": [
            {"name": h.name, "role": h.role, "target_kind": h.target_kind,
             "resolved_index": h.resolved_index, "waypoint_index": h.waypoint_index,
             "event_seq": h.event_seq}
            for h in result.milestone_hits
        ],
    }
    report = stored_dsl_witness_report(asg, task["grounded_dsl"], task["selected_route"])
    if report is None or not report["ok"]:
        raise ValueError(f"{task_path}: DSL witness replay failed: {report}")
    gold = grounded_gold(task)
    route, errors = replayable_route(stored_route_replay_input(task["selected_route"]), gold, asg, task["robot_position"])
    score = evaluate_grounded_route(asg, gold, route, task["robot_position"])
    if errors or not score["ok"]:
        raise ValueError(f"{task_path}: task annotation replay failed: {errors}, {score}")
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
    return {"task_id": task["task_id"], "status": "passed", "waypoints": len(result.waypoints)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, nargs="+", required=True, help="Authored task JSON files to solve, validate and update")
    parser.add_argument("--data-root", type=Path, help="Data directory containing benchmark/; defaults to RECONAV_DATA_ROOT or ./data")
    parser.add_argument("--solver-seconds", type=float, default=90.0)
    args = parser.parse_args(argv)
    for task in args.task:
        print(json.dumps(construct_task(task, data_root=args.data_root, solver_seconds=args.solver_seconds)), flush=True)


if __name__ == "__main__":
    main()
