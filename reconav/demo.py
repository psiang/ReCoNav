"""Small self-contained scene demonstrating real solving and route scoring."""
from __future__ import annotations

import json
from pathlib import Path
import time


def smoke(output: Path) -> dict:
    import numpy as np
    import trimesh
    from reconav.scene.asg import ASG, ASGLevelEntity, ASGRegionEntity, ASGObjectEntity
    from reconav.scene.geometry import ConvexPolygon2, Point2
    from reconav.scene.io import save_asg
    from reconav.scene.navigation import load_or_build_geometry_graph
    from reconav.solver.backend import solve_dsl_route
    from reconav.solver.execution_profiles import ExecutionProfile
    from reconav.solver.query_evidence import FEASIBLE
    import os

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    polygon = lambda x0, z0, x1, z1: ConvexPolygon2.new([Point2(x0, z0), Point2(x1, z0), Point2(x1, z1), Point2(x0, z1)])
    asg = ASG("example-room", [ASGRegionEntity(1, "room", 0, 0.0, [polygon(0, 0, 6, 4)])],
              [ASGObjectEntity(10, "table", 1, [polygon(4.8, 1.8, 5.2, 2.2)], [], level_index=0)], [], [],
              levels=[ASGLevelEntity(0, "ground floor", floor_elevation_mean=0.0, ceiling_elevation_mean=3.0, height_mean=3.0)], data_source="example")
    # This generated floor is independent of HM3D and needs no dataset download.
    vertices = np.asarray([[0., 0., 0.], [6., 0., 0.], [6., 0., 4.], [0., 0., 4.]])
    floor = trimesh.Trimesh(vertices=vertices, faces=[[0, 2, 1], [0, 3, 2]], process=False)
    old_root = os.environ.get("RECONAV_SCENE_ROOT")
    scene_root = output / "scenes"
    os.environ["RECONAV_SCENE_ROOT"] = str(scene_root)
    example_mesh = scene_root / "example/example-room/viewer/geometry.glb"
    example_mesh.parent.mkdir(parents=True, exist_ok=True)
    floor.export(example_mesh)
    try:
        load_or_build_geometry_graph(data_source="example", asg=asg, cache_root=output / "cache")
        save_asg(output / "scene/asg", asg)
        dsl = "START POS 1 0 2\nGOAL OBJ 10"
        route = solve_dsl_route(dsl, asg, profile=ExecutionProfile.FULL_HARD_SOFT, request=FEASIBLE)
        if route.get("status") != "SAT" or not route.get("waypoints"):
            raise RuntimeError(f"Example route was not found: {route.get('status')}")
        from reconav.benchmark.metrics import replayable_route
        from reconav.benchmark.route_evaluator import evaluate_grounded_route
        gold = {"goal": {"kind": "obj", "grounded_ids": [10]}, "via": [], "order": [], "avoid": [], "count": [], "width": [], "prefer": [], "prefer_order": []}
        start = {"x": 1., "floor_height": 0., "y": 2.}
        replay, errors = replayable_route(route, gold, asg, start)
        score = evaluate_grounded_route(asg, gold, replay, start)
        passed = not errors and score["ok"]
        result = {"status": "passed" if passed else "failed", "solver_status": route["status"], "waypoints": len(route["waypoints"]), "route_scoring_ok": score["ok"], "seconds": round(time.perf_counter() - started, 3), "model_calls": 0}
        (output / "route.json").write_text(json.dumps(route, indent=2) + "\n")
        (output / "score.json").write_text(json.dumps(score, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"Example route did not pass scoring: {score}")
        return result
    finally:
        if old_root is None:
            os.environ.pop("RECONAV_SCENE_ROOT", None)
        else:
            os.environ["RECONAV_SCENE_ROOT"] = old_root
