"""Record GC-VLN runtime counters and exceptions."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable


class RuntimeInstrumentation:
    def __init__(self, output: Path, *, run_kind: str) -> None:
        self.output = Path(output)
        self.write_seconds = 0.0
        self.write_count = 0
        self.value: dict[str, Any] = {
            "run_kind": run_kind,
            "scenegraph_call_count": 0,
            "scenegraph_success_count": 0,
            "scenegraph_detection_count": 0,
            "region_solver_call_count": 0,
            "region_solver_success_count": 0,
            "fmm_call_count": 0,
            "primitive_action_count": 0,
            "runtime_exception_count": 0,
            "runtime_exceptions": [],
            "stop_followed_caught_runtime_exception": False,
        }
        self.write()

    def wrap_scenegraph(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.value["scenegraph_call_count"] += 1
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                self.exception(exc, source="SceneGraph.get_scenegraph")
                raise
            self.value["scenegraph_success_count"] += 1
            if result:
                self.value["scenegraph_detection_count"] += 1
            owner = getattr(function, '__self__', None)
            if owner is not None and hasattr(owner, 'map_draft'):
                draft = owner.map_draft
                row = {'scene_nodes': draft.scene_graph.number_of_nodes(),
                       'agent_coordinate': [float(v) for v in args[5]],
                       'depth_min': float(args[2].min()), 'depth_max': float(args[2].max())}
                for name in ('bev_map', 'bev_map_fmm', 'bev_wall', 'bev_thin'):
                    row[name + '_nonzero'] = int(getattr(draft, name).count_nonzero().item())
                self.value.setdefault('map_diagnostics', []).append(row)
            return result

        return wrapped

    def wrap_region_solver(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.value["region_solver_call_count"] += 1
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                self.exception(exc, source="RegionSolver.get_next_point")
                raise
            self.value["region_solver_success_count"] += 1
            if isinstance(result, tuple) and len(result) == 2:
                point, reason = result
                self.value.setdefault('solver_outputs', []).append({
                    'next_point': None if point is None else [float(x) for x in point],
                    'stage_result': str(reason)})
            return result

        return wrapped

    def primitive(self) -> None:
        self.value["primitive_action_count"] += 1

    def fmm(self) -> None:
        """Record one native FMMPlanner invocation without changing its inputs."""
        self.value["fmm_call_count"] += 1

    def exception(self, exc: BaseException, *, source: str) -> None:
        record = {"source": source, "type": type(exc).__name__, "message": str(exc)}
        import traceback
        record['traceback'] = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if record not in self.value["runtime_exceptions"]:
            self.value["runtime_exceptions"].append(record)
            self.value["runtime_exception_count"] += 1
        self.write()

    def mark_stop_after_exception(self, stopped: bool) -> None:
        self.value["stop_followed_caught_runtime_exception"] = bool(
            stopped and self.value["runtime_exception_count"]
        )
        self.write()

    def write(self) -> None:
        started = time.perf_counter()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.output)
        self.write_seconds += time.perf_counter() - started
        self.write_count += 1
        self.output.with_suffix('.timing.json').write_text(json.dumps({
            'seconds': self.write_seconds, 'write_count': self.write_count,
        }, indent=2) + '\n', encoding='utf-8')
