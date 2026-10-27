"""GC-VLN episode setup, motion checks and trajectory recording."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from habitat_baselines.common.baseline_registry import baseline_registry
from src.agent.environments import GCVLNEnv
from src.agent.agent_policy import Agent

from reconav.baselines.gcvln.trajectory import ActualPrimitiveTrajectory
from reconav.baselines.gcvln.fairness import TeleportGuard
from reconav.baselines.gcvln.runtime_instrumentation import RuntimeInstrumentation


@baseline_registry.register_env(name="GCVLNBenchmarkEnv")
class GCVLNBenchmarkEnv(GCVLNEnv):
    def __init__(self, config: Any, dataset: Any = None) -> None:
        super().__init__(config, dataset)
        if self.tele_flag:
            raise RuntimeError("scored GC-VLN benchmark requires TELEFLAG=False")
        output = os.environ.get("GCVLN_BENCHMARK_OUTPUT")
        task_id = os.environ.get("GCVLN_BENCHMARK_TASK_ID")
        if not output or not task_id:
            raise RuntimeError("GC-VLN benchmark output/task environment is missing")
        run_kind = os.environ.get("GCVLN_RUN_KIND", "adapted_official_prompt_deterministic_lowering")
        valid_for_scoring = os.environ.get("GCVLN_VALID_FOR_SCORING", "1") == "1"
        self._trajectory = ActualPrimitiveTrajectory(
            Path(output) / "trajectory.json",
            task_id,
            run_kind=run_kind,
            valid_for_scoring=valid_for_scoring,
        )
        self._runtime_instrumentation = RuntimeInstrumentation(
            Path(output) / "runtime_instrumentation_env.json", run_kind=run_kind
        )
        self._teleport_guard = TeleportGuard(self._trajectory)

    def _actual_pose(self) -> tuple[list[float], list[float], float]:
        state = self._env.sim.get_agent_state()
        rotation = [*state.rotation.imag, state.rotation.real]
        heading = float(self.get_agent_info()["heading"])
        return state.position.tolist(), rotation, heading

    def reset(self) -> Any:
        observations = super().reset()
        import numpy as np
        expected = np.asarray(self._env.current_episode.start_position, dtype=float)
        actual = np.asarray(self._env.sim.get_agent_state().position, dtype=float)
        if np.linalg.norm(actual - expected) > 1e-5:
            raise RuntimeError('simulator changed the exact benchmark reset pose')
        simulator = self._env.sim
        pathfinder = getattr(simulator, 'pathfinder', None)
        if pathfinder is None:
            pathfinder = simulator._sim.pathfinder
        if not pathfinder.is_navigable(expected):
            raise RuntimeError('unsupported exact start: refusing native navmesh snapping')
        self._trajectory.reset(*self._actual_pose())
        self._trajectory.write()
        from reconav.baselines.common.rendering import configuration
        from reconav.baselines.common.protocol import write
        write(self._trajectory.output.parent / 'rendering.json', configuration())
        return observations

    def wrap_act(self, act: Any, vis_info: Any) -> Any:
        observations = super().wrap_act(act, vis_info)
        self._trajectory.primitive(int(act), *self._actual_pose(), done=False)
        self._runtime_instrumentation.primitive()
        return observations

    def step(self, action: dict[str, Any], vis_info: Any, bev: Any, *args: Any, **kwargs: Any) -> Any:
        if action.get("next_point") is not None:
            self._runtime_instrumentation.fmm()
        result = super().step(action, vis_info, bev, *args, **kwargs)
        if action["act"] == 0:
            self._trajectory.primitive(0, *self._actual_pose(), done=bool(result[2]))
            self._runtime_instrumentation.primitive()
        return result

    def teleport(self, pos: Any, rotation: Any) -> None:
        self._teleport_guard(pos, rotation)

    def close(self) -> None:
        self._trajectory.close()
        self._runtime_instrumentation.write()
        super().close()


class GoalFreeBenchmarkAgent(Agent):
    """Keep GC-VLN policy intact while replacing its goal-dependent result shell."""

    def __init__(self, env: Any, config: Any) -> None:
        super().__init__(env, config)
        output = Path(os.environ["GCVLN_BENCHMARK_OUTPUT"])
        run_kind = os.environ.get("GCVLN_RUN_KIND", "adapted_official_prompt_deterministic_lowering")
        self._instrumentation = RuntimeInstrumentation(
            output / "runtime_instrumentation.json", run_kind=run_kind
        )
        self._instrumented_scenegraphs: set[int] = set()
        self._instrumented_solvers: set[int] = set()
        self.visualization_save_dir = str(output / "runtime/visualization")
        self.log_save_dir = str(output / "runtime/native_logs")
        Path(self.visualization_save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_save_dir).mkdir(parents=True, exist_ok=True)

    def _reset_policy(self, *args: Any, **kwargs: Any) -> None:
        super()._reset_policy(*args, **kwargs)
        for scenegraph in self.sg_list:
            if id(scenegraph) not in self._instrumented_scenegraphs:
                scenegraph.get_scenegraph = self._instrumentation.wrap_scenegraph(
                    scenegraph.get_scenegraph
                )
                self._instrumented_scenegraphs.add(id(scenegraph))
        for solver in self.rs_list:
            if id(solver) not in self._instrumented_solvers:
                solver.get_next_point = self._instrumentation.wrap_region_solver(
                    solver.get_next_point
                )
                self._instrumented_solvers.add(id(solver))

    def rollout(self) -> None:
        # Upstream catches *all* exceptions around planning + visualization and
        # replaces the action with STOP. Wrapping just the solver misses e.g.
        # visualization API failures. Observe this frame without changing it.
        previous_trace = sys.gettrace()
        def trace(frame, event, arg):
            if frame.f_code is not Agent.rollout.__code__:
                return None
            if event == 'exception' and not isinstance(arg[1], (StopIteration, GeneratorExit)):
                self._instrumentation.exception(arg[1], source='Agent.rollout_caught_or_propagated')
            return trace
        sys.settrace(trace)
        try:
            super().rollout()
        except Exception as exc:
            self._instrumentation.exception(exc, source="Agent.rollout_uncaught")
            raise
        finally:
            sys.settrace(previous_trace)
        trajectory_path = Path(os.environ["GCVLN_BENCHMARK_OUTPUT"]) / "trajectory.json"
        stopped = False
        if trajectory_path.is_file():
            trajectory = __import__("json").loads(trajectory_path.read_text(encoding="utf-8"))
            trace = trajectory.get("action_trace", [])
            stopped = bool(trace and trace[-1].get("action") == 0)
        self._instrumentation.mark_stop_after_exception(stopped)

    def calculate_metric(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "steps_taken": float(info.get("steps_taken", 0.0)),
            "distance_to_goal": 10.0,
            "success": 0.0,
            "oracle_success": 0.0,
            "path_length": float(info.get("path_length", 0.0)),
            "spl": 0.0,
        }
