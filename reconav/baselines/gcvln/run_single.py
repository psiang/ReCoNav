"""Run one GC-VLN episode with its benchmark environment adapter."""

from __future__ import annotations

import argparse
import json
import os
import random
import site
import subprocess
import sys
import signal
import threading
import types
from pathlib import Path

import numpy as np

from reconav.baselines.canav.export_task import load_policy_input
from reconav.baselines.gcvln.export_task import export_episode, write_preflight_manifest
from reconav.baselines.gcvln.preprocess import released_parser_gate


AUDITED_SHA = "1beddca8d90d79855c110c5a433f7b8856ca263a"


def verify_upstream(root: Path) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if head != AUDITED_SHA:
        raise RuntimeError(f"GC-VLN HEAD {head} != audited {AUDITED_SHA}")
    if status:
        raise RuntimeError("official GC-VLN worktree is not clean")


def configure_legacy_glip_runtime(torch: object) -> None:
    """Supply PyTorch aliases and expose the separately built GLIP extension."""
    if not hasattr(torch, "_six"):
        torch_six = types.ModuleType("torch._six")
        torch_six.PY3 = True
        torch_six.PY37 = True
        torch._six = torch_six
        sys.modules["torch._six"] = torch_six
    if not hasattr(torch.hub, "_download_url_to_file"):
        torch.hub._download_url_to_file = torch.hub.download_url_to_file

    import maskrcnn_benchmark

    for root in site.getsitepackages():
        installed_package = Path(root) / "maskrcnn_benchmark"
        installed_extension = next(installed_package.glob("_C*.so"), None)
        if installed_extension and str(installed_package) not in maskrcnn_benchmark.__path__:
            maskrcnn_benchmark.__path__.append(str(installed_package))
    try:
        from maskrcnn_benchmark import _C  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("the official GLIP native extension is not installed") from exc


def init_single_threaded_env(init_env: object, config: object) -> object:
    """Select Habitat's equivalent single-process vector shell for one episode.

    The released helper otherwise uses ``forkserver`` even for one environment;
    that child cannot inherit the import-only GLIP compatibility aliases.  The
    upstream helper already selects ``ThreadedVectorEnv`` while a trace hook is
    active, so use that published branch only during construction.
    """
    previous_trace = sys.gettrace()
    sys.settrace(lambda *_args, **_kwargs: None)
    try:
        return init_env(config)
    finally:
        sys.settrace(previous_trace)


def require_valid_preprocessing(output: Path) -> tuple[dict, Path]:
    metadata_path = output / "preprocess_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "valid_first_response":
        raise RuntimeError(
            "GC-VLN runtime blocked by invalid first Appendix-G response; semantic retry/repair is forbidden: "
            + str(metadata.get("error"))
        )
    if metadata.get("semantic_retry_count") != 0 or metadata.get("semantic_refinement") is not None:
        raise RuntimeError("preprocessing metadata indicates forbidden semantic retry/refinement")
    runtime_path = output / "runtime_dag.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(runtime_path)
    return metadata, runtime_path


def write_preprocessing_empty_prediction(
    output: Path,
    *,
    task_id: str,
    start_position: tuple[float, float, float],
    start_heading: float,
    reason: str,
) -> Path:
    """Persist the evaluator-visible empty route required after one-shot validation failure."""
    trajectory = {
        "task_id": task_id,
        "method": "GC-VLN-GPT5.6-adapted",
        "run_kind": "adapted_official_prompt_deterministic_lowering",
        "valid_for_scoring": True,
        "coordinate_frame": "Habitat world xyz",
        "trajectory_source": (
            "formal empty prediction; rollout blocked by one-shot preprocessing validation"
        ),
        "failure_reason": "preprocessing_invalid",
        "termination": "preprocessing_invalid",
        "executed_actions": 0,
        "recorded_poses": 0,
        "pose_action_invariant": None,
        "path_length_m": 0.0,
        "teleflag": False,
        "teleport_attempts_blocked": 0,
        "configured_start_position": {
            "x": float(start_position[0]),
            "y": float(start_position[1]),
            "z": float(start_position[2]),
        },
        "configured_start_heading": float(start_heading),
        "route": {"waypoints": []},
        "action_trace": [],
    }
    trajectory_path = output / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "formal_empty_prediction",
        "valid_for_scoring": True,
        "phase": "paper_schema_validation",
        "failure_reason": "preprocessing_invalid",
        "validation_error": reason,
        "semantic_retry_performed": False,
        "lowering_performed": False,
        "released_parser_reached": False,
        "habitat_started": False,
        "scenegraph_call_count": 0,
        "scenegraph_success_count": 0,
        "region_solver_call_count": 0,
        "region_solver_success_count": 0,
        "fmm_call_count": 0,
        "primitive_action_count": 0,
        "waypoint_count": 0,
        "pose_count": 0,
        "path_length_m": 0.0,
        "termination": "preprocessing_invalid",
        "prediction_empty": True,
        "teleflag": False,
        "teleport_attempts_blocked": 0,
        "runtime_exception_count": 0,
        "runtime_exceptions": [],
        "method_code_changed": False,
    }
    (output / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return trajectory_path


def score_trajectory(args: argparse.Namespace, trajectory_path: Path) -> None:
    from reconav.baselines.common.time_budget import mark_policy_complete
    mark_policy_complete()
    project_root = Path(__file__).resolve().parents[3]
    scoring_env = dict(os.environ)
    scoring_env["PYTHONPATH"] = str(project_root)
    subprocess.run(
        [
            str(args.evaluator_python),
            "-m",
            "reconav.baselines.gcvln.score_trajectory",
            "--task",
            str(args.task),
            "--asg",
            str(args.asg),
            "--trajectory",
            str(trajectory_path),
            "--output",
            str(args.output / "metrics.json"),
        ],
        cwd=project_root,
        env=scoring_env,
        check=True,
    )


def normalize_native_stop_without_route(output: Path, trajectory: dict) -> bool:
    """Expose a zero-motion native STOP as the protocol's formal empty prediction.

    The action trace remains intact as execution evidence. Only evaluator-visible
    waypoints are emptied so reset/STOP poses cannot masquerade as progress.
    """
    trace = trajectory.get("action_trace", [])
    primitive_actions = [
        record.get("action") for record in trace if record.get("action") != "RESET"
    ]
    no_executable_route = (
        trajectory.get("termination") == "native_stop"
        and float(trajectory.get("path_length_m", 0.0)) == 0.0
        and primitive_actions == [0]
    )
    if not no_executable_route:
        return False

    native_path = output / "trajectory_native_stop.json"
    native_trajectory = dict(trajectory)
    native_trajectory["route"] = trajectory.get("raw_route", trajectory["route"])
    native_path.write_text(
        json.dumps(native_trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    trajectory["trajectory_source"] = (
        "formal empty prediction after zero-motion native STOP; actual action/pose trace preserved"
    )
    trajectory["failure_reason"] = "no_executable_plan"
    trajectory["policy_termination_reason"] = "normal_policy_stop"
    trajectory["native_trajectory_archive"] = native_path.name
    trajectory["route"] = {"waypoints": []}
    (output / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return True


def build_config(args: argparse.Namespace, dataset_path: Path):
    from config.default import _C as POLICY_CONFIG
    from habitat.config.default import get_config

    config = get_config(str(args.upstream / "config/r2r_vlnce.yaml"), None, POLICY_CONFIG)
    config.defrost()
    from reconav.baselines.common.robot import PROFILE
    config.TASK_CONFIG.SIMULATOR.AGENT_0.RADIUS = PROFILE['radius_m']
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = PROFILE['height_m']
    config.ENV_NAME = "GCVLNBenchmarkEnv"
    config.NUM_ENVIRONMENTS = 1
    config.GPU_NUMBERS = 1
    config.TORCH_GPU_ID = int(args.gpu_id)
    config.TORCH_GPU_IDS = [int(args.gpu_id)]
    simulator_gpu_id = args.simulator_gpu_id
    if simulator_gpu_id is None:
        simulator_gpu_id = args.gpu_id
    config.SIMULATOR_GPU_IDS = [int(simulator_gpu_id)]
    config.local_rank = 0
    config.EVAL.EPISODE_COUNT = 1
    config.DATASET.SPLIT_NUM = 1
    config.DATASET.SPLIT_INDEX = 0
    config.DATASET.DATASET_TYPE = "r2r"
    config.DATASET.EXPERIMENT_ID = "gcvln_gpt56_sol_high_T1_0001"
    config.TASK_CONFIG.DATASET.SPLIT = "gcvln_hm3d_single"
    config.TASK_CONFIG.DATASET.DATA_PATH = str(dataset_path.resolve())
    config.TASK_CONFIG.DATASET.SCENES_DIR = str(args.scene_root.resolve())
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = ["*"]
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [1]
    config.TASK_CONFIG.DATASET.SPLIT_NUM = 1
    config.TASK_CONFIG.DATASET.SPLIT_INDEX = 0
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["STEPS_TAKEN", "PATH_LENGTH"]
    forbidden_sensors = {"SHORTEST_PATH_SENSOR", "VLN_ORACLE_PROGRESS_SENSOR"}
    if forbidden_sensors.intersection(config.TASK_CONFIG.TASK.SENSORS):
        raise RuntimeError("oracle sensor present in GC-VLN benchmark config")
    config.POLICY_CONFIG.UNI.TELEFLAG = False
    config.POLICY_CONFIG.UNI.VIDEO_OPTION = []
    config.POLICY_CONFIG.GSAM2_SERVER_PORT = int(args.server_port)
    config.freeze()
    if config.POLICY_CONFIG.UNI.TELEFLAG:
        raise AssertionError("TELEFLAG was not disabled")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--asg", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="checkpoint/config overlay used as cwd; defaults to the frozen upstream root",
    )
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-port", type=int, default=7000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--simulator-gpu-id",
        type=int,
        help="Habitat renderer device; use -1 for the Mesa display-build path",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument('--asg-policy-assets', type=Path,
                        help='Explicit +ASG condition: scene-only graph and physical map assets')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_upstream(args.upstream)
    policy_input = load_policy_input(args.task)
    preprocessing_status = json.loads(
        (args.output / "preprocess_metadata.json").read_text(encoding="utf-8")
    ).get("status", "missing")
    write_preflight_manifest(
        policy_input,
        scene_path=args.scene,
        manifest_path=args.output / "policy_input_manifest.json",
        preprocessing_status=str(preprocessing_status),
    )
    metadata = None
    try:
        metadata, runtime_path = require_valid_preprocessing(args.output)
    except RuntimeError as exc:
        trajectory_path = write_preprocessing_empty_prediction(
            args.output,
            task_id=policy_input.benchmark_task_id,
            start_position=policy_input.start_position,
            start_heading=policy_input.start_heading,
            reason=str(exc),
        )
        score_trajectory(args, trajectory_path)
        return 0
    parser_gate = released_parser_gate(args.upstream, runtime_path)
    if not parser_gate.get("parser_success") or not parser_gate.get("navigation_tree_success"):
        raise RuntimeError(f"released parser gate failed: {parser_gate}")
    dataset_path = args.output / "gcvln_episode.json.gz"
    export_episode(
        policy_input,
        scene_path=args.scene,
        scene_root=args.scene_root,
        runtime_dag_path=runtime_path,
        dataset_path=dataset_path,
        manifest_path=args.output / "policy_input_manifest.json",
        llm_model=metadata.get('requested_model', 'gpt-5.6-sol'),
    )
    if args.prepare_only:
        return 0
    if not args.asg_policy_assets:
        from reconav.baselines.common.robot import install_habitat_profile
        install_habitat_profile(args.output / 'robot_geometry.json')

    os.environ["GCVLN_BENCHMARK_OUTPUT"] = str(args.output)
    os.environ["GCVLN_BENCHMARK_TASK_ID"] = policy_input.benchmark_task_id
    os.environ["GCVLN_RUN_KIND"] = "adapted_official_prompt_deterministic_lowering"
    os.environ["GCVLN_VALID_FOR_SCORING"] = "1"
    os.environ['GCVLN_BENCHMARK_MODEL'] = metadata.get('requested_model', 'gpt-5.6-sol')
    previous_cwd = Path.cwd()
    runtime_root = (args.runtime_root or args.upstream).resolve()
    os.chdir(runtime_root)
    previous_thread_hook = threading.excepthook
    def abort_failed_thread(event):
        (args.output/'runtime_thread_failure.json').write_text(json.dumps({
            'thread': event.thread.name, 'type': event.exc_type.__name__,
            'message': str(event.exc_value)}, indent=2)+'\n')
        previous_thread_hook(event)
        # ThreadedVectorEnv otherwise blocks forever after a reset/step worker
        # dies. Stop our own worker; the parent records infrastructure failure.
        os.kill(os.getpid(), signal.SIGTERM)
    threading.excepthook = abort_failed_thread
    try:
        import torch
        configure_legacy_glip_runtime(torch)
        from reconav.baselines.gcvln.evaluator_adapter import GoalFreeBenchmarkAgent
        from src.agent import init_env

        config = build_config(args, dataset_path)
        random.seed(config.TASK_CONFIG.SEED)
        np.random.seed(config.TASK_CONFIG.SEED)
        torch.manual_seed(config.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if args.asg_policy_assets:
            from reconav.baselines.common.protocol import policy_payload, read, write
            from reconav.baselines.gcvln.asg_scenegraph import ASGSceneGraph
            from reconav.baselines.gcvln.g0_env import G0VectorEnv
            import src.agent.agent_policy as policy_module
            policy = policy_payload(args.task)
            env = G0VectorEnv(config, policy, args.asg_policy_assets, args.asg,
                              args.scene, args.output, runtime_path)
            def provided_scenegraph(cfg, dataset):
                scenegraph = ASGSceneGraph(cfg, policy, args.asg_policy_assets, args.asg, args.output,
                    lambda: env.call(['get_pos_ori'])[0])
                env.asg_scenegraph = scenegraph
                return scenegraph
            policy_module.SceneGraph = provided_scenegraph
            manifest = read(args.output/'policy_input_manifest.json')
            manifest['condition'] = 'asg'
            manifest['provided_scene_information'] = ['all scene ASG entities', 'ASG physical free space', 'G0 portal connectivity']
            manifest['physical_backend'] = 'shared-G0-r025-h150-v1'
            manifest['native_navmesh_for_reachability'] = False
            manifest['explicitly_absent'] = [v for v in manifest.get('explicitly_absent', []) if v != 'GT ASG']
            write(args.output/'policy_input_manifest.json', manifest)
        else:
            env = init_single_threaded_env(init_env, config)
        try:
            agent = GoalFreeBenchmarkAgent(env, config)
            agent.act()
        finally:
            if args.asg_policy_assets:
                env.close()
    finally:
        threading.excepthook = previous_thread_hook
        os.chdir(previous_cwd)

    trajectory_path = args.output / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory["recorded_poses"] != trajectory["executed_actions"] + 1:
        raise RuntimeError("actual trajectory pose/action invariant failed")
    if trajectory["teleport_attempts_blocked"]:
        raise RuntimeError("scored rollout attempted teleport/recovery")
    main_instrumentation = json.loads(
        (args.output / "runtime_instrumentation.json").read_text(encoding="utf-8")
    )
    env_instrumentation = json.loads(
        (args.output / "runtime_instrumentation_env.json").read_text(encoding="utf-8")
    )
    instrumentation = dict(main_instrumentation)
    instrumentation["primitive_action_count"] = env_instrumentation["primitive_action_count"]
    instrumentation["fmm_call_count"] = env_instrumentation["fmm_call_count"]
    instrumentation['g0_search_call_count'] = env_instrumentation.get('g0_search_call_count', 0)
    instrumentation["environment_runtime_exceptions"] = env_instrumentation[
        "runtime_exceptions"
    ]
    if instrumentation['runtime_exception_count'] or instrumentation['environment_runtime_exceptions']:
        # The original policy catches exceptions and returns STOP. That is an
        # infrastructure failure, never evidence of unsuccessful reasoning.
        trajectory['runtime_exception'] = {'type': 'CaughtNativeRuntimeException',
            'message': json.dumps(instrumentation['runtime_exceptions'] +
                                  instrumentation['environment_runtime_exceptions'], ensure_ascii=False)}
        (args.output/'trajectory.json').write_text(json.dumps(trajectory, indent=2)+'\n')
        (args.output/'runtime_summary.json').write_text(json.dumps({
            'status': 'runtime_failure', 'instrumentation': instrumentation,
            'scoring_completed': False, 'trajectory': {key: trajectory[key] for key in
                ('executed_actions', 'recorded_poses', 'termination', 'path_length_m')}}, indent=2)+'\n')
        (args.output/'metrics.json').write_text(json.dumps({
            'full_is': None, 'hard_ns': None, 'prefer': None,
            'failure_category': 'runtime_failure', 'scoring_completed': False}, indent=2)+'\n')
        return 2
    prediction_empty = normalize_native_stop_without_route(args.output, trajectory)
    score_trajectory(args, trajectory_path)
    summary = {
        "status": "formal_empty_prediction" if prediction_empty else "completed",
        "parser_gate": parser_gate,
        "preprocess_response_id": metadata["call"].get("response_id") if metadata else None,
        "instrumentation": instrumentation,
        "teleflag": False,
        "oracle_sensors": [],
        "failure_reason": "no_executable_plan" if prediction_empty else None,
        "termination_reason": "normal_policy_stop" if prediction_empty else trajectory["termination"],
        "prediction_empty": prediction_empty,
        "trajectory": {key: trajectory[key] for key in ("executed_actions", "recorded_poses", "termination", "path_length_m")},
    }
    (args.output / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
