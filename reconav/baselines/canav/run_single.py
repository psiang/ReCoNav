"""Prepare and run the faithful CA-Nav HM3D single-task integration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from reconav.baselines.canav.compat import install_habitat_sensor_spec_compat, configure_groundingdino_extension
from reconav.baselines.canav.export_task import export_episode, load_policy_input
from reconav.baselines.canav.generate_reply import validate_reply


AUDITED_SHA = "8430359ff930880ccc74131d4c31d74a598a9b6e"
DEBUG_OUTPUT_DIRS = (
    "debug/results",
    "debug/checkpoints",
    "debug/tensorboard",
    "debug/video",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--asg", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--habitat-lab", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--evaluator-python",
        type=Path,
        required=True,
        help="existing project environment containing the unified evaluator dependencies",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=('qwen-local',),
        default='qwen-local',
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def verify_upstream(root: Path) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if head != AUDITED_SHA:
        raise RuntimeError(f"CA-Nav HEAD {head} != audited {AUDITED_SHA}")
    if status:
        raise RuntimeError("official CA-Nav worktree is not clean")


def load_cached_reply(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reply = payload.get("1") if isinstance(payload, dict) else None
    if not isinstance(reply, dict):
        raise ValueError("llm_reply.json must contain numeric CA-Nav episode key '1'")
    validate_reply(reply)
    return reply


def reply_is_faithful(metadata_path: Path, variant: str = 'qwen-local') -> bool:
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    return (variant == 'qwen-local'
            and metadata.get('source') == 'official_prompt_live_api'
            and metadata.get('input_fields') == ['instruction']
            and metadata.get('variant') == variant
            and metadata.get('model') == 'qwen3.8-27b'
            and metadata.get('reasoning_effort') == 'none'
            and metadata.get('status') == 'completed')


def remove_empty_output_dirs(output: Path) -> None:
    """Remove empty CA-Nav scaffolding while preserving every generated artifact."""

    for relative in DEBUG_OUTPUT_DIRS:
        path = output / relative
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    debug = output / "debug"
    if debug.is_dir():
        try:
            debug.rmdir()
        except OSError:
            pass


def build_config(args: argparse.Namespace, dataset_path: Path):
    from vlnce_baselines.config.default import get_config

    previous_cwd = Path.cwd()
    try:
        os.chdir(args.upstream)
        config = get_config(str(args.upstream / "vlnce_baselines/config/exp1.yaml"))
    finally:
        os.chdir(previous_cwd)
    config.defrost()
    from reconav.baselines.common.robot import PROFILE
    config.TASK_CONFIG.SIMULATOR.AGENT_0.RADIUS = PROFILE['radius_m']
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = PROFILE['height_m']
    config.TRAINER_NAME = "CANav-HM3D-Single"
    config.NUM_ENVIRONMENTS = 1
    config.TORCH_GPU_ID = 0
    config.TORCH_GPU_IDS = [0]
    config.SIMULATOR_GPU_IDS = [0]
    config.local_rank = 0
    config.world_size = 1
    config.EVAL.EPISODE_COUNT = 1
    config.TASK_CONFIG.DATASET.SPLIT = "canav_hm3d_single"
    config.TASK_CONFIG.DATASET.DATA_PATH = str(dataset_path)
    config.TASK_CONFIG.DATASET.SCENES_DIR = str(args.scene_root.resolve())
    config.TASK_CONFIG.DATASET.LLM_REPLYS_PATH = str((args.output / "llm_reply.json").resolve())
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = ["*"]
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [1]
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["STEPS_TAKEN"]
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 250
    config.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    data = args.artifacts
    config.BLIP2_MODEL_DIR = str(data / "blip2/blip2_model.pt")
    config.BLIP2_VIS_PROCESSORS_DIR = str(data / "blip2/blip2_vis_processors.pt")
    config.BLIP2_TEXT_PROCESSORS_DIR = str(data / "blip2/blip2_text_processors.pt")
    config.VQA_MODEL_DIR = str(data / "vqa/vqa_model.pt")
    config.VQA_VIS_PROCESSORS_DIR = str(data / "vqa/vqa_vis_processors.pt")
    config.VQA_TEXT_PROCESSORS_DIR = str(data / "vqa/vqa_text_processors.pt")
    config.MAP.GROUNDING_DINO_CONFIG_PATH = str(data / "grounded_sam/GroundingDINO_SwinT_OGC.py")
    config.MAP.GROUNDING_DINO_CHECKPOINT_PATH = str(data / "grounded_sam/groundingdino_swint_ogc.pth")
    config.MAP.SAM_CHECKPOINT_PATH = str(data / "grounded_sam/sam_vit_h_4b8939.pth")
    config.MAP.RepViTSAM_CHECKPOINT_PATH = str(data / "grounded_sam/repvit_sam.pt")
    config.RESULTS_DIR = str(args.output / "debug/results")
    config.CHECKPOINT_FOLDER = str(args.output / "debug/checkpoints")
    config.EVAL_CKPT_PATH_DIR = str(args.output / "debug/checkpoints")
    config.TENSORBOARD_DIR = str(args.output / "debug/tensorboard")
    config.VIDEO_DIR = str(args.output / "debug/video")
    config.freeze()
    return config


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for relative in DEBUG_OUTPUT_DIRS:
        (args.output / relative).mkdir(parents=True, exist_ok=True)
    verify_upstream(args.upstream)
    policy_input = load_policy_input(args.task)
    dataset_path = args.output / "canav_episode.json.gz"
    export_episode(
        policy_input,
        scene_path=args.scene,
        scene_root=args.scene_root,
        dataset_path=dataset_path,
        manifest_path=args.output / "policy_input_manifest.json",
    )
    if args.prepare_only:
        return 0
    from reconav.baselines.common.robot import install_habitat_profile
    install_habitat_profile(args.output / 'robot_geometry.json')
    load_cached_reply(args.output / "llm_reply.json")
    faithful_reply = reply_is_faithful(args.output / "llm_reply_metadata.json", args.variant)
    if not faithful_reply:
        raise RuntimeError(
            f"full run requires instruction-only output from the official prompt and {args.variant}; "
            "the paper uses the qwen-local variant"
        )
    sys.path.insert(0, str(args.habitat_lab.resolve()))
    sys.path.insert(0, str(args.upstream.resolve()))
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    sensor_spec_compat = install_habitat_sensor_spec_compat()
    configure_groundingdino_extension()
    from reconav.baselines.canav.evaluator_adapter import CANavHM3DSingleEvaluator

    evaluator = CANavHM3DSingleEvaluator(
        build_config(args, dataset_path),
        task_path=args.task,
        output_dir=args.output,
        method='CA-Nav-Qwen',
        run_kind='full_adapted_qwen',
        valid_for_scoring=faithful_reply,
        habitat_sensor_spec_compat=sensor_spec_compat,
    )
    try:
        evaluator.eval()
    finally:
        remove_empty_output_dirs(args.output)
    from reconav.baselines.common.trajectory_io import trajectory_path_length
    trajectory_path = args.output / 'trajectory.json'
    trajectory = json.loads(trajectory_path.read_text())
    if trajectory.get('termination') not in {'stop', 'native_max_steps'}:
        raise RuntimeError('CA-Nav did not complete its native episode')
    trajectory['path_length_m'] = trajectory_path_length(trajectory)
    trajectory_path.write_text(json.dumps(trajectory, indent=2) + '\n')
    (args.output/'runtime_summary.json').write_text(json.dumps({
        'status': 'completed', 'run_completed': True, 'variant': args.variant,
        'termination': trajectory['termination'], 'executed_actions': trajectory['executed_actions'],
        'path_length_m': trajectory['path_length_m']}, indent=2) + '\n')
    from reconav.baselines.common.time_budget import mark_policy_complete
    mark_policy_complete()
    project_root = Path(__file__).resolve().parents[3]
    scoring_env = dict(os.environ)
    scoring_env["PYTHONPATH"] = str(project_root)
    for name in ("LD_PRELOAD", "__GLX_VENDOR_LIBRARY_NAME", "LIBGL_ALWAYS_SOFTWARE"):
        scoring_env.pop(name, None)
    subprocess.run(
        [
            str(args.evaluator_python),
            "-m",
            "reconav.baselines.canav.score_trajectory",
            "--task",
            str(args.task),
            "--asg",
            str(args.asg),
            "--trajectory",
            str(args.output / "trajectory.json"),
            "--output",
            str(args.output / "metrics.json"),
        ],
        cwd=project_root,
        env=scoring_env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
