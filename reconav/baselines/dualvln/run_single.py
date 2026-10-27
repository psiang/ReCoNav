"""Run the released full DualVLN model on one goal-free HM3D task."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reconav.baselines.canav.export_task import heading_to_habitat_xyzw, load_policy_input
from reconav.baselines.dualvln.contracts import (
    SanitizedObservation,
    released_dualvln_config,
    write_policy_manifest,
)
from reconav.baselines.dualvln.instrumentation import S2InferenceInstrumentation
from reconav.baselines.dualvln.trajectory import DualVLNTrajectory


AUDITED_INTERNNAV_HEAD = "7a5c62400ac45b313d9b709c740b64191556a242"
AUDITED_CHECKPOINT_HEAD = "a698a9e898b4001621a319e1bc89f02ec715cc86"
DEPTH_CHECKPOINT = 'depth_anything_v2_metric_hypersim_vits.pth'


def depth_checkpoint(runtime_workdir: Path) -> Path:
    path = runtime_workdir / 'checkpoints' / DEPTH_CHECKPOINT
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise FileNotFoundError('Download the DualVLN depth checkpoint to ' + str(path))
    return path


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def verify_release(upstream: Path, checkpoint: Path) -> None:
    if _git_value(upstream, "rev-parse", "HEAD") != AUDITED_INTERNNAV_HEAD:
        raise RuntimeError("InternNav checkout differs from the audited release")
    if _git_value(upstream, "status", "--porcelain"):
        raise RuntimeError("InternNav worktree is not clean")
    if _git_value(checkpoint, "rev-parse", "HEAD") != AUDITED_CHECKPOINT_HEAD:
        raise RuntimeError("DualVLN checkpoint differs from the audited release")
    checkpoint_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if checkpoint_config.get("system1") != "nextdit_async":
        raise RuntimeError("released checkpoint is not the required nextdit_async variant")
    shards = sorted(checkpoint.glob("model-*-of-*.safetensors"))
    if len(shards) != 4 or any(path.stat().st_size < 1_000_000_000 for path in shards):
        raise RuntimeError("full four-shard DualVLN checkpoint is unavailable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--asg", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-workdir", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-primitive-actions", type=int, default=500)
    return parser.parse_args()


def _simulator(scene: Path, config: dict[str, Any]) -> Any:
    import habitat_sim
    from reconav.baselines.common.robot import PROFILE, install_habitat_profile
    install_habitat_profile()

    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_id = str(scene.resolve())
    simulator.gpu_device_id = -1
    simulator.enable_physics = False

    camera = config["camera"]
    sensor_specs = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [camera["resolution"][1], camera["resolution"][0]]
        spec.position = [0.0, camera["height_m"], 0.0]
        spec.hfov = camera["hfov"]
        if uuid == "depth":
            # The released normalization minimum is zero, but OpenGL requires
            # a strictly positive clipping plane.
            spec.near = camera["render_near_m"]
            spec.far = camera["depth_max_m"]
        sensor_specs.append(spec)

    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = PROFILE['height_m']
    agent.radius = PROFILE['radius_m']
    agent.sensor_specifications = sensor_specs
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=config["actions"]["forward_step_m"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=config["actions"]["turn_angle_deg"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=config["actions"]["turn_angle_deg"])
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=config["actions"]["tilt_angle_deg"])
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=config["actions"]["tilt_angle_deg"])
        ),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))


def _snapshot(state: Any) -> Any:
    return copy.deepcopy(state)


def restore_camera_after_look_down(sim: Any, recorder: DualVLNTrajectory) -> int:
    """Restore the native camera after the look-down observation was processed."""

    for action_name in ("look_up", "look_up"):
        before = _snapshot(sim.get_agent(0).get_state())
        sim.step(action_name)
        after = _snapshot(sim.get_agent(0).get_state())
        recorder.action(action_name, before, after)
    return 2


def main() -> int:
    args = parse_args()
    from reconav.baselines.common.robot import install_habitat_profile
    install_habitat_profile(args.output / 'robot_geometry.json')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_release(args.upstream, args.checkpoint)
    depth_checkpoint(args.runtime_workdir)
    policy_input = load_policy_input(args.task)
    config = released_dualvln_config(args.checkpoint, device=args.device)
    write_policy_manifest(
        policy_input,
        scene_path=args.scene,
        model_path=args.checkpoint,
        output=args.output / "policy_input_manifest.json",
    )
    (args.output / "released_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    upstream_path = str(args.upstream.resolve())
    diffusion_path = str((args.upstream / "third_party" / "diffusion-policy").resolve())
    for path in (upstream_path, diffusion_path):
        if path not in sys.path:
            sys.path.insert(0, path)

    args.runtime_workdir.mkdir(parents=True, exist_ok=True)
    recorder = DualVLNTrajectory(args.output / "trajectory.json", policy_input.benchmark_task_id)
    s2_instrumentation = S2InferenceInstrumentation(args.output / "s2_inference_provenance.json")
    sim = None
    agent = None
    policy_trace: list[dict[str, Any]] = []
    primitive_count = 0
    look_down_active = False
    previous_cwd = Path.cwd()
    os.chdir(args.runtime_workdir)
    try:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs
        from internnav.agent.internvla_n1_agent import InternVLAN1Agent
        from internnav.configs.agent import AgentCfg

        sim = _simulator(args.scene, config)
        state = habitat_sim.AgentState()
        state.position = np.asarray(policy_input.start_position, dtype=np.float32)
        state.rotation = quat_from_coeffs(np.asarray(heading_to_habitat_xyzw(policy_input.start_heading)))
        sim.initialize_agent(0, state)
        recorder.reset(sim.get_agent(0).get_state())
        if not sim.pathfinder.is_navigable(np.asarray(policy_input.start_position, dtype=np.float32)):
            raise RuntimeError('unsupported exact start: refusing snap before loading DualVLN')

        model_settings = {key: value for key, value in config.items() if key not in {"camera", "actions", "checkpoint_variant", "checkpoint_system1"}}
        agent = InternVLAN1Agent(AgentCfg(
            model_name="internvla_n1",
            model_settings=model_settings,
        ))
        s2_instrumentation.instrument_policy(agent.policy)
        agent.reset()

        while primitive_count < args.max_primitive_actions:
            observations = sim.get_sensor_observations()
            rgb = np.asarray(observations["rgb"])[..., :3].astype(np.uint8, copy=False)
            depth_m = np.asarray(observations["depth"], dtype=np.float32)
            depth_normalized = np.clip(depth_m / config["camera"]["depth_max_m"], 0.0, 1.0)[..., None]
            sanitized = SanitizedObservation(policy_input.instruction, rgb, depth_normalized)
            was_look_down = bool(agent.look_down)
            result = agent.step(sanitized.as_upstream_input())
            action_code = int(result[0]["action"][0])
            stop_origin = s2_instrumentation.classify_returned_action(action_code)
            llm_output = str(getattr(agent.policy, "llm_output", ""))
            output_type = None
            if llm_output:
                output_type = "pixel_goal" if any(ch.isdigit() for ch in llm_output) else "action_tokens"
            policy_trace.append({
                "policy_step": len(policy_trace),
                "look_down_input": was_look_down,
                "system2_output_type": output_type,
                "system2_text": llm_output,
                "returned_action_code": action_code,
                "stop_origin": stop_origin,
                "look_down_requested": bool(agent.look_down and action_code == -1),
                "dual_forward_step": int(agent.dual_forward_step),
                "system1_inference_count": int(agent.sys1_infer_times),
            })

            if look_down_active:
                primitive_count += restore_camera_after_look_down(sim, recorder)
                look_down_active = False

            if action_code == -1:
                # The released agent also returns -1 for an empty System 1
                # action list.  Only its explicit state flag denotes LOOKDOWN;
                # the other case is a native stand-still with no sim action.
                if agent.look_down:
                    for action_name in ("look_down", "look_down"):
                        before = _snapshot(sim.get_agent(0).get_state())
                        sim.step(action_name)
                        after = _snapshot(sim.get_agent(0).get_state())
                        recorder.action(action_name, before, after)
                        primitive_count += 1
                    look_down_active = True
                continue

            if action_code == 0:
                current = _snapshot(sim.get_agent(0).get_state())
                recorder.action("stop", current, current, native_stop=True)
                break

            action_name = {1: "move_forward", 2: "turn_left", 3: "turn_right"}.get(action_code)
            if action_name is None:
                raise RuntimeError(f"released policy returned unsupported action code {action_code}")
            before = _snapshot(sim.get_agent(0).get_state())
            sim.step(action_name)
            after = _snapshot(sim.get_agent(0).get_state())
            recorder.action(action_name, before, after)
            primitive_count += 1
        else:
            recorder.finish_max_steps()
    except Exception as exc:
        recorder.fail(exc)
    finally:
        if sim is not None:
            sim.close()
        os.chdir(previous_cwd)

    trajectory = recorder.payload()
    recorder.write()
    s2_instrumentation.write()
    (args.output / "policy_trace.json").write_text(
        json.dumps(policy_trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_types = sorted({row["system2_output_type"] for row in policy_trace if row["system2_output_type"]})
    summary = {
        "status": "completed" if trajectory["runtime_exception"] is None else "runtime_exception",
        "internnav_head": AUDITED_INTERNNAV_HEAD,
        "checkpoint_head": AUDITED_CHECKPOINT_HEAD,
        "checkpoint": "InternRobotics/InternVLA-N1-DualVLN",
        "system1": "nextdit_async",
        "system2_output_types": output_types,
        "usable_pixel_or_local_goal_produced": "pixel_goal" in output_types,
        "native_look_down_requests": sum(row["look_down_requested"] for row in policy_trace),
        "native_stand_still_outputs": sum(
            row["returned_action_code"] == -1 and not row["look_down_requested"] for row in policy_trace
        ),
        "policy_step_count": len(policy_trace),
        "s2_inference_provenance": s2_instrumentation.counters(),
        "genuine_model_stop_count": sum(
            row["stop_origin"] == "genuine_model_stop" for row in policy_trace
        ),
        "s2_failure_fallback_stop_observed_count": sum(
            row["stop_origin"] == "s2_failure_fallback" for row in policy_trace
        ),
        "trajectory": {key: trajectory[key] for key in (
            "executed_actions", "movement_primitive_count", "recorded_poses",
            "path_length_m", "termination", "prediction_empty"
        )},
        "method_code_changed": False,
    }
    (args.output / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    from reconav.baselines.common.time_budget import mark_policy_complete
    mark_policy_complete()
    scoring_env = dict(os.environ)
    scoring_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    subprocess.run(
        [
            str(args.evaluator_python), "-m", "reconav.baselines.dualvln.score_trajectory",
            "--task", str(args.task), "--asg", str(args.asg),
            "--trajectory", str(args.output / "trajectory.json"),
            "--output", str(args.output / "metrics.json"),
        ],
        cwd=Path(__file__).resolve().parents[3], env=scoring_env, check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
