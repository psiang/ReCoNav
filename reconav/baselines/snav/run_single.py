"""Run SNav inference on a benchmark episode and record its action/pose trace."""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from reconav.baselines.canav.export_task import load_policy_input, heading_to_habitat_xyzw
from reconav.baselines.dualvln.trajectory import DualVLNTrajectory
from reconav.baselines.common.protocol import write, file_identity


AUDITED_NAVSPACE_HEAD = '976bc48df4875139c96f257d8ec63272a50ad219'


def release_assets(checkpoint, vision_tower):
    """Read downloaded SNav checkpoint and encoder metadata."""
    rows = []
    for root, repository, revision in [
        (checkpoint, 'TidalYang/SNav-7B', '04d236f5125dd8783e0547188b6d664bfe94e45b'),
        (vision_tower, 'google/siglip-so400m-patch14-384', '9fdffc58afc957d1a03a25b10dba0329ab15c2a3'),
    ]:
        root = Path(root)
        config = root / 'config.json'
        index = root / 'model.safetensors.index.json'
        shards = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        files = [config, *(root / name for name in sorted(shards))]
        if not config.is_file() or any(not p.is_file() or p.stat().st_size < 1_000_000 for p in files[1:]):
            raise FileNotFoundError('Full SNav model assets are missing: ' + str(root))
        rows.append({'repository': repository, 'revision': revision,
                     'files': [file_identity(p) for p in files]})
    return rows


class SNavTrajectory(DualVLNTrajectory):
    def payload(self):
        result = super().payload()
        result.update(method='SNav', run_kind='released_SNav_goal_free_HM3D')
        return result


def upstream_imports(upstream):
    sys.path[:0] = [str(upstream), str(upstream/'snav_training')]
    # Eval imports llava.*, whereas the same release vendors its trained model
    # under snav_llava.*. Alias module names only; no weights/layers are replaced.
    for suffix in ('', '.constants', '.conversation', '.mm_utils', '.model', '.model.builder',
                   '.model.multimodal_encoder', '.model.multimodal_encoder.builder',
                   '.model.multimodal_encoder.siglip_encoder', '.model.llava_arch'):
        sys.modules['llava'+suffix] = importlib.import_module('snav_llava'+suffix)
    return importlib.import_module('evaluation.eval_snav')


def verify_release_tensors(model, checkpoint):
    import torch
    from safetensors import safe_open
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    state = model.state_dict()
    names = ['lm_head.weight','model.embed_tokens.weight','model.image_newline',
             'model.layers.0.self_attn.q_proj.weight','model.layers.27.self_attn.o_proj.weight',
             'model.mm_projector.0.weight']
    names += [n for n in index if 'vision_tower' in n and n.endswith('patch_embedding.weight')]
    checks = []
    for name in names:
        actual = state[name].detach().reshape(-1)[:32].cpu()
        with safe_open(checkpoint/index[name],framework='pt',device='cpu') as handle:
            expected = handle.get_tensor(name).reshape(-1)[:32].to(actual.dtype)
        if not torch.equal(actual,expected):
            raise RuntimeError('loaded tensor does not match released checkpoint: '+name)
        checks.append({'tensor':name,'verified_elements':len(actual),'matches':True})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('task','scene','asg','upstream','checkpoint','vision-tower','output','evaluator-python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=70)
    args = p.parse_args()
    head = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != AUDITED_NAVSPACE_HEAD:
        raise RuntimeError('NavSpace checkout differs from the pinned SNav release')
    release_assets(args.checkpoint, args.vision_tower)
    from reconav.baselines.common.robot import install_habitat_profile
    install_habitat_profile(args.output / 'robot_geometry.json')
    args.output.mkdir(parents=True, exist_ok=True)
    item = load_policy_input(args.task)
    native = upstream_imports(args.upstream)
    import torch
    from habitat_sim.utils.common import quat_from_coeffs
    from snav_llava.model.builder import load_pretrained_model
    from snav_llava.model.multimodal_encoder import builder as vt
    from snav_llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    from snav_llava.model import llava_arch
    from evaluation.common import create_simulator, get_rgb, ensure_size_rgb
    original_builder = vt.build_vision_tower
    def builder(cfg, **kw):
        if 'siglip' in str(getattr(cfg,'mm_vision_tower','')).lower():
            return SigLipVisionTower(cfg.mm_vision_tower, vision_tower_cfg=cfg, **kw)
        return original_builder(cfg, **kw)
    vt.build_vision_tower = llava_arch.build_vision_tower = builder
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    # Keep upstream's -1 EGL selection: this host provides Mesa CPU rendering.
    # CUDA model inference remains isolated by the launcher's device selection.
    sim, agent = create_simulator(args.scene, 384, 384, False, -1)
    recorder = SNavTrajectory(args.output/'trajectory.json', item.benchmark_task_id)
    calls = []
    started = time.monotonic()
    try:
        if not sim.pathfinder.is_navigable(np.asarray(item.start_position, dtype=np.float32)):
            raise RuntimeError('unsupported exact start; no snapping')
        state = agent.get_state()
        state.position = np.asarray(item.start_position, dtype=np.float32)
        state.rotation = quat_from_coeffs(heading_to_habitat_xyzw(item.start_heading))
        agent.set_state(state)
        recorder.reset(agent.get_state())
        tokenizer, model, processor, _ = load_pretrained_model(str(args.checkpoint), None, 'llava_qwen',
            torch_dtype='bfloat16', device_map={'':0}, attn_implementation='sdpa',
            overwrite_config={'mm_vision_tower':str(args.vision_tower)})
        if model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
            raise RuntimeError('Released Qwen checkpoint requires untied input/output embeddings')
        tensor_checks = verify_release_tensors(model,args.checkpoint)
        model.eval()
        write(args.output/'policy_input_manifest.json', {'instruction':item.instruction,
            'start_position':list(item.start_position),'start_heading':item.start_heading,
            'checkpoint_files':[file_identity(f) for f in sorted(args.checkpoint.glob('*.safetensors'))],
            'forbidden_benchmark_fields_exported':[], 'max_steps':args.max_steps,
            'history_frames':16, 'future_steps_prompt':6, 'actions_per_inference':4,
            'frame_resolution':[384,384], 'hfov':120, 'original_prompt_and_parser':True,
            'transformers_version':__import__('transformers').__version__,
            'input_output_embeddings_tied':False,
            'release_tensor_checks':tensor_checks,
            'rendering':'upstream EGL -1, Mesa llvmpipe CPU; model on isolated CUDA device'})
        original_generate = model.generate
        original_prepare = model.prepare_inputs_labels_for_multimodal
        effective = {}
        def prepare(*a, **kw):
            result = original_prepare(*a, **kw)
            if result[4] is not None:
                effective['input_tokens'] = int(result[4].shape[1])
            return result
        model.prepare_inputs_labels_for_multimodal = prepare
        def generate(input_ids, *a, **kw):
            effective.clear()
            start = time.monotonic()
            result = original_generate(input_ids, *a, **kw)
            count = int(result.shape[-1])
            if count > input_ids.shape[-1]:
                count -= int(input_ids.shape[-1])
            calls.append({'provider_called':False, 'provider':'local-released-checkpoint',
                'seconds':time.monotonic()-start, 'text_input_tokens':int(input_ids.shape[-1]),
                'multimodal_input_tokens':effective.get('input_tokens'), 'output_tokens':count,
                'images':len(kw['images']), 'max_new_tokens':kw['max_new_tokens']})
            write(args.output/'calls.json', calls)
            return result
        model.generate = generate
        history = [ensure_size_rgb(get_rgb(sim.get_sensor_observations()),384,384)]
        steps, stopped = 0, False
        while steps < args.max_steps and not stopped:
            raw = native.snav_inference(model, tokenizer, processor, 'qwen_1_5', history,
                item.instruction, 6, 16, 128, 'cuda:0', torch.bfloat16)
            actions = native.parse_actions_trainaligned(raw, max_actions=4)
            calls[-1].update(response=raw, parsed_actions=actions, parser_fallback_stop=not bool(actions))
            write(args.output/'calls.json', calls)
            for action in actions or ['stop']:
                if steps >= args.max_steps:
                    break
                before = copy.deepcopy(agent.get_state())
                if action == 'stop':
                    recorder.action('stop', before, before, native_stop=True)
                    if not actions:
                        recorder.termination = 'native_parser_empty_action_stop'
                        recorder.write()
                    stopped = True
                    break
                agent.act(action)
                recorder.action(action, before, agent.get_state())
                steps += 1
                history.append(ensure_size_rgb(get_rgb(sim.get_sensor_observations()),384,384))
        if not stopped:
            recorder.finish_max_steps()
        write(args.output/'runtime_summary.json', {'status':'completed','run_completed':True,
            'inference_calls':len(calls), 'seconds':time.monotonic()-started,
            'termination':recorder.termination, 'cloud_api_calls':0})
    except BaseException as error:
        recorder.fail(error)
        raise
    finally:
        sim.close()
    from reconav.baselines.common.time_budget import mark_policy_complete
    mark_policy_complete()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    env['LD_LIBRARY_PATH'] = str(args.evaluator_python.parent.parent/'lib')
    subprocess.run([str(args.evaluator_python),'-m','reconav.baselines.canav.score_trajectory',
        '--task',str(args.task),'--asg',str(args.asg),'--trajectory',str(args.output/'trajectory.json'),
        '--output',str(args.output/'metrics.json')], env=env, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
