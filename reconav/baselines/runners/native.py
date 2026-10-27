"""Launch CA-Nav, GC-VLN, SNav or DualVLN on selected benchmark tasks.

Use smoke for a sample episode and run for a full selection."""
from __future__ import annotations
from reconav import paths as release_paths
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from reconav.baselines.common.protocol import BENCHMARK, ROOT, asg_identity, digest, discover, file_digest, file_identity, policy_payload, read, scene_assets, write
from reconav.baselines.common.runtime import _environment, _preprocess_command, _runtime_command
from reconav.baselines.common.robot import CACHE, PROFILE, VERSIONS, mesh_identity
from reconav.baselines.common.g0 import validate_assets, identity as g0_identity
from reconav.baselines.common.time_budget import add_arguments, TimePolicy, BudgetTimeout, ResourceExhausted, wait_process, resource_reason, incomplete, record_plan, reporting, TERMINAL, diagnostic_read

def runtime_environment(method, gpu, base_url):
    old = os.environ.get('EXTERNAL_LOCAL_QWEN_URL')
    os.environ['EXTERNAL_LOCAL_QWEN_URL'] = base_url
    try:
        env = _environment(method, gpu)
    finally:
        if old is None:
            os.environ.pop('EXTERNAL_LOCAL_QWEN_URL', None)
        else:
            os.environ['EXTERNAL_LOCAL_QWEN_URL'] = old
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    if method == 'gcvln':
        env['PYTHONPATH'] = ':'.join(map(str, [ROOT, release_paths.upstream_path('GC-VLN'), release_paths.compat_path('gcvln'), release_paths.upstream_path('GC-VLN/third_party/GLIP'), release_paths.upstream_path('GC-VLN/third_party/ModelServer/src'), release_paths.runtime_path('gcvln/third_party/ModelServer/src'), release_paths.upstream_path('GC-VLN/third_party/habitat-lab')]))
    return env

def execute(command, cell, name, env, timeout, budget=None):
    env = dict(env)
    if budget is not None:
        env = budget.environment(env, cell)
    executable = Path(next((v for v in command if v.endswith('/bin/python')), command[0]))
    if executable.parent.name == 'bin':
        env['LD_LIBRARY_PATH'] = str(executable.parent.parent / 'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if env.get('RECONAV_BASELINE') == 'canav' or '/envs/canav/' in str(executable):
        env.update(__GLX_VENDOR_LIBRARY_NAME='mesa', LIBGL_ALWAYS_SOFTWARE='1', LD_PRELOAD=(str(release_paths.environment_path('canav', 'lib/libstdc++.so.6')) + ':' + os.environ.get('RECONAV_MESA_GLX_LIBRARY', 'libGLX_mesa.so.0')), PYTHONPATH=':'.join(map(str, [ROOT, release_paths.upstream_path('CA-Nav-code'), release_paths.upstream_path('habitat-lab-canav'), release_paths.upstream_path('GroundingDINO-canav'), release_paths.upstream_path('segment-anything-canav')])))
    with (cell / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        code = wait_process(proc, timeout, budget, cell,
                            fatal_log=cell / (name + '.log') if name == 'rollout' else None)
    if code:
        raise RuntimeError(f"{name} exited {code}; see {cell / (name + '.log')}")

def validate_native_completion(method, cell):
    summary = read(cell / 'runtime_summary.json')
    trajectory = read(cell / 'trajectory.json')
    if trajectory.get('runtime_exception') or summary.get('status') == 'runtime_failure':
        raise RuntimeError('Native exception is not a scored task failure')
    if method == 'snav':
        if not summary.get('run_completed') or not (cell / 'calls.json').exists():
            raise RuntimeError('SNav did not execute the released model')
    elif method == 'canav':
        if not summary.get('run_completed') or trajectory.get('termination') not in {'stop', 'native_max_steps', 'invalid_first_response'}:
            raise RuntimeError('CA-Nav did not complete its native policy')
    elif method == 'gcvln':
        instrumentation = summary.get('instrumentation', {})
        if instrumentation.get('runtime_exception_count') or instrumentation.get('environment_runtime_exceptions'):
            raise RuntimeError('GC-VLN swallowed a runtime exception before STOP')
    else:
        counters = summary.get('s2_inference_provenance', {})
        if counters.get('s2_exception_count') or summary.get('s2_failure_fallback_stop_observed_count'):
            raise RuntimeError('DualVLN model inference failed; no fake successful STOP')
    return (summary, trajectory)

def summarize(output):
    rows = []
    for path in sorted(output.glob('*/*/*/status.json')):
        row = read(path)
        if path.parent.name != row.get('task_id'):
            continue
        if row.get('execution_status') == 'completed' and (not resource_reason(row)):
            metrics = read(path.parent / 'metrics.json')
            row.update({k: metrics.get(k) for k in ('full_is', 'hard_ns', 'prefer', 'route_replay_valid')})
        rows.append(row)
    result = {'attempted': len(rows), 'rows': rows, 'groups': []}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        scored = [r for r in group if r['execution_status'] == 'completed' and r.get('scoring_completed', True) and (not resource_reason(r))]
        means, denominators = ({}, {})
        for key in ('full_is', 'hard_ns', 'prefer', 'seconds', 'input_tokens', 'output_tokens', 'total_tokens'):
            values = [r[key] for r in scored if r.get(key) is not None]
            means[key] = sum(values) / len(values) if values else None
            denominators[key] = len(values)
        result['groups'].append({'method': method, 'attempted': len(group), 'scored': len(scored), 'unsupported': sum((r['execution_status'] == 'unsupported_input' for r in group)), 'means': means, 'denominators': denominators})
    result['budget_reporting'] = reporting(output, rows)
    write(output / 'summary.json', result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'smoke', 'run', 'summary'])
    parser.add_argument('--method', choices=['dualvln', 'gcvln', 'canav', 'snav'])
    parser.add_argument('--condition', choices=['native', 'asg'], default='native')
    parser.add_argument('--asset-dir', type=Path, default=(release_paths.cache_root() / 'external/assets'))
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--server-port', type=int, default=7012)
    parser.add_argument('--local-base-url', default='http://127.0.0.1:8008/v1')
    parser.add_argument('--exact-starts', type=Path)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--preprocess-cache', type=Path, help='replay saved first CA-Nav/GC-VLN language predictions; freshly execute perception and navigation')
    add_arguments(parser)
    args = parser.parse_args()
    if args.mode == 'summary':
        print(json.dumps(summarize(args.output)), flush=True)
        return 0
    if not args.method:
        parser.error('select exactly one method; never implicitly rerun a matrix')
    if args.preprocess_cache and args.method not in {'canav', 'gcvln'}:
        parser.error('--preprocess-cache applies to CA-Nav and GC-VLN')
    if args.method == 'canav':
        from reconav.baselines.common.upstream import verify_recorded_source
        verify_recorded_source(release_paths.upstream_path('GroundingDINO-canav'), 'GroundingDINO-canav')
    if args.condition == 'asg' and args.method != 'gcvln':
        parser.error('this native dispatcher supports +ASG only for GC-VLN')
    tasks = discover(args.benchmark, args.task, split=args.split)
    time_policy = TimePolicy(args, tasks)
    tasks = time_policy.tasks
    if time_policy.enabled and args.preprocess_cache:
        parser.error('timed protocol requires fresh language predictions; no --preprocess-cache')
    if args.mode == 'smoke' and (not args.task or len(tasks) > 6):
        parser.error('smoke requires 1–6 explicit task IDs')
    starts = {}
    if args.condition == 'native':
        starts_path = args.exact_starts or CACHE / VERSIONS[args.method] / 'exact_starts.json'
        selected_scenes = {policy_payload(task)['scene_id'] for task in tasks}
        recorded_ids = {row['task_id'] for row in read(starts_path)['rows']} if starts_path.exists() else set()
        if args.exact_starts is None and (not starts_path.exists() or {task.stem for task in tasks} - recorded_ids):
            from reconav.baselines.common.native_assets import prepare_native_assets
            prepare_native_assets(VERSIONS[args.method], scene_ids=selected_scenes)
        start_doc = read(starts_path)
        if start_doc.get('robot_profile') != PROFILE or start_doc.get('habitat_version') != VERSIONS[args.method]:
            raise RuntimeError('exact-start audit has old robot size or another simulator version')
        starts = {r['task_id']: r for r in start_doc['rows']}
    from reconav.baselines.common.client import local_model_identity
    server_identity = local_model_identity(args.local_base_url) if args.method not in {'dualvln', 'snav'} and not args.preprocess_cache else None
    sources = {str(p.relative_to(ROOT)): file_digest(p) for folder in (ROOT / 'reconav/baselines' / args.method, ROOT / 'reconav/baselines/common', ROOT / 'reconav/baselines/runners') for p in sorted(folder.glob('*.py'))}
    rows = []
    for task in tasks:
        policy = policy_payload(task)
        assets = scene_assets(policy['scene_id'])
        physical_mesh = None
        if args.condition == 'asg':
            graph, check = validate_assets(policy, args.asset_dir, assets['asg'])
        else:
            physical_mesh = mesh_identity(assets['scene'], VERSIONS[args.method])
            assets['navmesh'] = Path(physical_mesh['navmesh_identity']['path'])
            check = starts.get(task.stem)
            if not check or check['policy_fingerprint'] != digest(policy) or any((check[k + '_identity'] != file_identity(assets[k]) for k in ('scene', 'navmesh'))):
                raise RuntimeError('missing/stale per-method exact-start audit: ' + task.stem)
        identity = {'task_sha256': file_digest(task), 'policy': policy, 'method': args.method, 'condition': args.condition, 'robot_profile': PROFILE, 'physical_mesh': physical_mesh, 'exact_start': check, 'adapter_sources': sources, 'asg': asg_identity(assets['asg']), 'language_model': 'qwen3.8-27b/none' if args.method not in {'dualvln', 'snav'} else 'released-checkpoint', 'local_base_url': args.local_base_url if args.method not in {'dualvln', 'snav'} else None}
        identity.update(time_policy.settings(task.stem))
        identity['server_identity'] = server_identity
        if args.method in {'canav', 'gcvln'}:
            identity['language_prediction_mode'] = 'saved_first_response' if args.preprocess_cache else 'fresh'
        if args.method == 'snav':
            from reconav.baselines.snav.run_single import release_assets
            runtime = release_paths.runtime_path('snav')
            identity['release_assets'] = release_assets(runtime / 'SNav-7B', runtime / 'siglip-so400m-patch14-384')
            identity['snav_protocol'] = {'max_steps': 70, 'history_frames': 16, 'actions_per_inference': 4, 'max_new_tokens': 128, 'dtype': 'bfloat16', 'attention': 'sdpa'}
        if args.method == 'dualvln':
            from reconav.baselines.dualvln.run_single import depth_checkpoint
            identity['depth_checkpoint'] = file_identity(depth_checkpoint(release_paths.runtime_path('dualvln')))
        if args.condition == 'asg':
            physical = read(args.asset_dir / policy['scene_id'] / 'physical_free_space.json')
            if physical['source_identity'] != asg_identity(assets['asg']) or physical['graph_id'] != graph['graph_id']:
                raise RuntimeError('stale physical ASG assets')
            identity['asg_physical_graph_id'] = physical['graph_id']
            identity['physical_backend'] = g0_identity(graph)
            identity['render_scene'] = file_identity(assets['scene'])
        rows.append({'task': str(task), 'supported': check['supported'], 'identity': identity})
    write(args.output / 'preflight.json', {'total': len(rows), 'supported': sum((r['supported'] for r in rows)), 'rows': rows})
    record_plan(args.output, [{'task_id': Path(r['task']).stem, 'method': args.method, 'condition': args.condition, 'supported': r['supported'], **time_policy.settings(Path(r['task']).stem)} for r in rows])
    if args.mode == 'preflight':
        print({'tasks': len(rows), 'supported': sum((r['supported'] for r in rows))}, flush=True)
        return 0
    env = runtime_environment(args.method, args.gpu, args.local_base_url)
    for row in rows:
        task = Path(row['task'])
        cell = args.output / args.method / task.parent.name / task.stem
        automatic_preprocess_cache = None
        if (cell / 'identity.json').exists():
            if read(cell / 'identity.json') != row['identity']:
                raise RuntimeError('inputs/code changed; use a new output directory: ' + str(cell))
            old = read(cell / 'status.json') if (cell / 'status.json').exists() else {}
            if old.get('execution_status') in TERMINAL:
                print('resume ' + task.stem, flush=True)
                continue
            if not args.retry_failed:
                raise RuntimeError('Inspect failed task and explicitly use --retry-failed: ' + str(cell))
            automatic_preprocess_cache = cell.with_name(cell.name + '.attempt-' + str(time.time_ns()))
            cell.rename(automatic_preprocess_cache)
        write(cell / 'identity.json', row['identity'])
        status = {'task_id': task.stem, 'method': args.method, 'condition': args.condition, **time_policy.settings(task.stem), 'execution_status': 'running', 'scoring_completed': False}
        if not row['supported']:
            status.update(execution_status='unsupported_input', api_calls=0, reason='exact_start_outside_native_navmesh' if args.condition == 'native' else 'exact_start_outside_G0', snap_applied=False)
            write(cell / 'metrics.json', {'full_is': None, 'hard_ns': None, 'prefer': None})
            write(cell / 'status.json', status)
            summarize(args.output)
            continue
        device_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free = {int(a): int(b) for a, b in (s.split(',') for s in device_rows.splitlines())}
        required_memory = 2048 if args.method == 'gcvln' and args.condition == 'asg' else 24000 if args.method in {'dualvln', 'snav'} else 12000
        if free.get(args.gpu, 0) < required_memory:
            raise RuntimeError('insufficient free GPU memory; no service was stopped')
        write(cell / 'status.json', status)
        started = time.monotonic()
        budget = time_policy.deadline(task.stem)
        if time_policy.enabled:
            automatic_preprocess_cache = None
        try:
            canav_invalid = False
            if args.method == 'canav':
                try:
                    from reconav.baselines.common.preprocess_cache import reuse_canav
                    cache_source = automatic_preprocess_cache or (args.preprocess_cache / args.method / task.parent.name / task.stem if args.preprocess_cache else None)
                    cached = cache_source and reuse_canav(cache_source, cell, row['identity']['policy'], release_paths.upstream_path('CA-Nav-code'))
                    if args.preprocess_cache and not cached:
                        raise FileNotFoundError('missing matching saved first prediction: ' + str(cache_source))
                    if not cached:
                        execute([sys.executable, '-m', 'reconav.baselines.canav.qwen_reply', '--task', str(task), '--upstream', str(release_paths.upstream_path('CA-Nav-code')), '--output', str(cell), '--base-url', args.local_base_url], cell, 'preprocess', env, args.timeout, budget)
                    else:
                        canav_invalid = read(cell / 'llm_reply_metadata.json')['status'] == 'invalid_first_response'
                except RuntimeError:
                    metadata = read(cell / 'llm_reply_metadata.json') if (cell / 'llm_reply_metadata.json').exists() else {}
                    if metadata.get('status') != 'invalid_first_response':
                        raise
                    canav_invalid = True
            if args.method == 'gcvln':
                from urllib.request import urlopen
                if args.condition == 'native':
                    with urlopen(f'http://127.0.0.1:{args.server_port}/', timeout=5) as response:
                        if json.load(response).get('process') != '/api/process':
                            raise RuntimeError('wrong GSAM2 service')
                try:
                    from reconav.baselines.common.preprocess_cache import reuse_gcvln
                    cache_source = automatic_preprocess_cache or (args.preprocess_cache / args.method / task.parent.name / task.stem if args.preprocess_cache else None)
                    cached = cache_source and reuse_gcvln(cache_source, cell, row['identity']['policy'])
                    if args.preprocess_cache and not cached:
                        raise FileNotFoundError('missing matching saved first prediction: ' + str(cache_source))
                    if not cached:
                        execute(_preprocess_command(args.method, task, cell), cell, 'preprocess', env, args.timeout, budget)
                except RuntimeError:
                    metadata = read(cell / 'preprocess_metadata.json') if (cell / 'preprocess_metadata.json').exists() else {}
                    if metadata.get('status') != 'invalid_first_response':
                        raise
            if args.method == 'snav':
                assets = scene_assets(task.parent.name)
                runtime = release_paths.runtime_path('snav')
                command = [str(runtime / 'venv/bin/python'), '-m', 'reconav.baselines.snav.run_single', '--task', str(task), '--scene', str(assets['scene']), '--asg', str(assets['asg']), '--upstream', str(release_paths.upstream_path('NavSpace')), '--checkpoint', str(runtime / 'SNav-7B'), '--vision-tower', str(runtime / 'siglip-so400m-patch14-384'), '--output', str(cell), '--evaluator-python', sys.executable]
            else:
                command = _runtime_command(args.method, task, cell, args.gpu, args.server_port)
            if args.condition == 'asg':
                command += ['--asg-policy-assets', str(args.asset_dir)]
            if canav_invalid:
                from reconav.baselines.common.trajectory_io import write_zero_motion_trajectory
                write_zero_motion_trajectory(cell, read(task), 'CA-Nav-Qwen', 'invalid_first_response')
                trajectory = read(cell / 'trajectory.json')
                trajectory.pop('runtime_exception')
                trajectory['run_kind'] = 'completed_invalid_model_prediction'
                write(cell / 'trajectory.json', trajectory)
                write(cell / 'runtime_summary.json', {'status': 'completed', 'run_completed': True, 'model_prediction_valid': False, 'native_rollout_skipped': 'unparseable model prediction'})
                if budget is not None:
                    left = budget.remaining()
                    if left is not None and left <= 0:
                        raise BudgetTimeout('episode wall budget exhausted')
                    budget.finished = time.monotonic()
                execute([sys.executable, '-m', 'reconav.baselines.canav.score_trajectory', '--task', str(task), '--asg', str(scene_assets(task.parent.name)['asg']), '--trajectory', str(cell / 'trajectory.json'), '--output', str(cell / 'metrics.json')], cell, 'score', env, args.timeout)
            else:
                execute(command, cell, 'rollout', env, args.timeout, budget)
            native, trajectory = validate_native_completion(args.method, cell)
            if args.method == 'gcvln' and 'recording' in native:
                status['recording'] = native['recording']
            reason = resource_reason({**status, 'termination': trajectory.get('termination')}, cell)
            if reason:
                raise ResourceExhausted(reason)
            status.update(execution_status='completed', scoring_completed=True, termination=trajectory['termination'], path_length_m=trajectory['path_length_m'], executed_actions=trajectory['executed_actions'])
        except BudgetTimeout as error:
            incomplete(status, cell, 'timeout', str(error))
        except Exception as error:
            reason = str(error) if isinstance(error, ResourceExhausted) else resource_reason(status, cell)
            if reason:
                incomplete(status, cell, 'resource_exhausted', reason)
            else:
                status.update(execution_status='failed', error=f'{type(error).__name__}: {error}')
        status['seconds'] = budget.elapsed() if budget is not None else time.monotonic() - started
        status['token_usage_complete'] = status['execution_status'] == 'completed'
        calls = diagnostic_read(cell / 'calls.json', [])
        status['api_calls'] = sum((c.get('provider_called', True) for c in calls))
        status['cached_nominal_language_seconds'] = sum((c.get('seconds', 0) for c in calls if c.get('cache_source')))
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            status[key] = sum((c.get('usage', {}).get(key, 0) for c in calls)) if status['token_usage_complete'] and args.method in {'gcvln', 'canav'} and all(('usage' in c for c in calls)) else None
        if not status['token_usage_complete']:
            status['recorded_usage_lower_bound'] = {key: sum((c.get('usage', {}).get(key, 0) for c in calls)) for key in ('input_tokens', 'output_tokens', 'total_tokens')}
        write(cell / 'status.json', status)
        summarize(args.output)
        print(json.dumps(status), flush=True)
        if status['execution_status'] == 'failed':
            return 2
    summarize(args.output)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
