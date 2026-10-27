"""Resume-safe current 192-task launcher for official NavGPT/MapGPT agents.

Preflight checks inputs. Smoke/run execute full native policy episodes.
The same limits and scoring apply to smoke and run; only task count differs.
"""
from __future__ import annotations
from reconav import paths as release_paths
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from urllib.parse import urlparse
from reconav.baselines.common.protocol import ASG_ROOT, BENCHMARK, ROOT, SCENE_ROOT, COMMITS, LLM_METHODS, asg_identity, cell_identity, digest, discover, eligible_resume, file_digest, file_identity, policy_payload, read, scene_assets, write
from reconav.baselines.common.robot import CACHE, PROFILE, mesh_identity, require_asg_profile
from reconav.baselines.common.time_budget import add_arguments, TimePolicy, BudgetTimeout, ResourceExhausted, wait_process, resource_reason, incomplete, record_plan, reporting, diagnostic_read

def worker_environment(worker_python=None):
    release_paths.configure()
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', PYOPENGL_PLATFORM='osmesa', LIBGL_ALWAYS_SOFTWARE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH'] = (str(Path(worker_python).parent.parent / 'lib') if worker_python else str(release_paths.environment_path('navgpt', 'lib'))) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, [env.get(key), '127.0.0.1', 'localhost']))
    return env

def execute(command, output, env, timeout, budget=None):
    if budget is not None:
        env = budget.environment(env, output)
    with (output / 'worker.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return wait_process(proc, timeout, budget, output)

def summary(output):
    rows = []
    for p in sorted(Path(output).glob('*/*/*/*/status.json')):
        status = read(p)
        if p.parent.name != status.get('task_id'):
            continue
        metrics = read(p.parent / 'metrics.json') if (p.parent / 'metrics.json').exists() else {}
        calls = diagnostic_read(p.parent / 'calls.json', [])
        tokens = {name: sum((c.get('usage', {}).get(name, 0) for c in calls)) for name in ('input_tokens', 'output_tokens', 'total_tokens')}
        if status.get('token_usage_complete') is False:
            status['recorded_usage_lower_bound'] = dict(tokens)
            tokens = {name: None for name in tokens}
        rows.append({'cell': str(p.parent), **status, 'full_is': metrics.get('full_is'), 'hard_ns': metrics.get('hard_ns'), 'prefer': metrics.get('prefer'), 'route_replay_valid': metrics.get('route_replay_valid'), **tokens, 'reasoning_tokens': sum((c.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0) for c in calls)), 'accounted_llm_seconds': sum((c.get('seconds', 0) for c in calls)), 'actual_provider_calls': sum((c.get('provider_called', True) for c in calls)), 'actual_usd_upper_or_reserved': sum((c.get('charged_usd_upper', c.get('reserved_usd', 0)) for c in calls if c.get('provider_called', True)))})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method'], row['condition']].append(row)
    aggregates = []
    for (method, condition), group in sorted(grouped.items()):
        scored = [r for r in group if r.get('execution_status') == 'completed' and r.get('scoring_completed') and (not resource_reason(r))]
        means, denominators = ({}, {})
        for field in ('full_is', 'hard_ns', 'prefer', 'input_tokens', 'output_tokens', 'total_tokens', 'reasoning_tokens', 'seconds', 'accounted_llm_seconds'):
            values = [r[field] for r in scored if r.get(field) is not None]
            means[field] = sum(values) / len(values) if values else None
            denominators[field] = len(values)
        aggregates.append({'method': method, 'condition': condition, 'attempted': len(group), 'statuses': dict(Counter((r['execution_status'] for r in group))), 'scored': len(scored), 'means': means, 'denominators': denominators, 'actual_usd_upper_or_reserved': sum((r['actual_usd_upper_or_reserved'] for r in group))})
    result = {'attempted': len(rows), 'completed': sum((x.get('execution_status') == 'completed' for x in rows)), 'groups': aggregates, 'rows': rows, 'notes': ['No unsupported/partial/plumbing run is silently scored as zero.', 'Prefer is averaged only over scored preference tasks (non-null Prefer).', 'Replay tokens and LLM time remain accounted; actual provider spending is separate.', 'seconds is measured episode wall time; cached image loading differs from cold rendering.']}
    result['budget_reporting'] = reporting(output, rows)
    write(Path(output) / 'summary.json', result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'smoke', 'run', 'summary'])
    parser.add_argument('--method', action='append', choices=LLM_METHODS)
    parser.add_argument('--condition', action='append', choices=['native', 'asg'])
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--asg-root', type=Path, default=ASG_ROOT)
    parser.add_argument('--scene-root', type=Path, default=SCENE_ROOT)
    parser.add_argument('--assets', type=Path, default=(release_paths.cache_root() / 'external/assets'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--provider', choices=['local-qwen'], default='local-qwen')
    parser.add_argument('--base-url', default='http://127.0.0.1:8008/v1')
    parser.add_argument('--model', default='qwen3.8-27b', choices=['qwen3.8-27b'])
    parser.add_argument('--reasoning-effort', default='none', choices=['none'])
    parser.add_argument('--max-output-tokens', type=int, default=8192)
    parser.add_argument('--mapgpt-image-limit', type=int, default=64, help='Paper image-slot cap; does not prune history.')
    parser.add_argument('--max-episode-tokens', type=int, default=0, help='Local measured multimodal admission budget; 0 disables it.')
    parser.add_argument('--api-timeout', type=float, default=None, help='Transport timeout in seconds (default 900).')
    parser.add_argument('--episode-timeout', type=float, default=3600)
    parser.add_argument('--worker-python', type=Path, default=Path(str(release_paths.environment_path('navgpt', 'bin/python'))))
    parser.add_argument('--navgpt-deps', type=Path, default=release_paths.compat_path('navgpt'))
    parser.add_argument('--retry-failed', action='store_true')
    add_arguments(parser)
    args = parser.parse_args()
    args.api_timeout = args.api_timeout or 900
    if args.max_output_tokens < 1:
        parser.error('positive output limit required')
    if args.mapgpt_image_limit < 1 or args.max_episode_tokens < 0:
        parser.error('invalid image/token budget')
    if args.mode == 'summary':
        print(json.dumps(summary(args.output)), flush=True)
        return 0
    tasks = discover(args.benchmark, args.task, split=args.split)
    time_policy = TimePolicy(args, tasks)
    tasks = time_policy.tasks
    if args.mode == 'smoke' and (not args.task or len(tasks) > 6):
        parser.error('smoke requires 1–6 explicit task IDs; never implicitly select all 192')
    if args.mode == 'run' and (not args.method):
        parser.error('full run requires an explicit --method selection')
    methods = args.method or list(LLM_METHODS)
    conditions = args.condition or ['native', 'asg']
    args.output.mkdir(parents=True, exist_ok=True)
    env = worker_environment(args.worker_python)
    server_identity = None
    if args.mode in {'smoke', 'run'} and args.provider == 'local-qwen':
        parsed = urlparse(args.base_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.username:
            parser.error('local backend must use an unauthenticated loopback HTTP endpoint')
        with urlopen(args.base_url.rstrip('/') + '/models', timeout=10) as response:
            models = json.load(response)['data']
        selected = next((m for m in models if m['id'] == args.model))
        server_identity = {k: selected.get(k) for k in ('id', 'root', 'max_model_len')}
        for key in list(env):
            if any((word in key.upper() for word in ('API_KEY', 'OPENAI_KEY', 'GEMINI_KEY'))):
                env.pop(key)
    preflight = []
    native_check_file = CACHE / '0.3.3/exact_starts.json'
    if 'native' in conditions and not native_check_file.exists():
        from reconav.baselines.common.native_assets import prepare_native_assets
        prepare_native_assets('0.3.3')
    native_checks = {r['task_id']: r for r in read(native_check_file)['rows']} if native_check_file.exists() else {}
    prepared_graphs = {}
    for path in tasks:
        policy = policy_payload(path)
        assets = scene_assets(policy['scene_id'], args.scene_root, args.asg_root)
        required = [assets['scene'], assets['render_geometry'], *[assets['asg'] / n for n in ('entity.json', 'geometry.json', 'topology.json')]]
        if 'native' in conditions:
            required.append(assets['navmesh'])
        row = {'task_id': path.stem, 'scene_id': policy['scene_id'], 'missing_assets': [str(p) for p in required if not p.is_file()]}
        start_path = args.assets / policy['scene_id'] / 'starts' / (path.stem + '.json')
        start = read(start_path) if start_path.exists() else {}
        graph_path = args.assets / policy['scene_id'] / 'asg_graph.json'
        if policy['scene_id'] not in prepared_graphs:
            prepared_graphs[policy['scene_id']] = read(graph_path) if graph_path.exists() else {}
        graph = prepared_graphs[policy['scene_id']]
        if 'asg' in conditions:
            require_asg_profile(policy['scene_id'], graph)
        asg_current = not row['missing_assets'] and graph.get('source_identity') == asg_identity(assets['asg']) and (start.get('policy_fingerprint') == digest(policy)) and (start.get('graph_id') == graph.get('graph_id'))
        if 'asg' in conditions and (not asg_current):
            raise RuntimeError('missing/stale ASG start audit: ' + path.stem)
        row['asg_start_supported'] = start.get('supported') if asg_current else None
        native = native_checks.get(path.stem, {})
        aligned_mesh = mesh_identity(assets['scene'], '0.3.3') if 'native' in conditions else None
        native_current = aligned_mesh is not None and native.get('robot_profile') == PROFILE and (native.get('policy_fingerprint') == digest(policy)) and (native.get('navmesh_identity') == aligned_mesh['navmesh_identity'])
        row['native_start_supported'] = native.get('supported') if native_current else None
        if 'native' in conditions and (not native_current):
            raise RuntimeError('missing/stale aligned native start audit: ' + path.stem)
        preflight.append(row)
    write(args.output / 'preflight.json', {'tasks': len(tasks), 'rows': preflight})
    record_plan(args.output, [{'task_id': r['task_id'], 'method': method, 'condition': condition, 'supported': r[condition + '_start_supported'], **time_policy.settings(r['task_id'])} for r in preflight for method in methods for condition in conditions])
    if args.mode == 'preflight':
        print(json.dumps({'tasks': len(tasks), 'missing_asset_tasks': sum((bool(r['missing_assets']) for r in preflight)), 'asg_starts': sum((r['asg_start_supported'] is True for r in preflight)), 'native_starts': sum((r['native_start_supported'] is True for r in preflight)), 'asg_start_checks_missing': sum((r['asg_start_supported'] is None for r in preflight)), 'native_start_checks_missing': sum((r['native_start_supported'] is None for r in preflight))}))
        return 0 if not any((r['missing_assets'] for r in preflight)) else 2
    for path in tasks:
        for method in methods:
            for condition in conditions:
                policy = policy_payload(path)
                assets = scene_assets(policy['scene_id'], args.scene_root, args.asg_root)
                upstream = release_paths.upstream_path('NavGPT' if method == 'navgpt' else 'MapGPT')
                commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
                if not commit.startswith(COMMITS[method]):
                    raise RuntimeError(f'{method} upstream is not the audited revision')
                source_hashes = {name: file_digest(Path(__file__).resolve().parents[1] / "common" / name) for name in ('protocol.py', 'world.py', 'worker.py', 'client.py', 'robot.py', 'g0.py')}
                settings = {'model': args.model, 'reasoning_effort': args.reasoning_effort, 'robot_profile': PROFILE, 'api_timeout': args.api_timeout, 'max_output_tokens': args.max_output_tokens, 'max_steps': 10 if method == 'navgpt' else 15, 'upstream_commit': commit, 'adapter_sources': source_hashes, 'mode': 'full_episode'}
                settings.update(time_policy.settings(path.stem))
                source_hashes['time_budget.py'] = file_digest(Path(__file__).resolve().parents[1] / 'common/time_budget.py')
                source_hashes['run.py'] = file_digest(Path(__file__))
                source_hashes['context.py'] = file_digest(Path(__file__).resolve().parents[1] / 'common/context.py')
                source_hashes[method + '/agent.py'] = file_digest(Path(__file__).resolve().parents[1] / method / 'agent.py')
                if method == 'mapgpt':
                    settings['mapgpt_image_limit'] = args.mapgpt_image_limit
                if args.max_episode_tokens:
                    settings['max_episode_tokens'] = args.max_episode_tokens
                if args.provider == 'local-qwen':
                    settings.update(provider=args.provider, base_url=args.base_url, server_identity=server_identity)
                if condition == 'native':
                    settings['physical_mesh'] = mesh_identity(assets['scene'], '0.3.3')
                    assets['navmesh'] = Path(settings['physical_mesh']['navmesh_identity']['path'])
                else:
                    from reconav.baselines.common.g0 import identity as g0_identity
                    settings['physical_backend'] = g0_identity(prepared_graphs[policy['scene_id']])
                identity = cell_identity(path, method, condition, settings, assets)
                cell = args.output / method / condition / policy['scene_id'] / path.stem
                if eligible_resume(cell, identity):
                    print(f'resume completed {method}/{condition}/{path.stem}', flush=True)
                    continue
                if (cell / 'identity.json').exists():
                    if read(cell / 'identity.json') != identity:
                        raise RuntimeError(f'input/config/code changed; use a new output directory: {cell}')
                    if (cell / 'status.json').exists() and read(cell / 'status.json').get('execution_status') in {'timeout', 'resource_exhausted'}:
                        print(f'resume censored {method}/{condition}/{path.stem}', flush=True)
                        continue
                    if not args.retry_failed:
                        raise RuntimeError(f'previous incomplete attempt; inspect it and use --retry-failed: {cell}')
                    archive = cell.with_name(cell.name + '.attempt-' + str(time.time_ns()))
                    cell.rename(archive)
                cell.mkdir(parents=True, exist_ok=True)
                write(cell / 'identity.json', identity)
                write(cell / 'policy.json', policy)
                check = native_checks.get(path.stem, {})
                if condition == 'native' and check.get('policy_fingerprint') == digest(policy) and (check.get('navmesh_identity') == file_identity(assets['navmesh'])) and (check.get('supported') is False):
                    status = {'method': method, 'condition': condition, 'task_id': path.stem, **time_policy.settings(path.stem), 'execution_status': 'unsupported_input', 'reason': 'exact_start_outside_native_navmesh', 'snap_applied': False, 'api_calls': 0, 'scoring_completed': False}
                    write(cell / 'status.json', status)
                    write(cell / 'metrics.json', {'hard_ns': None, 'full_is': None, 'prefer': None, 'failure_category': 'unsupported_input'})
                    summary(args.output)
                    print(json.dumps(status), flush=True)
                    continue
                config = {**settings, 'method': method, 'condition': condition, 'policy': str(cell / 'policy.json'), 'output': str(cell), 'upstream': str(upstream), 'asset_dir': str(args.assets), 'navgpt_deps': str(args.navgpt_deps), 'navmesh': str(assets['navmesh']), 'render_geometry': str(assets['render_geometry']), 'native_graph_cache': str(CACHE / 'native_assets' / policy['scene_id'] / 'native_graph.json'), 'asg_source': str(assets['asg'])}
                write(cell / 'config.json', config)
                cmd = [str(args.worker_python), '-m', 'reconav.baselines.common.worker', '--config', str(cell / 'config.json')]
                status = {'method': method, 'condition': condition, 'task_id': path.stem, **time_policy.settings(path.stem), 'execution_status': 'running', 'scoring_completed': False}
                write(cell / 'status.json', status)
                budget = time_policy.deadline(path.stem)
                try:
                    code = execute(cmd, cell, env, args.episode_timeout, budget)
                    worker_status = read(cell / 'worker_status.json') if (cell / 'worker_status.json').exists() else {}
                    status.update(worker_status, worker_exit_code=code)
                    reason = resource_reason(status, cell)
                    if reason:
                        raise ResourceExhausted(reason)
                    if code != 0:
                        status['execution_status'] = worker_status.get('execution_status', 'failed')
                        raise RuntimeError(worker_status.get('error', 'worker failed; see worker.log'))
                    from reconav.baselines.canav.score_trajectory import score_recorded_trajectory
                    from reconav.scene.io import load_asg
                    metrics = score_recorded_trajectory(read(path), read(cell / 'trajectory.json'), load_asg(assets['asg'], data_source='hm3dsem'))
                    write(cell / 'metrics.json', metrics)
                    status['scoring_completed'] = True
                    status['execution_status'] = 'completed'
                except (BudgetTimeout, ResourceExhausted) as error:
                    incomplete(status, cell, 'timeout' if isinstance(error, BudgetTimeout) else 'resource_exhausted', str(error))
                except Exception as error:
                    status.update(execution_status='failed' if status['execution_status'] in {'running', 'completed'} else status['execution_status'], error=f'{type(error).__name__}: {error}')
                    write(cell / 'status.json', status)
                    summary(args.output)
                    print(json.dumps(status), flush=True)
                    return 2
                if budget is not None:
                    status['seconds'] = budget.elapsed()
                    status['token_usage_complete'] = status['execution_status'] == 'completed'
                write(cell / 'status.json', status)
                summary(args.output)
                print(json.dumps(status), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
