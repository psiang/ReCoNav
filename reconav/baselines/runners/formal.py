"""Explicit, resumable Lang2LTL-2 / LIMP + ASG benchmark runner.

Policy workers only read allow-listed policy JSON and scene maps. The parent
alone reads full tasks when calling the common evaluator after execution.
"""
from reconav import paths as release_paths
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from reconav.baselines.common.protocol import ROOT, BENCHMARK, asg_identity, digest, discover, file_digest, file_identity, policy_payload, read, scene_assets, write
from reconav.baselines.common.robot import PROFILE, require_asg_profile
from reconav.baselines.common.time_budget import add_arguments, TimePolicy, BudgetTimeout, ResourceExhausted, resource_reason, incomplete, record_plan, mark_policy_complete, TERMINAL

def worker(config_path):
    from reconav.baselines.common.formal_language import Frontend, PredictionFailure
    from reconav.baselines.common.formal_scene import landmarks, write_trajectory
    from reconav.baselines.common.world import World
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('formal workers must remain CPU-only')
    config = read(config_path)
    output = Path(config['output'])
    started = time.monotonic()
    frontend = None
    embeddings = None
    status = {'execution_status': 'running', 'cloud_generative_calls': 0}
    try:
        world = World(config)
        records, unlocated = landmarks(world, Path(config['asg_source']))
        write(output / 'scene_input_manifest.json', {'scene_only': True, 'landmarks': records, 'unlocated_semantic_entities': unlocated, 'gold_task_fields': [], 'search_graph': 'provided ASG G0, exact start attachment; no navigation oracle'})
        frontend = Frontend(config['method'], world.policy, output, config['base_url'], config.get('frontend_cache'), config.get('frontend_cache_cell'))
        with (output / 'native.stdout.log').open('w') as log, contextlib.redirect_stdout(log):
            try:
                language = frontend.run()
                if config['method'] == 'lang2ltl2':
                    from reconav.baselines.common.embeddings import Embeddings
                    from reconav.baselines.lang2ltl2.grounding import ground
                    from reconav.baselines.common.ltl_graph import plan
                    embeddings = Embeddings(config['embedding_cache'], config['embedding_ledger'], config['embedding_key_file'], config['embedding_budget_usd'])
                    bindings = ground(language, records, world.policy, output, embeddings, config['grounding_cache'])
                    write(output / 'embedding_calls.json', embeddings.calls)
                    path, planning = plan(world, language['lifted_formula'], bindings, records)
                else:
                    from reconav.baselines.limp.asg import run
                    path, planning = run(language, records, world, output)
                write(output / 'planning.json', planning)
                termination = planning['termination']
            except PredictionFailure as error:
                termination = 'method_prediction_failure'
                path = ['START']
                write(output / 'prediction_failure.json', {'error': str(error), 'semantic_retry_count': 0, 'scope': 'completed first method prediction, not a transport or adapter exception'})
            trajectory = write_trajectory(world, output, config['method'], termination, path)
        mark_policy_complete()
        status.update(execution_status='completed', run_completed=True, termination=termination, executed_actions=trajectory['executed_actions'], path_length_m=trajectory['path_length_m'])
    except Exception as error:
        status.update(execution_status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    calls = frontend.client.calls if frontend else []
    if embeddings is not None:
        write(output / 'embedding_calls.json', embeddings.calls)
    status.update(seconds=time.monotonic() - started, api_calls=sum((c.get('provider_called', True) for c in calls)), nominal_api_calls=len(calls), unknown_usage_calls=sum(('usage' not in c for c in calls)))
    status['cached_nominal_language_seconds'] = sum((c.get('seconds', 0) for c in calls if c.get('cache_source')))
    for key in ('input_tokens', 'output_tokens', 'total_tokens'):
        status[key] = sum((c.get('usage', {}).get(key, 0) for c in calls))
    write(output / 'worker_status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['execution_status'] == 'completed' else 2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'smoke', 'run', 'worker', 'summary'])
    parser.add_argument('--method', choices=['lang2ltl2', 'limp'])
    parser.add_argument('--condition', choices=['asg'], default='asg')
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--asset-dir', type=Path, default=(release_paths.cache_root() / 'external/assets'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8009/v1')
    parser.add_argument('--worker-python', type=Path, default=Path(sys.executable),
                        help='Python environment containing the original formal-policy dependencies')
    parser.add_argument('--frontend-cache', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--embedding-budget-usd', type=float, default=0.05)
    parser.add_argument('--embedding-key-file', type=Path, default=None)
    parser.add_argument('--embedding-cache', type=Path,
                        default=release_paths.cache_root() / 'external/formal/embeddings')
    parser.add_argument('--grounding-cache', type=Path,
                        default=release_paths.cache_root() / 'external/formal/original_reg_spg')
    add_arguments(parser)
    args = parser.parse_args()
    if args.mode == 'worker':
        return worker(args.config)
    from reconav.baselines.runners.native import execute, summarize
    if args.mode == 'summary':
        print(json.dumps(summarize(args.output)))
        return 0
    if not args.method or not args.output:
        parser.error('explicit method and output are required; no implicit matrix rerun')
    tasks = discover(args.benchmark, args.task, split=args.split)
    time_policy = TimePolicy(args, tasks)
    tasks = time_policy.tasks
    if time_policy.enabled and args.frontend_cache:
        parser.error('timed protocol requires fresh language predictions; no --frontend-cache')
    if args.mode == 'smoke' and (not args.task or len(tasks) > 6):
        parser.error('smoke requires 1-6 explicit task IDs')
    rows = []
    from reconav.baselines.common.client import local_model_identity
    server_identity = local_model_identity(args.base_url)
    source_root = release_paths.upstream_path('Lang2LTL-2' if args.method == 'lang2ltl2' else 'robotlimp')
    source_files = [source_root / n for n in ('srer.py', 'reg.py', 'spg.py', 'lt_s2s_sup_tcd.py')] if args.method == 'lang2ltl2' else [source_root / n for n in ('limp/language/llm4tl.py', 'limp/utils/gen_utils.py', 'limp/planner/multi_level_planner.py', 'limp/planner/fmt.py', 'limp/language/temporal_logic/dfa.py', 'limp/language/temporal_logic/ltl_progression.py', 'open-spatial-grounding/osg/spatial_relationships.py')]
    source_files.append(release_paths.resource_path('lang2ltl2_srer_prompt.txt') if args.method == 'lang2ltl2' else source_root / 'limp/language/prompts/singlestage_limp_prompt.txt')
    sources = {str(p): file_digest(p) for p in source_files + list((ROOT / 'reconav/baselines/common').glob('*.py')) + list((ROOT / 'reconav/baselines/runners').glob('*.py')) + list((ROOT / 'reconav/baselines' / ('lang2ltl2' if args.method == 'lang2ltl2' else 'limp')).glob('*.py'))}
    for task in tasks:
        policy = policy_payload(task)
        assets = scene_assets(policy['scene_id'])
        graph = read(args.asset_dir / policy['scene_id'] / 'asg_graph.json')
        require_asg_profile(policy['scene_id'], graph)
        start = read(args.asset_dir / policy['scene_id'] / 'starts' / (task.stem + '.json'))
        if graph['source_identity'] != asg_identity(assets['asg']) or start['policy_fingerprint'] != digest(policy) or start['graph_id'] != graph['graph_id']:
            raise RuntimeError('stale ASG graph/start audit: ' + task.stem)
        identity = {'method': args.method, 'condition': 'asg', 'policy': policy, 'task_sha256': file_digest(task), 'graph_id': graph['graph_id'], 'exact_start': start, 'sources': sources, 'robot_profile': PROFILE, 'generative_model': 'qwen3.8-27b/none', 'base_url': args.base_url, 'server_identity': server_identity, 'protocol': 'original-formal-frontends-ASG-graph-v1'}
        identity.update(time_policy.settings(task.stem))
        from reconav.baselines.common.g0 import identity as g0_identity
        identity['physical_backend'] = g0_identity(graph)
        if args.method == 'lang2ltl2':
            identity['T5_assets'] = [file_identity(p) for p in sorted((release_paths.model_root() / 'lang2ltl2/checkpoint-best').glob('*')) if p.is_file()]
        rows.append({'task': str(task), 'supported': start['supported'], 'identity': identity})
    write(args.output / 'preflight.json', {'total': len(rows), 'supported': sum((r['supported'] for r in rows)), 'rows': rows})
    record_plan(args.output, [{'task_id': Path(r['task']).stem, 'method': args.method, 'condition': 'asg', 'supported': r['supported'], **time_policy.settings(Path(r['task']).stem)} for r in rows])
    if args.mode == 'preflight':
        print({'tasks': len(rows), 'supported': sum((r['supported'] for r in rows))})
        return 0
    release_paths.configure()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT))
    for key in list(env):
        if any((s in key.upper() for s in ('API_KEY', 'OPENAI_KEY', 'GEMINI_KEY'))):
            env.pop(key, None)
    if args.method == 'lang2ltl2':
        embedding_key = os.environ.get('RECONAV_EMBEDDING_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if embedding_key:
            env['RECONAV_EMBEDDING_API_KEY'] = embedding_key
    cache = release_paths.cache_root() / 'external/formal'
    for row in rows:
        task = Path(row['task'])
        policy = row['identity']['policy']
        assets = scene_assets(policy['scene_id'])
        cell = args.output / args.method / policy['scene_id'] / task.stem
        automatic_frontend_cache = None
        if (cell / 'identity.json').exists():
            if read(cell / 'identity.json') != row['identity']:
                raise RuntimeError('code/config changed; use a new output directory: ' + str(cell))
            old = read(cell / 'status.json') if (cell / 'status.json').exists() else {}
            retry_embedding = args.retry_failed and resource_reason(old) == 'dollar_budget_exhausted'
            if old.get('execution_status') in TERMINAL and not retry_embedding:
                print('resume ' + task.stem, flush=True)
                continue
            if not args.retry_failed:
                raise RuntimeError('inspect failed cell, then explicitly use --retry-failed')
            automatic_frontend_cache = cell.with_name(cell.name + '.attempt-' + str(time.time_ns()))
            cell.rename(automatic_frontend_cache)
        write(cell / 'identity.json', row['identity'])
        write(cell / 'policy.json', policy)
        status = {'method': args.method, 'condition': 'asg', 'task_id': task.stem, 'execution_status': 'running', 'scoring_completed': False, **time_policy.settings(task.stem)}
        if not row['supported']:
            status.update(execution_status='unsupported_input', reason='exact_start_not_connected_to_G0', api_calls=0)
            write(cell / 'status.json', status)
            continue
        config = {'method': args.method, 'condition': 'asg', 'output': str(cell), 'policy': str(cell / 'policy.json'), 'robot_profile': PROFILE, 'asset_dir': str(args.asset_dir), 'asg_source': str(assets['asg']), 'base_url': args.base_url, 'frontend_cache': str(args.frontend_cache) if args.frontend_cache else None, 'frontend_cache_cell': str(automatic_frontend_cache) if automatic_frontend_cache and (not time_policy.enabled) else None, 'embedding_cache': str(args.embedding_cache), 'embedding_ledger': str(cache / 'embedding_budget.json'), 'embedding_key_file': str(args.embedding_key_file) if args.embedding_key_file else None, 'embedding_budget_usd': args.embedding_budget_usd, 'grounding_cache': str(args.grounding_cache)}
        write(cell / 'config.json', config)
        write(cell / 'status.json', status)
        budget = time_policy.deadline(task.stem)
        try:
            execute([str(args.worker_python), '-m', 'reconav.baselines.runners.formal', 'worker', '--config', str(cell / 'config.json')], cell, 'worker', env, args.timeout, budget)
            status.update(read(cell / 'worker_status.json'))
            reason = resource_reason(status, cell)
            if reason:
                raise ResourceExhausted(reason)
            if status['execution_status'] != 'completed':
                raise RuntimeError('formal policy did not complete')
            from reconav.baselines.canav.score_trajectory import score_recorded_trajectory
            from reconav.scene.io import load_asg
            metrics = score_recorded_trajectory(read(task), read(cell / 'trajectory.json'), load_asg(assets['asg'], data_source='hm3dsem'))
            write(cell / 'metrics.json', metrics)
            status['scoring_completed'] = True
        except BudgetTimeout as error:
            incomplete(status, cell, 'timeout', str(error))
        except Exception as error:
            if (cell / 'worker_status.json').exists():
                status.update(read(cell / 'worker_status.json'))
            reason = str(error) if isinstance(error, ResourceExhausted) else resource_reason(status, cell)
            if reason:
                incomplete(status, cell, 'resource_exhausted', reason)
            else:
                status.update(execution_status='failed', error=f'{type(error).__name__}: {error}')
        if budget is not None:
            status['seconds'] = budget.elapsed()
        status['token_usage_complete'] = status['execution_status'] == 'completed'
        write(cell / 'status.json', status)
        summarize(args.output)
        print(json.dumps(status), flush=True)
        if resource_reason(status) == 'dollar_budget_exhausted':
            print('Embedding budget exhausted. Increase --embedding-budget-usd and repeat with --retry-failed.', flush=True)
            return 2
        if status['execution_status'] == 'failed':
            return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
