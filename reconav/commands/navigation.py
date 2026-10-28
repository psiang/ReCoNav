"""Run the paper main methods and paired grounding/assurance ablations."""
from __future__ import annotations
from reconav import paths as release_paths
import argparse
import json
import os
from pathlib import Path
from reconav.llm.client import current_model_config
from reconav.method.runtime.experiment import run_experiment, qwen_preflight, select_tasks
from reconav.method.runtime.models import METHODS, RuntimeConfig, method_provider, method_backbone, methods_for_provider, normalize_provider, normalize_method

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Real NL -> model -> method -> shared solver -> evaluator')
    run.add_argument('--tasks-root', type=Path, required=True, help='Contains <scene>/*.json, e.g. benchmark/tasks/hm3dsem')
    run.add_argument('--asg-root', type=Path, required=True, help='Contains <scene>/asg/')
    run.add_argument('--task-data-source', default='hm3dsem')
    run.add_argument('--asg-data-source', default='hm3dsem')
    run.add_argument('--output', type=Path, required=True, help='Independent experiment run directory')
    run.add_argument('--results-root', type=Path, help='Write experiment records as <root>/<source_model_method>/<scene>/<task>/result.json')
    run.add_argument('--object-vocabulary', type=Path, required=True)
    run.add_argument('--provider', choices=('qwen', 'openai', 'gpt', 'gemini'), default=os.environ.get('LLM_PROVIDER', 'qwen'), help='One provider owns every learned call in this run.')
    run.add_argument('--qwen-base-url', nargs='+', action='extend', default=None, help='One or more OpenAI-compatible Qwen endpoints. The flag may be repeated. Task workers are assigned round-robin across endpoints; one task stays on one endpoint.')
    run.add_argument('--qwen-model', default=os.environ.get('QWEN_SERVED_MODEL_NAME'))
    run.add_argument('--openai-base-url', default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    run.add_argument('--openai-model', default=os.environ.get('OPENAI_MODEL'))
    run.add_argument('--openai-reasoning-effort', default=os.environ.get('OPENAI_REASONING_EFFORT', 'high'))
    run.add_argument('--gemini-base-url', default=os.environ.get('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta'))
    run.add_argument('--gemini-model', default=os.environ.get('GEMINI_MODEL'))
    run.add_argument('--methods', nargs='+', type=normalize_method, choices=METHODS, default=None)
    run.add_argument('--task-workers', type=int, default=1, help='Run distinct tasks in parallel worker processes; use this as the primary GPU/CPU parallelism knob')
    run.add_argument('--method-workers', type=int, default=1, help='Within one task, run ready method pipelines concurrently; keep 1 for task-level parallel runs')
    run.add_argument('--scenes', nargs='*', default=[])
    run.add_argument('--task-ids', nargs='*', default=[])
    run.add_argument('--split', choices=('all', 'hard44'), default='all')
    run.add_argument('--dry-run', action='store_true', help='Show selected methods and tasks without model calls')
    run.add_argument('--limit', type=int)
    run.add_argument('--task-order', choices=('mechanism-first', 'filename'), default='mechanism-first')
    run.add_argument('--resume', action='store_true')
    run.add_argument('--prediction-only', action='store_true', help='Do not load oracle values or compute benchmark scores')
    run.add_argument('--scene-geometry', choices=('compact',), default='compact', help='Uniform task-independent ASG view for every scene-facing learned call')
    run.add_argument('--max-input-characters', type=int, default=512000, help='Anomaly guard; actual local context admission uses exact tokens, including feedback')
    run.add_argument('--tokenizer-path', help='Tokenizer matching the served Qwen model (defaults to /models root)')
    run.add_argument('--served-max-model-len', type=int, help='Optional local-model context ceiling; limited to the endpoint-advertised context length')
    run.add_argument('--context-safety-margin', type=int, default=4096)
    run.add_argument('--max-tokens', type=int, default=8192)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--temperature', type=float, default=0.7)
    run.add_argument('--top-p', type=float, default=0.8)
    run.add_argument('--top-k', type=int, default=20)
    run.add_argument('--min-p', type=float, default=0.0)
    run.add_argument('--presence-penalty', type=float, default=0.0)
    run.add_argument('--repetition-penalty', type=float, default=1.0)
    run.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)
    run.add_argument('--solver-seconds', type=float, default=90.0)
    run.add_argument('--resolution-seconds', type=float, default=360.0)
    run.add_argument('--max-assignments', type=int, default=128)
    run.add_argument('--repair-rounds', type=int, choices=(0, 1, 2), default=2, help='Bounded learned recovery rounds for ReCoNav/Forced-Early')
    run.add_argument('--feedback-rounds', type=int, choices=(0, 1, 2), default=2, help='Additional Direct+Feedback retries after the shared Direct first attempt')
    run.add_argument('--reference-seconds', type=float, default=90.0)
    run.add_argument('--geometry-cache-root', type=Path, default=release_paths.cache_root() / 'geometry')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.reference_seconds <= 0:
        parser.error('--reference-seconds must be positive')
    if args.task_workers < 1:
        parser.error('--task-workers must be positive')
    if args.method_workers < 1:
        parser.error('--method-workers must be positive')
    if args.context_safety_margin < 0:
        parser.error('--context-safety-margin must not be negative')
    if args.served_max_model_len is not None and args.served_max_model_len < 1:
        parser.error('--served-max-model-len must be positive')
    provider = normalize_provider(args.provider)
    for name, value in (("QWEN_TIMEOUT_S", "1800"), ("OPENAI_TIMEOUT_S", "240"), ("GEMINI_TIMEOUT_S", "600")):
        os.environ.setdefault(name, value)
    if provider != 'qwen' and (args.tokenizer_path is not None or args.served_max_model_len is not None):
        parser.error('--tokenizer-path and --served-max-model-len require the local qwen provider (Qwen or Gemma)')
    methods = tuple(args.methods or methods_for_provider(provider))
    if any((method_provider(method) != provider for method in methods)):
        parser.error('--methods must all use the same provider selected by --provider')
    from reconav.benchmark.selection import task_ids
    try:
        args.task_ids = task_ids(args.tasks_root, args.split, args.task_ids)
        if args.dry_run:
            tasks = select_tasks(args.tasks_root, args.scenes, args.task_ids, args.limit, args.task_order)
            print(json.dumps({'methods': list(methods), 'split': args.split, 'task_count': len(tasks),
                              'task_ids': [request.task_id for _, _, request in tasks],
                              'model_calls': 0, 'output': str(args.output.resolve())}, indent=2))
            return
    except (OSError, ValueError) as error:
        parser.error(str(error))
    os.environ['HM3DSEM_OBJECT_VOCABULARY'] = str(args.object_vocabulary.resolve())
    os.environ['LLM_PROVIDER'] = provider
    os.environ['NAV_GEOMETRY_CACHE_ROOT'] = str(args.geometry_cache_root.resolve())
    tokenizer_path = None
    served_max_model_len = None
    qwen_base_urls: tuple[str, ...] = ()
    if provider == 'qwen':
        requested_qwen_urls = args.qwen_base_url or [os.environ.get('QWEN_BASE_URL', 'http://127.0.0.1:8008/v1')]
        qwen_base_urls = tuple(dict.fromkeys((str(url).rstrip('/') for url in requested_qwen_urls if str(url).strip())))
        if not qwen_base_urls:
            parser.error('--qwen-base-url requires at least one endpoint')
        if args.qwen_model:
            os.environ['QWEN_SERVED_MODEL_NAME'] = args.qwen_model
        preflights = [qwen_preflight(url, args.qwen_model) for url in qwen_base_urls]
        served_models = {item['served_model'] for item in preflights}
        if len(served_models) != 1:
            parser.error('all Qwen endpoints must serve the same model: ' + repr(sorted(served_models)))
        runtime_signatures = {(item.get('runtime_model', {}).get('root'), item.get('runtime_model', {}).get('parent')) for item in preflights}
        if len(runtime_signatures) != 1:
            parser.error(f'all Qwen endpoints must expose the same model root/parent signature, got {sorted(map(repr, runtime_signatures))}')
        served_model = next(iter(served_models))
        os.environ['QWEN_SERVED_MODEL_NAME'] = served_model
        os.environ['QWEN_BASE_URL'] = qwen_base_urls[0]
        runtime_models = [item.get('runtime_model', {}) for item in preflights]
        local_identity = ' '.join([served_model, *[str(item.get('root', '')) for item in runtime_models]]).lower()
        backbone = 'gemma' if 'gemma' in local_identity else 'qwen'
        if args.methods is None:
            methods = methods_for_provider(backbone)
        elif any((method_backbone(method) != backbone for method in methods)):
            parser.error(f'method backbone must match the served {backbone} model')
        max_model_lens = [int(item['max_model_len']) for item in runtime_models if item.get('max_model_len') is not None]
        if len(max_model_lens) != len(runtime_models):
            parser.error('every Qwen endpoint /models response must expose max_model_len')
        served_max_model_len = min(max_model_lens)
        runtime_model = dict(runtime_models[0])
        runtime_model['max_model_len'] = served_max_model_len
        if args.served_max_model_len is not None:
            served_max_model_len = min(served_max_model_len, args.served_max_model_len)
        tokenizer_path = args.tokenizer_path or runtime_model.get('root') or runtime_model.get('parent')
        if not tokenizer_path:
            parser.error('Qwen /models must expose tokenizer root/parent, or pass --tokenizer-path')
        model_info = {'provider': 'qwen', 'endpoint': qwen_base_urls[0], 'endpoints': list(qwen_base_urls), 'served_model': served_model, 'runtime_model': runtime_model, 'services': preflights}
    elif provider == 'openai':
        os.environ['OPENAI_BASE_URL'] = args.openai_base_url
        if args.openai_model:
            os.environ['OPENAI_MODEL'] = args.openai_model
        os.environ['OPENAI_REASONING_EFFORT'] = args.openai_reasoning_effort
        try:
            runtime_model = current_model_config('openai')
        except ValueError as exc:
            parser.error(str(exc))
        if not os.environ.get('OPENAI_API_KEY'):
            parser.error('OPENAI_API_KEY is required for --provider openai')
        model_info = {'provider': 'openai', 'endpoint': args.openai_base_url, 'served_model': runtime_model['model'], 'runtime_model': runtime_model}
    else:
        os.environ['GEMINI_BASE_URL'] = args.gemini_base_url
        if args.gemini_model:
            os.environ['GEMINI_MODEL'] = args.gemini_model
        try:
            runtime_model = current_model_config('gemini')
        except ValueError as exc:
            parser.error(str(exc))
        if not os.environ.get('GEMINI_API_KEY'):
            parser.error('GEMINI_API_KEY is required for --provider gemini')
        model_info = {'provider': 'gemini', 'endpoint': args.gemini_base_url, 'served_model': runtime_model['model'], 'runtime_model': runtime_model}
    timeout_variable = {'qwen': 'QWEN_TIMEOUT_S', 'openai': 'OPENAI_TIMEOUT_S', 'gemini': 'GEMINI_TIMEOUT_S'}[provider]
    model_info['request_timeout_seconds'] = float(os.environ[timeout_variable])
    config = RuntimeConfig(max_tokens=args.max_tokens, seed=args.seed, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty, enable_thinking=args.enable_thinking, max_input_characters=args.max_input_characters, scene_geometry=args.scene_geometry, solver_seconds=args.solver_seconds, resolution_seconds=args.resolution_seconds, max_assignments=args.max_assignments, repair_rounds=args.repair_rounds, feedback_rounds=args.feedback_rounds, tokenizer_name_or_path=None if tokenizer_path is None else str(tokenizer_path), served_max_model_len=None if served_max_model_len is None else int(served_max_model_len), context_safety_margin=args.context_safety_margin)
    summary = run_experiment(tasks_root=args.tasks_root, asg_root=args.asg_root, task_data_source=args.task_data_source, asg_data_source=args.asg_data_source, output=args.output, results_root=args.results_root, methods=methods, config=config, scenes=args.scenes, task_ids=args.task_ids, limit=args.limit, resume=args.resume, prediction_only=args.prediction_only, reference_seconds=args.reference_seconds, model_info=model_info, method_workers=args.method_workers, task_workers=args.task_workers, qwen_base_urls=qwen_base_urls, task_order=args.task_order)
    print(json.dumps(summary['main_table_metrics'], indent=2, ensure_ascii=False))
    if summary.get('provider_interruption'):
        raise SystemExit('Provider quota exhausted. Restore quota, then repeat the same command with --resume; saved results are preserved.')
    if not args.prediction_only and (not summary['valid_for_reporting']):
        raise SystemExit('Run is incomplete: inspect summary.health, correct infrastructure/evaluation errors, and resume.')
if __name__ == '__main__':
    main()
