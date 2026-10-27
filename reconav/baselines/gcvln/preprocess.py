"""Official Appendix-G GPT preprocessing and lossless runtime-schema lowering."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from reconav.baselines.canav.export_task import load_policy_input
DIRECTIONS = {'front', 'right', 'left', 'back', 'unknown'}
RELATIONS = {'right', 'left', 'through', 'weave', 'pass', 'near', 'back'}
PAPER_STAGE_FIELDS = {'waypoint position', 'connected nodes'}
PAPER_OBJECT_FIELDS = {'node', 'object position'}

def load_official_prompt(path: Path) -> tuple[str, str]:
    prompt = path.read_text(encoding='utf-8').strip()
    required = ('Parse navigation instructions into movement stages in JSON format', '"stage 1"', '"waypoint position"', '"connected nodes"', '"object position"', '<Instruction>')
    missing = [value for value in required if value not in prompt]
    if missing:
        raise ValueError(f'official Appendix-G prompt is missing markers: {missing}')
    if prompt.count('<Instruction>') != 1:
        raise ValueError('official prompt must contain exactly one <Instruction> placeholder')
    return (prompt, hashlib.sha256(prompt.encode('utf-8')).hexdigest())

def compose_request(prompt: str, instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be a non-empty string')
    if prompt.count('<Instruction>') != 1:
        raise ValueError('prompt placeholder count changed')
    return prompt.replace('<Instruction>', instruction)

def decode_paper_json(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch('```(?:json)?\\s*(.*?)\\s*```', value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError('Appendix-G output must be a top-level JSON object')
    return parsed


def validate_paper_dag(value: dict[str, Any]) -> None:
    if not value:
        raise ValueError('paper DAG must contain at least one stage')
    expected_stages = [f'stage {index}' for index in range(1, len(value) + 1)]
    if list(value) != expected_stages:
        raise ValueError(f'stages must be continuous and ordered: {expected_stages}')
    for stage_key, stage in value.items():
        if not isinstance(stage, dict) or set(stage) != PAPER_STAGE_FIELDS:
            raise ValueError(f'{stage_key} fields must be exactly {sorted(PAPER_STAGE_FIELDS)}')
        direction = stage['waypoint position']
        if direction not in DIRECTIONS:
            raise ValueError(f'invalid waypoint direction in {stage_key}: {direction!r}')
        nodes = stage['connected nodes']
        if not isinstance(nodes, list):
            raise ValueError(f'connected nodes must be a list in {stage_key}')
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or set(node) != PAPER_OBJECT_FIELDS:
                raise ValueError(f'object {index} in {stage_key} must have exactly {sorted(PAPER_OBJECT_FIELDS)}')
            if not isinstance(node['node'], str) or not node['node'].strip():
                raise ValueError(f'object {index} in {stage_key} has an empty node')
            relation = node['object position']
            if relation not in RELATIONS:
                raise ValueError(f'invalid relation in {stage_key} object {index}: {relation!r}')

def paper_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(((key, stage['waypoint position'], tuple(((node['node'], node['object position']) for node in stage['connected nodes']))) for key, stage in value.items()))

def runtime_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(((key.replace('_', ' ', 1), stage['navigation_direction'], tuple(((node['node'], node['relation']) for node in stage['connected_nodes']))) for key, stage in value.items()))

def lower_paper_dag(value: dict[str, Any]) -> dict[str, Any]:
    validate_paper_dag(value)
    lowered: dict[str, Any] = {}
    for index, (stage_key, stage) in enumerate(value.items(), start=1):
        if stage_key != f'stage {index}':
            raise AssertionError('stage ordering changed during lowering')
        lowered[f'stage_{index}'] = {'navigation_direction': stage['waypoint position'], 'connected_nodes': [{'node': node['node'], 'relation': node['object position']} for node in stage['connected nodes']]}
    if paper_signature(value) != runtime_signature(lowered):
        raise AssertionError('deterministic lowering changed DAG semantics')
    return lowered

class ParserEnvironmentError(RuntimeError):
    """The released parser could not execute in its native environment."""


def released_parser_gate(upstream: Path, runtime_path: Path, python: Path | None = None) -> dict[str, Any]:
    code = """
import json,sys,traceback
try:
    from src.solver.instruction_graph import get_instruction_graphs
    from src.solver.navigation_tree import NavigationTree
except Exception:
    traceback.print_exc()
    sys.exit(2)
try:
    value=json.load(open(sys.argv[1], encoding='utf-8'))
    obs=[{'instruction': {'DAG': json.dumps(value, ensure_ascii=False)}}]
    stages,graphs,objects,wrong=get_instruction_graphs(obs)
    if wrong: raise ValueError(f'wrong_dags={wrong}')
    NavigationTree(graphs[0], stages[0], 0.5, device='cpu')
except (ValueError,KeyError,IndexError,TypeError,AssertionError):
    traceback.print_exc()
    sys.exit(3)
print(json.dumps({'parser_success': True, 'navigation_tree_success': True, 'stage_count': stages[0], 'objects': objects[0]}))
"""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(upstream.resolve())
    prefix = os.environ.get('RECONAV_ENV_GCVLN_RUNTIME_PY39')
    executable = (python or (Path(prefix) / 'bin/python' if prefix else Path(sys.executable))).expanduser().resolve()
    library = str(executable.parent.parent / 'lib')
    env['LD_LIBRARY_PATH'] = library + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    completed = subprocess.run([str(executable), '-c', code, str(runtime_path.resolve())], cwd=upstream, env=env, text=True, capture_output=True)
    if completed.returncode == 3:
        raise ValueError('released parser rejected the first response: ' + completed.stderr.strip())
    if completed.returncode:
        raise ParserEnvironmentError(f'released parser environment failed ({executable}): {completed.stderr.strip()}')
    return json.loads(completed.stdout.strip().splitlines()[-1])

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-base-url', default=os.environ.get('EXTERNAL_LOCAL_QWEN_URL', 'http://127.0.0.1:8008/v1'))
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    item = load_policy_input(args.task)
    prompt, prompt_hash = load_official_prompt(args.prompt)
    request = compose_request(prompt, item.instruction)
    (output / 'llm_request.txt').write_text(request, encoding='utf-8')
    from reconav.baselines.common.client import LocalQwenClient
    client = LocalQwenClient({'model': 'qwen3.8-27b', 'base_url': args.local_base_url, 'max_output_tokens': 2048}, output)
    raw = client.generate('', request, json_mode=True, stage='official_instruction_parser')
    call_record = client.calls[-1]
    requested_model, reasoning = ('qwen3.8-27b', 'none')
    (output / 'llm_raw_response.txt').write_text(raw, encoding='utf-8')
    base_metadata = {'benchmark_task_id': item.benchmark_task_id, 'input_fields': ['instruction'], 'source': 'official_appendix_g_live_api', 'official_prompt_path': str(args.prompt.resolve()), 'official_prompt_sha256': prompt_hash, 'official_publication': 'GC-VLN, CoRL 2025, PMLR 305, Appendix G, official PDF page 16', 'official_source_url': 'https://proceedings.mlr.press/v305/yin25a.html', 'requested_model': requested_model, 'returned_model': call_record.get('response_model'), 'reasoning_effort': reasoning, 'call': call_record, 'semantic_retry_count': 0, 'semantic_refinement': None}
    decoded: dict[str, Any] | None = None
    try:
        decoded = decode_paper_json(raw)
        validate_paper_dag(decoded)
        paper = decoded
        runtime = lower_paper_dag(paper)
        paper_path = output / 'paper_dag.json'
        runtime_path = output / 'runtime_dag.json'
        _write_json(paper_path, paper)
        _write_json(runtime_path, runtime)
        gate = released_parser_gate(args.upstream, runtime_path)
    except (ParserEnvironmentError, OSError) as exc:
        _write_json(output / 'preprocess_metadata.json', {**base_metadata, 'status': 'infrastructure_error', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    except Exception as exc:
        if decoded is not None:
            _write_json(output / 'rejected_paper_dag.json', decoded)
        failure = {**base_metadata, 'status': 'invalid_first_response', 'error': f'{type(exc).__name__}: {exc}'}
        _write_json(output / 'preprocess_metadata.json', failure)
        raise
    metadata = {**base_metadata, 'status': 'valid_first_response', 'paper_dag': paper, 'runtime_dag': runtime, 'semantic_signature_preserved': paper_signature(paper) == runtime_signature(runtime), 'released_parser_gate': gate}
    _write_json(output / 'preprocess_metadata.json', metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
