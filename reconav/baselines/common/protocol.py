"""Shared experiment settings, task inputs and result files."""
from __future__ import annotations
from reconav import paths as release_paths
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from reconav.baselines.canav.export_task import load_policy_input
ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = release_paths.benchmark_path('tasks', 'hm3dsem')
ASG_ROOT = release_paths.benchmark_path('scenes/hm3dsem')
SCENE_ROOT = release_paths.hm3d_root()
VERSION = 'external-current-benchmark-20260924'
LLM_METHODS = ('navgpt', 'mapgpt')
COMMITS = {'navgpt': 'b3fc8a21', 'mapgpt': '7c642f4507a8'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def file_identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}

def asg_identity(directory):
    return {name: file_identity(Path(directory) / name) for name in ('entity.json', 'geometry.json', 'topology.json')}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def discover(root=BENCHMARK, ids=(), split='all'):
    from reconav.benchmark.selection import task_ids
    ids = task_ids(Path(root), split, ids)
    tasks = sorted(Path(root).glob('*/*.json'))
    result, seen = ([], set())
    for path in tasks:
        policy = load_policy_input(path)
        key = (policy.scene_id, policy.benchmark_task_id)
        if key in seen:
            raise ValueError(f'duplicate task {key}')
        seen.add(key)
        if ids and policy.benchmark_task_id not in ids:
            continue
        if policy.scene_id != path.parent.name or policy.benchmark_task_id != path.stem:
            raise ValueError(f'task/path identity mismatch: {path}')
        result.append(path)
    missing = set(ids) - {p.stem for p in result}
    if missing or not result:
        raise ValueError(f'empty selection or unknown tasks: {sorted(missing)}')
    return result

def scene_assets(scene_id, scene_root=SCENE_ROOT, asg_root=ASG_ROOT):
    if Path(scene_id).name != scene_id or '-' not in scene_id:
        raise ValueError('invalid scene ID')
    stem = scene_id.split('-', 1)[1]
    scene = Path(scene_root) / scene_id
    return {'scene': scene / (stem + '.basis.glb'), 'navmesh': scene / (stem + '.basis.navmesh'), 'render_geometry': scene / (stem + '.glb'), 'asg': Path(asg_root) / scene_id / 'asg'}

def policy_payload(path):
    result = asdict(load_policy_input(path))
    result['start_position'] = list(result['start_position'])
    return result

def cell_identity(task, method, condition, settings, assets):
    scene_files = [assets['scene'], assets['render_geometry']]
    if condition == 'native':
        scene_files.append(assets['navmesh'])
    if condition == 'asg':
        scene_files += [assets['asg'] / n for n in ('entity.json', 'geometry.json', 'topology.json')]
    return {'version': VERSION, 'task_sha256': file_digest(task), 'method': method, 'condition': condition, 'settings': settings, 'scene_assets': {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in scene_files}, 'policy_input': policy_payload(task)}

def eligible_resume(cell, identity):
    cell = Path(cell)
    if not all(((cell / name).is_file() for name in ('identity.json', 'status.json', 'metrics.json'))):
        return False
    if read(cell / 'identity.json') != identity:
        return False
    status = read(cell / 'status.json')
    if status.get('execution_status') == 'unsupported_input':
        return status.get('reason') == 'exact_start_outside_native_navmesh' and status.get('api_calls') == 0
    return status.get('execution_status') == 'completed' and (cell / 'trajectory.json').is_file()
