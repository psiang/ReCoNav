"""Run the complete raw-instruction benchmark for external navigation baselines."""
from __future__ import annotations
from reconav import paths as release_paths
from reconav.baselines.common.trajectory_io import read_json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    'scene_root': release_paths.hm3d_root(),
    'evaluator_python': Path(os.environ.get('RECONAV_EVALUATOR_PYTHON', sys.executable)),
}
PYTHONS = {'canav': Path(str(release_paths.environment_path('canav', 'bin/python'))), 'gcvln': Path(str(release_paths.environment_path('gcvln-runtime-py39', 'bin/python'))), 'dualvln': Path(str(release_paths.environment_path('dualvln-official', 'bin/python')))}


def _environment(baseline: str, gpu: int) -> dict[str, str]:
    release_paths.configure()
    env = dict(os.environ)
    env['RECONAV_BASELINE'] = baseline
    env['PYTHONPATH'] = str(ROOT)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    for key in list(env):
        if any(word in key.upper() for word in ('API_KEY', 'OPENAI_KEY', 'GEMINI_KEY')):
            env.pop(key)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['GLOG_minloglevel'] = '2'
    for name in ('NO_PROXY', 'no_proxy'):
        env[name] = ','.join(filter(None, [env.get(name), '127.0.0.1', 'localhost']))
    return env


def _preprocess_command(baseline: str, task: Path, cell: Path) -> list[str] | None:
    python = sys.executable
    if baseline == 'gcvln':
        return [python, '-m', 'reconav.baselines.gcvln.preprocess', '--task', str(task), '--prompt', str(ROOT / 'reconav/baselines/gcvln/appendix_g_prompt.txt'), '--upstream', str(release_paths.upstream_path('GC-VLN')), '--output', str(cell)]
    return None

def _assets_for_task(task: Path) -> dict[str, Path]:
    """Resolve the scene from a released multi-scene task."""
    from reconav.baselines.common.protocol import scene_assets
    return scene_assets(str(read_json(task)['scene_id']))


def _runtime_command(baseline: str, task: Path, cell: Path, gpu: int, server_port: int) -> list[str]:
    python = str(PYTHONS[baseline])
    assets = _assets_for_task(task)
    common = ['--task', str(task), '--scene', str(assets['scene']), '--asg', str(assets['asg']), '--evaluator-python', str(DEFAULTS['evaluator_python']), '--output', str(cell)]
    if baseline == 'canav':
        return ['xvfb-run', '-a', '-s', '-screen 0 1280x1024x24 +extension GLX +render -noreset', python, '-m', 'reconav.baselines.canav.run_single', *common, '--scene-root', str(DEFAULTS['scene_root']), '--upstream', str(release_paths.upstream_path('CA-Nav-code')), '--habitat-lab', str(release_paths.upstream_path('habitat-lab-canav')), '--artifacts', (str(release_paths.external_root() / 'artifacts/canav')), '--variant', 'qwen-local']
    if baseline == 'gcvln':
        upstream = release_paths.upstream_path('GC-VLN')
        runtime = release_paths.runtime_path('gcvln')
        if not (runtime / 'src').is_dir():
            runtime = upstream
        return ['xvfb-run', '-a', '-s', '-screen 0 1280x1024x24 +extension GLX +render -noreset', python, '-m', 'reconav.baselines.gcvln.run_single', *common, '--scene-root', str(DEFAULTS['scene_root']), '--upstream', str(upstream), '--runtime-root', str(runtime), '--server-port', str(server_port), '--gpu-id', '0', '--simulator-gpu-id', '-1']
    if baseline == 'dualvln':
        return [python, '-m', 'reconav.baselines.dualvln.run_single', *common, '--upstream', str(release_paths.upstream_path('InternNav')), '--checkpoint', str(release_paths.runtime_path('dualvln/InternVLA-N1-DualVLN')), '--runtime-workdir', str(release_paths.runtime_path('dualvln')), '--device', 'cuda:0']
    raise ValueError('Unsupported paper baseline')

