"""Dispatch language-agent policies with shared scene I/O and accounting."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from reconav.baselines.common.protocol import read, write
from reconav.baselines.common.client import LocalQwenClient, BudgetExceeded
from reconav.baselines.common.world import World
from reconav.baselines.mapgpt.agent import mapgpt
from reconav.baselines.navgpt.agent import navgpt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('this worker requires CPU-only rendering and empty CUDA_VISIBLE_DEVICES')
    upstream = Path(config['upstream'])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    if head != config['upstream_commit']:
        raise RuntimeError('upstream source version changed')
    if config.get('provider') != 'local-qwen':
        raise ValueError('paper agents require the released local Qwen backend')
    client = LocalQwenClient(config, output)
    world = None
    started = time.monotonic()
    status = {'execution_status': 'running'}
    write(output / 'worker_status.json', status)
    try:
        world = World(config, client.captions)
        with (output / 'native.stdout.log').open('w') as log, contextlib.redirect_stdout(log):
            result, reason = {'navgpt': navgpt, 'mapgpt': mapgpt}[config['method']](world, client, config)
        write(output / 'native_trace.json', result)
        from reconav.baselines.common.time_budget import mark_policy_complete
        mark_policy_complete()
        status.update(execution_status='completed', termination=reason)
    except Exception as error:
        status.update(execution_status='blocked' if isinstance(error, BudgetExceeded) else 'failed',
                      error=f'{type(error).__name__}: {error}')
    finally:
        if world is not None:
            positions = [{'x': float(x), 'y': float(y), 'z': float(z)} for x, y, z in world.positions]
            trajectory = {'method': config['method'] + '-' + config['condition'],
                          'task_id': world.policy['benchmark_task_id'], 'route': {'waypoints': positions},
                          'executed_actions': len(world.moves), 'recorded_poses': len(positions),
                          'path_length_m': sum(math.dist(a, b) for a, b in zip(world.positions, world.positions[1:])),
                          'termination': status.get('termination'), 'valid_for_scoring': True,
                          'runtime_exception': status.get('error'), 'navigation_backend': world.condition,
                          'moves': world.moves, 'physical_backend': world.physical_backend}
            write(output / 'trajectory.json', trajectory)
            world.close()
        status['seconds'] = time.monotonic() - started
        status['api_calls'] = sum(r.get('provider_called', True) for r in client.calls)
        status['reused_calls'] = sum(r.get('provider_called') is False for r in client.calls)
        status['accounted_tokens'] = sum(r.get('usage', {}).get('total_tokens', 0) for r in client.calls)
        status['accounted_llm_seconds'] = sum(r.get('seconds', 0) for r in client.calls)
        write(output / 'worker_status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['execution_status'] == 'completed' else 2

if __name__ == "__main__":
    raise SystemExit(main())
