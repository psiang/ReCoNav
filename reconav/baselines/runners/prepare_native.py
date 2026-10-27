"""Build native viewpoint graphs and check exact task starts on CPU."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from reconav.baselines.common.protocol import BENCHMARK, digest, discover, file_identity, policy_payload, scene_assets, write
from reconav.baselines.common.world import native_graph
from reconav.baselines.common.robot import CACHE, PROFILE, mesh_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--assets', type=Path, default=CACHE/'native_assets')
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--split', choices=['all', 'hard44'], default='all')
    args = parser.parse_args()
    import habitat_sim
    groups = defaultdict(list)
    for path in discover(args.benchmark, args.task, split=args.split):
        groups[path.parent.name].append(path)
    rows = []
    for scene_id, tasks in groups.items():
        doc = mesh_identity(scene_assets(scene_id)['scene'], habitat_sim.__version__)
        navmesh = Path(doc['navmesh_identity']['path'])
        graph = native_graph(navmesh, args.assets/scene_id/'native_graph.json')
        pf = habitat_sim.PathFinder()
        if not pf.load_nav_mesh(str(navmesh)):
            raise RuntimeError(f'failed to load {navmesh}')
        for task in tasks:
            policy = policy_payload(task)
            position = np.asarray(policy['start_position'], dtype=np.float32)
            navigable = bool(pf.is_navigable(position))
            # A diagnostic only: no actual snapping changes the policy input.
            distance = float(np.linalg.norm(pf.snap_point(position)-position))
            rows.append({'task_id': task.stem, 'scene_id': scene_id,
                         'robot_profile': PROFILE,
                         'supported': navigable, 'policy_fingerprint': digest(policy),
                         'navmesh_identity': file_identity(navmesh),
                         'snap_distance_m': distance, 'snap_applied': False})
        print(f'{scene_id}: native graph {len(graph["nodes"])} viewpoints, {len(graph["edges"])} edges', flush=True)
    write(args.assets/'native_preflight.json', {'total': len(rows), 'supported': sum(r['supported'] for r in rows), 'rows': rows})


if __name__ == '__main__':
    main()
