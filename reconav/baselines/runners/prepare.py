"""Export ASG scene connectivity and attach exact task starts on CPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict

from reconav.baselines.common.protocol import ASG_ROOT, BENCHMARK, asg_identity, digest, discover, policy_payload, read, write


def export_scene(scene_id, output, asg_root=ASG_ROOT):
    from reconav.method.runtime.scene import DirectorySceneProvider
    from reconav.scene.navigation import get_geometry_graph
    context = DirectorySceneProvider(asg_root, 'hm3dsem').get(scene_id)
    graph = get_geometry_graph(context.asg)
    from reconav.baselines.common.robot import PROFILE
    config = asdict(graph.config)
    for field, expected in (("robot_radius_m", PROFILE["radius_m"]),
                            ("robot_height_m", PROFILE["height_m"]),
                            ("robot_diameter_m", PROFILE["diameter_m"])):
        if config[field] != expected:
            raise ValueError(f"G0 robot dimensions differ for {scene_id}: {field}")
    nodes = {str(n.node_id): {'position': [n.x, graph.floor_height(n.region_id), n.y],
                            'region': n.region_id, 'kind': n.kind, 'entity_id': n.entity_id}
             for n in graph.nodes}
    # All physical edges, including portal/stair connectors, are retained.
    payload = {'scene_id': scene_id, 'source': 'ASG_scene_only_G0', 'graph_id': graph.graph_id,
               'source_identity': asg_identity(Path(asg_root)/scene_id/'asg'),
               'scene_fingerprint': context.fingerprint, 'nodes': nodes,
               'adjacency': {str(k): [[str(j), float(w)] for j, w in rows]
                             for k, rows in graph.adjacency.items()},
               'public_asg': context.public_view('compact'),
               'policy_restrictions': ['no task contract', 'no target binding', 'no gold route', 'no constraint solver']}
    write(Path(output) / scene_id / 'asg_graph.json', payload)
    from shapely.geometry import mapping
    write(Path(output) / scene_id / 'physical_free_space.json', {
        'source_identity': payload['source_identity'], 'graph_id': graph.graph_id,
        'scene_only': True,
        'regions': {str(k): {'floor_height': graph.floor_height(k),
                            'free_space': mapping(region.free_space)}
                    for k, region in graph.regions.items()},
    })
    return context, graph


def attach_start(policy, graph):
    from shapely.geometry import Point
    from reconav.scene.navigation import visible_component_nodes
    x, height, z = policy['start_position']
    point = Point(x, z)
    candidates = []
    for rid, physical in graph.regions.items():
        if abs(graph.floor_height(rid) - height) > 0.75:
            continue
        component = graph.locate_local_component(rid, point)
        if component is None:
            continue
        links = visible_component_nodes(graph, rid, component, (x, z))
        candidates.append((abs(graph.floor_height(rid) - height), rid, links))
    if not candidates:
        return {'supported': False, 'reason': 'exact start outside ASG physical free space', 'position': [x, height, z]}
    _, rid, links = min(candidates)
    if not links:
        return {'supported': False, 'reason': 'no collision-free start attachment', 'position': [x, height, z]}
    return {'supported': True, 'position': [x, height, z], 'region': rid,
            'links': [[str(n), float(w)] for n, w in links], 'snap_applied': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--asg-root', type=Path, default=ASG_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--split', choices=['all', 'hard44'], default='all')
    parser.add_argument('--native', action='store_true', help='Also restore the released native navmeshes; requires official HM3D scene files')
    parser.add_argument('--native-method', choices=['canav', 'gcvln', 'snav', 'dualvln', 'navgpt', 'mapgpt'], help='Restore only this method\'s Habitat version (with --native)')
    args = parser.parse_args()
    if args.native_method and not args.native:
        parser.error('--native-method requires --native')
    groups = defaultdict(list)
    for path in discover(args.benchmark, args.task, split=args.split):
        groups[path.parent.name].append(path)
    rows = []
    from reconav.baselines.common.robot import CACHE, PROFILE
    checks_path = CACHE / 'asg_checks.json'
    previous = read(checks_path) if checks_path.exists() else {}
    scenes = dict(previous.get('scenes', {})) if previous.get('robot_profile') == PROFILE else {}
    for scene_id, paths in groups.items():
        _, graph = export_scene(scene_id, args.output, args.asg_root)
        scenes[scene_id] = {
            'graph_id': graph.graph_id,
            'source_identity': asg_identity(args.asg_root / scene_id / 'asg'),
            'g0_config': asdict(graph.config), 'gc_physical_graph_matches': True,
        }
        for path in paths:
            policy = policy_payload(path)
            start = {**attach_start(policy, graph), 'policy_fingerprint': digest(policy), 'graph_id': graph.graph_id}
            write(args.output / scene_id / 'starts' / (path.stem + '.json'), start)
            rows.append({'task_id': path.stem, 'scene_id': scene_id, **start})
        print(f'{scene_id}: {len(paths)} starts; graph={len(graph.nodes)} nodes', flush=True)
        write(checks_path, {'robot_profile': PROFILE, 'scenes': scenes,
                            'api_calls': 0, 'graph_regeneration': False})
    write(args.output / 'asg_preflight.json', {'total': len(rows), 'supported': sum(r['supported'] for r in rows),
                                             'scene_count': len(groups), 'rows': rows})
    if args.native:
        from reconav.baselines.common.native_assets import prepare_native_assets
        from reconav.baselines.common.robot import VERSIONS
        versions = [VERSIONS[args.native_method]] if args.native_method else ('0.2.1', '0.2.4', '0.3.3')
        for version in versions:
            prepare_native_assets(version, scene_ids=groups)


if __name__ == '__main__':
    main()
