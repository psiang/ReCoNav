"""Build native navmeshes and validate starts and motion on CPU."""
import argparse
from collections import defaultdict
import os

import numpy as np
from reconav.baselines.common.protocol import (BENCHMARK, discover, digest, policy_payload,
                                        scene_assets, file_identity, write)
from reconav.baselines.common.robot import CACHE, PROFILE, mesh_paths, mesh_identity, navmesh_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append', default=[])
    args = parser.parse_args()
    import habitat_sim
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('geometry preparation must not allocate CUDA')
    version = habitat_sim.__version__
    groups = defaultdict(list)
    for task in discover(BENCHMARK):
        if not args.scene or task.parent.name in args.scene:
            groups[task.parent.name].append(task)
    rows = []
    for scene_id, tasks in groups.items():
        scene = scene_assets(scene_id)['scene']
        mesh, manifest = mesh_paths(scene, version)
        if manifest.exists():
            record = mesh_identity(scene, version)
        else:
            cfg = habitat_sim.SimulatorConfiguration()
            cfg.scene_id = str(scene)
            cfg.create_renderer = version == '0.2.1'
            cfg.requires_textures = False
            cfg.enable_physics = False
            cfg.gpu_device_id = 0 if version == '0.2.1' else -1
            agent = habitat_sim.agent.AgentConfiguration()
            agent.radius, agent.height = PROFILE['radius_m'], PROFILE['height_m']
            agent.sensor_specifications = []
            if version == '0.2.1':
                sensor = habitat_sim.CameraSensorSpec()
                sensor.uuid = 'geometry_loader'
                sensor.sensor_type = habitat_sim.SensorType.COLOR
                sensor.resolution = [16, 16]
                agent.sensor_specifications = [sensor]
            sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
            try:
                settings = navmesh_settings(habitat_sim)
                if not sim.recompute_navmesh(sim.pathfinder, settings):
                    raise RuntimeError('navmesh recompute failed: ' + scene_id)
                mesh.parent.mkdir(parents=True, exist_ok=True)
                sim.pathfinder.save_nav_mesh(str(mesh))
                record = {'robot_profile': PROFILE, 'habitat_version': version,
                          'scene_identity': file_identity(scene), 'navmesh_identity': file_identity(mesh),
                          'explicit_recompute': True, 'source': 'native scene collision mesh; no ASG',
                          'settings': {name: getattr(settings, name) for name in (
                              'agent_radius', 'agent_height', 'agent_max_climb', 'agent_max_slope',
                              'cell_size', 'cell_height')},
                          'navigable_area': float(sim.pathfinder.navigable_area)}
                write(manifest, record)
            finally:
                sim.close()
        pf = habitat_sim.PathFinder()
        if not pf.load_nav_mesh(str(mesh)):
            raise RuntimeError('saved navmesh reload failed')
        for task in tasks:
            policy = policy_payload(task)
            position = np.asarray(policy['start_position'], dtype=np.float32)
            supported = bool(pf.is_navigable(position))
            snapped = pf.snap_point(position)
            distance = float(np.linalg.norm(snapped - position))
            # Exercise native collision filtering without model, reward or goal.
            stepped = pf.try_step(position, position + np.asarray([.25, 0, 0])) if supported else None
            if stepped is not None and not np.all(np.isfinite(stepped)):
                raise RuntimeError('nonfinite native collision-filter output')
            rows.append({'task_id': task.stem, 'scene_id': scene_id, 'supported': supported,
                         'policy_fingerprint': digest(policy), 'robot_profile': PROFILE,
                         'scene_identity': file_identity(scene), 'navmesh_identity': file_identity(mesh),
                         'snap_applied': False, 'snap_distance_m': distance if np.isfinite(distance) else None,
                         'motion_probe_finite': stepped is not None})
        print(scene_id + ': rebuilt/loaded aligned mesh and audited ' + str(len(tasks)) + ' starts', flush=True)
    write(CACHE / version / 'exact_starts.json', {'robot_profile': PROFILE, 'habitat_version': version,
        'total': len(rows), 'supported': sum(r['supported'] for r in rows), 'rows': rows,
        'api_calls': 0, 'gpu_model_loads': 0})
    print({'version': version, 'total': len(rows), 'supported': sum(r['supported'] for r in rows)}, flush=True)


if __name__ == '__main__':
    main()
