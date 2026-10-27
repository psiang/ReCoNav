"""Robot dimensions and version-matched Habitat navmesh configuration."""
from reconav import paths as release_paths
import functools
from pathlib import Path
from reconav.scene.physical import ROBOT_RADIUS_M, ROBOT_HEIGHT_M, ROBOT_DIAMETER_M
from reconav.baselines.common.protocol import file_identity, read, write
CACHE = (release_paths.cache_root() / 'external/robot')
PROFILE = {'version': 'robot-size-r025-h150-v1', 'radius_m': ROBOT_RADIUS_M, 'diameter_m': ROBOT_DIAMETER_M, 'height_m': ROBOT_HEIGHT_M}
VERSIONS = {'navgpt': '0.3.3', 'mapgpt': '0.3.3', 'dualvln': '0.3.3', 'snav': '0.3.3', 'gcvln': '0.2.4', 'canav': '0.2.1'}

def mesh_paths(scene, version, cache=CACHE):
    root = Path(cache) / version / Path(scene).parent.name
    return (root / 'robot.navmesh', root / 'mesh.json')

def mesh_identity(scene, version, cache=CACHE):
    mesh, manifest = mesh_paths(scene, version, cache)
    if not manifest.exists() or not mesh.exists():
        from reconav.baselines.common.native_assets import prepare_native_assets
        prepare_native_assets(version, cache, scene_ids=[Path(scene).parent.name])
    doc = read(manifest)
    if doc.get('robot_profile') != PROFILE or doc.get('habitat_version') != version or doc.get('scene_identity') != file_identity(scene) or (doc.get('navmesh_identity') != file_identity(mesh)):
        raise RuntimeError('stale/mismatched robot navmesh: ' + str(manifest))
    return doc

def require_asg_profile(scene_id, graph, cache=CACHE):
    checks = read(Path(cache) / 'asg_checks.json')
    record = checks['scenes'].get(scene_id, {})
    if checks.get('robot_profile') != PROFILE or record.get('graph_id') != graph.get('graph_id') or record.get('source_identity') != graph.get('source_identity'):
        raise RuntimeError('missing/stale G0 robot-size audit: ' + scene_id)

def navmesh_settings(habitat_sim):
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = ROBOT_RADIUS_M
    settings.agent_height = ROBOT_HEIGHT_M
    return settings

def install_habitat_profile(audit_path=None):
    """Set robot dimensions and reload the navmesh after simulator reconfiguration.

    Install this hook before creating the simulator, including threaded envs.
    """
    import habitat_sim
    cls = habitat_sim.Simulator
    if getattr(cls.reconfigure, '_benchmark_robot_profile', False):
        return
    original = cls.reconfigure
    version = habitat_sim.__version__

    @functools.wraps(original)
    def reconfigure(sim, config):
        scene = Path(config.sim_cfg.scene_id)
        record = None if str(config.sim_cfg.scene_id) in {'', 'NONE', 'none'} else mesh_identity(scene, version)
        for agent in config.agents:
            agent.radius, agent.height = (ROBOT_RADIUS_M, ROBOT_HEIGHT_M)
        original(sim, config)
        if record is None:
            return
        mesh = record['navmesh_identity']['path']
        if not sim.pathfinder.load_nav_mesh(mesh):
            raise RuntimeError('failed loading aligned navmesh: ' + mesh)
        for agent in config.agents:
            if (agent.radius, agent.height) != (ROBOT_RADIUS_M, ROBOT_HEIGHT_M):
                raise RuntimeError('native simulator changed robot size')
        sim._benchmark_robot_profile = record
        if audit_path:
            write(Path(audit_path), {'robot_profile': PROFILE, 'mesh': record, 'reconfigure_guard': True, 'snap_applied': False})
    reconfigure._benchmark_robot_profile = True
    cls.reconfigure = reconfigure
