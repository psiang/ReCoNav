"""Adapt GC-VLN's vector environment to G0 waypoint execution.

Habitat provides RGB-D observations; G0 provides motion and connectivity."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reconav.baselines.common.g0 import WaypointExecutor, identity
from reconav.baselines.common.protocol import write
from reconav.baselines.common.robot import PROFILE
from reconav.baselines.gcvln.trajectory import ActualPrimitiveTrajectory
from reconav.baselines.gcvln.runtime_instrumentation import RuntimeInstrumentation


def world_to_gc(points, policy):
    points = np.asarray(points, dtype=float)
    dx = points[..., 0] - policy['start_position'][0]
    dz = points[..., 1] - policy['start_position'][2]
    c, s = math.cos(policy['start_heading']), math.sin(policy['start_heading'])
    return np.stack([-s*dx-c*dz, -c*dx+s*dz], axis=-1)


def gc_to_world(points, policy):
    points = np.asarray(points, dtype=float)
    c, s = math.cos(policy['start_heading']), math.sin(policy['start_heading'])
    x, y = points[..., 0], points[..., 1]
    return np.stack([-s*x-c*y+policy['start_position'][0],
                     -c*x+s*y+policy['start_position'][2]], axis=-1)


def active_map_floor(height, floor_heights):
    """Use the same current BEV layer at reset, on a landing and mid-stair."""
    return min(floor_heights, key=lambda floor:abs(floor-height))


class G0Trajectory(ActualPrimitiveTrajectory):
    def __init__(self, output, policy, executor):
        super().__init__(output, policy['benchmark_task_id'], run_kind='GC-VLN_ASG_G0')
        self.executor = executor

    def payload(self):
        value = super().payload()
        value.update(trajectory_source='executed G0 physical edges and exact endpoint attachments',
                     physical_backend=identity(self.executor.source),
                     g0_waypoint_moves=self.executor.moves)
        return value


class RenderOnlyHabitat:
    def __init__(self, config, scene):
        import habitat_sim
        sim = habitat_sim.SimulatorConfiguration()
        sim.scene_id = str(scene)
        sim.gpu_device_id = -1
        sim.enable_physics = False
        agent = habitat_sim.agent.AgentConfiguration()
        agent.radius, agent.height = PROFILE['radius_m'], PROFILE['height_m']
        agent.sensor_specifications = []
        for degrees in range(0, 360, 30):
            for key, typ in [('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)]:
                source = getattr(config.TASK_CONFIG.SIMULATOR, key.upper()+'_SENSOR')
                sensor = habitat_sim.CameraSensorSpec()
                sensor.uuid = key + (('_'+str(degrees)) if degrees else '')
                sensor.sensor_type = typ
                sensor.resolution = [source.HEIGHT, source.WIDTH]
                sensor.position = list(source.POSITION)
                sensor.orientation = [0, math.radians(degrees), 0]
                sensor.hfov = source.HFOV
                agent.sensor_specifications.append(sensor)
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim, [agent]))
        self.agent = self.sim.get_agent(0)

    def observe(self, position, yaw):
        import habitat_sim
        import quaternion
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0, yaw, 0])
        self.agent.set_state(state)  # Camera pose, NEVER motion/collision authority.
        observations = self.sim.get_sensor_observations()
        return {k: (v[..., :3] if k.startswith('rgb') else v[..., None])
                for k, v in observations.items()}

    def close(self):
        self.sim.close()


class G0VectorEnv:
    """Single-episode vector interface for the GC-VLN agent."""
    def __init__(self, config, policy, asset_dir, asg_dir, scene, output, runtime_dag):
        self.config, self.policy, self.output = config, policy, Path(output)
        self.executor = WaypointExecutor(policy, asset_dir, asg_dir)
        self.renderer = None
        self.asg_scenegraph = None
        self.scene = scene
        self.dag = Path(runtime_dag).read_text()
        self.size = config.POLICY_CONFIG.MAP.SIZE
        self.resolution = config.POLICY_CONFIG.MAP.RESOLUTION
        self.position = self.executor.position
        self.yaw = float(policy['start_heading'])
        self.num_envs, self.number_of_episodes = 1, [1]
        self.episode = SimpleNamespace(episode_id='1', scene_id=str(scene))
        self.trajectory = G0Trajectory(self.output/'trajectory.json', policy, self.executor)
        self.instrumentation = RuntimeInstrumentation(self.output/'runtime_instrumentation_env.json',
                                                       run_kind='GC-VLN_ASG_G0')
        write(self.output/'robot_geometry.json', {**identity(self.executor.source),
            'renderer': 'Habitat RGB-D only; no sim.step/pathfinder motion',
            'low_level_change': 'FMM waypoint follower replaced by shared G0 graph paths',
            'camera_height_m': config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION[1]})

    def resume_all(self):
        self.num_envs = 1

    def current_episodes(self):
        return [self.episode] if self.num_envs else []

    def pause_at(self, index):
        assert index == 0 and self.num_envs == 1
        self.num_envs = 0

    def pose(self):
        return list(self.position), [0., math.sin(self.yaw/2), 0., math.cos(self.yaw/2)], self.yaw

    def call(self, names, *args, **kwargs):
        if names != ['get_pos_ori']:
            raise ValueError('unsupported G0 environment call: ' + str(names))
        return [self.pose()[:2]]

    def observation(self):
        if self.renderer is None:
            self.renderer = RenderOnlyHabitat(self.config, self.scene)
        obs = self.renderer.observe(self.position, self.yaw)
        gc = world_to_gc([self.position[0], self.position[2]], self.policy)
        obs.update(gps=np.array([gc[0], -gc[1]], dtype=np.float32),
                   compass=np.array([(self.yaw-self.policy['start_heading']+math.pi)%math.tau-math.pi], dtype=np.float32),
                   instruction={'DAG': self.dag, 'text': self.policy['instruction'],
                                'text_English': self.policy['instruction']})
        return obs

    def reset(self):
        self.trajectory.reset(*self.pose())
        self.trajectory.write()
        return [self.observation()]

    def step(self, actions):
        assert len(actions) == self.num_envs == 1
        action = actions[0]['action']
        result = 'complete'
        point = action.get('next_point')
        try:
            if point is not None:
                point = np.asarray(point, dtype=float)
                gc = (point[::-1]-self.size//2)/self.resolution
                target = gc_to_world(gc, self.policy)
                self.instrumentation.value['g0_search_call_count'] = self.instrumentation.value.get('g0_search_call_count', 0)+1
                # The current pose may be halfway up a connector and > .75 m
                # away from either floor. Resolve normal BEV cell heights from
                # the SAME nearest layer rendered to the original solver.
                floor = active_map_floor(self.position[1],
                    self.executor.physical.region_floor_elevations.values())
                if self.asg_scenegraph is None:
                    raise RuntimeError('GC G0 executor requires the actual solver raster interface')
                path = self.executor.follow(target, floor, projection_tolerance_m=math.sqrt(2)/self.resolution,
                    connector_cell=self.asg_scenegraph.is_connector_cell(point,self.position[1]))
                result = 'complete' if self.executor.moves[-1]['accepted'] else 'incomplete'
                for end in path:
                    before = self.position
                    delta = np.asarray(end)-before
                    if math.hypot(delta[0], delta[2]) > 1e-9:
                        self.yaw = math.atan2(-delta[0], -delta[2])
                    # Full certified polyline is recorded at <= native step spacing.
                    count = max(1, math.ceil(math.dist(before, end)/self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE))
                    for i in range(1, count+1):
                        self.position = (np.asarray(before)+(np.asarray(end)-before)*(i/count)).tolist()
                        self.trajectory.primitive('G0_MOVE', *self.pose(), done=False)
                        self.instrumentation.primitive()
            done = action['act'] == 0
            if done:
                self.trajectory.primitive(0, *self.pose(), done=True)
                self.instrumentation.primitive()
            return [(self.observation(), 0., done,
                     {'action_valid': result, 'pre_point': ','.join(str(int(x)) for x in point) if point is not None else '',
                      'steps_taken': len(self.trajectory.records)-1,
                      'path_length': self.trajectory.path_length_m})]
        except Exception as exc:
            self.instrumentation.exception(exc, source='G0VectorEnv.step')
            raise

    def close(self):
        self.trajectory.close()
        self.instrumentation.write()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
