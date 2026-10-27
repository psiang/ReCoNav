"""Physical navigation graph and waypoint execution for +ASG baselines."""
from pathlib import Path
from functools import lru_cache

from reconav.baselines.common.protocol import asg_identity, digest, read
from reconav.baselines.common.robot import PROFILE, require_asg_profile

BACKEND = 'shared-G0-r025-h150-v1'


@lru_cache(maxsize=9)
def _physical_graph(asg_dir, source_fingerprint, graph_id):
    from reconav.scene.io import load_asg
    from reconav.scene.navigation import get_geometry_graph
    graph = get_geometry_graph(load_asg(asg_dir))
    if graph.graph_id != graph_id:
        raise RuntimeError('runtime G0 differs from exported G0')
    return graph


def validate_assets(policy, asset_dir, asg_dir):
    root = Path(asset_dir) / policy['scene_id']
    graph = read(root / 'asg_graph.json')
    require_asg_profile(policy['scene_id'], graph)
    if graph['source_identity'] != asg_identity(asg_dir):
        raise RuntimeError('stale G0 ASG source')
    start = read(root / 'starts' / (policy['benchmark_task_id'] + '.json'))
    if (start.get('policy_fingerprint') != digest(policy)
            or start.get('graph_id') != graph['graph_id']):
        raise RuntimeError('stale G0 exact-start attachment')
    return graph, start


def identity(graph):
    return {'backend': BACKEND, 'graph_id': graph['graph_id'],
            'source_identity': graph['source_identity'], 'robot_profile': PROFILE,
            'native_navmesh_for_reachability': False, 'snap_start': False}


class WaypointExecutor:
    """Follow method-selected coordinates using G0 endpoint and connectivity rules.

    Reject nontraversable waypoints and retain portal and stair edges.
    """
    def __init__(self, policy, asset_dir, asg_dir):
        import networkx as nx
        self.policy = policy
        self.source, start = validate_assets(policy, asset_dir, asg_dir)
        if not start['supported']:
            raise RuntimeError('unsupported exact G0 start')
        self.physical = _physical_graph(str(asg_dir),digest(self.source['source_identity']),self.source['graph_id'])
        if self.physical.graph_id != self.source['graph_id']:
            raise RuntimeError('runtime G0 differs from exported G0')
        cfg = self.physical.config
        if (cfg.robot_radius_m, cfg.robot_diameter_m, cfg.robot_height_m) != (.25, .5, 1.5):
            raise RuntimeError('runtime G0 robot dimensions differ')
        self.graph = nx.Graph()
        self.graph.add_nodes_from((k, dict(v)) for k, v in self.source['nodes'].items())
        for a, edges in self.source['adjacency'].items():
            self.graph.add_weighted_edges_from((a, b, w) for b, w in edges)
        self.connectors = [(a,b,w) for a,edges in self.source['adjacency'].items() for b,w in edges
                           if a < b and self.source['nodes'][a]['region'] != self.source['nodes'][b]['region']]
        self.graph.add_node('START', position=policy['start_position'], region=start['region'])
        for n, w in start['links']:
            self.graph.add_edge('START', n, weight=w, attachment=True)
        self.current = 'START'
        self.moves = []
        self.serial = 0

    @property
    def position(self):
        return list(self.graph.nodes[self.current]['position'])

    def follow(self, xz, height, *, projection_tolerance_m=0., connector_cell=False):
        import math
        from shapely.geometry import Point
        from reconav.scene.navigation import visible_component_nodes, segment_is_in_free_space
        x, z = map(float, xz)
        if not all(math.isfinite(v) for v in (x, z, height)):
            raise ValueError('nonfinite G0 waypoint')
        candidates = []
        for rid in self.physical.regions:
            floor = self.physical.floor_height(rid)
            if abs(floor - height) > .75:
                continue
            component = self.physical.locate_local_component(rid, Point(x, z))
            if component is not None:
                candidates.append((abs(floor-height), rid, component, floor))
        record = {'requested_xz': [x, z], 'requested_height': float(height),
                  'accepted': False, 'snap_applied': False, 'path': [], 'segments': []}
        self.moves.append(record)
        if connector_cell:
            record['waypoint_interface'] = 'rasterized_G0_connector_cell'
            return self._follow_connector(xz,height,projection_tolerance_m,record)
        if not candidates:
            if projection_tolerance_m:
                return self._follow_connector(xz, height, projection_tolerance_m, record)
            record['reason'] = 'waypoint_outside_G0_free_space'
            return []
        _, rid, component, floor = min(candidates)
        links = visible_component_nodes(self.physical, rid, component, (x, z))
        self.serial += 1
        target = 'WAYPOINT_' + str(self.serial)
        self.graph.add_node(target, position=[x, floor, z], region=rid)
        for n, w in links:
            self.graph.add_edge(target, str(n), weight=w, attachment=True)
        current = self.graph.nodes[self.current]
        if (current.get('region') == rid and abs(current['position'][1]-floor) <= .75
                and segment_is_in_free_space(self.physical, rid,
                    (current['position'][0], current['position'][2]), (x, z))):
            self.graph.add_edge(self.current, target,
                                weight=math.dist(current['position'], [x, floor, z]), attachment=True)
        return self._execute_target(target, record)

    def _follow_connector(self, xz, height, tolerance, record):
        """Project a rasterized portal/stair cell onto its G0 connector.

The tolerance is the native BEV cell quantization."""
        import math
        import numpy as np
        import networkx as nx
        if not 0 < tolerance <= .071:
            raise ValueError('connector quantization exceeds one 5 cm raster-cell diagonal')
        choices = []
        point = np.asarray(xz, dtype=float)
        # Stacked stair flights can overlap in XZ. Resolve by physical access
        # from the actual current pose, NOT by 2D projection error alone.
        distances = nx.single_source_dijkstra_path_length(self.graph,self.current,weight='weight')
        current = self.graph.nodes[self.current]
        current_portal = current.get('entity_id') if current.get('kind') == 'portal_anchor' else None
        if current.get('connector_edge'):
            endpoints = [self.source['nodes'][n] for n in current['connector_edge']]
            # Once at a landing, another flight can be entered normally.
            if min(abs(current['position'][1]-n['position'][1]) for n in endpoints) > .10:
                current_portal = endpoints[0].get('entity_id')
        for a, b, weight in self.connectors:
            pa = np.asarray(self.source['nodes'][a]['position'])
            pb = np.asarray(self.source['nodes'][b]['position'])
            if not min(pa[1],pb[1])-.75 <= height <= max(pa[1],pb[1])+.75:
                continue
            delta = (pb-pa)[[0,2]]
            denom = float(delta @ delta)
            if denom <= 1e-12:
                continue
            fraction = float(np.clip((point-pa[[0,2]]) @ delta / denom, 0, 1))
            pose = pa + fraction*(pb-pa)
            distance = float(np.linalg.norm(pose[[0,2]]-point))
            if distance <= tolerance:
                cost = min(distances.get(a,math.inf)+fraction*weight,
                           distances.get(b,math.inf)+(1-fraction)*weight)
                if current.get('connector_edge') == (a,b):
                    cost = min(cost,weight*abs(current['fraction']-fraction))
                if math.isfinite(cost):
                    continuity = int(current_portal is not None and
                        self.source['nodes'][a].get('entity_id') != current_portal)
                    # Within a known flight, minimize raster error before
                    # travel cost: otherwise a slightly nearer point can stop
                    # short of a landing merely to save centimeters of travel.
                    rank = (distance,cost) if current_portal is not None else (cost,distance)
                    choices.append((continuity,rank,cost,distance,a,b,fraction,pose.tolist(),weight))
        for continuity,rank,cost,distance,a,b,fraction,pose,weight in sorted(choices):
            self.serial += 1
            target = 'WAYPOINT_' + str(self.serial)
            self.graph.add_node(target, position=pose, connector_edge=(a,b), fraction=fraction)
            for n, w in [(a,weight*fraction),(b,weight*(1-fraction))]:
                self.graph.add_edge(target,n,weight=w,attachment=True,connector=True)
            current = self.graph.nodes[self.current]
            if current.get('connector_edge') == (a,b):
                self.graph.add_edge(self.current,target,weight=weight*abs(current['fraction']-fraction),
                                    attachment=True,connector=True)
            if not nx.has_path(self.graph,self.current,target):
                self.graph.remove_node(target)
                continue
            record.update(connector_projection={'edge':[a,b],'fraction':fraction,
                'quantization_error_m':distance,'maximum_quantization_error_m':tolerance,
                'physical_access_length_m':cost,
                'ambiguity_rule':'current connector continuity, then minimum G0 access length from actual pose'},
                endpoint_position=pose)
            return self._execute_target(target,record)
        record['reason'] = 'waypoint_outside_G0_free_space_and_connectors'
        return []

    def _execute_target(self, target, record):
        import networkx as nx
        try:
            path = nx.shortest_path(self.graph, self.current, target, weight='weight')
        except nx.NetworkXNoPath:
            self.graph.remove_node(target)
            record['reason'] = 'waypoint_unreachable_in_G0'
            return []
        positions = []
        for a, b in zip(path, path[1:]):
            first, second = self.graph.nodes[a]['position'], self.graph.nodes[b]['position']
            record['segments'].append({'from': a, 'to': b, 'start': list(first), 'end': list(second),
                'kind': ('G0_connector_subsegment' if self.graph[a][b].get('connector') else
                         'G0_endpoint_attachment' if self.graph[a][b].get('attachment') else 'G0_edge')})
            positions.append(list(second))
        record.update(accepted=True, reason='G0_path', path=path)
        previous = self.current
        self.current = target
        if previous.startswith('WAYPOINT_'):
            self.graph.remove_node(previous)
        return positions
