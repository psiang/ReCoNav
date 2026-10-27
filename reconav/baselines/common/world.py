"""CPU navigation/observation bridge for discrete external agents on HM3D.

Native mode uses Habitat navmesh only. ASG mode uses the shared scene-only G0
as its search graph. Both execute and record physical graph edge polylines;
neither can query the benchmark goal, gold constraints or gold route.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from reconav.baselines.common.protocol import asg_identity, digest, file_identity, read, write


def viewpoint(key):
    return hashlib.md5(str(key).encode()).hexdigest()


def bearing(first, second):
    # Matterport heading increases clockwise; world forward at zero is -Z.
    d = np.asarray(second) - np.asarray(first)
    return math.atan2(float(d[0]), float(-d[2])) % math.tau


def native_graph(navmesh, cache):
    import habitat_sim
    from scipy.spatial import cKDTree
    identity = {'navmesh': str(navmesh), 'size': navmesh.stat().st_size,
                'mtime_ns': navmesh.stat().st_mtime_ns, 'spacing_m': 1.5, 'radius_m': 3.5}
    if cache.exists() and read(cache).get('identity') == identity:
        return read(cache)
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(navmesh)):
        raise RuntimeError('native navmesh load failed')
    vertices = np.asarray(pf.build_navmesh_vertices())
    # Scene-only deterministic sampling, independent of instructions/targets.
    bins = {}
    for p in vertices:
        key = tuple(np.floor(p / [1.5, 0.5, 1.5]).astype(int))
        if key not in bins and pf.is_navigable(p):
            bins[key] = p
    points = np.asarray(list(bins.values()))
    tree = cKDTree(points)
    nodes = {str(i): {'position': p.tolist()} for i, p in enumerate(points)}
    edges = []
    for i, p in enumerate(points):
        for j in sorted(tree.query_ball_point(p, 3.5)):
            if j <= i:
                continue
            query = habitat_sim.ShortestPath()
            query.requested_start, query.requested_end = p, points[j]
            if pf.find_path(query) and query.geodesic_distance <= 4.0:
                edges.append([str(i), str(j), float(query.geodesic_distance),
                              [np.asarray(x).tolist() for x in query.points]])
    result = {'identity': identity, 'nodes': nodes, 'edges': edges, 'source': 'Habitat_native_navmesh'}
    write(cache, result)
    return result


class World:
    def __init__(self, config, captioner=None):
        self.config = config
        self.policy = read(config['policy'])
        self.scene_id = self.policy['scene_id']
        self.condition = config['condition']
        self.cache = Path(config['asset_dir']) / self.scene_id
        self.cache.mkdir(parents=True, exist_ok=True)
        self.graph = nx.Graph()
        self.public_asg = None
        self.physical_backend = None
        if self.condition == 'asg':
            source = read(self.cache / 'asg_graph.json')
            if config.get('robot_profile'):
                from reconav.baselines.common.robot import PROFILE, require_asg_profile
                if config['robot_profile'] != PROFILE:
                    raise RuntimeError('old/mismatched G0 robot profile')
                require_asg_profile(self.scene_id, source)
                from reconav.baselines.common.g0 import identity
                self.physical_backend = identity(source)
            if config.get('asg_source') and source.get('source_identity') != asg_identity(config['asg_source']):
                raise RuntimeError('stale ASG graph; rerun reconav.baselines.runners.prepare')
            self.public_asg = source['public_asg']
            self.graph.add_nodes_from((str(k), dict(v)) for k, v in source['nodes'].items())
            for k, neighbors in source['adjacency'].items():
                self.graph.add_weighted_edges_from((str(k), str(j), w) for j, w in neighbors)
            start = read(self.cache / 'starts' / (self.policy['benchmark_task_id'] + '.json'))
            if start.get('policy_fingerprint') != digest(self.policy) or start.get('graph_id') != source.get('graph_id'):
                raise RuntimeError('stale ASG start attachment; rerun reconav.baselines.runners.prepare')
            if not start['supported']:
                raise RuntimeError('unsupported_exact_start: ' + start['reason'])
            links = [(str(n), w, None) for n, w in start['links']]
        else:
            import habitat_sim
            source = native_graph(Path(config['navmesh']),
                                  Path(config.get('native_graph_cache', self.cache / 'native_graph.json')))
            self.graph.add_nodes_from((str(k), dict(v)) for k, v in source['nodes'].items())
            for a, b, w, poly in source['edges']:
                self.graph.add_edge(a, b, weight=w, polyline=poly, direction=a)
            pf = habitat_sim.PathFinder()
            pf.load_nav_mesh(config['navmesh'])
            pos = np.asarray(self.policy['start_position'], dtype=np.float32)
            if not pf.is_navigable(pos):
                raise RuntimeError('unsupported_exact_start: outside native navmesh; no snapping applied')
            nearest = sorted(self.graph, key=lambda n: math.dist(pos, self.graph.nodes[n]['position']))[:64]
            links = []
            for n in nearest:
                query = habitat_sim.ShortestPath()
                query.requested_start, query.requested_end = pos, np.asarray(self.graph.nodes[n]['position'], dtype=np.float32)
                if pf.find_path(query) and query.geodesic_distance <= 4.0:
                    links.append((n, float(query.geodesic_distance), [np.asarray(p).tolist() for p in query.points]))
            if not links:
                raise RuntimeError('unsupported_exact_start: no connected native viewpoints')
        self.graph.add_node('START', position=self.policy['start_position'],
                            region=start.get('region') if self.condition == 'asg' else None)
        for n, weight, poly in links:
            self.graph.add_edge('START', n, weight=weight, polyline=poly, direction='START')
        self.raw_keys = {viewpoint(k): k for k in self.graph}
        self.heading = (-float(self.policy['start_heading'])) % math.tau
        self.current = 'START'
        self.positions = [list(self.policy['start_position'])]
        self.moves = []
        self.visited = ['START']
        self.captioner = captioner
        self.renderer = None
        self.caption_cache = {}
        self.candidate_cache = {}
        self.env = SimpleNamespace(sims=[SimpleNamespace(navigable_dict={}, gmap=self)])

    def candidates(self):
        if self.current in self.candidate_cache:
            return self.candidate_cache[self.current]
        lengths, paths = nx.single_source_dijkstra(self.graph, self.current, cutoff=4.0, weight='weight')
        origin = self.graph.nodes[self.current]['position']
        buckets = {}
        for n, distance in lengths.items():
            if n == self.current or distance < 0.4:
                continue
            pos = self.graph.nodes[n]['position']
            angle = bearing(origin, pos)
            bucket = round(angle / (math.pi / 4)) % 8
            rank = (abs(distance - 2.0), n)
            if bucket not in buckets or rank < buckets[bucket][0]:
                buckets[bucket] = (rank, n, distance, angle, paths[n])
        result = {viewpoint(n): {'heading': angle, 'elevation': math.atan2(
                    self.graph.nodes[n]['position'][1] - origin[1], max(distance, 1e-6)),
                    'distance': distance, 'path': path}
                  for _, n, distance, angle, path in [buckets[k] for k in sorted(buckets)]}
        if self.condition == 'asg':
            # A staircase is one physical G0 connector and may exceed the
            # local 4 m viewpoint radius. Never delete it via range/bearing
            # pruning. Offer one nearest landing per portal/destination level
            # whenever its entry is locally reachable. This is scene-only
            # action availability, not language/goal-conditioned planning.
            levels = {r['id']: r['level'] for r in self.public_asg.get('regions', [])}
            current_level = levels.get(self.graph.nodes[self.current].get('region'))
            crossings = {}
            for a, distance in lengths.items():
                data_a = self.graph.nodes[a]
                if data_a.get('kind') != 'portal_anchor':
                    continue
                level_a = levels.get(data_a.get('region'))
                if current_level is not None and level_a != current_level:
                    continue
                for b in self.graph[a]:
                    data_b = self.graph.nodes[b]
                    level_b = levels.get(data_b.get('region'))
                    if (data_b.get('kind') != 'portal_anchor' or level_a is None or
                            level_b is None or level_a == level_b or b in paths[a]):
                        continue
                    total = distance + self.graph[a][b]['weight']
                    key = (data_a['entity_id'], level_b)
                    rank = (total, a, b)
                    if key not in crossings or rank < crossings[key][0]:
                        crossings[key] = (rank, paths[a] + [b])
            for (total, a, b), path in crossings.values():
                pos = self.graph.nodes[b]['position']
                result[viewpoint(b)] = {'heading': bearing(origin, pos),
                    'elevation': math.atan2(pos[1]-origin[1], math.hypot(pos[0]-origin[0], pos[2]-origin[2])),
                    'distance': total, 'path': path}
        self.candidate_cache[self.current] = result
        return result

    def image(self, angle):
        from PIL import Image
        directory = self.cache / 'rgb' / self.condition
        directory.mkdir(parents=True, exist_ok=True)
        pos = self.graph.nodes[self.current]['position']
        key = digest([file_identity(self.config['render_geometry']), pos, round(angle, 8), 320, 240, 75, 1.25])
        path = directory / (key + '.jpg')
        depth_path = path.with_suffix('.depth.npy')
        if not path.exists() or (self.config.get('method') == 'navgpt' and not depth_path.exists()):
            if self.renderer is None:
                from reconav.baselines.common.renderer import OSMesaRendererProcess
                self.renderer = OSMesaRendererProcess(Path(self.config['render_geometry']), width=320, height=240, hfov=75)
            c, s = math.cos(angle), math.sin(angle)
            # Heading clockwise is rotation about the world Y axis by -angle.
            pose = np.array([[c, 0, -s, pos[0]], [0, 1, 0, pos[1] + 1.25],
                             [s, 0, c, pos[2]], [0, 0, 0, 1]], dtype=float)
            rgb, depth = self.renderer.render(pose)
            if not np.isfinite(depth).all() or not (depth > 0).any():
                raise RuntimeError('invalid CPU-rendered RGB-D observation')
            Image.fromarray(rgb).save(path, quality=90)
            np.save(depth_path, depth, allow_pickle=False)
        return str(path)

    def reset(self):
        if self.moves:
            raise RuntimeError('one World instance is one episode')
        return self._get_obs()

    def _get_obs(self):
        candidates = self.candidates()
        self.env.sims[0].navigable_dict = candidates
        base = {'instr_id': self.policy['benchmark_task_id'], 'instruction': self.policy['instruction'],
                'scan': self.scene_id, 'viewpoint': viewpoint(self.current), 'heading': self.heading,
                'elevation': 0.0, 'viewIndex': 12}
        if self.config['method'] == 'mapgpt':
            base['candidate'] = [dict(viewpointId=vp, pointId=12, idx=i + 1,
                                     absolute_heading=c['heading'], absolute_elevation=c['elevation'],
                                     image=self.image(c['heading']))
                                 for i, (vp, c) in enumerate(candidates.items())]
        else:
            if self.current not in self.caption_cache:
                images = [self.image(i * math.pi / 4) for i in range(8)]
                views = self.captioner(images)
                captions, detections = [], []
                for i, (view, image) in enumerate(zip(views, images)):
                    captions.append(view['description'])
                    depth = np.load(Path(image).with_suffix('.depth.npy'), allow_pickle=False)
                    objects = {}
                    for obj in view['objects']:
                        if not isinstance(obj, dict) or not isinstance(obj.get('label'), str):
                            continue
                        try:
                            u, v = float(obj['u']), float(obj['v'])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if not (0 <= u <= 1 and 0 <= v <= 1):
                            continue
                        row, col = min(depth.shape[0]-1, int(v*depth.shape[0])), min(depth.shape[1]-1, int(u*depth.shape[1]))
                        d = float(depth[row, col])
                        horizontal = (2*u-1)*math.tan(math.radians(75)/2)
                        vertical = (2*v-1)*math.tan(math.radians(75)/2)*240/320
                        distance = d*math.sqrt(1+horizontal**2+vertical**2)
                        if not 0 < distance <= 3:
                            continue
                        heading = (i*45+math.degrees(math.atan(horizontal))) % 360
                        label = obj['label'].strip()
                        if label and (label not in objects or distance < objects[label]['distance']):
                            objects[label] = {'heading': heading, 'distance': distance}
                    detections.append(objects)
                self.caption_cache[self.current] = (captions, detections)
            captions, detections = self.caption_cache[self.current]
            base.update(candidate={vp: {k: v for k, v in c.items() if k != 'path'} for vp, c in candidates.items()},
                        obs=captions, obs_summary=' '.join(captions), objects=detections)
        return [base]

    def move(self, vp):
        if vp not in self.candidates():
            raise ValueError('action is not among current navigable candidates')
        action = self.candidates()[vp]
        path = action['path']
        for a, b in zip(path, path[1:]):
            edge = self.graph.edges[a, b]
            poly = edge.get('polyline')
            if poly:
                if edge.get('direction') != a:
                    poly = list(reversed(poly))
                self.positions.extend(poly[1:])
            else:
                self.positions.append(self.graph.nodes[b]['position'])
        before = self.current
        self.current = self.raw_keys[vp]
        self.heading = action['heading']
        self.visited.append(self.current)
        self.moves.append({'from': viewpoint(before), 'to': vp, 'distance_m': action['distance'],
                           'physical_search_path': path})

    def step(self, actions):
        self.move(actions[0])
        return self._get_obs()

    def bfs_shortest_path(self, before, after):
        # Native NavGPT records logical viewpoints; full physical path is saved separately.
        return [before, after]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
