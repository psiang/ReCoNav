"""Execute LIMP's language, OSG, DFA and bi-level planning pipeline on ASG scenes.

Scene entities and G0 vertices provide the spatial and motion interfaces."""
from reconav import paths as release_paths
import ast
import copy
import math
import os
import re
import random
import string
import sys
import time
import traceback
from collections import defaultdict
from types import SimpleNamespace
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from reconav.baselines.common.protocol import file_digest, write
from reconav.baselines.common.upstream import definitions, module
from reconav.baselines.common.formal_language import PredictionFailure

def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value

def osg_elements(records):
    result = []
    for key, record in records.items():
        labels = {record['label'], record['label'].replace(' ', '_')}
        if record['kind'] == 'portal':
            labels.add('portal')
        for label in sorted(labels):
            result.append({'mask_id': key, 'mask_label': label, 'source_label': record['label'], 'worldframe_3d_position': [record['position'][0], record['position'][2], record['position'][1]], 'spatial_constraints': []})
    return result

def physical_quotient(source):
    """FMT identifies vertices by coordinates; co-located ASG entity anchors
    must not arbitrarily discard each other's physical edges.
    Never merge different rooms that merely touch at a geometric boundary.
    """
    groups = defaultdict(list)
    for n, d in source.nodes(data=True):
        groups[tuple(d['position'])].append(n)
    mapping = {}
    graph = nx.Graph()
    aliases = {}
    for pos, names in groups.items():
        regions = {source.nodes[n].get('region') for n in names} - {None}
        if len(regions) > 1:
            raise RuntimeError('ambiguous cross-region coincident graph vertices')
        representative = 'START' if 'START' in names else names[0]
        aliases[representative] = names
        graph.add_node(representative, position=list(pos))
        mapping.update({n: representative for n in names})
    for a, b, data in source.edges(data=True):
        u, v = (mapping[a], mapping[b])
        if u != v and (not graph.has_edge(u, v) or data['weight'] < graph.edges[u, v]['weight']):
            graph.add_edge(u, v, weight=data['weight'], original_edge=[a, b], direction=u)
    return (graph, aliases)

def expand_physical_walk(canonical_path, graph, world):
    result = ['START']
    for u, v in zip(canonical_path, canonical_path[1:]):
        if u == v:
            continue
        edge = graph.edges[u, v]
        a, b = edge['original_edge'] if edge['direction'] == u else edge['original_edge'][::-1]
        if result[-1] != a:
            if world.graph.nodes[result[-1]]['position'] != world.graph.nodes[a]['position']:
                raise RuntimeError('nonzero jump when expanding coincident ASG anchors')
            world.graph.add_edge(result[-1], a, weight=0.0, duplicate_pose_equivalence=True)
            result.append(a)
        result.append(b)
    return result

def run(frontend, records, world, output):
    import spot
    root = release_paths.upstream_path('robotlimp')
    sys.path.insert(0, str(root))
    from limp.language.temporal_logic import dfa
    common = {'re': re, 'spot': spot, 'copy': copy, 'string': string, 'np': np, 'nx': nx, 'dfa': dfa, 'eval': ast.literal_eval}
    gen = definitions(root / 'limp/utils/gen_utils.py', ['_get_spot_format', 'spotify', 'get_std_format', 'ltl2dfa', 'build_dfa_graph', 'get_spatial_referents', 'extract_instruction_referent_details', 'parse_spatial_relations'], common)
    try:
        formula, encoding = gen['spotify'](frontend['lifted_formula'], frontend['spatial_predicates'])
        task_dfa, dfa_graph = gen['ltl2dfa'](encoding, formula)
    except (KeyError, IndexError, ValueError, AssertionError, SyntaxError, UnboundLocalError, TypeError) as error:
        raise PredictionFailure('original_LIMP_DFA_cannot_convert_prediction: ' + str(error)) from error
    referents = gen['get_spatial_referents'](encoding)
    spatial = definitions(root / 'open-spatial-grounding/osg/spatial_relationships.py', None, {'re': re, 'np': np, 'parse_spatial_relations': gen['parse_spatial_relations']})
    all_elements = osg_elements(records)
    for element in all_elements:
        for descriptor in referents.get(element['mask_label'], []):
            try:
                satisfied, comparatives = spatial['check_spatial_predicate_satisfaction'](element, descriptor, all_elements)
            except Exception as error:
                if str(error).startswith('Unrecognized spatial constraint:'):
                    raise PredictionFailure('original_OSG_unsupported_emitted_predicate: ' + descriptor) from error
                raise
            element['spatial_constraints'].append({'constraint': descriptor, 'satisfies_spatial_constraint': bool(satisfied), 'satisfying_comparative_instances': comparatives})
    write(output / 'grounding.json', {'encoding': encoding, 'original_spotify_formula': str(formula), 'referents': referents, 'elements': jsonable(all_elements), 'human_verification': False, 'spatial_grounder': 'original OSG exact labels and spatial predicates; no semantic synonyms/oracle rerank', 'scene_label_interface': 'space/underscore spelling aliases; typed ASG portals also exposed as portal; same instance ID', 'DFA_edges': task_dfa.edges, 'DFA_accepting': task_dfa.accepting_states})
    planner_elements = copy.deepcopy(all_elements)
    for element in planner_elements:
        element['worldframe_3d_position'][2] = records[element['mask_id']]['floor_height']
    fmt = module(root / 'limp/planner/fmt.py', 'limp_original_fmt')
    random.seed(0)
    fmt_utils = definitions(root / 'limp/utils/fmt_utils.py', ['sample_goals'], {'np': np, 'random': random})
    planner = definitions(root / 'limp/planner/multi_level_planner.py', None, {'nx': nx, 'np': np, 'copy': copy, 'time': time, 'os': os, 'traceback': traceback, 'all_element_with_label': spatial['all_element_with_label'], 'fmt_utils': SimpleNamespace(**fmt_utils)})
    graph, aliases = physical_quotient(world.graph)
    ids = list(graph)
    positions = np.asarray([graph.nodes[n]['position'] for n in ids])[:, [0, 2, 1]]
    index = {n: i for i, n in enumerate(ids)}
    position_tree = cKDTree(positions)
    plan_attempts = []
    latest_values = {}
    nav_positions = {}
    native_positions = planner['get_predicate_satisfying_positions']

    def capture_positions(*args):
        result = native_positions(*args)
        nav_positions.update(result[0])
        return result
    planner['get_predicate_satisfying_positions'] = capture_positions

    def node(point):
        distance, i = position_tree.query(point)
        if distance > 1e-07:
            raise RuntimeError('LIMP graph interface received a non-G0 pose')
        return ids[int(i)]

    def generate_tpsm(pointcloud, pois, values, nearness_threshold, **kwargs):
        latest_values.clear()
        latest_values.update(values)
        entries = []
        for pred, points in pois.items():
            for j, p in enumerate(points):
                for i in position_tree.query_ball_point(p, nearness_threshold):
                    entries.append({'point_index': i, 'point_position': positions[i].tolist(), 'point_value': values.get(pred, 0), 'origin_predicate': pred, 'origin_predicate_index': j, 'origin_predicate_position': np.asarray(p).tolist()})
        if not entries:
            return ({}, [])
        return ({}, entries)
    original_truth = planner['get_all_value_map_point_true_predicates']
    planner['get_all_value_map_point_true_predicates'] = lambda value_map, *a: original_truth(value_map, *a) if value_map else []

    def obstacles(pointcloud, truth, *args, **kwargs):
        allowed = {task_dfa.state}
        allowed.update((task_dfa.check_progression_state(task_dfa.state, p['all_true_predicates']) for p in truth if p['point_value'] == 1))
        return {'blocked': {ids[int(p['point_index'])] for p in truth if p['point_value'] == -1}, 'state': task_dfa.state, 'allowed_states': allowed, 'values': dict(latest_values)}

    def obstacle_map(domain, grouped, resolution, *args):
        goals = {}
        for key, points in grouped.items():
            for p in points:
                goals[tuple(p)] = key
        return (domain, goals, np.zeros(3), positions.max(0))

    class GraphFMT(fmt.FMTPlanner):

        def add_map_design(self, domain):
            self.domain = domain
            self.node_list = [p.copy() for p, n in zip(positions, ids) if n not in domain['blocked']]
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(len(self.node_list)))

        def check_collision(self, src, dst):
            a = node(src)
            if a in self.domain['blocked']:
                return False
            if dst is None:
                return True
            b = node(dst)
            if b in self.domain['blocked'] or not (a == b or graph.has_edge(a, b)):
                return False
            samples = np.linspace(src, dst, max(2, math.ceil(np.linalg.norm(dst - src) / 0.1) + 1))
            for point in samples:
                true = [p for p, locations in nav_positions.items() if np.any(np.linalg.norm(np.asarray(locations) - point, axis=1) <= 0.5)]
                if any((self.domain['values'].get(p) == -1 for p in true)):
                    return False
                if task_dfa.check_progression_state(self.domain['state'], true) not in self.domain['allowed_states']:
                    return False
            return True

        def plan(self, start, goals, heuristic_weight=0.0):
            if not any(goals.values()) or not self.check_collision(start, None):
                return {'path': np.array([]), 'cost': 0, 'n_steps': 0, 'goal_flag': 0, 'goal': np.asarray(start), 'goal_origin_predicate': 'Unknown'}
            class GraphNeighbors:

                def __init__(self, points):
                    self.points = points
                    self.lookup = {}
                    for i, p in enumerate(points):
                        self.lookup.setdefault(node(p), []).append(i)

                def query_ball_point(self, p, r):
                    center = node(p)
                    adjacent = [center, *graph[center]]
                    return [i for n in adjacent for i in self.lookup.get(n, []) if np.linalg.norm(self.points[i] - p) <= r]
            previous = fmt.cKDTree
            fmt.cKDTree = GraphNeighbors
            try:
                result = super().plan(start, goals, heuristic_weight)
            finally:
                fmt.cKDTree = previous
            plan_attempts.append(jsonable(result))
            return result
    planner.update(fmt=SimpleNamespace(FMTPlanner=GraphFMT), generate_tpsm=generate_tpsm, get_obstacle_map_pointcloud=obstacles, generate_obstacle_map=obstacle_map, save_progressive_planning_step_artifacts=lambda *a, **kw: None)
    native_motion = planner['motion_level_planning']

    def motion(*args, **kwargs):
        result = native_motion(*args, **kwargs)
        if result[0] is None:
            raise planner['NoDFAGoalNodes']('original FMT/backtracking exhausted physical goals')
        return result
    planner['motion_level_planning'] = motion
    write(output / 'asg_bridge_manifest.json', {'scene_interfaces': ['OSG observations -> ASG entities', 'raster point-cloud/sampling -> complete scene G0 vertices', 'KD radius/collision -> G0 adjacency', 'rover destinations projected to their scene floor'], 'coincident_scene_anchor_equivalence': {'raw_vertices': world.graph.number_of_nodes(), 'physical_vertices': graph.number_of_nodes()}, 'source_sha256': {str(root / p): file_digest(root / p) for p in ('limp/planner/multi_level_planner.py', 'limp/planner/fmt.py', 'open-spatial-grounding/osg/spatial_relationships.py')}})
    initial_pos = positions[index['START']]
    saved = planner['progressive_motion_planner'](initial_pos, task_dfa, dfa_graph, None, planner_elements, encoding, nearness_value=0.5, map_resolution=1.0, robot_motion_type='2D', visualize=False, tmp_fldr=str(output) + '/', show_color_bars=False)
    path = ['START']
    for step in saved['world_plan'].values():
        if step['action'] != 'navigation':
            raise PredictionFailure('generated_non_navigation_skill_not_executable_in_navigation_benchmark')
        for p in step['path']:
            n = node(p)
            if n != path[-1]:
                path.append(n)
    write(output / 'original_plan.json', jsonable(saved))
    path = expand_physical_walk(path, graph, world)
    return (path, {'termination': 'limp_plan_complete' if task_dfa.in_terminal_state() else 'limp_no_feasible_plan', 'final_dfa_state': task_dfa.state, 'accepting_states': task_dfa.accepting_states, 'FMT_attempts': plan_attempts, 'executor': 'original LIMP bi-level/FMT search + G0 map interface', 'goal_nearness_m': 0.5, 'human_verification': False})
