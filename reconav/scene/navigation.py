"""Region-bounded physical graph and query-local semantic overlays.

Each annotated/generated region polygon is its own feasible-space envelope.
Complete scene-mesh occupancy in the robot body-height band is subtracted
inside that envelope.  Registered portals are the only inter-region edges and
their transition bridges are intentionally exempt from ordinary occupancy.
Object collision footprints remain semantic geometry for goal/via/avoid.
"""
from __future__ import annotations
from reconav import paths as release_paths
import fcntl
import hashlib
import heapq
import json
import math
import os
import pickle
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, triangulate, unary_union
from shapely.prepared import prep
from reconav.scene.asg import ASG, ASGObjectEntity, ASGPortalEntity
from reconav.scene.physical import BODY_BOTTOM_OFFSET_M, ENDPOINT_OBJECT_CLEARANCE_M, ROBOT_DIAMETER_M, ROBOT_HEIGHT_M, ROBOT_RADIUS_M
from reconav.scene.mesh_io import load_glb_triangles
from reconav.scene.mesh_physical import MeshFloorEvidence, MeshHeightPatch, MeshProjectionConfig, MeshSource, build_mesh_height_patches, estimate_mesh_floor_elevations, height_patch_specs, locate_scene_mesh, mesh_file_sha256, scene_mesh_source
DEFAULT_GEOMETRY_CACHE_ROOT = Path((str(release_paths.cache_root() / 'geometry')))
GEOMETRY_CACHE_ENV = 'NAV_GEOMETRY_CACHE_ROOT'
GEOMETRY_ALGORITHM_VERSION = 'region-portal-mesh-band-g0-v21'
GRAPH_STORAGE_VERSION = 5
ROUND_QUAD_SEGS = 16
NUMERIC_TOLERANCE_M = 1e-08
POINT_LOCATION_TOLERANCE_M = 1e-06
PORTAL_LANDING_SEARCH_DEPTH_M = 1.25

@dataclass(frozen=True)
class PhysicalGraphConfig:
    robot_radius_m: float = ROBOT_RADIUS_M
    robot_diameter_m: float = ROBOT_DIAMETER_M
    robot_height_m: float = ROBOT_HEIGHT_M
    body_bottom_offset_m: float = BODY_BOTTOM_OFFSET_M
    endpoint_clearance_m: float = ENDPOINT_OBJECT_CLEARANCE_M
    round_quad_segs: int = ROUND_QUAD_SEGS
    numeric_tolerance_m: float = NUMERIC_TOLERANCE_M
    point_location_tolerance_m: float = POINT_LOCATION_TOLERANCE_M
    portal_landing_search_depth_m: float = PORTAL_LANDING_SEARCH_DEPTH_M
    mesh_projection: MeshProjectionConfig = MeshProjectionConfig()
    algorithm_version: str = GEOMETRY_ALGORITHM_VERSION
    storage_version: int = GRAPH_STORAGE_VERSION

    def validate(self) -> None:
        if self.robot_radius_m <= 0 or self.robot_height_m <= 0:
            raise ValueError('robot dimensions must be positive')
        if abs(self.robot_diameter_m - 2.0 * self.robot_radius_m) > 1e-12:
            raise ValueError('robot diameter must equal twice its radius')
        if self.body_bottom_offset_m < 0:
            raise ValueError('body bottom offset cannot be negative')
        if self.endpoint_clearance_m < self.robot_radius_m:
            raise ValueError('endpoint clearance cannot be smaller than robot radius')
        if self.round_quad_segs < 4 or self.numeric_tolerance_m <= 0:
            raise ValueError('invalid polygonization configuration')
        if self.mesh_projection.raster_resolution_m <= 0:
            raise ValueError('mesh raster resolution must be positive')

    def conservative_buffer_distance(self, nominal_distance_m: float) -> float:
        return float(nominal_distance_m) / math.cos(math.pi / (4.0 * float(self.round_quad_segs)))

@dataclass(frozen=True)
class GraphNode:
    node_id: int
    region_id: int
    sheet_id: int
    local_component_id: int
    x: float
    y: float
    kind: str
    entity_id: int | None = None

@dataclass(frozen=True)
class AnchorRecord:
    node_id: int
    kind: str
    entity_id: int
    region_id: int
    sheet_id: int
    local_component_id: int
    x: float
    y: float
    support_component_area_m2: float

@dataclass(frozen=True)
class PortalConnector:
    portal_id: int
    region_a: int
    region_b: int
    node_a: int
    node_b: int
    cross_floor: bool
    length_2d_m: float
    validated: bool
    reason: str | None = None
    annotation_anchor_a: tuple[float, float] | None = None
    annotation_anchor_b: tuple[float, float] | None = None
    runtime_anchor_a: tuple[float, float] | None = None
    runtime_anchor_b: tuple[float, float] | None = None
    anchor_move_a_m: float | None = None
    anchor_move_b_m: float | None = None
    source_segment_a: tuple[tuple[float, float], tuple[float, float]] | None = None
    source_segment_b: tuple[tuple[float, float], tuple[float, float]] | None = None
    minimum_clearance_m: float | None = None

@dataclass
class PhysicalSheetGeometry:
    sheet_id: int
    region_ids: tuple[int, ...]
    level_ids: tuple[int, ...]
    floor_elevations_m: tuple[float, ...]
    body_band_m: tuple[float, float]
    support: BaseGeometry
    center_support: BaseGeometry
    original_blockers: BaseGeometry
    inflated_blockers: BaseGeometry
    free_space: BaseGeometry
    components: tuple[Polygon, ...]
    support_face_indices: tuple[int, ...]
    occupancy_face_indices: tuple[int, ...]
    metrics: dict[str, object] = field(default_factory=dict)

@dataclass
class RegionPhysicalGeometry:
    """One region's independently bounded collision-free navigation domain."""
    region_id: int
    level_id: int
    sheet_id: int
    support: BaseGeometry
    original_blockers: BaseGeometry
    inflated_blockers: BaseGeometry
    stair_footprint_obstacles: BaseGeometry
    free_space: BaseGeometry
    components: tuple[Polygon, ...]
    metrics: dict[str, object] = field(default_factory=dict)

@dataclass
class GeometryGraph:
    data_source: str
    scene_id: str
    graph_id: str
    physical_id: str
    semantic_id: str
    config: PhysicalGraphConfig
    mesh_source: MeshSource
    region_floor_elevations: dict[int, float]
    floor_evidence: dict[int, MeshFloorEvidence]
    sheets: dict[int, PhysicalSheetGeometry]
    regions: dict[int, RegionPhysicalGeometry]
    nodes: tuple[GraphNode, ...]
    adjacency: dict[int, tuple[tuple[int, float], ...]]
    region_local_nodes: dict[tuple[int, int], tuple[int, ...]]
    object_anchors: dict[int, tuple[AnchorRecord, ...]]
    portal_anchors: dict[int, dict[int, tuple[AnchorRecord, ...]]]
    region_anchors: dict[int, tuple[AnchorRecord, ...]]
    portal_connectors: tuple[PortalConnector, ...]
    region_to_global_component: dict[tuple[int, int], int]
    node_to_global_component: dict[int, int]
    build_metrics: dict[str, object]
    cache_dir: str | None = None
    cache_hit: bool = False

    def region_geometry(self, region_id: int) -> RegionPhysicalGeometry:
        try:
            return self.regions[int(region_id)]
        except KeyError as error:
            raise ValueError(f'region {region_id} has no physical navigation domain') from error

    def floor_height(self, region_id: int) -> float:
        try:
            return float(self.region_floor_elevations[int(region_id)])
        except KeyError as error:
            raise ValueError(f'region {region_id} has no resolved mesh floor') from error

    def locate_local_component(self, region_id: int, point: Point) -> int | None:
        physical = self.region_geometry(region_id)
        for index, component in enumerate(physical.components):
            if component.covers(point):
                return index
        close = [(index, component.distance(point)) for index, component in enumerate(physical.components) if component.distance(point) <= self.config.point_location_tolerance_m]
        return min(close, key=lambda item: (item[1], item[0]))[0] if close else None

    def global_component_for_point(self, region_id: int, point: Point) -> int | None:
        local = self.locate_local_component(region_id, point)
        if local is None:
            return None
        return self.region_to_global_component.get((int(region_id), local))

    def entity_global_components(self, kind: str, entity_id: int) -> frozenset[int]:
        anchors: Iterable[AnchorRecord]
        if kind in {'obj', 'object'}:
            anchors = self.object_anchors.get(int(entity_id), ())
        elif kind == 'portal':
            anchors = (anchor for values in self.portal_anchors.get(int(entity_id), {}).values() for anchor in values)
        elif kind == 'region':
            anchors = self.region_anchors.get(int(entity_id), ())
        elif kind == 'level':
            anchors = (anchor for region_id, values in self.region_anchors.items() if self.regions[region_id].level_id == int(entity_id) for anchor in values)
        else:
            return frozenset()
        return frozenset((self.region_to_global_component[anchor.region_id, anchor.local_component_id] for anchor in anchors if (anchor.region_id, anchor.local_component_id) in self.region_to_global_component))

    def reachable_entities(self, source_kind: str, source_id: int, target_kind: str, target_id: int) -> bool:
        return bool(self.entity_global_components(source_kind, source_id) & self.entity_global_components(target_kind, target_id))

    def portal_crossing(self, portal_id: int, from_region: int, to_region: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
        values = self.portal_crossings(portal_id, from_region, to_region)
        return values[0] if values else None

    def portal_crossings(self, portal_id: int, from_region: int, to_region: int) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
        """Return every validated bounded runtime-anchor pair."""
        candidates = [item for item in self.portal_connectors if item.portal_id == int(portal_id) and item.validated and ({item.region_a, item.region_b} == {int(from_region), int(to_region)})]
        if not candidates:
            return ()
        result = []
        for connector in sorted(candidates, key=lambda item: (item.length_2d_m, item.node_a, item.node_b)):
            first = self.nodes[connector.node_a]
            second = self.nodes[connector.node_b]
            if connector.region_a == int(from_region):
                result.append(((first.x, first.y), (second.x, second.y)))
            else:
                result.append(((second.x, second.y), (first.x, first.y)))
        return tuple(result)
_MEMORY_CACHE: dict[tuple[str, str, str], GeometryGraph] = {}
_OVERLAY_CACHE: dict[tuple[str, int, tuple[int, ...]], BaseGeometry] = {}
_PREPARED_GEOMETRY: dict[int, object] = {}
_TOLERANT_GEOMETRY: dict[tuple[int, float], BaseGeometry] = {}
_ENDPOINT_VISIBILITY_CACHE: dict[tuple[str, int, int, float, float], tuple[tuple[int, float], ...]] = {}

def geometry_cache_root(value: Path | None=None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get(GEOMETRY_CACHE_ENV)
    return Path(configured).expanduser().resolve() if configured else DEFAULT_GEOMETRY_CACHE_ROOT

def load_or_build_geometry_graph(*, data_source: str, asg: ASG, cache_root: Path | None=None, config: PhysicalGraphConfig=PhysicalGraphConfig(), force_rebuild: bool=False) -> GeometryGraph:
    """Load the shared region/portal G0 for a scene, publishing atomically."""
    config.validate()
    source = str(data_source).strip()
    if not source:
        raise ValueError('data_source is required for physical cache isolation')
    asg.data_source = source
    if not force_rebuild:
        from reconav.scene.graph_io import load_if_available
        portable = load_if_available(asg, config)
        if portable is not None:
            return portable
    ids = _fingerprints(asg, config)
    graph_id = ids['graph_id']
    memory_key = (source, str(asg.scene_id), graph_id)
    if not force_rebuild and memory_key in _MEMORY_CACHE:
        graph = _MEMORY_CACHE[memory_key]
        graph.cache_hit = True
        setattr(asg, '_geometry_graph_handle', graph)
        return graph
    root = geometry_cache_root(cache_root)
    scene_dir = root / source / str(asg.scene_id)
    scene_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = scene_dir.parent / f'.{asg.scene_id}.lock'
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not force_rebuild:
            loaded = _load_completed_cache(scene_dir, source, asg.scene_id, graph_id)
            if loaded is not None:
                loaded.cache_hit = True
                _MEMORY_CACHE[memory_key] = loaded
                setattr(asg, '_geometry_graph_handle', loaded)
                return loaded
        started = time.perf_counter()
        graph = build_geometry_graph(data_source=source, asg=asg, graph_id=graph_id, physical_id=ids['physical_id'], semantic_id=ids['semantic_id'], config=config)
        graph.build_metrics['build_seconds'] = time.perf_counter() - started
        _publish_cache(scene_dir, graph)
        graph.cache_dir = str(scene_dir)
        graph.cache_hit = False
        _MEMORY_CACHE[memory_key] = graph
        setattr(asg, '_geometry_graph_handle', graph)
        return graph

def get_geometry_graph(asg: ASG) -> GeometryGraph:
    source = getattr(asg, 'data_source', None) or os.environ.get('NAV_DATA_SOURCE')
    if not source:
        raise ValueError('ASG has no data_source; load it with reconav.scene.io.load_asg or call load_or_build_geometry_graph(data_source=..., asg=...)')
    loaded = getattr(asg, '_geometry_graph_handle', None)
    if isinstance(loaded, GeometryGraph) and loaded.data_source == str(source) and (loaded.scene_id == str(asg.scene_id)) and (loaded.config.algorithm_version == GEOMETRY_ALGORITHM_VERSION):
        return loaded
    graph = load_or_build_geometry_graph(data_source=str(source), asg=asg)
    setattr(asg, '_geometry_graph_handle', graph)
    return graph


def _fingerprints(asg: ASG, config: PhysicalGraphConfig) -> dict[str, str]:
    mesh_path = locate_scene_mesh(asg)
    physical_payload = {'scene_id': str(asg.scene_id), 'mesh_sha256': mesh_file_sha256(mesh_path), 'mesh_resolved_path': str(mesh_path.resolve()), 'config': asdict(config), 'floor_height_inputs': {str(region.index): {'level_id': int(region.level), 'level_floor_hint_m': float(asg.get_level(region.level).floor_elevation_mean), 'annotation_floor_elevation_m': float(region.get_height())} for region in sorted(asg.regions, key=lambda value: value.index)}, 'region_feasible_boundaries': {str(region.index): _polygon_payload(region.polygon_2d) for region in sorted(asg.regions, key=lambda value: value.index)}, 'stair_footprint_obstacles': {str(obj.index): {'label': str(obj.label), 'footprint': _polygon_payload(obj.navigation_footprint_2d())} for obj in sorted(asg.objects, key=lambda value: value.index) if _is_stair_surface(obj)}, 'cross_height_channels': [_portal_physical_payload(value) for value in asg.portals]}
    physical_id = _json_digest(physical_payload)
    semantic_payload = {**({'object_endpoint_reachability_fallback': True} if asg.object_endpoint_reachability_fallback else {}), **({'object_endpoint_clearance_m': object_approach_clearance(asg, config)} if asg.object_endpoint_clearance_m is not None else {}), **({'endpoint_region_inset_m': endpoint_region_inset(asg)} if asg.endpoint_region_inset_m is not None else {}), 'regions': {str(region.index): _polygon_payload(region.polygon_2d) for region in sorted(asg.regions, key=lambda value: value.index)}, 'objects': {str(obj.index): {'region': int(obj.region_index), 'raw': _polygon_payload(obj.raw_footprint_2d), 'collision': _polygon_payload(obj.collision_footprint_2d), 'body': {str(region_id): _polygon_payload(values) for region_id, values in sorted(obj.body_band_footprint_2d_by_region.items())}} for obj in sorted(asg.objects, key=lambda value: value.index)}, 'portals': [_portal_semantic_payload(value) for value in asg.portals]}
    semantic_id = _json_digest(semantic_payload)
    graph_id = _json_digest({'physical_id': physical_id, 'semantic_id': semantic_id, 'storage_version': GRAPH_STORAGE_VERSION})
    return {'physical_id': physical_id, 'semantic_id': semantic_id, 'graph_id': graph_id}

def build_geometry_graph(*, data_source: str, asg: ASG, graph_id: str | None=None, physical_id: str | None=None, semantic_id: str | None=None, config: PhysicalGraphConfig=PhysicalGraphConfig()) -> GeometryGraph:
    """Build region-local free domains, warm graphs, and portal channels."""
    config.validate()
    asg.data_source = str(data_source)
    ids = _fingerprints(asg, config)
    graph_id = graph_id or ids['graph_id']
    physical_id = physical_id or ids['physical_id']
    semantic_id = semantic_id or ids['semantic_id']
    mesh_path = locate_scene_mesh(asg)
    triangles, _colors = load_glb_triangles(mesh_path)
    mesh_source = scene_mesh_source(mesh_path, triangles)
    floor_evidence = estimate_mesh_floor_elevations(asg, triangles, config=config.mesh_projection)
    region_floor_elevations = {region_id: evidence.resolved_floor_elevation_m for region_id, evidence in floor_evidence.items()}
    specs = height_patch_specs(asg, bottom_offset_m=config.body_bottom_offset_m, robot_height_m=config.robot_height_m, config=config.mesh_projection, region_floor_elevations=region_floor_elevations)
    mesh_patches = build_mesh_height_patches(asg, triangles, specs, floor_evidence=floor_evidence, robot_radius_m=config.robot_radius_m, conservative_buffer_radius_m=config.conservative_buffer_distance(config.robot_radius_m), round_quad_segs=config.round_quad_segs, config=config.mesh_projection)
    region_levels = {int(region.index): int(region.level) for region in asg.regions}
    sheets = {patch_id: _sheet_from_mesh_patch(patch, region_levels) for patch_id, patch in mesh_patches.items()}
    region_to_sheet = {region_id: sheet_id for sheet_id, sheet in sheets.items() for region_id in sheet.region_ids}
    regions: dict[int, RegionPhysicalGeometry] = {}
    for region in asg.regions:
        sheet = sheets[region_to_sheet[int(region.index)]]
        semantic_shape = _region_shape(region)
        stair_footprints = _stair_footprint_obstacles(asg, semantic_shape)
        region_free = _clean(semantic_shape.difference(sheet.inflated_blockers).difference(stair_footprints))
        region_components = tuple(_polygons(region_free))
        metrics = dict(sheet.metrics)
        metrics.update({'semantic_region_id': int(region.index), 'resolved_floor_elevation_m': region_floor_elevations[int(region.index)], 'floor_evidence': asdict(floor_evidence[int(region.index)]), 'semantic_region_area_m2': float(semantic_shape.area), 'semantic_region_free_area_m2': float(region_free.area), 'region_boundary_used_as_physical_edge': True, 'region_boundary_center_inset_m': 0.0, 'stair_footprint_obstacle_margin_m': 0.0, 'stair_footprint_obstacle_area_m2': float(stair_footprints.area), 'portal_transition_exempt_from_ordinary_obstacles': True, 'ordinary_object_footprints_used_as_physical_blockers': False, 'stair_object_footprints_used_as_physical_blockers': True, 'stair_guards_used_as_physical_blockers': False})
        regions[int(region.index)] = RegionPhysicalGeometry(int(region.index), int(region.level), sheet.sheet_id, semantic_shape, sheet.original_blockers, sheet.inflated_blockers, stair_footprints, region_free, region_components, metrics)
    nodes: list[GraphNode] = []
    edge_weights: dict[tuple[int, int], float] = {}
    region_local_nodes: dict[tuple[int, int], list[int]] = defaultdict(list)

    def add_node(region_id: int, component_id: int, point: Point, kind: str, *, entity_id: int | None=None) -> int:
        if kind in {'boundary', 'steiner'}:
            for node_id in region_local_nodes[region_id, component_id]:
                prior = nodes[node_id]
                if prior.kind in {'boundary', 'steiner'} and abs(prior.x - point.x) <= 1e-10 and (abs(prior.y - point.y) <= 1e-10):
                    return node_id
        physical = regions[int(region_id)]
        node_id = len(nodes)
        nodes.append(GraphNode(node_id, int(region_id), physical.sheet_id, int(component_id), round(float(point.x), 8), round(float(point.y), 8), str(kind), None if entity_id is None else int(entity_id)))
        region_local_nodes[region_id, component_id].append(node_id)
        return node_id

    def add_edge(first: int, second: int, weight: float | None=None) -> None:
        if first == second:
            return
        a, b = (nodes[first], nodes[second])
        value = math.hypot(a.x - b.x, a.y - b.y) if weight is None else float(weight)
        key = (min(first, second), max(first, second))
        edge_weights[key] = min(value, edge_weights.get(key, math.inf))
    visibility_bridge_failures: list[dict[str, int]] = []
    for region_id, physical in sorted(regions.items()):
        for component_id, component in enumerate(physical.components):
            points = _component_vertices(component)
            representative = component.representative_point()
            points.append(representative)
            coordinate_to_node: dict[tuple[float, float], int] = {}
            for point in points:
                coordinate_to_node.setdefault(_coord(point), add_node(region_id, component_id, point, 'steiner' if point.equals(representative) else 'boundary'))
            for first, second in _sparse_visibility_segments(component, points):
                first_id, second_id = (coordinate_to_node[_coord(first)], coordinate_to_node[_coord(second)])
                line = LineString([(first.x, first.y), (second.x, second.y)])
                if _base_segment_is_valid(line, physical, config):
                    add_edge(first_id, second_id)
            remaining = _routing_cluster_count(region_local_nodes[region_id, component_id], edge_weights)
            if remaining > 1:
                visibility_bridge_failures.append({'region_id': region_id, 'local_component_id': component_id, 'routing_clusters': remaining})
    object_anchors: dict[int, list[AnchorRecord]] = defaultdict(list)
    for obj in sorted(asg.objects, key=lambda value: value.index):
        candidate_regions = set(obj.body_band_footprint_2d_by_region)
        if obj.region_index in regions:
            candidate_regions.add(int(obj.region_index))
        for region_id in sorted(candidate_regions):
            physical = regions.get(region_id)
            if physical is None:
                continue
            footprint = _semantic_object_geometry(obj)
            if footprint.is_empty:
                continue
            approach = object_approach_domain(asg, config, region_id, physical.free_space, footprint)
            for part in _polygons(approach):
                component_id = _component_for_geometry(physical.components, part)
                if component_id is None:
                    continue
                for point in _anchor_points(part, footprint):
                    node_id = add_node(region_id, component_id, point, 'object_anchor', entity_id=obj.index)
                    if not _attach_node(node_id, nodes, region_local_nodes[region_id, component_id], edge_weights, physical, config):
                        continue
                    object_anchors[obj.index].append(AnchorRecord(node_id, 'object', int(obj.index), region_id, physical.sheet_id, component_id, float(point.x), float(point.y), float(part.area)))
    region_anchors: dict[int, list[AnchorRecord]] = defaultdict(list)
    for region in sorted(asg.regions, key=lambda value: value.index):
        physical = regions[int(region.index)]
        for component_id, component in enumerate(physical.components):
            domains = _polygons(endpoint_region_domain(asg, region.index, component))
            for part in domains:
                for point in _anchor_points(part, GeometryCollection()):
                    node_id = add_node(region.index, component_id, point, 'region_anchor', entity_id=region.index)
                    if not _attach_node(node_id, nodes, region_local_nodes[region.index, component_id], edge_weights, physical, config):
                        continue
                    region_anchors[region.index].append(AnchorRecord(node_id, 'region', int(region.index), int(region.index), physical.sheet_id, component_id, float(point.x), float(point.y), float(part.area)))
    portal_anchors: dict[int, dict[int, list[AnchorRecord]]] = defaultdict(lambda: defaultdict(list))
    portal_attempts: dict[int, dict[int, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    portal_segments: dict[tuple[int, int], LineString] = {}
    for portal in sorted(asg.portals, key=lambda value: value.index):
        for region_id in (int(portal.region_a), int(portal.region_b)):
            physical, region = (regions.get(region_id), asg.get_region(region_id))
            if physical is None or region is None:
                continue
            segment = _portal_segment(portal, region_id)
            if segment is None:
                portal_attempts[portal.index][region_id].append({'status': 'rejected', 'reason': 'cannot resolve bounded portal cross-section from segment/endpoints/direction/anchors'})
                continue
            portal_segments[portal.index, region_id] = segment
            raw_anchor = portal.get_anchor(region_id)
            annotation = Point(raw_anchor.x, raw_anchor.y)
            candidates = _portal_side_candidates(annotation, segment, _region_shape(region), physical, config)
            if portal.label == 'stair' and candidates:
                reference = _region_shape(region).representative_point()
                preferred_component = min({component_id for component_id, _point, _source in candidates}, key=lambda component_id: (physical.components[component_id].distance(reference), -physical.components[component_id].area, component_id))
                candidates = [item for item in candidates if item[0] == preferred_component]
            for component_id, point, source in candidates:
                node_id = add_node(region_id, component_id, point, 'portal_anchor', entity_id=portal.index)
                attached = _attach_node(node_id, nodes, region_local_nodes[region_id, component_id], edge_weights, physical, config)
                portal_attempts[portal.index][region_id].append({'runtime_anchor': [float(point.x), float(point.y)], 'annotation_anchor': [float(annotation.x), float(annotation.y)], 'move_m': float(point.distance(annotation)), 'source': source, 'component_id': component_id, 'attached': attached})
                if attached:
                    portal_anchors[portal.index][region_id].append(AnchorRecord(node_id, 'portal', int(portal.index), region_id, physical.sheet_id, component_id, float(point.x), float(point.y), 0.0))
    connectors: list[PortalConnector] = []
    for portal in sorted(asg.portals, key=lambda value: value.index):
        side_a = portal_anchors.get(portal.index, {}).get(portal.region_a, ())
        side_b = portal_anchors.get(portal.index, {}).get(portal.region_b, ())
        if not side_a or not side_b:
            connectors.append(PortalConnector(int(portal.index), int(portal.region_a), int(portal.region_b), side_a[0].node_id if side_a else -1, side_b[0].node_id if side_b else -1, regions[portal.region_a].level_id != regions[portal.region_b].level_id or portal.label == 'stair', math.inf, False, 'one or both landing domains have no graph-attached runtime anchor'))
            continue
        segment_a = portal_segments[portal.index, portal.region_a]
        segment_b = portal_segments[portal.index, portal.region_b]
        for anchor_a, anchor_b in _portal_anchor_pairs(side_a, side_b, segment_a, segment_b):
            valid, reason, clearance = _portal_pair_is_valid(portal, anchor_a, anchor_b, segment_a, segment_b, regions, nodes, config)
            raw_a, raw_b = (portal.get_anchor(portal.region_a), portal.get_anchor(portal.region_b))
            runtime_a, runtime_b = (Point(anchor_a.x, anchor_a.y), Point(anchor_b.x, anchor_b.y))
            connector = PortalConnector(int(portal.index), int(portal.region_a), int(portal.region_b), anchor_a.node_id, anchor_b.node_id, regions[portal.region_a].level_id != regions[portal.region_b].level_id or portal.label == 'stair', float(runtime_a.distance(runtime_b)), valid, reason, (float(raw_a.x), float(raw_a.y)), (float(raw_b.x), float(raw_b.y)), (anchor_a.x, anchor_a.y), (anchor_b.x, anchor_b.y), float(runtime_a.distance(Point(raw_a.x, raw_a.y))), float(runtime_b.distance(Point(raw_b.x, raw_b.y))), _line_tuple(segment_a), _line_tuple(segment_b), clearance)
            connectors.append(connector)
            if valid:
                region_a = asg.get_region(portal.region_a)
                region_b = asg.get_region(portal.region_b)
                if region_a is None or region_b is None:
                    continue
                transition_length = math.sqrt(runtime_a.distance(runtime_b) ** 2 + (region_floor_elevations[region_a.index] - region_floor_elevations[region_b.index]) ** 2)
                add_edge(anchor_a.node_id, anchor_b.node_id, transition_length)
    adjacency_lists: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (first, second), weight in edge_weights.items():
        adjacency_lists[first].append((second, weight))
        adjacency_lists[second].append((first, weight))
    adjacency = {node.node_id: tuple(sorted(adjacency_lists.get(node.node_id, ()))) for node in nodes}
    region_to_global = _global_component_index(regions, connectors, nodes)
    node_to_global = {node.node_id: region_to_global[node.region_id, node.local_component_id] for node in nodes}
    build_metrics: dict[str, object] = {'physical_id': physical_id, 'semantic_id': semantic_id, 'mesh_source': asdict(mesh_source), 'height_patch_count': len(sheets), 'floor_evidence': {str(region_id): asdict(value) for region_id, value in sorted(floor_evidence.items())}, 'region_count': len(regions), 'local_component_count': sum((len(region.components) for region in regions.values())), 'global_component_count': len(set(region_to_global.values())), 'node_count': len(nodes), 'edge_count': len(edge_weights), 'validated_portal_connector_count': sum((item.validated for item in connectors)), 'portal_connector_failure_count': sum((not item.validated for item in connectors)), 'portal_failures': {str(portal_id): {str(region_id): attempts for region_id, attempts in sorted(sides.items()) if not any((bool(item.get('attached')) for item in attempts))} for portal_id, sides in sorted(portal_attempts.items()) if any((not any((bool(item.get('attached')) for item in attempts)) for attempts in sides.values()))}, 'portal_runtime_anchor_attempts': {str(portal_id): {str(region_id): attempts for region_id, attempts in sorted(sides.items())} for portal_id, sides in sorted(portal_attempts.items())}, 'object_anchor_count': sum((len(value) for value in object_anchors.values())), 'anchored_object_count': sum((bool(value) for value in object_anchors.values())), 'total_free_area_m2': sum((float(region.free_space.area) for region in regions.values())), 'sheet_metrics': {str(key): value.metrics for key, value in sorted(sheets.items())}, 'connectivity_basis': 'region-local free components connected only by registered portal/stair transition contracts; portal bridges are exempt from ordinary body-band obstacles', 'visibility_materialization': 'Delaunay warm graph with component bridging; query endpoints receive complete visible attachment and direct endpoint chords', 'visibility_bridge_failures': visibility_bridge_failures, 'old_physical_paths_active': {'floor_mesh_support': False, 'object_traversal_blocker_union': False, 'object_collision_fallback': False, 'stair_guards': False}, 'stair_protection_contract': 'stair-object navigation footprints are zero-extra-margin in-region obstacles; only registered stair portal bridges cross them'}
    return GeometryGraph(str(data_source), str(asg.scene_id), graph_id, physical_id, semantic_id, config, mesh_source, region_floor_elevations, floor_evidence, sheets, regions, tuple(nodes), adjacency, {key: tuple(values) for key, values in region_local_nodes.items()}, {key: tuple(values) for key, values in object_anchors.items()}, {portal_id: {region_id: tuple(values) for region_id, values in sides.items()} for portal_id, sides in portal_anchors.items()}, {key: tuple(values) for key, values in region_anchors.items()}, tuple(connectors), region_to_global, node_to_global, build_metrics)


def segment_has_physical_clearance(graph: GeometryGraph, region_id: int, first: tuple[float, float], second: tuple[float, float]) -> bool:
    line = LineString([first, second])
    physical = graph.region_geometry(region_id)
    return _base_segment_is_valid(line, physical, graph.config) and _line_clear_of_geometry(line, physical.original_blockers, graph.config.robot_radius_m, graph.config.numeric_tolerance_m)

def segment_is_in_free_space(graph: GeometryGraph, region_id: int, first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Fast conservative predicate used while materializing candidate edges."""
    return _base_segment_is_valid(LineString([first, second]), graph.region_geometry(region_id), graph.config)

def visible_component_nodes(graph: GeometryGraph, region_id: int, component_id: int, point: tuple[float, float]) -> tuple[tuple[int, float], ...]:
    """Return and cache the complete visible warm-node set for an endpoint."""
    key = (graph.graph_id, int(region_id), int(component_id), round(float(point[0]), 7), round(float(point[1]), 7))
    cached = _ENDPOINT_VISIBILITY_CACHE.get(key)
    if cached is not None:
        return cached
    result: list[tuple[int, float]] = []
    for target in graph.region_local_nodes.get((int(region_id), int(component_id)), ()):
        node = graph.nodes[target]
        if node.kind not in {'boundary', 'steiner'}:
            continue
        target_xy = (node.x, node.y)
        if segment_is_in_free_space(graph, region_id, point, target_xy):
            result.append((target, math.hypot(float(point[0]) - node.x, float(point[1]) - node.y)))
    value = tuple(result)
    _ENDPOINT_VISIBILITY_CACHE[key] = value
    return value


def _sheet_from_mesh_patch(patch: MeshHeightPatch, region_levels: dict[int, int]) -> PhysicalSheetGeometry:
    return PhysicalSheetGeometry(patch.spec.patch_id, patch.spec.region_ids, tuple(sorted({region_levels[value] for value in patch.spec.region_ids})), patch.spec.floor_elevations_m, (patch.spec.body_band_bottom_m, patch.spec.body_band_top_m), patch.support, patch.center_support, patch.raw_occupancy, patch.inflated_occupancy, patch.free_space, tuple(_polygons(patch.free_space)), patch.support_face_indices, patch.occupancy_face_indices, patch.metrics)

def _semantic_object_geometry(obj: ASGObjectEntity) -> BaseGeometry:
    """Task geometry retained independently from ordinary mesh obstacles."""
    return _union((polygon.to_shapely() for polygon in obj.navigation_footprint_2d()))

def _is_stair_surface(obj: ASGObjectEntity) -> bool:
    """Return whether an object describes the walkable stair/ramp surface.

    Railings and handrails remain governed by the scene-mesh body band.  They
    must not enlarge the zero-margin stair footprint used to protect the
    opening.
    """
    label = ' '.join(str(obj.label).strip().lower().replace('_', ' ').replace('-', ' ').split())
    if 'rail' in label:
        return False
    words = set(label.split())
    return bool(words.intersection({'stair', 'stairs', 'staircase', 'step', 'steps', 'ramp'}))

def _stair_footprint_obstacles(asg: ASG, semantic_region: BaseGeometry) -> BaseGeometry:
    """Clip stair navigation footprints to a region with no added margin."""
    return _union((_semantic_object_geometry(obj).intersection(semantic_region) for obj in asg.objects if _is_stair_surface(obj)))

def _portal_segment(portal: ASGPortalEntity, region_id: int) -> LineString | None:
    """Resolve a bounded portal cross-section without requiring manual data.

    Manual side-specific clearance segments are accepted as the strongest
    evidence.  ASGs built from scratch normally provide opening endpoints or
    a tangent/normal plus ``clear_width``; the final fallback recovers the
    cross-section from the two landing anchors and measured width.
    """
    if int(region_id) not in {int(portal.region_a), int(portal.region_b)}:
        return None
    side = 'a' if int(region_id) == int(portal.region_a) else 'b'
    candidates = [item for item in portal.clearance_segments if str(item.get('side')) in {side, 'shared'} and isinstance(item.get('endpoints'), list) and (len(item['endpoints']) == 2)]
    anchor = portal.get_anchor(region_id)
    anchor_point = Point(anchor.x, anchor.y)
    segments = [LineString(item['endpoints']) for item in candidates if _valid_line_endpoints(item['endpoints'])]
    if segments:
        return min(segments, key=lambda value: (value.distance(anchor_point), value.length))
    if portal.endpoint_a is not None and portal.endpoint_b is not None:
        segment = LineString([(float(portal.endpoint_a.x), float(portal.endpoint_a.y)), (float(portal.endpoint_b.x), float(portal.endpoint_b.y))])
        if segment.length > NUMERIC_TOLERANCE_M:
            return segment
    width = None if portal.clear_width is None else float(portal.clear_width)
    if width is None or not math.isfinite(width) or width <= NUMERIC_TOLERANCE_M:
        return None
    direction: tuple[float, float] | None = None
    if portal.tangent is not None:
        direction = (float(portal.tangent.x), float(portal.tangent.y))
    elif portal.normal is not None:
        direction = (-float(portal.normal.y), float(portal.normal.x))
    else:
        other = portal.anchor_b if int(region_id) == int(portal.region_a) else portal.anchor_a
        travel_x = float(other.x) - float(anchor.x)
        travel_y = float(other.y) - float(anchor.y)
        direction = (-travel_y, travel_x)
    norm = math.hypot(*direction)
    if norm <= NUMERIC_TOLERANCE_M:
        return None
    unit_x, unit_y = (direction[0] / norm, direction[1] / norm)
    half = width / 2.0
    return LineString([(float(anchor.x) - half * unit_x, float(anchor.y) - half * unit_y), (float(anchor.x) + half * unit_x, float(anchor.y) + half * unit_y)])

def _valid_line_endpoints(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        points = [tuple((float(coordinate) for coordinate in point)) for point in value]
    except (TypeError, ValueError):
        return False
    return all((len(point) >= 2 and math.isfinite(point[0]) and math.isfinite(point[1]) for point in points)) and math.dist(points[0][:2], points[1][:2]) > NUMERIC_TOLERANCE_M

def _portal_side_candidates(annotation: Point, segment: LineString, semantic_region: BaseGeometry, physical: RegionPhysicalGeometry, config: PhysicalGraphConfig, *, free_space: BaseGeometry | None=None) -> list[tuple[int, Point, str]]:
    search_depth = min(config.portal_landing_search_depth_m, max(0.5, float(annotation.distance(segment)) + 0.35))
    active_free = physical.free_space if free_space is None else free_space
    domain = _clean(active_free.intersection(segment.buffer(search_depth, cap_style=2, join_style=1)))
    semantic_domain = _clean(domain.intersection(semantic_region.buffer(max(0.05, config.mesh_projection.raster_resolution_m * 2.0), join_style=1)))
    source_prefix = 'semantic-side'
    if semantic_domain.is_empty:
        semantic_domain, source_prefix = (domain, 'physical-side-semantic-mapping-uncertain')
    candidates: list[tuple[int, Point, str]] = []
    sample_points = [segment.interpolate(fraction, normalized=True) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)] + [annotation]
    for part_index, part in enumerate(_polygons(semantic_domain)):
        component_id = _component_for_geometry(physical.components, part)
        if component_id is None:
            continue
        point_sources: list[tuple[Point, str]] = [(part.representative_point(), 'component-representative'), (nearest_points(annotation, part)[1], 'annotation-nearest'), (nearest_points(segment.interpolate(0.5, normalized=True), part)[1], 'segment-center-nearest')]
        point_sources.extend(((nearest_points(sample, part)[1], f'segment-sample-{index:02d}') for index, sample in enumerate(sample_points) if sample.distance(part) <= search_depth + config.numeric_tolerance_m))
        for point, source in point_sources:
            if _prepared_covers(active_free, point):
                candidates.append((component_id, point, f'{source_prefix}:{part_index}:{source}'))
    deduped: dict[tuple[int, float, float], tuple[int, Point, str]] = {}
    for item in candidates:
        deduped.setdefault((item[0], round(float(item[1].x), 4), round(float(item[1].y), 4)), item)
    return sorted(deduped.values(), key=lambda item: (item[1].distance(annotation), item[0], item[1].x, item[1].y))

def _portal_pair_is_valid(portal: ASGPortalEntity, anchor_a: AnchorRecord, anchor_b: AnchorRecord, segment_a: LineString, segment_b: LineString, regions: dict[int, RegionPhysicalGeometry], nodes: Sequence[GraphNode], config: PhysicalGraphConfig) -> tuple[bool, str | None, float | None]:
    measured_width = float(portal.clear_width) if portal.clear_width is not None else min(float(segment_a.length), float(segment_b.length))
    if measured_width + config.numeric_tolerance_m < config.robot_diameter_m:
        return (False, 'measured clear width is smaller than robot diameter', None)
    first, second = (Point(nodes[anchor_a.node_id].x, nodes[anchor_a.node_id].y), Point(nodes[anchor_b.node_id].x, nodes[anchor_b.node_id].y))
    if portal.label == 'stair':
        return (True, None, measured_width / 2.0)
    line = LineString([(first.x, first.y), (second.x, second.y)])
    if line.length > 2.0 * config.portal_landing_search_depth_m + segment_a.length:
        return (False, 'runtime side pair exceeds bounded opening/landing domain', None)
    blockers = _union([regions[portal.region_a].original_blockers, regions[portal.region_b].original_blockers])
    clearance = math.inf if blockers.is_empty else float(line.distance(blockers))
    return (True, None, clearance)

def _portal_anchor_pairs(side_a: Sequence[AnchorRecord], side_b: Sequence[AnchorRecord], segment_a: LineString, segment_b: LineString) -> list[tuple[AnchorRecord, AnchorRecord]]:
    """Enumerate the complete bounded runtime-anchor product.

    Candidate sets have already been clipped to the registered segment/landing
    domains.  Rejecting a pair before the physical connector check would make
    the channel incomplete and could turn a graph-search miss into a false
    portal failure.
    """

    def fraction(anchor: AnchorRecord, segment: LineString) -> float:
        return float(segment.project(Point(anchor.x, anchor.y), normalized=True))
    fractions_a = {anchor.node_id: fraction(anchor, segment_a) for anchor in side_a}
    fractions_b = {anchor.node_id: fraction(anchor, segment_b) for anchor in side_b}
    pairs = [(anchor_a, anchor_b) for anchor_a in side_a for anchor_b in side_b]
    return sorted(pairs, key=lambda value: (abs(fractions_a[value[0].node_id] - fractions_b[value[1].node_id]), value[0].node_id, value[1].node_id))

def _base_segment_is_valid(line: LineString, physical: RegionPhysicalGeometry, config: PhysicalGraphConfig) -> bool:
    return _covers_with_tolerance(physical.free_space, line, config.numeric_tolerance_m)

def _line_clear_of_geometry(line: LineString, geometry: BaseGeometry, clearance_m: float, tolerance_m: float) -> bool:
    return geometry.is_empty or line.distance(geometry) + tolerance_m >= clearance_m

def _prepared_covers(geometry: BaseGeometry, candidate: BaseGeometry) -> bool:
    key = id(geometry)
    prepared = _PREPARED_GEOMETRY.get(key)
    if prepared is None:
        prepared = prep(geometry)
        _PREPARED_GEOMETRY[key] = prepared
    return bool(prepared.covers(candidate))

def _covers_with_tolerance(geometry: BaseGeometry, candidate: BaseGeometry, tolerance_m: float) -> bool:
    key = (id(geometry), float(tolerance_m))
    expanded = _TOLERANT_GEOMETRY.get(key)
    if expanded is None:
        expanded = geometry.buffer(float(tolerance_m), join_style=1)
        _TOLERANT_GEOMETRY[key] = expanded
    return _prepared_covers(expanded, candidate)

def _attach_node(node_id: int, nodes: list[GraphNode], candidates: Sequence[int], edge_weights: dict[tuple[int, int], float], physical: RegionPhysicalGeometry, config: PhysicalGraphConfig) -> bool:
    point = Point(nodes[node_id].x, nodes[node_id].y)
    attached = 0
    ranked = heapq.nsmallest(64, (((nodes[node_id].x - nodes[candidate].x) ** 2 + (nodes[node_id].y - nodes[candidate].y) ** 2, candidate) for candidate in candidates if candidate != node_id and nodes[candidate].kind in {'boundary', 'steiner'}))
    examined = {candidate for _distance_squared, candidate in ranked}

    def attach(ranked_candidates: Iterable[tuple[float, int]]) -> None:
        nonlocal attached
        for _distance_squared, candidate in ranked_candidates:
            if attached >= 8:
                return
            if candidate == node_id:
                continue
            other = nodes[candidate]
            line = LineString([(point.x, point.y), (other.x, other.y)])
            if not _base_segment_is_valid(line, physical, config):
                continue
            key = (min(node_id, candidate), max(node_id, candidate))
            edge_weights[key] = min(float(line.length), edge_weights.get(key, math.inf))
            attached += 1
    attach(ranked)
    if not attached:
        attach(sorted((((nodes[node_id].x - nodes[candidate].x) ** 2 + (nodes[node_id].y - nodes[candidate].y) ** 2, candidate) for candidate in candidates if candidate != node_id and candidate not in examined and (nodes[candidate].kind in {'boundary', 'steiner'})), key=lambda item: (item[0], item[1])))
    return attached > 0 or _prepared_covers(physical.free_space, point)

def _routing_cluster_count(node_ids: Sequence[int], edge_weights: dict[tuple[int, int], float]) -> int:
    """Report sparse warm-graph clusters without changing physical truth."""
    selected = set(node_ids)
    active = {value for edge in edge_weights for value in edge if value in selected}
    if not active:
        return 0
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in edge_weights:
        if first in active and second in active:
            adjacency[first].append(second)
            adjacency[second].append(first)
    unseen = set(active)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for following in adjacency.get(current, ()):
                if following in unseen:
                    unseen.remove(following)
                    stack.append(following)
    return count

def _sparse_visibility_segments(component: Polygon, points: Sequence[Point]) -> Iterator[tuple[Point, Point]]:
    candidates = GeometryCollection([component, MultiPoint(points)])
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for triangle in triangulate(candidates, tolerance=0.0, edges=False):
        coordinates = list(triangle.exterior.coords[:-1])
        for first, second in zip(coordinates, coordinates[1:] + coordinates[:1]):
            a, b = ((round(float(first[0]), 8), round(float(first[1]), 8)), (round(float(second[0]), 8), round(float(second[1]), 8)))
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                yield (Point(*key[0]), Point(*key[1]))

def _global_component_index(regions: dict[int, RegionPhysicalGeometry], connectors: Sequence[PortalConnector], nodes: Sequence[GraphNode]) -> dict[tuple[int, int], int]:
    keys = [(region_id, component_id) for region_id, region in sorted(regions.items()) for component_id in range(len(region.components))]
    parent = {key: key for key in keys}

    def find(key: tuple[int, int]) -> tuple[int, int]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        a, b = (find(first), find(second))
        if a != b:
            parent[max(a, b)] = min(a, b)
    for connector in connectors:
        if connector.validated and connector.node_a >= 0 and (connector.node_b >= 0):
            first, second = (nodes[connector.node_a], nodes[connector.node_b])
            union((first.region_id, first.local_component_id), (second.region_id, second.local_component_id))
    roots = sorted({find(key) for key in keys})
    root_ids = {root: index for index, root in enumerate(roots)}
    return {key: root_ids[find(key)] for key in keys}

def _load_completed_cache(scene_dir: Path, data_source: str, scene_id: str, graph_id: str) -> GeometryGraph | None:
    manifest_path, graph_path = (scene_dir / 'manifest.json', scene_dir / 'g0.pkl')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'complete' or manifest.get('data_source') != str(data_source) or manifest.get('scene_id') != str(scene_id) or (manifest.get('graph_id') != graph_id) or (int(manifest.get('storage_version', -1)) != GRAPH_STORAGE_VERSION) or (_file_sha256(graph_path) != manifest.get('g0_sha256')):
            return None
        with graph_path.open('rb') as stream:
            graph = pickle.load(stream)
        if not isinstance(graph, GeometryGraph) or graph.graph_id != graph_id:
            return None
        graph.cache_dir = str(scene_dir)
        return graph
    except (OSError, ValueError, TypeError, KeyError, pickle.PickleError, EOFError):
        return None

def _publish_cache(scene_dir: Path, graph: GeometryGraph) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.g0.', suffix='.tmp', dir=scene_dir)
    temporary, graph_path = (Path(temporary_name), scene_dir / 'g0.pkl')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _file_sha256(temporary)
        os.replace(temporary, graph_path)
        _atomic_json(scene_dir / 'manifest.json', {'status': 'complete', 'data_source': graph.data_source, 'scene_id': graph.scene_id, 'graph_id': graph.graph_id, 'physical_id': graph.physical_id, 'semantic_id': graph.semantic_id, 'storage_version': GRAPH_STORAGE_VERSION, 'algorithm_version': GEOMETRY_ALGORITHM_VERSION, 'physical_config': asdict(graph.config), 'mesh_source': asdict(graph.mesh_source), 'derived_conservative_robot_buffer_m': graph.config.conservative_buffer_distance(graph.config.robot_radius_m), 'g0_file': graph_path.name, 'g0_sha256': digest, 'build_metrics': graph.build_metrics})
    finally:
        temporary.unlink(missing_ok=True)

def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _portal_physical_payload(portal: ASGPortalEntity) -> dict[str, object]:

    def point(value: object) -> list[float] | None:
        if value is None:
            return None
        return [round(float(value.x), 8), round(float(value.y), 8)]
    return {'index': int(portal.index), 'label': str(portal.label), 'regions': [int(portal.region_a), int(portal.region_b)], 'anchors': [point(portal.anchor_a), point(portal.anchor_b)], 'clear_width': portal.clear_width, 'center': point(portal.center), 'tangent': point(portal.tangent), 'normal': point(portal.normal), 'endpoints': [point(portal.endpoint_a), point(portal.endpoint_b)], 'clearance_segments': portal.clearance_segments}

def _portal_semantic_payload(portal: ASGPortalEntity) -> dict[str, object]:
    return {**_portal_physical_payload(portal), 'anchors': [[round(float(portal.anchor_a.x), 8), round(float(portal.anchor_a.y), 8)], [round(float(portal.anchor_b.x), 8), round(float(portal.anchor_b.y), 8)]]}

def _polygon_payload(values: Iterable[object]) -> list[list[list[float]]]:
    return [[[round(float(point.x), 8), round(float(point.y), 8)] for point in polygon.vertices] for polygon in values]

def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def _region_shape(region: object) -> BaseGeometry:
    return _union((polygon.to_shapely() for polygon in region.polygon_2d))

def _anchor_points(domain: Polygon, target: BaseGeometry) -> list[Point]:
    points = [domain.representative_point()]
    if not target.is_empty:
        points.append(nearest_points(target, domain)[1])
    if domain.exterior.length > 0:
        points.extend((domain.exterior.interpolate(fraction, normalized=True) for fraction in (0.0, 0.25, 0.5, 0.75)))
    return _dedupe_points((point for point in points if domain.covers(point)))

def _component_for_geometry(components: Sequence[Polygon], geometry: BaseGeometry) -> int | None:
    point = geometry.representative_point()
    candidates = [(index, float(component.intersection(geometry).area)) for index, component in enumerate(components) if component.covers(point) or component.intersects(geometry)]
    return max(candidates, key=lambda item: (item[1], -item[0]))[0] if candidates else None

def _component_vertices(component: Polygon) -> list[Point]:
    points = [Point(x, y) for x, y in component.exterior.coords[:-1]]
    for interior in component.interiors:
        points.extend((Point(x, y) for x, y in interior.coords[:-1]))
    return _dedupe_points(points)

def _coord(point: Point) -> tuple[float, float]:
    return (round(float(point.x), 8), round(float(point.y), 8))

def _dedupe_points(points: Iterable[Point]) -> list[Point]:
    return list({_coord(point): point for point in points}.values())


def _line_tuple(line: LineString) -> tuple[tuple[float, float], tuple[float, float]]:
    coordinates = list(line.coords)
    return ((float(coordinates[0][0]), float(coordinates[0][1])), (float(coordinates[-1][0]), float(coordinates[-1][1])))

def _union(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    values = [value for value in geometries if value is not None and (not value.is_empty)]
    if not values:
        return GeometryCollection()
    return _clean(unary_union(values) if len(values) > 1 else values[0])

def _clean(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty or geometry.is_valid:
        return geometry
    cleaned = geometry.buffer(0)
    return cleaned if not cleaned.is_empty else geometry

def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 1e-07 else []
    return [value for value in getattr(geometry, 'geoms', ()) if isinstance(value, Polygon) and (not value.is_empty) and (value.area > 1e-07)]

def object_approach_clearance(asg: ASG, config: PhysicalGraphConfig) -> float:
    """A map may request closer object endpoints without weakening semantics."""
    requested = asg.object_endpoint_clearance_m
    if requested is None:
        return config.endpoint_clearance_m
    requested = float(requested)
    if not math.isfinite(requested) or not config.robot_radius_m <= requested <= config.endpoint_clearance_m:
        raise ValueError('map object approach clearance must be finite and no looser than the semantic threshold')
    return requested

def endpoint_region_inset(asg: ASG) -> float:
    requested = asg.endpoint_region_inset_m
    if requested is None:
        return 0.0
    requested = float(requested)
    if not math.isfinite(requested) or not 0.0 <= requested <= 0.5:
        raise ValueError('map endpoint region inset must be finite and between 0 and 0.5m')
    return requested

def endpoint_region_domain(asg: ASG, region_id: int, free: BaseGeometry) -> BaseGeometry:
    """Keep destination candidates away from uncertain reconstructed room cuts.

    Physical free space, semantic encounters and avoidance domains stay intact.
    A map without the optional policy follows the original endpoint domain.
    """
    inset = endpoint_region_inset(asg)
    if not inset:
        return free
    region = asg.get_region(int(region_id))
    if region is None:
        return GeometryCollection()
    return _clean(free.intersection(_region_shape(region).buffer(-inset, join_style=2)))

def object_approach_domain(asg, config, region_id, free, footprint):
    """Honor a map's close approach policy without removing reachable objects.

    The optional reconstruction policy adapts each free component's destination
    radius only when its close approach is empty. It never exceeds the original
    semantic encounter radius or relaxes robot collision geometry.
    """
    domain = endpoint_region_domain(asg, region_id, free)
    radius = object_approach_clearance(asg, config)
    near = footprint.buffer(radius, quad_segs=config.round_quad_segs, join_style=1, cap_style=1)
    if not asg.object_endpoint_reachability_fallback:
        return _clean(domain.intersection(near))
    if domain.is_empty:
        domain = free
    parts = []
    for component in _polygons(domain):
        approach = component.intersection(near)
        if approach.area < 1e-08:
            distance = component.distance(footprint)
            adaptive = min(config.endpoint_clearance_m, max(radius, distance + 0.04))
            approach = component.intersection(footprint.buffer(adaptive, quad_segs=config.round_quad_segs, join_style=1, cap_style=1))
        if not approach.is_empty:
            parts.append(approach)
    return _clean(unary_union(parts))
