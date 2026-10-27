"""Reusable, lazily complete visibility queries on the fixed physical domain.

G0 is a warm graph, not an optimization certificate. This layer adds semantic
encounter boundaries without removing physical space, and enumerates *every*
legal same-component visibility edge of the published finite vertex domain.
Constraints are interpreted by the product search, not baked into G0.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import math
import os
import weakref

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.ops import nearest_points

from reconav.scene.geometry import Point3
from reconav.solver.qualifier_geometry import object_geometry, UnknownRelationGeometry
from reconav.scene.navigation import get_geometry_graph

QUERY_GRAPH_VERSION = "complete-visibility-relevant-semantic-vertices-v2"
_QUERY_GRAPHS: dict[tuple[int, str], object] = {}
_TOTAL_QUERY_CACHE_BYTES = int(
    os.environ.get("BENCHMARK_QUERY_CACHE_BYTES", str(256 * 1024 * 1024))
)
_VISIBILITY_CACHE_BYTES = max(1, _TOTAL_QUERY_CACHE_BYTES // 2)


@dataclass(frozen=True)
class QueryNode:
    region: int
    component: int
    x: float
    z: float


@dataclass(frozen=True)
class QueryEdge:
    target: int
    length: float
    portal_id: int | None = None


def _parts(shape):
    if hasattr(shape, "geoms"):
        for child in shape.geoms:
            yield from _parts(child)
    else:
        yield shape


def _vertices(shape):
    for part in _parts(shape):
        if part.is_empty:
            continue
        if part.geom_type == "Polygon":
            for ring in (part.exterior, *part.interiors):
                yield from list(ring.coords)[:-1]
        elif part.geom_type in {"LineString", "LinearRing"}:
            yield from part.coords
        elif part.geom_type == "Point":
            yield (part.x, part.y)


class CompleteQueryGraph:
    """Complete visibility is materialized by source, never by nearest-k.

    Node universe is scene-wide and task-independent: changing AVOID/COUNT or
    selecting a different candidate cannot change the baseline's route domain.
    Physical boundaries/configuration and portal transition contracts are read
    without modification. Circle discretization and endpoint domain are named
    explicitly in the certificate; this is not a robot-dynamics certificate.
    """
    def __init__(self, asg, physical, cancel_token=None):
        self.asg_ref = weakref.ref(asg)
        self.physical = physical
        self.nodes: list[QueryNode] = []
        self.by_component: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.by_coord: dict[tuple[int, int, float, float], int] = {}
        self.special_edges: dict[int, list[QueryEdge]] = defaultdict(list)
        self.neighbor_cache: OrderedDict[int, tuple[QueryEdge, ...]] = OrderedDict()
        self.neighbor_cache_sizes: dict[int, int] = {}
        self.neighbor_cache_bytes = 0
        self.neighbor_cache_hits = 0
        self.neighbor_cache_misses = 0
        self.neighbor_cache_evictions = 0
        self.object_shapes = {}
        self.object_nodes: dict[int, set[int]] = defaultdict(set)
        self.missing_semantic_objects: set[int] = set()
        self.region_nodes: dict[int, list[int]] = defaultdict(list)
        self.levels = {r.index: r.level for r in asg.regions}
        self.free_prepared = {}
        for region_id, physical_region in physical.regions.items():
            for component_id, component in enumerate(physical_region.components):
                if cancel_token is not None:
                    cancel_token.raise_if_requested()
                domain = component.buffer(physical.config.numeric_tolerance_m)
                shapely.prepare(domain)
                self.free_prepared[(region_id, component_id)] = domain
                for x, z in _vertices(component):
                    self.add_node(region_id, component_id, x, z)
                rep = component.representative_point()
                self.add_node(region_id, component_id, rep.x, rep.y)
        # Include ALL registered anchors, never only the winner's anchor.
        old_to_new = {}
        for node in physical.nodes:
            if node.local_component_id < 0:
                continue
            point = Point(node.x, node.y)
            component = physical.locate_local_component(node.region_id, point)
            if component is not None:
                old_to_new[node.node_id] = self.add_node(node.region_id, component, node.x, node.y)
        for connector in physical.portal_connectors:
            if not connector.validated:
                continue
            first, second = old_to_new.get(connector.node_a), old_to_new.get(connector.node_b)
            if first is None or second is None:
                # Contract was not materialized; never silently turn this into UNSAT.
                raise ValueError(f"registered portal {connector.portal_id} is missing a runtime landing")
            pa, pb = self.point3(first), self.point3(second)
            length = math.dist((pa.x, pa.y, pa.z), (pb.x, pb.y, pb.z))
            self.special_edges[first].append(QueryEdge(second, length, connector.portal_id))
            self.special_edges[second].append(QueryEdge(first, length, connector.portal_id))
        # Obstacle/free-space vertices and registered physical landings are the
        # task-independent route domain. Semantic approach vertices are needed
        # only for objects named by the current query; unrelated object zones
        # are not obstacles and cannot bend a shortest path.
        self.structural_size = len(self.nodes)
        self.semantic_nodes_by_object: dict[int, set[int]] = defaultdict(set)
        radius = physical.config.endpoint_clearance_m
        for obj in asg.objects:
            if cancel_token is not None:
                cancel_token.raise_if_requested()
            try:
                shape, level, _region = object_geometry(asg, obj.index)
            except UnknownRelationGeometry:
                self.missing_semantic_objects.add(obj.index)
                continue
            self.object_shapes[obj.index] = (shape, level, obj.region_index)
            # These are semantic waypoints, not additional physical obstacles.
            zone = shape.buffer(radius, quad_segs=physical.config.round_quad_segs, join_style=1)
            for region_id, physical_region in physical.regions.items():
                if self.levels[region_id] != level or (
                    region_id != obj.region_index and region_id not in obj.body_band_footprint_2d_by_region
                ):
                    continue
                for component_id, component in enumerate(physical_region.components):
                    if not component.intersects(zone):
                        continue
                    approach = component.intersection(zone)
                    # Include a circumscribed offset boundary for collision-free
                    # routes around a round semantic zone. Exact-radius contacts
                    # remain separate endpoint candidates; semantic tests below
                    # use analytic distance, not this rendering approximation.
                    outside = shape.buffer(
                        radius / math.cos(math.pi/(4.0*physical.config.round_quad_segs)),
                        quad_segs=physical.config.round_quad_segs, join_style=1,
                    ).boundary.intersection(component)
                    for x, z in _vertices(outside):
                        node_id = self.add_node(region_id, component_id, x, z)
                        self.semantic_nodes_by_object[obj.index].add(node_id)
                    for x, z in _vertices(approach):
                        node_id = self.add_node(region_id, component_id, x, z)
                        self.semantic_nodes_by_object[obj.index].add(node_id)
                        if region_id == obj.region_index:
                            self.object_nodes[obj.index].add(node_id)
                    for part in _parts(approach):
                        if part.is_empty:
                            continue
                        p = part.representative_point()
                        node_id = self.add_node(region_id, component_id, p.x, p.y)
                        self.semantic_nodes_by_object[obj.index].add(node_id)
                        if region_id == obj.region_index:
                            self.object_nodes[obj.index].add(node_id)
        self.base_size = len(self.nodes)
        self.array_cache: dict[tuple[int, int], np.ndarray] = {}
        for key, values in self.by_component.items():
            self.array_cache[key] = np.asarray([(self.nodes[i].x, self.nodes[i].z) for i in values], dtype=float)

    def add_node(self, region, component, x, z):
        key = (int(region), int(component), round(float(x), 9), round(float(z), 9))
        if key in self.by_coord:
            return self.by_coord[key]
        node_id = len(self.nodes)
        self.nodes.append(QueryNode(int(region), int(component), float(x), float(z)))
        self.by_coord[key] = node_id
        self.by_component[(int(region), int(component))].append(node_id)
        self.region_nodes[int(region)].append(node_id)
        return node_id

    def point3(self, node_id):
        node = self.nodes[node_id]
        return Point3(node.x, self.physical.floor_height(node.region), node.z)

    def neighbors(self, node_id, cancel_token=None):
        cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            self.neighbor_cache_hits += 1
            self.neighbor_cache.move_to_end(node_id)
            return cached
        self.neighbor_cache_misses += 1
        node = self.nodes[node_id]
        key = (node.region, node.component)
        ids = self.by_component[key]
        xy = self.array_cache[key]
        origin = np.asarray([node.x, node.z])
        output = list(self.special_edges.get(node_id, ()))
        domain = self.free_prepared[key]
        # Prepared/vectorized exact polygon predicates; cancellation remains bounded.
        for offset in range(0, len(ids), 256):
            if cancel_token is not None:
                cancel_token.raise_if_requested()
            subset = xy[offset:offset+256]
            lines = shapely.linestrings(np.stack((np.broadcast_to(origin, subset.shape), subset), axis=1))
            visible = shapely.covers(domain, lines)
            for j in np.flatnonzero(visible):
                target = ids[offset + int(j)]
                if target == node_id:
                    continue
                length = math.hypot(subset[j, 0] - node.x, subset[j, 1] - node.z)
                if length > 1e-12:
                    output.append(QueryEdge(target, length))
        result = tuple(output)
        estimated = 64 + 48 * len(result)
        self.neighbor_cache[node_id] = result
        self.neighbor_cache_sizes[node_id] = estimated
        self.neighbor_cache_bytes += estimated
        while (
            self.neighbor_cache_bytes > _VISIBILITY_CACHE_BYTES
            and len(self.neighbor_cache) > 1
        ):
            evicted, _value = self.neighbor_cache.popitem(last=False)
            self.neighbor_cache_bytes -= self.neighbor_cache_sizes.pop(evicted, 0)
            self.neighbor_cache_evictions += 1
        return result

    def cache_metrics(self):
        return {
            "visibility_budget_bytes": _VISIBILITY_CACHE_BYTES,
            "visibility_estimated_bytes": self.neighbor_cache_bytes,
            "visibility_entries": len(self.neighbor_cache),
            "visibility_hits": self.neighbor_cache_hits,
            "visibility_misses": self.neighbor_cache_misses,
            "visibility_evictions": self.neighbor_cache_evictions,
        }


class AttachedQuery:
    """A query adds the actual START without modifying a shared cached graph."""
    def __init__(
        self,
        base,
        asg,
        start_target,
        cancel_token=None,
        *,
        active_object_ids=(),
    ):
        self.base = base
        self.nodes = list(base.nodes)
        self.extra_neighbors: dict[int, list[QueryEdge]] = defaultdict(list)
        self.neighbor_cache: dict[int, tuple[QueryEdge, ...]] = {}
        active = set(range(base.structural_size))
        for object_id in active_object_ids:
            active.update(base.semantic_nodes_by_object.get(int(object_id), ()))
        self.active_base_nodes = frozenset(active)
        self.active_by_component = {
            key: tuple(node_id for node_id in values if node_id in active)
            for key, values in base.by_component.items()
        }
        self.active_array_cache = {
            key: np.asarray(
                [(base.nodes[node_id].x, base.nodes[node_id].z) for node_id in values],
                dtype=float,
            )
            for key, values in self.active_by_component.items()
        }
        self.start_ids: list[int] = []
        if start_target.kind == "pos":
            x, height, z = (float(v) for v in start_target.values)
            candidates = []
            for region_id in base.physical.regions:
                component = base.physical.locate_local_component(region_id, Point(x, z))
                if component is not None:
                    candidates.append((abs(base.physical.floor_height(region_id)-height), region_id, component))
            if not candidates:
                raise ValueError("actual START is outside the declared physical free-space domain")
            best = min(item[0] for item in candidates)
            # Geometry can overlap at a threshold, but it cannot change START level.
            selected = [item for item in candidates if item[0] <= best+1e-6]
            for _delta, region, component in selected:
                self.start_ids.append(self.attach(region, component, x, z, cancel_token))
                # Exact target-boundary contacts are legal query endpoints. Add
                # radial projections for every semantic object, not only GOAL.
                point = Point(x, z)
                for _obj_id in active_object_ids:
                    shape, level, obj_region = base.object_shapes[int(_obj_id)]
                    if obj_region != region:
                        continue
                    nearest = nearest_points(point, shape)[1]
                    length = point.distance(nearest)
                    radius = base.physical.config.endpoint_clearance_m
                    if length <= radius+1e-10:
                        continue
                    factor = radius / length
                    p = Point(nearest.x+(x-nearest.x)*factor, nearest.y+(z-nearest.y)*factor)
                    comp = base.physical.locate_local_component(region, p)
                    if comp is not None:
                        self.attach(region, comp, p.x, p.y, cancel_token)
        else:
            from reconav.solver.target_grounding import ground_target
            grounded = ground_target(asg, start_target, 1, start_target=start_target)
            if grounded.kind != "obj":
                raise ValueError("START supports POS or an object")
            if len(grounded.ids) != 1:
                raise ValueError("START object must be unique")
            object_id = next(iter(grounded.ids))
            self.start_ids = sorted(base.object_nodes.get(object_id, ()))
        if not self.start_ids:
            raise ValueError("START has no legal endpoint")

    def attach(self, region, component, x, z, cancel_token):
        key = (int(region), int(component), round(float(x), 9), round(float(z), 9))
        existing = self.base.by_coord.get(key)
        if existing is not None:
            return existing
        for i in range(self.base.base_size, len(self.nodes)):
            node = self.nodes[i]
            if node.region == region and node.component == component and math.hypot(node.x-x, node.z-z) <= 1e-9:
                return i
        node_id = len(self.nodes)
        node = QueryNode(region, component, x, z)
        self.nodes.append(node)
        candidates = [*self.active_by_component[(region, component)],
                      *(i for i in range(self.base.base_size, node_id)
                        if self.nodes[i].region == region and self.nodes[i].component == component)]
        domain = self.base.free_prepared[(region, component)]
        for offset in range(0, len(candidates), 256):
            if cancel_token is not None:
                cancel_token.raise_if_requested()
            ids = candidates[offset:offset+256]
            xy = np.asarray([(self.nodes[i].x,self.nodes[i].z) for i in ids])
            if not len(xy):
                continue
            lines = shapely.linestrings(np.stack((np.broadcast_to([x,z],xy.shape),xy),axis=1))
            for j in np.flatnonzero(shapely.covers(domain,lines)):
                target = ids[int(j)]
                length = math.hypot(self.nodes[target].x-x,self.nodes[target].z-z)
                self.extra_neighbors[node_id].append(QueryEdge(target,length))
                self.extra_neighbors[target].append(QueryEdge(node_id,length))
        return node_id

    def neighbors(self, node_id, token=None):
        static = self._active_neighbors(node_id, token) if node_id < self.base.base_size else ()
        return (*static,*self.extra_neighbors.get(node_id,()))

    def _active_neighbors(self, node_id, token=None):
        cached = self.neighbor_cache.get(node_id)
        if cached is not None:
            return cached
        node = self.nodes[node_id]
        key = (node.region, node.component)
        ids = self.active_by_component[key]
        xy = self.active_array_cache[key]
        origin = np.asarray([node.x, node.z])
        output = list(self.base.special_edges.get(node_id, ()))
        domain = self.base.free_prepared[key]
        for offset in range(0, len(ids), 256):
            if token is not None:
                token.raise_if_requested()
            subset = xy[offset:offset+256]
            if not len(subset):
                continue
            lines = shapely.linestrings(
                np.stack((np.broadcast_to(origin, subset.shape), subset), axis=1)
            )
            visible = shapely.covers(domain, lines)
            for j in np.flatnonzero(visible):
                target = ids[offset + int(j)]
                if target == node_id:
                    continue
                length = math.hypot(
                    subset[j, 0] - node.x,
                    subset[j, 1] - node.z,
                )
                if length > 1e-12:
                    output.append(QueryEdge(target, length))
        result = tuple(output)
        self.neighbor_cache[node_id] = result
        return result

    def point3(self,node_id):
        node=self.nodes[node_id]
        return Point3(node.x,self.base.physical.floor_height(node.region),node.z)


def get_complete_query_graph(asg, token=None):
    physical=get_geometry_graph(asg)
    key=(id(asg),physical.graph_id)
    graph=_QUERY_GRAPHS.get(key)
    if graph is None or graph.asg_ref() is not asg:
        graph=CompleteQueryGraph(asg,physical,token)
        # Bound cross-scene process memory; the disk G0 remains available.
        if len(_QUERY_GRAPHS)>=3:
            _QUERY_GRAPHS.pop(next(iter(_QUERY_GRAPHS)))
        _QUERY_GRAPHS[key]=graph
    return graph
