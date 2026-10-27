"""Bounded route steering on G0; only original-program replay certifies a route.

The internal visits below are navigation hints, not DSL milestones. They never
change a name, scope, candidate denotation, physical edge, or proof domain.
Missing a route here is inconclusive and falls through to the complete product.
"""
from __future__ import annotations

import math

from shapely.geometry import LineString, Point

from reconav.scene.physical import within_semantic_clearance
from reconav.solver.cancellation import check_local_search_budget
from reconav.solver.qualifier_geometry import object_geometry, UnknownRelationGeometry
from reconav.solver.target_grounding import GroundedTarget, ground_target, target_from_avoid


def navigation_plans(asg, program, physical, sequence, grounded, requirements):
    """Yield ordinary and bounded requirement-directed physical itineraries."""
    ordinary = [(name, grounded[name], i) for i, name in enumerate(sequence)]
    yield ordinary, ()
    names = ("START", *sequence)
    start = program.original_start_target or program.start().target
    visits = []
    blocks = []

    def scoped(selector, spec, amount=0, avoid=False, exact=False):
        check_local_search_budget()
        target = ground_target(asg, selector, 0, start_target=start)
        after, until = getattr(spec, "after", "START"), getattr(spec, "until", "GOAL")
        if after not in names or until not in names:
            return
        lo, hi = names.index(after), names.index(until)
        if lo >= hi:
            return
        if avoid:
            blocks.append((target, lo, hi))
        if amount > 0:
            visits.append((target, lo, hi, amount, exact))

    def count(spec, expected):
        # Under-count witnesses avoid the whole class; over-count witnesses
        # deliberately enter k+1 distinct members. These are sufficient
        # constructions only, never completeness arguments.
        if expected == 0:
            if spec.op == "at_least":
                scoped(spec.target, spec, spec.count)
            elif spec.op == "exactly":
                scoped(spec.target, spec, spec.count, avoid=spec.count == 0, exact=True)
            else:
                scoped(spec.target, spec, avoid=True)
        elif spec.op == "at_least":
            scoped(spec.target, spec, avoid=True)
        else:
            scoped(spec.target, spec, spec.count + 1)

    for spec in program.avoid_specs:
        scoped(target_from_avoid(spec), spec, avoid=True)
    for spec in program.count_specs:
        count(spec, 0)
    for kind, index, expected in requirements:
        specs = {
            "avoid": program.soft_avoid_specs, "count": program.soft_count_specs,
            "via": program.soft_via_specs, "order": program.soft_order_specs,
        }
        if kind not in specs or not 0 <= index < len(specs[kind]):
            return
        if kind == "avoid":
            spec = program.soft_avoid_specs[index]
            scoped(spec.target, spec, int(expected == 1), avoid=expected == 0)
        elif kind == "count":
            count(program.soft_count_specs[index], expected)
        elif kind == "via":
            spec = program.soft_via_specs[index]
            scoped(spec.target, spec, int(expected == 0), avoid=expected == 1)
        # ORDER outcomes are checked by replay over original milestones. Their
        # valid permutations are already enumerated by fast_witness.
    if not visits and not blocks:
        return

    def rank(target, entity_id):
        check_local_search_budget()
        single = GroundedTarget(target.kind, frozenset((entity_id,)))
        from reconav.solver.fast_witness import _target_nodes
        nodes = _target_nodes(physical, asg, single)
        if not nodes:
            return math.inf
        if start.kind == "pos":
            x, _, z = map(float, start.values)
            return min(math.hypot(physical.nodes[n].x - x, physical.nodes[n].y - z) for n in nodes)
        return float(entity_id)

    ranked = [sorted(t.ids, key=lambda i: (rank(t, i), i)) for t, _, _, _, _ in visits]
    seen = set()
    for variant in range(12):
        check_local_search_budget()
        inserted = {}
        signature = []
        possible = True
        plan_blocks = list(blocks)
        for (target, lo, hi, amount, exact), ids in zip(visits, ranked):
            if amount > len(ids):
                possible = False
                break
            offset = variant % len(ids)
            chosen = (ids[offset:] + ids[:offset])[:amount]
            if variant % 2:
                chosen = list(reversed(chosen))
            if exact and (excluded := target.ids.difference(chosen)):
                plan_blocks.append((GroundedTarget(target.kind, frozenset(excluded)), lo, hi))
            leg = lo + (variant // max(1, len(ids))) % (hi - lo)
            for entity_id in chosen:
                hint = GroundedTarget(target.kind, frozenset((entity_id,)))
                inserted.setdefault(leg, []).append((None, hint, leg))
                signature.append((target.kind, entity_id, leg))
        if not possible:
            continue
        key = tuple(signature)
        if key in seen:
            continue
        seen.add(key)
        stops = []
        for name, target, leg in ordinary:
            stops.extend(inserted.get(leg, ()))
            stops.append((name, target, leg))
        yield stops, tuple(plan_blocks)


def edge_filter(asg, physical, blocks, leg):
    active = [target for target, lo, hi in blocks if lo <= leg < hi]
    if not active:
        return None
    region_ids, level_ids, portal_ids, shapes = set(), set(), set(), []
    for target in active:
        check_local_search_budget()
        if target.kind == "region":
            region_ids.update(target.ids)
        elif target.kind == "level":
            level_ids.update(target.ids)
        elif target.kind == "portal":
            portal_ids.update(target.ids)
        elif target.kind == "obj":
            for entity_id in target.ids:
                check_local_search_budget()
                try:
                    shape, level, _region = object_geometry(asg, entity_id)
                except UnknownRelationGeometry:
                    # Steering cannot reason about this geometry. No path
                    # from this restricted attempt is evidence of impossibility.
                    return lambda *_args: False
                obj = asg.get_object(entity_id)
                regions = {obj.region_index, *obj.body_band_footprint_2d_by_region}
                shapes.append((shape, level, regions))
    blocked_edges = {
        (a, b)
        for connector in physical.portal_connectors
        if connector.validated and connector.portal_id in portal_ids
        for a, b in ((connector.node_a, connector.node_b), (connector.node_b, connector.node_a))
    }
    radius = physical.config.endpoint_clearance_m
    cache = {}

    def allowed(first, second, start_point=None):
        check_local_search_budget()
        key = (first, second)
        if key in cache:
            return cache[key]
        destination = physical.nodes[second]
        if first is None:
            point, source_region = start_point
            origin = Point(point.x, point.z)
        else:
            source = physical.nodes[first]
            source_region = source.region_id
            origin = Point(source.x, source.y)
        target_region = destination.region_id
        source_level = asg.get_region(source_region).level
        target_level = asg.get_region(target_region).level
        accept = not (
            (target_region in region_ids and source_region != target_region)
            or (target_level in level_ids and source_level != target_level)
            or (first, second) in blocked_edges
        )
        if accept and shapes:
            end = Point(destination.x, destination.y)
            line = LineString((origin, end))
            same_level = source_level == target_level
            for shape, level, regions in shapes:
                check_local_search_budget()
                relevant_source = source_region in regions
                relevant_target = target_region in regions
                if not relevant_source and not relevant_target:
                    continue
                if relevant_source and within_semantic_clearance(origin.distance(shape), radius):
                    continue
                geometry = line if same_level else end
                if (same_level or relevant_target) and within_semantic_clearance(geometry.distance(shape), radius):
                    accept = False
                    break
        cache[key] = accept
        return accept
    return allowed


def path_to_target(
    asg, physical, sources, target, *, require_entry, start_point,
    deadline, cancel_token, allowed,
):
    from reconav.solver.fast_witness import _shortest_path, _target_nodes

    def shortest(origins, destinations):
        return _shortest_path(
            physical, origins, destinations, deadline=deadline,
            cancel_token=cancel_token, allowed=allowed,
        )

    if target.kind == "portal":
        best = None
        best_cost = math.inf
        for connector in physical.portal_connectors:
            if not connector.validated or connector.portal_id not in target.ids:
                continue
            for first, second in ((connector.node_a, connector.node_b), (connector.node_b, connector.node_a)):
                if allowed is not None and not allowed(first, second):
                    continue
                path = shortest(sources, (first,))
                if path is None:
                    continue
                cost = sources[path[0]] + sum(
                    math.hypot(physical.nodes[a].x - physical.nodes[b].x, physical.nodes[a].y - physical.nodes[b].y)
                    for a, b in zip(path, path[1:])
                )
                if cost < best_cost:
                    best, best_cost = [*path, second], cost
        return best

    targets = _target_nodes(physical, asg, target)
    if not targets:
        return None
    prefix = []
    # Initially-inside is not an entry. An explicit physical excursion permits
    # a later re-entry without creating a synthetic milestone or COUNT event.
    if require_entry:
        if start_point is not None:
            point, region = start_point
            current = (point.x, point.z, region)
        elif len(sources) == 1:
            node = physical.nodes[next(iter(sources))]
            current = (node.x, node.y, node.region_id)
        else:
            current = None
        if current is not None and _inside(asg, physical, target, *current):
            outside = (
                node.node_id for node in physical.nodes
                if not _inside(asg, physical, target, node.x, node.y, node.region_id)
            )
            prefix = shortest(sources, outside)
            if prefix is None:
                return None
            sources = {prefix[-1]: 0.0}
    path = shortest(sources, targets)
    if path is None:
        return None
    return prefix[:-1] + path if prefix else path


def _inside(asg, physical, target, x, z, region):
    check_local_search_budget()
    if target.kind == "region":
        return region in target.ids
    if target.kind == "level":
        return asg.get_region(region).level in target.ids
    if target.kind == "obj":
        point = Point(x, z)
        for entity_id in target.ids:
            obj = asg.get_object(entity_id)
            if region != obj.region_index and region not in obj.body_band_footprint_2d_by_region:
                continue
            try:
                shape, _level, _region = object_geometry(asg, entity_id)
            except UnknownRelationGeometry:
                continue
            if within_semantic_clearance(point.distance(shape), physical.config.endpoint_clearance_m):
                return True
    return False
