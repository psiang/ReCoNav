"""Component/portal relaxation used only for sound penalty lower bounds."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import time

from shapely.geometry import Point
from reconav.scene.physical import within_semantic_clearance

from reconav.solver.avoid_runtime import AvoidRuntime
from reconav.solver.count_runtime import CountRuntime
from reconav.solver.order_runtime import OrderRuntime
from reconav.solver.prefer_runtime import PreferRuntime
from reconav.solver.segment_events import EVENT_SEMANTICS_VERSION
from reconav.solver.target_grounding import ground_target
from reconav.solver.qualifier_geometry import object_geometry, UnknownRelationGeometry
from reconav.solver.cancellation import (
    LocalSearchBudgetExpired, check_local_search_budget, local_search_budget,
)


@dataclass(frozen=True)
class TopologyState:
    component: tuple[int, int]
    progress: tuple[str, ...]
    counts: object
    preference: object
    penalty: float


@dataclass(frozen=True)
class TopologyBoundResult:
    outcome: str
    certificate: dict
    states_visited: int


def prove_penalty_bound(asg, program, physical, bound: int, *, deadline=None, cancel_token=None):
    try:
        with local_search_budget(deadline, cancel_token):
            return _prove_penalty_bound(asg, program, physical, bound, deadline=deadline)
    except (LocalSearchBudgetExpired, UnknownRelationGeometry) as error:
        return TopologyBoundResult("NO_CERTIFICATE", {
            "graph_id": physical.graph_id, "bound": int(bound),
            "no_certificate_reason": str(error),
        }, 0)


def _prove_penalty_bound(asg, program, physical, bound: int, *, deadline=None):
    """Exclude ``penalty <= bound`` on an over-approximating topology product.

    Unsupported semantic watchers return NO_CERTIFICATE; they are never silently
    interpreted as absent constraints.
    """

    unsupported = _unsupported_conditions(program, asg)
    graph_nodes = [
        (region_id, component_id)
        for region_id, region in physical.regions.items()
        for component_id in range(len(region.components))
    ]
    graph = defaultdict(list)
    for connector in physical.portal_connectors:
        check_local_search_budget()
        if not connector.validated:
            continue
        first = physical.nodes[connector.node_a]
        second = physical.nodes[connector.node_b]
        a = (int(first.region_id), int(first.local_component_id))
        b = (int(second.region_id), int(second.local_component_id))
        graph[a].append((b, int(connector.portal_id), connector.node_b))
        graph[b].append((a, int(connector.portal_id), connector.node_a))
    manifest = {
        "graph_id": physical.graph_id,
        "nodes": len(graph_nodes),
        "directed_edges": sum(len(values) for values in graph.values()),
        "parallel_portals_preserved": True,
        "object_component_domain": "all semantic-clearance component intersections",
        "object_portal_endpoint_events": "actual registered arrival clearance",
        "initial_region_level_events": True,
        "bound": int(bound),
        "event_semantics": EVENT_SEMANTICS_VERSION,
        "retained_conditions": [
            "hard ORDER",
            "hard portal/region/level AVOID and COUNT",
            "soft portal/region/level AVOID and COUNT",
            "hard named object/region/level milestone component domains",
        ],
        "ignored_conditions": unsupported,
    }
    if unsupported:
        return TopologyBoundResult("NO_CERTIFICATE", manifest, 0)

    original_start = program.original_start_target or program.start().target
    grounded = {
        item.name: ground_target(
            asg,
            item.target,
            item.source.number,
            start_target=original_start,
        )
        for item in program.milestones
        if item.name != "START"
    }
    components_by_name = {
        name: _grounded_components(physical, asg, target)
        for name, target in grounded.items()
    }
    if any(not values for values in components_by_name.values()):
        manifest["no_certificate_reason"] = "a named endpoint has no complete component domain"
        return TopologyBoundResult("NO_CERTIFICATE", manifest, 0)
    start_components = _start_components(physical, original_start, asg)
    if not start_components:
        manifest["no_certificate_reason"] = "START has no component"
        return TopologyBoundResult("NO_CERTIFICATE", manifest, 0)

    order = OrderRuntime(program, asg)
    avoid = AvoidRuntime(program, asg)
    count = CountRuntime(program, asg)
    prefer = PreferRuntime(program, asg)
    scope_specs = [
        item.spec
        for item in (*avoid.avoids, *count.counts, *prefer.avoids, *prefer.counts)
    ]

    def choices(progress, available):
        output = [(progress, ())]
        seen = {(progress, ())}
        cursor = 0
        while cursor < len(output):
            check_local_search_budget()
            prior, hits = output[cursor]
            cursor += 1
            if prior and prior[-1] == "GOAL":
                continue
            for name, kind, entity_id in available:
                if name in prior:
                    continue
                if any(name == spec.until and spec.after not in prior for spec in scope_specs):
                    continue
                following = order._observe_one(prior, name)
                if following is None:
                    continue
                if name == "GOAL" and not order.accepting(following):
                    continue
                value = (following, (*hits, (name, kind, entity_id)))
                if value not in seen:
                    seen.add(value)
                    output.append(value)
        return output

    def internal_available(component):
        output = []
        for name, target in grounded.items():
            if component not in components_by_name[name]:
                continue
            if target.kind == "obj" or name == "GOAL":
                output.extend((name, target.kind, int(value)) for value in target.ids)
        return tuple(output)

    def edge_available(component, portal_id, arrival_node=None):
        region_id, _local = component
        level_id = physical.regions[region_id].level_id
        output = []
        for name, target in grounded.items():
            if target.kind == "portal" and portal_id in target.ids:
                output.append((name, "portal", portal_id))
            elif target.kind == "obj" and arrival_node is not None:
                node = physical.nodes[arrival_node]
                point = Point(node.x, node.y)
                for entity_id in target.ids:
                    shape, _level, object_region = object_geometry(asg, entity_id)
                    if object_region == region_id and within_semantic_clearance(
                        point.distance(shape), physical.config.endpoint_clearance_m,
                    ):
                        # An object boundary event may coincide with this
                        # portal crossing and exclude it from an open scope.
                        output.append((name, "obj", int(entity_id)))
            elif target.kind == "region" and region_id in target.ids:
                output.append((name, "region", region_id))
            elif target.kind == "level" and level_id in target.ids:
                output.append((name, "level", level_id))
        return tuple(output)

    def forbidden(progress, names, entities):
        targets = set(names)
        for item in avoid.avoids:
            if not _active(item.spec, progress, targets):
                continue
            pools = {
                "obj": item.object_ids,
                "region": item.region_ids,
                "portal": item.portal_ids,
                "level": item.level_ids,
            }
            if any(entity_id in pools.get(kind, ()) for kind, entity_id in entities):
                return True
        return False

    def transition(state, progress, names, entities, component):
        if forbidden(state.progress, names, entities):
            return None
        counts = count.observe(state.counts, state.progress, names, entities)
        if counts is None:
            return None
        step = prefer.observe(state.preference, state.progress, names, entities)
        if not math.isfinite(step.penalty):
            return None
        penalty = state.penalty + step.penalty
        if penalty > bound:
            return None
        return TopologyState(component, progress, counts, step.state, penalty)

    queue = deque()
    best = {}
    trace = {}
    start_progress = order._observe_one((), "START")
    if start_progress is None:
        manifest["no_certificate_reason"] = "START cannot initialize hard ORDER"
        return TopologyBoundResult("NO_CERTIFICATE", manifest, 0)
    for component in sorted(start_components):
        preference = prefer.initial_state(start_progress, component[0])
        base_state = TopologyState(
            component,
            start_progress,
            count.initial_state(),
            preference,
            0.0,
        )
        initially_available = (*internal_available(component), *edge_available(component, None))
        for progress, hits in choices(start_progress, initially_available):
            state = transition(base_state, progress, tuple(v[0] for v in hits), (), component)
            if state is not None:
                _enqueue(queue, best, trace, state, None, {"kind": "initial", "hits": hits})

    visited = 0
    while queue:
        check_local_search_budget()
        if deadline is not None and time.perf_counter() >= deadline:
            manifest["states_visited"] = visited
            manifest["no_certificate_reason"] = "topology budget exhausted"
            return TopologyBoundResult("NO_CERTIFICATE", manifest, visited)
        state = queue.popleft()
        key = _state_key(state)
        if state.penalty != best.get(key):
            continue
        visited += 1
        if state.progress and state.progress[-1] == "GOAL":
            if order.accepting(state.progress) and count.accepting(state.counts):
                final = state.penalty + prefer.final_penalty(state.preference)
                if final <= bound:
                    manifest["states_visited"] = visited
                    manifest["relaxed_trace"] = _trace(trace, state)
                    return TopologyBoundResult("INCONCLUSIVE_WITH_TOPOLOGY_TRACE", manifest, visited)

        # Named object/GOAL events inside one component are optimistic zero-cost moves.
        for progress, hits in choices(state.progress, internal_available(state.component)):
            if progress == state.progress:
                continue
            following = transition(
                state,
                progress,
                tuple(value[0] for value in hits),
                (),
                state.component,
            )
            if following is not None:
                _enqueue(
                    queue,
                    best,
                    trace,
                    following,
                    state,
                    {"kind": "internal", "hits": hits},
                )

        for target_component, portal_id, arrival_node in graph.get(state.component, ()):
            source_region = state.component[0]
            target_region = target_component[0]
            entities = [("portal", portal_id)]
            if source_region != target_region:
                entities.append(("region", target_region))
            source_level = physical.regions[source_region].level_id
            target_level = physical.regions[target_region].level_id
            if source_level != target_level:
                entities.append(("level", target_level))
            for progress, hits in choices(
                state.progress,
                edge_available(target_component, portal_id, arrival_node),
            ):
                following = transition(
                    state,
                    progress,
                    tuple(value[0] for value in hits),
                    tuple(entities),
                    target_component,
                )
                if following is not None:
                    _enqueue(
                        queue,
                        best,
                        trace,
                        following,
                        state,
                        {
                            "kind": "portal",
                            "portal_id": portal_id,
                            "from": state.component,
                            "to": target_component,
                            "hits": hits,
                        },
                    )

    manifest["states_visited"] = visited
    manifest["excluded_penalty_bound"] = int(bound)
    manifest["input_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "graph_id": physical.graph_id,
                "dsl_lines": [
                    item.source.text
                    for item in (
                        *program.milestones,
                        *program.order_constraints,
                        *program.avoid_specs,
                        *program.count_specs,
                        *program.soft_via_specs,
                        *program.soft_avoid_specs,
                        *program.soft_count_specs,
                        *program.soft_order_specs,
                    )
                ],
                "bound": bound,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return TopologyBoundResult("BOUND_EXCLUDED", manifest, visited)


def _unsupported_conditions(program, asg):
    reasons = []
    original_start = program.original_start_target or program.start().target
    for label, specs in (
        ("hard AVOID", program.avoid_specs),
        ("hard COUNT", program.count_specs),
        ("soft AVOID", program.soft_avoid_specs),
        ("soft COUNT", program.soft_count_specs),
    ):
        for spec in specs:
            selector = spec.target if hasattr(spec, "target") else None
            if selector is None:
                from reconav.solver.target_grounding import target_from_avoid

                selector = target_from_avoid(spec)
            grounded = ground_target(
                asg,
                selector,
                spec.source.number,
                start_target=original_start,
            )
            if grounded.kind == "obj":
                reasons.append(f"{label} object encounter geometry")
    if program.soft_via_specs:
        reasons.append("soft VIA occurrence")
    if program.soft_order_specs:
        reasons.append("soft ORDER history")
    return sorted(set(reasons))


def _grounded_components(physical, asg, target):
    output = set()
    if target.kind == "obj":
        for entity_id in target.ids:
            check_local_search_budget()
            shape, _level, region_id = object_geometry(asg, entity_id)
            region = physical.regions.get(region_id)
            if region is None:
                continue
            # Registered warm anchors need not cover every semantic endpoint
            # component in the complete query graph. Include the whole legal
            # approach domain analytically; physical tolerance only enlarges
            # this relaxation and cannot exclude a valid finite-graph route.
            clearance = physical.config.endpoint_clearance_m + physical.config.numeric_tolerance_m
            for component_id, component in enumerate(region.components):
                check_local_search_budget()
                if within_semantic_clearance(component.distance(shape), clearance):
                    output.add((region_id, component_id))
    elif target.kind == "region":
        for entity_id in target.ids:
            check_local_search_budget()
            for component_id in range(len(physical.regions[entity_id].components)):
                output.add((entity_id, component_id))
    elif target.kind == "level":
        for region_id, region in physical.regions.items():
            check_local_search_budget()
            if region.level_id in target.ids:
                for component_id in range(len(region.components)):
                    output.add((region_id, component_id))
    elif target.kind == "portal":
        for connector in physical.portal_connectors:
            if connector.validated and connector.portal_id in target.ids:
                for node_id in (connector.node_a, connector.node_b):
                    node = physical.nodes[node_id]
                    output.add((int(node.region_id), int(node.local_component_id)))
    return frozenset(output)


def _start_components(physical, target, asg=None):
    if target.kind == "obj":
        if asg is not None:
            return _grounded_components(physical, asg, ground_target(asg, target, 0, start_target=target))
        return frozenset(
            (anchor.region_id, anchor.local_component_id)
            for anchor in physical.object_anchors.get(int(target.values[0]), ())
        )
    if target.kind != "pos":
        return frozenset()
    x, height, z = (float(value) for value in target.values)
    candidates = []
    for region_id in physical.regions:
        check_local_search_budget()
        local = physical.locate_local_component(region_id, Point(x, z))
        if local is not None:
            candidates.append((abs(physical.floor_height(region_id) - height), region_id, local))
    if not candidates:
        return frozenset()
    best = min(item[0] for item in candidates)
    return frozenset((region_id, local) for delta, region_id, local in candidates if delta <= best + 1e-6)


def _active(spec, progress, targets):
    return (
        spec.after in progress
        and spec.until not in progress
        and spec.after not in targets
        and spec.until not in targets
    )


def _state_key(state):
    return (state.component, state.progress, state.counts, state.preference)


def _enqueue(queue, best, trace, state, prior, event):
    key = _state_key(state)
    if state.penalty >= best.get(key, math.inf):
        return
    best[key] = state.penalty
    trace[state] = (prior, event)
    queue.append(state)


def _trace(trace, state):
    output = []
    current = state
    while current in trace:
        prior, event = trace[current]
        output.append(event)
        if prior is None:
            break
        current = prior
    output.reverse()
    return output
