"""Bounded hard-witness construction on the existing sparse physical G0."""

from __future__ import annotations

from dataclasses import replace
import heapq
import itertools
import math
import time

from shapely.geometry import Point

from reconav.scene.geometry import Point3
from reconav.solver.cancellation import (
    LocalSearchBudgetExpired, check_local_search_budget, local_search_budget,
)
from reconav.solver.order_runtime import OrderRuntime
from reconav.solver.target_grounding import ground_target
from reconav.scene.navigation import segment_is_in_free_space


def fast_complete_hard_witness(
    asg, program, physical, *, deadline=None, requirements=(), cancel_token=None,
):
    """Return a replayed witness or ``None``; failure is never a proof."""
    try:
        with local_search_budget(deadline, cancel_token):
            return _fast_complete_hard_witness(
                asg, program, physical, deadline=deadline,
                requirements=requirements, cancel_token=cancel_token,
            )
    except LocalSearchBudgetExpired:
        return None


def _fast_complete_hard_witness(
    asg, program, physical, *, deadline, requirements, cancel_token,
):
    check_local_search_budget()

    from reconav.solver.cross_region import CrossRegionPathResult, RouteMilestoneHit
    from reconav.solver.replay import replay_solution

    started = time.perf_counter()
    order = OrderRuntime(program, asg)
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
    has_object_watchers = bool(
        any(spec.kind == "obj" for spec in program.avoid_specs)
        or any(spec.target.kind == "obj" for spec in program.count_specs)
        or any(spec.target.kind == "obj" for spec in program.soft_avoid_specs)
        or any(spec.target.kind == "obj" for spec in program.soft_count_specs)
        or any(spec.target.kind == "obj" for spec in program.soft_via_specs)
    )
    via_names = tuple(
        item.name for item in program.milestones if item.role == "via"
    )
    sequences = []
    if len(via_names) > 6:
        return None
    for permutation in itertools.permutations(via_names):
        check_local_search_budget()
        progress = order._observe_one((), "START")
        if progress is None:
            continue
        valid = True
        for name in (*permutation, "GOAL"):
            progress = order._observe_one(progress, name)
            if progress is None:
                valid = False
                break
        if valid and order.accepting(progress):
            sequences.append((*permutation, "GOAL"))
    if not sequences:
        return None

    portal_by_edge = {}
    for connector in physical.portal_connectors:
        if connector.validated:
            portal_by_edge[(connector.node_a, connector.node_b)] = connector.portal_id
            portal_by_edge[(connector.node_b, connector.node_a)] = connector.portal_id

    start_nodes, start_point = _start_nodes(physical, asg, original_start)
    if not start_nodes:
        return None
    from reconav.solver.directed_witness import navigation_plans, edge_filter, path_to_target

    plans = (
        (sequence, stops, blocks)
        for sequence in sequences
        for stops, blocks in navigation_plans(
            asg, program, physical, sequence, grounded, requirements
        )
    )
    for sequence, stops, blocks in plans:
        if cancel_token is not None:
            cancel_token.raise_if_requested()
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        source_costs = dict(start_nodes)
        all_nodes = []
        milestone_positions = []
        failed = False
        for name, target, leg in stops:
            check_local_search_budget()
            allowed = edge_filter(asg, physical, blocks, leg)
            entry_milestone = name is not None and name != "GOAL" and target.kind in {"region", "level"}
            if entry_milestone and not all_nodes and start_point is not None and _matches_region_target(
                asg, target, start_point[1]
            ):
                # The actual START is an eligible region/level occurrence.
                # A local region anchor reached later is not another one.
                milestone_positions.append((name, -1))
                continue
            current_entry = bool(all_nodes) and (
                (len(all_nodes) > 1 and physical.nodes[all_nodes[-2]].region_id != physical.nodes[all_nodes[-1]].region_id)
                or (len(all_nodes) == 1 and start_point is None)
                or (
                    len(all_nodes) == 1 and start_point is not None
                    and abs(physical.nodes[all_nodes[-1]].x - start_point[0].x) <= 1e-10
                    and abs(physical.nodes[all_nodes[-1]].y - start_point[0].z) <= 1e-10
                )
            )
            if entry_milestone and current_entry and _matches_region_target(
                asg, target, physical.nodes[all_nodes[-1]].region_id
            ):
                milestone_positions.append((name, len(all_nodes) - 1))
                continue
            # START attachments are real route segments and must obey the same
            # steering restriction as physical graph edges.
            if not all_nodes and start_point is not None and allowed is not None:
                source_costs = {
                    node_id: cost for node_id, cost in source_costs.items()
                    if allowed(None, node_id, start_point)
                }
            entry_prefix = []
            query_sources = source_costs
            if entry_milestone and all_nodes and _matches_region_target(
                asg, target, physical.nodes[all_nodes[-1]].region_id
            ):
                # VIA LEVEL is eligible on *region* entry in the complete
                # product, including entering another room on the same level.
                current_region = physical.nodes[all_nodes[-1]].region_id
                entry_prefix = _shortest_path(
                    physical, source_costs,
                    (node.node_id for node in physical.nodes if node.region_id != current_region),
                    deadline=deadline, cancel_token=cancel_token, allowed=allowed,
                )
                if entry_prefix is None:
                    failed = True
                    break
                query_sources = {entry_prefix[-1]: 0.0}
            path = path_to_target(
                asg, physical, query_sources, target,
                require_entry=name is None,
                start_point=start_point if not all_nodes else None,
                deadline=deadline, cancel_token=cancel_token, allowed=allowed,
            )
            if path is None:
                failed = True
                break
            if entry_prefix:
                path = entry_prefix[:-1] + path
            if entry_milestone:
                previous_region = (
                    physical.nodes[all_nodes[-1]].region_id if all_nodes
                    else start_point[1] if start_point is not None else None
                )
                entry_index = None
                for index, node_id in enumerate(path):
                    region = physical.nodes[node_id].region_id
                    eligible = previous_region is None or previous_region != region
                    if _matches_region_target(asg, target, region) and eligible:
                        entry_index = index
                        break
                    previous_region = region
                if entry_index is None:
                    failed = True
                    break
                # Stop at the event, so the following leg uses its true scope.
                path = path[:entry_index + 1]
            if all_nodes and path[0] == all_nodes[-1]:
                all_nodes.extend(path[1:])
            else:
                all_nodes.extend(path)
            if name is not None:
                milestone_positions.append((name, len(all_nodes) - 1))
            source_costs = {path[-1]: 0.0}
        if failed or not all_nodes:
            continue

        points: list[Point3] = []
        regions: list[int] = []
        # A detour can revisit the same graph node. Milestones refer to path
        # occurrences, never to the last position of a node with that ID.
        path_to_position: list[int] = []
        if start_point is not None:
            points.append(start_point[0])
            regions.append(start_point[1])
        for node_id in all_nodes:
            node = physical.nodes[node_id]
            point = Point3(
                float(node.x),
                physical.floor_height(node.region_id),
                float(node.y),
            )
            if points and regions[-1] == node.region_id and math.dist(
                (points[-1].x, points[-1].y, points[-1].z),
                (point.x, point.y, point.z),
            ) <= 1e-10:
                path_to_position.append(len(points) - 1)
                continue
            points.append(point)
            regions.append(int(node.region_id))
            path_to_position.append(len(points) - 1)
        if len(points) < 1:
            continue

        portals = []
        for first, second in zip(all_nodes, all_nodes[1:]):
            portal_id = portal_by_edge.get((first, second))
            if portal_id is not None:
                portals.append(int(portal_id))

        milestones = []
        start_item = program.start()
        start_resolved = (
            int(start_item.target.values[0])
            if start_item.target.kind == "obj"
            else regions[0]
        )
        milestones.append(
            RouteMilestoneHit(
                "START",
                "start",
                start_item.target.kind,
                start_item.target.selector,
                start_item.target.values[0] if start_item.target.values else start_resolved,
                start_resolved,
                points[0],
                0,
                0,
            )
        )
        for event_seq, (name, sequence_position) in enumerate(
            milestone_positions,
            start=1,
        ):
            item = program.milestone_by_name()[name]
            target = grounded[name]
            if sequence_position == -1:
                resolved = regions[0] if target.kind == "region" else int(asg.get_region(regions[0]).level)
                milestones.append(RouteMilestoneHit(
                    name, item.role, target.kind, item.target.selector,
                    item.target.values[0] if item.target.values else resolved,
                    resolved, points[0], 0, event_seq,
                ))
                continue
            node_id = all_nodes[sequence_position]
            node = physical.nodes[node_id]
            resolved = (
                portal_by_edge[(all_nodes[sequence_position - 1], node_id)]
                if target.kind == "portal" and sequence_position > 0
                else _resolved_target_id(target, node, physical)
            )
            waypoint = path_to_position[sequence_position]
            milestones.append(
                RouteMilestoneHit(
                    name,
                    item.role,
                    target.kind,
                    item.target.selector,
                    item.target.values[0] if item.target.values else resolved,
                    resolved,
                    points[waypoint],
                    waypoint,
                    event_seq,
                )
            )
        length = sum(
            math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
            for a, b in zip(points, points[1:])
        )
        provisional = CrossRegionPathResult(
            points,
            regions,
            portals,
            milestones,
            0.0,
            [],
            status="SAT",
            status_reason="sparse G0 hard-witness candidate",
            query_graph_complete=False,
            state_search_complete=False,
            optimality_established=False,
            proof_domain=f"G0={physical.graph_id};sparse-physical-witness",
            requested_mode="FEASIBLE",
            query_outcome="SAT_WITNESS",
            witness_penalty=0.0,
            witness_length=length,
            penalty_lb=0.0,
            penalty_ub=0.0,
            length_lb=0.0,
            length_ub=length,
            certificate_kind="replayed-sparse-G0-witness",
            timing={"fast_witness_seconds": time.perf_counter() - started},
            counters={
                "fast_witness_nodes": len(all_nodes),
                "object_encounter_calls": 0 if not has_object_watchers else None,
                "watcher_objects_present": has_object_watchers,
                "directed_navigation_targets": sum(name is None for name, _, _ in stops),
                "directed_block_groups": len(blocks),
                "witness_requirements_checked": len(requirements),
            },
        )
        check_local_search_budget()
        replay = replay_solution(program, asg, provisional)
        check_local_search_budget()
        penalty = replay.get("preference_cost")
        if not replay.get("ok") or not isinstance(penalty, (int, float)):
            continue
        result = replace(
            provisional,
            optimal_soft_cost=float(penalty),
            witness_penalty=float(penalty),
            penalty_ub=float(penalty),
        )
        replay = replay_solution(program, asg, result)
        check_local_search_budget()
        if (
            replay.get("ok") and replay.get("penalty_matches_solver")
            and witness_satisfies_requirements(replay, program, requirements)
        ):
            return replace(result, semantic_replay=replay)
    return None


def _start_nodes(physical, asg, target):
    if target.kind == "obj":
        object_id = int(target.values[0])
        return {
            anchor.node_id: 0.0
            for anchor in physical.object_anchors.get(object_id, ())
        }, None
    if target.kind != "pos":
        return {}, None
    x, height, z = (float(value) for value in target.values)
    candidates = []
    for region_id in physical.regions:
        local = physical.locate_local_component(region_id, Point(x, z))
        if local is not None:
            candidates.append(
                (abs(physical.floor_height(region_id) - height), region_id, local)
            )
    if not candidates:
        return {}, None
    best = min(value[0] for value in candidates)
    selected = [value for value in candidates if value[0] <= best + 1e-6]
    costs = {}
    for _delta, region_id, component_id in selected:
        check_local_search_budget()
        # One complete cancellable scan suffices. The old preliminary visibility
        # helper repeated the same boundary predicates without cancellation.
        for node_id in physical.region_local_nodes.get((region_id, component_id), ()):
            check_local_search_budget()
            node = physical.nodes[node_id]
            if segment_is_in_free_space(
                physical,
                region_id,
                (x, z),
                (node.x, node.y),
            ):
                costs[node_id] = min(
                    costs.get(node_id, math.inf),
                    math.hypot(x - node.x, z - node.y),
                )
    _delta, region_id, _component_id = min(selected)
    return costs, (Point3(x, physical.floor_height(region_id), z), region_id)


def _target_nodes(physical, asg, target):
    nodes = set()
    if target.kind == "obj":
        for entity_id in target.ids:
            nodes.update(anchor.node_id for anchor in physical.object_anchors.get(entity_id, ()))
    elif target.kind == "region":
        for entity_id in target.ids:
            nodes.update(anchor.node_id for anchor in physical.region_anchors.get(entity_id, ()))
    elif target.kind == "level":
        for region in asg.regions:
            if region.level in target.ids:
                nodes.update(
                    anchor.node_id
                    for anchor in physical.region_anchors.get(region.index, ())
                )
    else:
        # A portal milestone is an edge event, not merely either side's anchor.
        return frozenset()
    return frozenset(nodes)


def _shortest_path(
    physical, source_costs, targets, *, deadline=None, cancel_token=None, allowed=None,
):
    queue = []
    distance = {}
    parent = {}
    serial = itertools.count()
    for node_id, cost in source_costs.items():
        if cost < distance.get(node_id, math.inf):
            distance[node_id] = float(cost)
            heapq.heappush(queue, (float(cost), next(serial), node_id))
    target_set = set()
    for node_id in targets:
        check_local_search_budget()
        target_set.add(node_id)
    while queue:
        if cancel_token is not None:
            cancel_token.raise_if_requested()
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        cost, _serial, node_id = heapq.heappop(queue)
        if cost != distance.get(node_id):
            continue
        if node_id in target_set:
            path = [node_id]
            while node_id in parent:
                node_id = parent[node_id]
                path.append(node_id)
            path.reverse()
            return path
        for neighbor, weight in physical.adjacency.get(node_id, ()):
            if allowed is not None and not allowed(node_id, neighbor):
                continue
            following = cost + float(weight)
            if following < distance.get(neighbor, math.inf):
                distance[neighbor] = following
                parent[neighbor] = node_id
                heapq.heappush(queue, (following, next(serial), neighbor))
    return None


def _resolved_target_id(target, node, physical):
    if target.kind == "obj":
        if node.kind == "object" and node.entity_id in target.ids:
            return int(node.entity_id)
        return int(next(iter(target.ids)))
    if target.kind == "region":
        return int(node.region_id)
    if target.kind == "level":
        return int(physical.regions[node.region_id].level_id)
    return int(next(iter(target.ids)))


def _matches_region_target(asg, target, region_id):
    return (
        region_id in target.ids if target.kind == "region"
        else asg.get_region(region_id).level in target.ids
    )


def witness_satisfies_requirements(replay, program, requirements):
    """Check exact per-clause outcomes from the independent original replay."""
    if not isinstance(replay, dict) or not replay.get("ok"):
        return False
    records = {
        str(item.get("name")): bool(item.get("satisfied"))
        for item in replay.get("constraints", ()) if isinstance(item, dict)
    }
    offsets = {
        "via": 0,
        "avoid": len(program.soft_via_specs),
        "count": len(program.soft_via_specs) + len(program.soft_avoid_specs),
    }
    lengths = {
        "via": len(program.soft_via_specs), "avoid": len(program.soft_avoid_specs),
        "count": len(program.soft_count_specs), "order": len(program.soft_order_specs),
    }
    for kind, index, expected in requirements:
        if index < 0 or index >= lengths.get(kind, 0):
            return False
        name = f"prefer_order[{index}]" if kind == "order" else f"prefer[{offsets[kind] + index}]"
        if name not in records or int(not records[name]) != expected:
            return False
    return True
