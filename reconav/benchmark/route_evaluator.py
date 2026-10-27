"""Evaluate a route against a grounded contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from reconav.scene.asg import ASG
from reconav.scene.physical import ENDPOINT_OBJECT_CLEARANCE_M
from reconav.solver.route_events import object_encounter_entries
from reconav.solver.temporal_semantics import (
    logical_scope_is_ordered,
    minimum_soft_event_penalty,
    transition_in_waypoint_scope,
)

START_POSITION_TOLERANCE_M = 0.25


@dataclass(frozen=True)
class RouteScope:
    after_seq: int
    until_seq: int
    after_waypoint: int
    until_waypoint: int


def evaluate_grounded_route(
    asg: ASG,
    grounded: dict[str, Any],
    route: dict[str, Any],
    robot_position: dict[str, Any],
) -> dict[str, Any]:
    if not route.get("waypoints"):
        return {
            "ok": False,
            "errors": [_failure("route_empty", "route", "route has no executable waypoints")],
            "constraints": [],
            "preference_cost": None,
        }

    start_distance = route_start_distance(route, robot_position)
    results: list[dict[str, Any]] = []
    if start_distance is not None:
        results.append(_result(
            "start",
            "START",
            start_distance <= START_POSITION_TOLERANCE_M,
            {"distance_m": start_distance, "tolerance_m": START_POSITION_TOLERANCE_M},
        ))

    event_times = _event_positions(route, "event_seq")
    waypoint_positions = _event_positions(route, "waypoint_index")
    portal_events = _portal_events(route)
    for item in grounded.get("via", []):
        name = str(item.get("name"))
        hit = _milestone(route, name)
        ok = hit is not None and _hit_matches(hit, item)
        results.append(_result("via", name, ok, hit or {}))

    goal_hit = _milestone(route, "GOAL")
    results.append(_result("goal", "GOAL", goal_hit is not None and _hit_matches(goal_hit or {}, grounded["goal"]), goal_hit or {}))

    soft_via_names = {
        str(item["name"])
        for item in grounded.get("prefer", [])
        if item.get("preference") == "via" and item.get("name")
    }
    conditional_orders = []
    for index, item in enumerate(grounded.get("order", [])):
        if soft_via_names.intersection(_order_names(item)):
            conditional_orders.append({**item, "_source_index": index})
            continue
        ok, witness = _order_satisfied(item, event_times)
        results.append(_result("order", f"order[{index}]", ok, witness))

    for index, item in enumerate(grounded.get("avoid", [])):
        ok, witness = _avoid_satisfied(
            asg, item, route, event_times, waypoint_positions, portal_events
        )
        results.append(_result("avoid", f"avoid[{index}]", ok, witness))

    for index, item in enumerate(grounded.get("count", [])):
        ok, witness = _count_satisfied(
            item, route, event_times, waypoint_positions, portal_events, asg=asg
        )
        results.append(_result("count", f"count[{index}]", ok, witness))

    for index, item in enumerate(grounded.get("width", [])):
        ok, witness = _width_satisfied(
            asg, item, route, event_times, waypoint_positions, portal_events
        )
        results.append(_result("width", f"width[{index}]", ok, witness))

    preference_cost, preferences = _preference_cost(
        asg,
        grounded,
        route,
        event_times,
        waypoint_positions,
        portal_events,
        conditional_orders,
    )
    results.extend(preferences)
    optimal_soft_cost = route.get("optimal_soft_cost")
    if route.get("optimality_established") is True and isinstance(optimal_soft_cost, (int, float)):
        results.append(_result(
            "preference_optimality",
            "preference_optimality",
            abs(preference_cost - float(optimal_soft_cost)) <= 1e-9,
            {"observed": preference_cost, "optimal": float(optimal_soft_cost)},
        ))
    errors = [
        _failure("constraint_violation", item["component"], f"{item['name']} failed", item["witness"])
        for item in results
        if not item["satisfied"] and item["component"] not in {"prefer", "prefer_order"}
    ]
    return {"ok": not errors, "errors": errors, "constraints": results, "preference_cost": preference_cost}


def route_start_distance(
    route: dict[str, Any],
    robot_position: dict[str, Any] | None,
) -> float | None:
    """Return planar distance from the task start to the first route waypoint."""
    if not isinstance(robot_position, dict):
        return None
    waypoints = route.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints or not isinstance(waypoints[0], dict):
        return None
    try:
        route_x = float(waypoints[0]["x"])
        route_z = float(waypoints[0]["z"])
        start_x = float(robot_position["x"])
        start_z = float(robot_position["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (route_x, route_z, start_x, start_z)):
        return math.inf
    return math.hypot(route_x - start_x, route_z - start_z)


def _event_positions(route: dict[str, Any], field: str) -> dict[str, int]:
    if field not in {"event_seq", "waypoint_index"}:
        raise ValueError(f"unsupported event position field: {field}")
    positions = {"START": 0}
    for hit in route.get("milestones", []):
        if (
            isinstance(hit, dict)
            and hit.get("name") is not None
            and hit.get(field) is not None
        ):
            positions[str(hit["name"])] = int(hit[field])
    if "GOAL" not in positions:
        positions["GOAL"] = (
            max(positions.values(), default=0) + 1
            if field == "event_seq"
            else max(len(route.get("waypoints", [])) - 1, 0)
        )
    return positions


def _portal_events(route: dict[str, Any]) -> list[tuple[int, int]]:
    regions = [int(value) for value in route.get("regions", [])]
    portals = iter(int(value) for value in route.get("portals", []) if int(value) >= 0)
    events = []
    for index, (before, after) in enumerate(zip(regions, regions[1:]), start=1):
        if before != after:
            portal = next(portals, None)
            if portal is not None:
                events.append((index, portal))
    return events


def _preferred_via_occurrences(
    asg: ASG,
    item: dict[str, Any],
    route: dict[str, Any],
    portal_events: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    ids = {int(value) for value in item.get("grounded_ids", [])}
    if item.get("kind") == "region":
        regions = [int(region) for region in route.get("regions", [])]
        return [
            {
                "waypoint_index": index,
                "entity_kind": "region",
                "entity_id": region,
            }
            for index, region in enumerate(regions)
            if region in ids and (index == 0 or regions[index - 1] != region)
        ]
    if item.get("kind") == "level":
        regions = [asg.get_region(int(region)) for region in route.get("regions", [])]
        levels = [region.level if region is not None else None for region in regions]
        return [
            {
                "waypoint_index": index,
                "entity_kind": "level",
                "entity_id": level,
            }
            for index, level in enumerate(levels)
            if level in ids and (index == 0 or levels[index - 1] != level)
        ]
    if item.get("kind") == "portal":
        return [
            {
                "waypoint_index": index,
                "entity_kind": "portal",
                "entity_id": portal,
            }
            for index, portal in portal_events
            if portal in ids
        ]
    waypoints = route.get("waypoints", [])
    regions = route.get("regions", [])
    return [
        {
            "waypoint_index": item.waypoint_index,
            "segment_fraction": item.segment_fraction,
            "entity_kind": "obj",
            "entity_id": item.object_id,
        }
        for item in object_encounter_entries(
            asg,
            ids,
            waypoints,
            regions,
        )
    ]


def _preference_event_trace(
    asg: ASG,
    grounded: dict[str, Any],
    route: dict[str, Any],
    portal_events: list[tuple[int, int]],
) -> tuple[tuple[str, ...], dict[str, list[dict[str, Any]]]]:
    hard_events = [
        {
            "name": str(item["name"]),
            "waypoint_index": int(item["waypoint_index"]),
            "event_seq": int(item["event_seq"]),
            "source": "hard_event",
        }
        for item in route.get("milestones", [])
        if isinstance(item, dict)
        and item.get("role") != "prefer"
        and item.get("name") is not None
        and item.get("waypoint_index") is not None
        and item.get("event_seq") is not None
    ]
    hard_events.sort(
        key=lambda item: (item["waypoint_index"], item["event_seq"])
    )
    records = list(hard_events)
    occurrences_by_name: dict[str, list[dict[str, Any]]] = {}
    for preference_index, item in enumerate(grounded.get("prefer", [])):
        if item.get("preference") != "via" or not item.get("name"):
            continue
        name = str(item["name"])
        candidates = _preferred_via_occurrences(asg, item, route, portal_events)
        seen_intervals: set[int] = set()
        admitted = []
        for candidate in candidates:
            candidate_record = {
                **candidate,
                "source": "implicit_route_visitation",
                "preference_index": preference_index,
            }
            candidate_key = _preference_event_sort_key(candidate_record)
            interval = sum(
                _preference_event_sort_key(event) < candidate_key
                for event in hard_events
            )
            if interval in seen_intervals:
                continue
            seen_intervals.add(interval)
            occurrence = {
                **candidate_record,
                "name": name,
                "hard_event_interval": interval,
            }
            admitted.append(occurrence)
            records.append(occurrence)
        occurrences_by_name[name] = admitted
    records.sort(key=_preference_event_sort_key)
    for trace_index, item in enumerate(records):
        item["event_trace_index"] = trace_index
    return tuple(str(item["name"]) for item in records), occurrences_by_name


def _preference_event_sort_key(item: dict[str, Any]) -> tuple[float, ...]:
    waypoint_index = int(item["waypoint_index"])
    source = str(item["source"])
    fraction = float(
        item.get(
            "segment_fraction",
            0.0 if source == "hard_event" and waypoint_index == 0 else 1.0,
        )
    )
    source_order = (
        int(source != "hard_event")
        if waypoint_index == 0
        else int(source == "hard_event")
    )
    ordinal = int(item.get("event_seq", item.get("preference_index", 0)))
    return float(waypoint_index), fraction, float(source_order), float(ordinal)


def _milestone(route: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((item for item in route.get("milestones", []) if str(item.get("name")) == name), None)


def _hit_matches(hit: dict[str, Any], target: dict[str, Any]) -> bool:
    kind = str(hit.get("target_kind", hit.get("kind", "")))
    return kind == str(target.get("kind")) and int(hit.get("resolved_index", hit.get("id", -1))) in {
        int(value) for value in target.get("grounded_ids", [])
    }


def _scope(
    item: dict[str, Any],
    event_times: dict[str, int],
    waypoint_positions: dict[str, int],
) -> RouteScope | None:
    after_name = str(item.get("after", "START"))
    until_name = str(item.get("until", "GOAL"))
    after_seq = event_times.get(after_name)
    until_seq = event_times.get(until_name)
    after_waypoint = waypoint_positions.get(after_name)
    until_waypoint = waypoint_positions.get(until_name)
    if (
        after_seq is None
        or until_seq is None
        or after_waypoint is None
        or until_waypoint is None
        or not logical_scope_is_ordered(after_seq, until_seq)
        or after_waypoint > until_waypoint
    ):
        return None
    return RouteScope(after_seq, until_seq, after_waypoint, until_waypoint)


def _order_satisfied(item: dict[str, Any], positions: dict[str, int]) -> tuple[bool, dict[str, Any]]:
    first, second = positions.get(str(item.get("first"))), positions.get(str(item.get("second")))
    witness = {"first": first, "second": second, "names": {name: positions.get(str(name)) for name in item.get("names", [])}}
    if first is None or second is None:
        return False, witness
    kind = item.get("kind")
    if kind == "after":
        return first > second, witness
    if first >= second:
        return False, witness
    named = [positions.get(str(name)) for name in item.get("names", [])]
    if kind == "before":
        return True, witness
    if kind == "next":
        event_order = sorted((pos, name) for name, pos in positions.items())
        sequence = [name for _pos, name in event_order]
        a, b = str(item.get("first")), str(item.get("second"))
        return a in sequence and b in sequence and sequence.index(b) == sequence.index(a) + 1, witness
    if any(pos is None for pos in named):
        return False, witness
    between = {name for name, pos in positions.items() if first < pos < second and name not in {"START", "GOAL"}}
    required = {str(name) for name in item.get("names", [])}
    if kind == "between_only":
        return between == required, witness
    if kind == "between_without":
        return not (between & required), witness
    return False, witness


def _avoid_satisfied(
    asg: ASG,
    item: dict[str, Any],
    route: dict[str, Any],
    event_times: dict[str, int],
    waypoint_positions: dict[str, int],
    portal_events: list[tuple[int, int]],
    *,
    include_until_transition: bool = True,
) -> tuple[bool, dict[str, Any]]:
    scope = _scope(item, event_times, waypoint_positions)
    if scope is None:
        return False, {"reason": "invalid scope"}
    ids = {int(value) for value in item.get("grounded_ids", [])}
    boundary = _scope_boundary_entity(route, str(item.get("until", "GOAL")))
    touched: set[int] = set()
    if item.get("kind") == "region":
        regions = [int(value) for value in route.get("regions", [])]
        touched = {
            regions[index]
            for index in range(1, len(regions))
            if regions[index] != regions[index - 1]
            and transition_in_waypoint_scope(
                index,
                scope.after_waypoint,
                scope.until_waypoint,
                include_until_transition=False,
            )
            and not (
                index == scope.until_waypoint
                and boundary == ("region", regions[index])
            )
            and regions[index] in ids
        }
    elif item.get("kind") == "portal":
        touched = ids & {
            portal
            for pos, portal in portal_events
            if transition_in_waypoint_scope(
                pos,
                scope.after_waypoint,
                scope.until_waypoint,
                include_until_transition=False,
            )
            and not (
                pos == scope.until_waypoint
                and boundary == ("portal", portal)
            )
        }
    elif item.get("kind") == "obj":
        touched = _objects_entered(
            asg,
            ids,
            route.get("waypoints", []),
            scope,
            include_until_transition=True,
            excluded_on_until=boundary,
            regions=route.get("regions", []),
        )
    elif item.get("kind") == "level":
        regions = [asg.get_region(int(value)) for value in route.get("regions", [])]
        touched = {
            region.level
            for index, region in enumerate(regions[1:], start=1)
            if region is not None
            and (regions[index - 1] is None or regions[index - 1].level != region.level)
            and transition_in_waypoint_scope(
                index,
                scope.after_waypoint,
                scope.until_waypoint,
                include_until_transition=False,
            )
            and not (
                index == scope.until_waypoint
                and boundary == ("level", region.level)
            )
            and region.level in ids
        }
    return not touched, {
        "event_scope": [scope.after_seq, scope.until_seq],
        "waypoint_scope": [scope.after_waypoint, scope.until_waypoint],
        "touched_ids": sorted(touched),
    }


def _count_satisfied(
    item: dict[str, Any],
    route: dict[str, Any],
    event_times: dict[str, int],
    waypoint_positions: dict[str, int],
    portal_events: list[tuple[int, int]],
    *,
    asg: ASG | None = None,
) -> tuple[bool, dict[str, Any]]:
    scope = _scope(item, event_times, waypoint_positions)
    if scope is None:
        return False, {"reason": "invalid scope"}
    ids = {int(value) for value in item.get("grounded_ids", [])}
    boundary = _scope_boundary_entity(route, str(item.get("until", "GOAL")))
    if item.get("kind") == "region":
        regions = [int(value) for value in route.get("regions", [])]
        seen = ids & {
            regions[index]
            for index in range(1, len(regions))
            if regions[index] != regions[index - 1]
            and transition_in_waypoint_scope(
                index,
                scope.after_waypoint,
                scope.until_waypoint,
                include_until_transition=False,
            )
            and not (
                index == scope.until_waypoint
                and boundary == ("region", regions[index])
            )
        }
    elif item.get("kind") == "portal":
        seen = ids & {
            portal
            for pos, portal in portal_events
            if transition_in_waypoint_scope(
                pos,
                scope.after_waypoint,
                scope.until_waypoint,
                include_until_transition=False,
            )
            and not (
                pos == scope.until_waypoint
                and boundary == ("portal", portal)
            )
        }
    elif item.get("kind") == "level":
        if asg is not None:
            region_ids = [int(value) for value in route.get("regions", [])]
            seen = set()
            for index in range(1, len(region_ids)):
                before = asg.get_region(region_ids[index - 1])
                region = asg.get_region(region_ids[index])
                if (
                    before is not None
                    and region is not None
                    and before.level != region.level
                    and region.level in ids
                    and transition_in_waypoint_scope(
                        index,
                        scope.after_waypoint,
                        scope.until_waypoint,
                        include_until_transition=False,
                    )
                ):
                    seen.add(region.level)
        else:
            seen = set()
    else:
        seen = _objects_entered(
            asg,
            ids,
            route.get("waypoints", []),
            scope,
            include_until_transition=True,
            excluded_on_until=boundary,
            regions=route.get("regions", []),
        ) if asg is not None else set()
    observed, expected = len(seen), int(item.get("count", -1))
    op = str(item.get("op"))
    ok = observed >= expected if op == "at_least" else observed <= expected if op == "at_most" else observed == expected
    return ok, {
        "event_scope": [scope.after_seq, scope.until_seq],
        "waypoint_scope": [scope.after_waypoint, scope.until_waypoint],
        "observed": observed,
        "expected": expected,
        "ids": sorted(seen),
    }


def _width_satisfied(
    asg: ASG,
    item: dict[str, Any],
    route: dict[str, Any],
    event_times: dict[str, int],
    waypoint_positions: dict[str, int],
    portal_events: list[tuple[int, int]],
) -> tuple[bool, dict[str, Any]]:
    scope = _scope(item, event_times, waypoint_positions)
    width = item.get("width_m", item.get("grounded_width_m"))
    if scope is None or width is None:
        return False, {"reason": "invalid width or scope"}
    failed_portals = []
    for pos, portal_id in portal_events:
        if not transition_in_waypoint_scope(
            pos,
            scope.after_waypoint,
            scope.until_waypoint,
            include_until_transition=True,
        ):
            continue
        portal = asg.get_portal(portal_id)
        if portal is None or portal.clear_width is None or float(portal.clear_width) + 1e-9 < float(width):
            failed_portals.append(portal_id)

    witness = {
        "event_scope": [scope.after_seq, scope.until_seq],
        "waypoint_scope": [scope.after_waypoint, scope.until_waypoint],
        "width_m": width,
        "failed_portals": failed_portals,
        # Ordinary physical clearance is established by the solver's shared
        # region/mesh-band graph.  ASG collision footprints are semantic
        # goal/VIA/AVOID/COUNT/PREFER geometry, never obstacle geometry.
        "ordinary_clearance_authority": "solver_region_mesh_body_band",
        "portal_transition_clearance": "portal_clear_width",
    }
    return not failed_portals, witness


def _preference_cost(
    asg: ASG,
    grounded: dict[str, Any],
    route: dict[str, Any],
    event_times: dict[str, int],
    waypoint_positions: dict[str, int],
    portal_events: list[tuple[int, int]],
    conditional_orders: list[dict[str, Any]] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    conditional_orders = conditional_orders or []
    cost = 0.0
    results = []
    event_trace, via_occurrences = _preference_event_trace(
        asg, grounded, route, portal_events
    )
    named_vias = [
        item
        for item in grounded.get("prefer", [])
        if item.get("preference") == "via" and item.get("name")
    ]
    order_evaluation = minimum_soft_event_penalty(
        named_vias,
        conditional_orders,
        grounded.get("prefer_order", []),
        event_trace,
    )
    cost += float(order_evaluation["penalty"])
    for index, item in enumerate(grounded.get("prefer", [])):
        preference = item.get("preference")
        if preference == "avoid":
            ok, witness = _avoid_satisfied(
                asg,
                item,
                route,
                event_times,
                waypoint_positions,
                portal_events,
                include_until_transition=True,
            )
        elif preference == "count":
            ok, witness = _count_satisfied(
                item,
                route,
                event_times,
                waypoint_positions,
                portal_events,
                asg=asg,
            )
        else:
            name = str(item["name"]) if item.get("name") else None
            occurrences = (
                via_occurrences.get(name, [])
                if name
                else _preferred_via_occurrences(asg, item, route, portal_events)
            )
            selected = (
                order_evaluation["selected_occurrences"].get(name)
                if name
                else None
            )
            ok = selected is not None if name else bool(occurrences)
            witness = {
                "candidate_occurrences": occurrences,
                "selected_occurrence": selected,
                "grounded_ids": sorted(
                    int(value) for value in item.get("grounded_ids", [])
                ),
            }
        if not ok and not (preference == "via" and item.get("name")):
            cost += float(item.get("weight", 1))
        results.append(_result("prefer", f"prefer[{index}]", ok, witness))
    for item, evaluation in zip(
        conditional_orders, order_evaluation["required_results"]
    ):
        witness = dict(evaluation["witness"])
        witness["selected_soft_occurrences"] = order_evaluation[
            "selected_occurrences"
        ]
        results.append(
            _result(
                "order",
                f"order[{item['_source_index']}]",
                bool(evaluation["satisfied"]),
                witness,
            )
        )
    order_items = grounded.get("prefer_order", [])
    for index, (item, evaluation) in enumerate(
        zip(order_items, order_evaluation["soft_order_results"])
    ):
        ok = bool(evaluation["satisfied"])
        witness = dict(evaluation["witness"])
        witness.update(
            {
                "selected_soft_occurrences": order_evaluation[
                    "selected_occurrences"
                ],
                "soft_via_occurrences": {
                    name: occurrences
                    for name, occurrences in via_occurrences.items()
                    if name in _order_names(item)
                },
            }
        )
        # ``minimum_soft_event_penalty`` already charged each soft ORDER once.
        results.append(_result("prefer_order", f"prefer_order[{index}]", ok, witness))
    return cost, results


def _objects_entered(
    asg: ASG,
    ids: set[int],
    waypoints: list[dict[str, Any]],
    scope: RouteScope,
    *,
    include_until_transition: bool,
    excluded_on_until: tuple[str, int] | None = None,
    regions: list[int] | None = None,
) -> set[int]:
    if regions is None or len(regions) != len(waypoints):
        # Never guess a floor/region from one arbitrary object. Current replay
        # supplies a complete region sequence; an incomplete legacy route has
        # no trustworthy object-encounter evidence.
        return set()
    last_segment = (
        scope.until_waypoint
        if include_until_transition
        else scope.until_waypoint - 1
    )
    return {
        item.object_id
        for item in object_encounter_entries(
            asg,
            ids,
            waypoints,
            [int(value) for value in regions],
            first_segment=scope.after_waypoint + 1,
            last_segment=last_segment,
            clearance_m=ENDPOINT_OBJECT_CLEARANCE_M,
            include_terminal_event=False,
            excluded_terminal_object_ids=(
                (excluded_on_until[1],)
                if excluded_on_until is not None
                and excluded_on_until[0] == "obj"
                else ()
            ),
        )
    }


def _scope_boundary_entity(
    route: dict[str, Any],
    name: str,
) -> tuple[str, int] | None:
    hit = _milestone(route, name)
    if hit is None:
        return None
    kind = str(hit.get("target_kind", hit.get("kind", "")))
    raw_id = hit.get("resolved_index", hit.get("id"))
    if kind not in {"obj", "region", "portal", "level"} or raw_id is None:
        return None
    return kind, int(raw_id)


def _order_names(item: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(item.get("first")),
        str(item.get("second")),
        *(str(name) for name in item.get("names", [])),
    )


def _result(component: str, name: str, satisfied: bool, witness: dict[str, Any]) -> dict[str, Any]:
    return {"component": component, "name": name, "satisfied": bool(satisfied), "witness": witness}


def _failure(kind: str, component: str, message: str, witness: Any = None) -> dict[str, Any]:
    return {
        "component": component,
        "type": kind,
        "message": message,
        "witness": witness,
    }
