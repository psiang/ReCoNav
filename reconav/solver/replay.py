"""Independent evaluator replay of a parsed program and a solver witness."""
from __future__ import annotations

from dataclasses import asdict
from reconav.solver.target_grounding import ground_target, target_from_avoid


def milestone_occurrence_errors(asg, route):
    """Validate supplied event positions against the complete solver domain.

    ``route`` uses waypoints/regions/portals/milestones. This validates real
    occurrences, not language bindings, which remain the original evaluator's
    responsibility. Interior REGION/LEVEL GOAL arrivals are allowed; VIA must
    occur initially or on a region transition (also for a same-level LEVEL).
    """
    from shapely.geometry import Point
    import math
    from reconav.scene.physical import ENDPOINT_OBJECT_CLEARANCE_M, within_semantic_clearance
    from reconav.solver.cancellation import check_local_search_budget
    from reconav.solver.qualifier_geometry import object_geometry, UnknownRelationGeometry

    points = route.get("waypoints", ())
    regions = route.get("regions", ())
    if not points or len(points) != len(regions):
        return ["milestone validation requires aligned physical waypoints and regions"]
    regions = [int(region) for region in regions]
    portal_events = {}
    portal_ids = iter(route.get("portals", ()))
    for index in range(1, len(regions)):
        if regions[index] != regions[index - 1]:
            portal = next(portal_ids, None)
            if portal is not None:
                portal_events[index] = int(portal)
    errors = []
    order = []
    names = set()
    sequences = set()
    def integer(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise ValueError("not an integer")
        return int(value)
    for hit in route.get("milestones", ()):
        check_local_search_budget()
        name = str(hit.get("name", ""))
        try:
            index = integer(hit["waypoint_index"])
            entity_id = integer(hit.get("resolved_index", hit.get("id")))
            event_seq = integer(hit["event_seq"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name}: missing or invalid milestone position/identity")
            continue
        if name in names:
            errors.append(f"{name}: duplicate named milestone")
        names.add(name)
        if event_seq < 0 or event_seq in sequences:
            errors.append(f"{name}: invalid or duplicate named event sequence")
        sequences.add(event_seq)
        if not 0 <= index < len(points):
            errors.append(f"{name}: milestone index outside route")
            continue
        order.append((event_seq, index))
        kind = hit.get("target_kind", hit.get("kind"))
        if kind not in {"pos", "obj", "region", "level", "portal"} or (kind == "pos" and name != "START"):
            errors.append(f"{name}: unknown or ineligible milestone kind")
        if name == "START" and (index != 0 or event_seq != 0):
            errors.append("START: must be the initial route occurrence")
        if name == "GOAL" and index != len(points) - 1:
            errors.append("GOAL: must be the terminal route occurrence")
        region = asg.get_region(regions[index])
        if region is None:
            errors.append(f"{name}: unknown physical region")
            continue
        if kind in {"region", "level"}:
            actual = regions[index] if kind == "region" else int(region.level)
            if actual != entity_id:
                errors.append(f"{name}: {kind} identity differs from actual waypoint")
            if name != "GOAL" and index != 0 and regions[index - 1] == regions[index]:
                errors.append(f"{name}: {kind} VIA is not an initial or region-entry occurrence")
        elif kind == "portal":
            portal = asg.get_portal(entity_id)
            valid = (
                index > 0 and portal_events.get(index) == entity_id
                and portal is not None
                and {int(portal.region_a), int(portal.region_b)} == {regions[index - 1], regions[index]}
                and regions[index - 1] != regions[index]
            )
            if not valid:
                errors.append(f"{name}: portal milestone is not its actual crossing")
        elif kind == "obj":
            point = points[index]
            try:
                x, z = (float(point["x"]), float(point["z"])) if isinstance(point, dict) else (float(point.x), float(point.z))
                shape, _level, object_region = object_geometry(asg, entity_id)
                valid = object_region == regions[index] and within_semantic_clearance(
                    Point(x, z).distance(shape), ENDPOINT_OBJECT_CLEARANCE_M,
                )
            except (UnknownRelationGeometry, KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"{name}: object milestone lacks an actual same-region encounter")
    ordered_positions = [index for _seq, index in sorted(order)]
    if ordered_positions != sorted(ordered_positions):
        errors.append("named event order disagrees with physical waypoint order")
    return errors


def replay_solution(program, asg, result):
    from reconav.benchmark.route_evaluator import evaluate_grounded_route
    start = program.original_start_target or program.start().target
    def target(selector):
        value=ground_target(asg,selector,0,start_target=start)
        return {"kind":value.kind,"grounded_ids":sorted(value.ids)}
    def scope(spec):return {"after":spec.after,"until":spec.until}
    def order(spec):return {"kind":spec.kind,"first":spec.first,"second":spec.second,"names":list(spec.names)}
    grounded={"goal":target(program.goal().target),"via":[],"order":[order(x) for x in program.order_constraints],"avoid":[],"count":[],"prefer":[],"prefer_order":[]}
    for item in program.milestones:
        if item.role=="via":grounded["via"].append({**target(item.target),"name":item.name})
    for item in program.avoid_specs:
        grounded["avoid"].append({**target(target_from_avoid(item)),**scope(item)})
    for item in program.count_specs:
        grounded["count"].append({**target(item.target),**scope(item),"op":item.op,"count":item.count})
    for index,item in enumerate(program.soft_via_specs):
        grounded["prefer"].append({**target(item.target),"preference":"via","name":item.name or f"_prefer_{index}","weight":1})
    for item in program.soft_avoid_specs:
        grounded["prefer"].append({**target(item.target),**scope(item),"preference":"avoid","weight":1})
    for item in program.soft_count_specs:
        grounded["prefer"].append({**target(item.target),**scope(item),"preference":"count","op":item.op,"count":item.count,"weight":1})
    for item in program.soft_order_specs:
        grounded["prefer_order"].append({**order(item.constraint),"weight":1})
    route={
        "waypoints":[{"x":p.x,"y":p.y,"z":p.z} for p in result.waypoints],
        "regions":list(result.traversed_region_ids),"portals":list(result.portal_indices),
        "milestones":[{**asdict(hit),"kind":hit.target_kind,"id":hit.resolved_index} for hit in result.milestone_hits],
        "execution_trace_complete":False,
        "optimality_established":result.optimality_established,
        "penalty_proven":result.penalty_proven,
        "length_proven":result.length_proven,
        "optimal_soft_cost":result.optimal_soft_cost if result.penalty_proven else None,
    }
    occurrence_errors = milestone_occurrence_errors(asg, route)
    if occurrence_errors:
        return {
            "ok": False,
            "errors": [{"code": "ineligible_milestone_occurrence", "message": error} for error in occurrence_errors],
            "constraints": [], "preference_cost": None,
            "solver_penalty": result.optimal_soft_cost, "penalty_matches_solver": False,
        }
    original=program.start().target
    if original.kind=="pos":
        x,y,z=(float(v) for v in original.values)
    elif result.waypoints:
        point=result.waypoints[0];x,y,z=point.x,point.y,point.z
    else:
        x=y=z=0.0
    report=evaluate_grounded_route(asg,grounded,route,{"x":x,"floor_height":y,"y":z})
    report["solver_penalty"]=result.optimal_soft_cost
    report["penalty_matches_solver"]=(report.get("preference_cost") is not None and result.optimal_soft_cost is not None and abs(float(report["preference_cost"])-float(result.optimal_soft_cost))<=1e-9)
    return report
