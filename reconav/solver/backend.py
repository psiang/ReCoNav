"""Reusable adapters for the existing DSL, solver, automaton, and geometry backend."""

from __future__ import annotations

import math
import time
from typing import Any

from reconav.scene.asg import ASG
from reconav.solver.cross_region import CrossRegionPathResult
from reconav.solver.dsl_ast import (
    DSLProgram,
    OrderConstraint,
    SourceLine,
)
from reconav.solver.dsl_parser import parse_dsl, solve_program
from reconav.solver.execution_profiles import (
    ExecutionProfile,
    coerce_profile,
    objective_for,
    program_features,
    validate_execution_profile,
)
from reconav.solver.query_evidence import FEASIBLE, MIN_PENALTY, SolveRequest


def solve_dsl_route(
    dsl: str,
    asg: ASG,
    *,
    profile: ExecutionProfile | str,
    cancel_token=None,
    request: SolveRequest | None = None,
) -> dict[str, Any]:
    """Validate and execute DSL under an explicit capability profile.

    WAYPOINT carries only a fixed typed-ID sequence. The sequence is executed
    by the same one-shot solve_program/state-search stack as the structured
    profiles, with no benchmark AVOID/COUNT/PREFERENCE semantics exposed.
    """
    started = time.perf_counter()
    requested_profile = coerce_profile(profile)
    program = parse_dsl(dsl)
    if request is None:
        features = program_features(program)
        request = (
            MIN_PENALTY
            if requested_profile is ExecutionProfile.FULL_HARD_SOFT
            and any(item in {"prefer", "prefer_order"} for item in features)
            else FEASIBLE
        )
    validation = validate_execution_profile(program, requested_profile)
    if requested_profile == ExecutionProfile.WAYPOINT:
        result = _solve_waypoint_product(
            program,
            asg,
            cancel_token=cancel_token,
            request=request,
        )
        execution_mode = "temporal_product_search_fixed_waypoint_sequence"
    else:
        result = solve_program(
            program,
            asg,
            cancel_token=cancel_token,
            request=request,
        )
        execution_mode = "temporal_product_search"
    selected_profile = validation["profile"]
    objective = list(objective_for(selected_profile))
    solver_regions = [int(x) for x in result.traversed_region_ids]
    return {
        "waypoints": _waypoints(result.waypoints),
        "regions": solver_regions,
        "solver_region_witness": solver_regions,
        "route_source": "dsl_solver",
        "portals": [int(x) for x in result.portal_indices],
        "targets": _targets(result.milestone_hits),
        "milestones": _milestones(result.milestone_hits),
        "execution_trace": _execution_trace(getattr(result, "execution_trace", [])),
        "execution_trace_complete": False,
        "cost": _path_length(result.waypoints),
        "optimal_soft_cost": result.optimal_soft_cost
        if selected_profile == ExecutionProfile.FULL_HARD_SOFT.value
        else None,
        "status": result.status,
        "status_reason": result.status_reason,
        "physical_domain_complete": result.physical_domain_complete,
        "query_graph_complete": result.query_graph_complete,
        "state_search_complete": result.state_search_complete,
        "optimality_established": result.optimality_established,
        "penalty_lb": result.penalty_lb,
        "penalty_ub": result.penalty_ub,
        "penalty_proven": result.penalty_proven,
        "length_lb": result.length_lb,
        "length_ub": result.length_ub,
        "length_proven": result.length_proven,
        "length_condition": result.length_condition,
        "requested_mode": result.requested_mode,
        "query_outcome": result.query_outcome,
        "certificate_kind": result.certificate_kind,
        "certificate_payload": result.certificate_payload,
        "proof_domain": result.proof_domain,
        "semantic_replay": result.semantic_replay,
        "backend": "asg_solver",
        "profile": selected_profile,
        "profile_validation": validation,
        "objective": objective,
        "execution_mode": execution_mode,
        "enforced_constraints": list(program_features(program)),
        "solver_time_ms": int((time.perf_counter() - started) * 1000),
    }


def _solve_waypoint_product(
    program: DSLProgram,
    asg: ASG,
    *,
    cancel_token=None,
    request: SolveRequest = FEASIBLE,
) -> CrossRegionPathResult:
    """Execute a Direct-ID-Sequence route with the shared ordered route search."""
    start = program.start()
    goal = program.goal()
    if start is None or goal is None:
        raise ValueError("WAYPOINT execution requires START and GOAL")

    vias = [item for item in program.milestones if item.role == "via"]
    ordered_names = ["START", *(item.name for item in vias), "GOAL"]
    internal_source = SourceLine(
        0,
        "internal fixed Direct-ID-Sequence route",
    )
    sequence_order = [
        OrderConstraint(
            kind="next",
            first=left,
            second=right,
            names=(),
            source=internal_source,
        )
        for left, right in zip(ordered_names, ordered_names[1:])
    ]
    ordered = DSLProgram(
        milestones=list(program.milestones),
        order_constraints=sequence_order,
        original_start_target=program.original_start_target,
        query_object_ids=program.query_object_ids,
    )
    return solve_program(
        ordered,
        asg,
        cancel_token=cancel_token,
        request=request,
    )


def _waypoints(points: list[Any]) -> list[dict[str, float]]:
    out = [{"x": float(p.x), "y": float(p.y) + 1.2, "z": float(p.z)} for p in points]
    for i, point in enumerate(out):
        a = point if i < len(out) - 1 else out[max(i - 1, 0)]
        b = out[i + 1] if i < len(out) - 1 else point
        dx, dz = b["x"] - a["x"], b["z"] - a["z"]
        point["heading_rad"] = math.atan2(dz, dx) if dx * dx + dz * dz > 1e-12 else 0.0
    return out


def _targets(hits: list[Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for hit in hits:
        kind = _target_kind(str(hit.target_kind))
        if kind not in {"obj", "region", "portal", "level"}:
            continue
        targets.append({
            "role": str(hit.role),
            "name": str(hit.name),
            "kind": kind,
            "id": int(hit.resolved_index),
            "waypoint_index": hit.waypoint_index,
            "event_seq": hit.event_seq,
        })
    return targets


def _milestones(hits: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(hit.name),
            "role": str(hit.role),
            "target_kind": _target_kind(str(hit.target_kind)),
            "target_selector": str(hit.target_selector),
            "target_value": hit.target_value,
            "resolved_index": int(hit.resolved_index),
            "waypoint_index": hit.waypoint_index,
            "event_seq": hit.event_seq,
            "point": {"x": float(hit.point.x), "y": float(hit.point.y) + 1.2, "z": float(hit.point.z)},
        }
        for hit in hits
    ]


def _execution_trace(observations: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "COUNT_OBSERVATION",
            "constraint_handle": str(item.constraint_handle),
            "entity_kind": _target_kind(str(item.entity_kind)),
            "entity_id": int(item.entity_id),
            "waypoint_index": item.waypoint_index,
            "event_seq_before": int(item.event_seq_before),
            "event_seq_after": int(item.event_seq_after),
            "trace_seq": item.trace_seq,
        }
        for item in observations
    ]


def _target_kind(kind: str) -> str:
    return {"object": "obj"}.get(kind, kind)


def _path_length(points: list[Any]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt((float(b.x) - float(a.x)) ** 2 + (float(b.z) - float(a.z)) ** 2)
    return total
