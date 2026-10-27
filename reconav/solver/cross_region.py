"""
Cross-region route planning over ASG topology, with route milestone progress.

The search state is the current region, the current anchor (object or portal), and
which DSL milestones have been witnessed. VIA OBJ is satisfied by an actual
in-region path to the object, not merely by passing through the object's region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


from reconav.scene.asg import (
    ASG,
)
from reconav.scene.geometry import Point3
from reconav.solver.cancellation import CancellationToken
from reconav.solver.dsl_ast import DSLProgram
from reconav.scene.physical import ROBOT_DIAMETER_M as DEFAULT_ROBOT_WIDTH_M
from reconav.solver.query_evidence import LEX_OPTIMAL, SolveRequest

@dataclass(frozen=True)
class RouteMilestoneHit:
    name: str
    role: str
    target_kind: str
    target_selector: str
    target_value: int | str
    resolved_index: int
    point: Point3
    waypoint_index: int | None = None
    event_seq: int | None = None


@dataclass(frozen=True)
class RouteCountObservation:
    constraint_handle: str
    entity_kind: str
    entity_id: int
    point: Point3
    event_seq_before: int
    event_seq_after: int
    waypoint_index: int | None = None
    trace_seq: int | None = None


@dataclass(frozen=True)
class CrossRegionPathResult:
    """Cross-region route result: waypoints, per-waypoint region ids, portals, and milestone witnesses."""

    waypoints: List[Point3]
    traversed_region_ids: List[int]
    portal_indices: List[int]
    milestone_hits: List[RouteMilestoneHit] = field(default_factory=list)
    optimal_soft_cost: float | None = None
    execution_trace: List[RouteCountObservation] = field(default_factory=list)
    status: str = "SAT"
    status_reason: str | None = None
    physical_domain_complete: bool = True
    query_graph_complete: bool = False
    state_search_complete: bool = False
    optimality_established: bool = False
    proof_domain: str = "region-envelope + mesh-body-band + portal-contract"
    semantic_replay: dict | None = None
    requested_mode: str = "LEX_OPTIMAL"
    requested_bound: int | None = None
    query_outcome: str = "INCOMPLETE"
    witness_penalty: float | None = None
    witness_length: float | None = None
    penalty_lb: float | None = None
    penalty_ub: float | None = None
    penalty_proven: bool = False
    length_lb: float | None = None
    length_ub: float | None = None
    length_proven: bool = False
    length_condition: str | None = None
    certificate_kind: str | None = None
    certificate_payload: dict | None = None
    cancelled: bool = False
    incomplete: bool = False
    timing: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)


def cross_region_nav(
    asg: ASG,
    program: DSLProgram,
    default_robot_width_m: float = DEFAULT_ROBOT_WIDTH_M,
    cancel_token: CancellationToken | None = None,
    request: SolveRequest = LEX_OPTIMAL,
) -> CrossRegionPathResult:
    """Use the shared physical graph with complete event-state queries."""
    if abs(default_robot_width_m - DEFAULT_ROBOT_WIDTH_M) > 1e-9:
        raise ValueError("the current benchmark uses one fixed robot width")
    from reconav.solver.state_search import solve_event_product
    return solve_event_product(asg, program, cancel_token, request=request)
