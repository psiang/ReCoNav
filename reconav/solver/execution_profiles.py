"""Capability gates for the shared navigation execution IR.

Profiles constrain which parts of :class:`DSLProgram` an adapter may expose.
They do not infer missing constraints and do not inspect natural language or
benchmark annotations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from reconav.solver.dsl_ast import DSLProgram


class ExecutionProfile(str, Enum):
    WAYPOINT = "WAYPOINT"
    HARD_TEMPORAL = "HARD_TEMPORAL"
    FULL_HARD_SOFT = "FULL_HARD_SOFT"


class ProfileCapabilityError(ValueError):
    """Raised when an execution IR exceeds its selected solver profile."""

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report or {}


_FEATURE_ORDER = (
    "via",
    "order",
    "avoid",
    "count",
    "prefer",
    "prefer_order",
)

_ALLOWED_FEATURES = {
    ExecutionProfile.WAYPOINT: frozenset({"via"}),
    ExecutionProfile.HARD_TEMPORAL: frozenset({"via", "order", "avoid", "count"}),
    ExecutionProfile.FULL_HARD_SOFT: frozenset(_FEATURE_ORDER),
}

_OBJECTIVES = {
    ExecutionProfile.WAYPOINT: ("feasibility",),
    ExecutionProfile.HARD_TEMPORAL: ("feasibility",),
    ExecutionProfile.FULL_HARD_SOFT: ("soft_violation_cost",),
}


def coerce_profile(profile: ExecutionProfile | str) -> ExecutionProfile:
    try:
        return profile if isinstance(profile, ExecutionProfile) else ExecutionProfile(str(profile))
    except ValueError as exc:
        valid = ", ".join(item.value for item in ExecutionProfile)
        raise ValueError(f"unknown execution profile {profile!r}; expected one of: {valid}") from exc


def program_features(program: DSLProgram) -> tuple[str, ...]:
    """Return solver-facing feature families actually present in the AST."""
    present = {
        "via": any(item.role == "via" for item in program.milestones),
        "order": bool(program.order_constraints),
        "avoid": bool(program.avoid_specs),
        "count": bool(program.count_specs),
        "prefer": bool(program.soft_via_specs or program.soft_avoid_specs or program.soft_count_specs),
        "prefer_order": bool(program.soft_order_specs),
    }
    return tuple(name for name in _FEATURE_ORDER if present[name])


def objective_for(profile: ExecutionProfile | str) -> tuple[str, ...]:
    return _OBJECTIVES[coerce_profile(profile)]


def validate_execution_profile(
    program: DSLProgram,
    profile: ExecutionProfile | str,
) -> dict[str, Any]:
    """Validate one parsed execution IR and return an auditable report.

    WAYPOINT accepts only concrete grounded milestones.  Its executor consumes
    VIA declarations in source order and then GOAL; order operators remain out
    of capability because they would expose temporal search semantics.
    """
    selected = coerce_profile(profile)
    present = program_features(program)
    allowed = _ALLOWED_FEATURES[selected]
    rejected = tuple(name for name in present if name not in allowed)
    report = {
        "profile": selected.value,
        "accepted": not rejected,
        "present_features": list(present),
        "allowed_features": [name for name in _FEATURE_ORDER if name in allowed],
        "rejected_features": list(rejected),
        "objective": list(_OBJECTIVES[selected]),
    }
    if rejected:
        raise ProfileCapabilityError(
            f"{selected.value} profile rejects execution IR feature(s): {', '.join(rejected)}",
            report,
        )
    if selected == ExecutionProfile.WAYPOINT:
        try:
            _validate_waypoint_milestones(program)
        except ProfileCapabilityError as exc:
            report["accepted"] = False
            report["rejected_features"].append("waypoint_grounding")
            raise ProfileCapabilityError(str(exc), report) from exc
    return report


def _validate_waypoint_milestones(program: DSLProgram) -> None:
    start = program.start()
    if start is None or start.target.kind != "pos" or start.target.selector != "coord":
        raise ProfileCapabilityError("WAYPOINT requires a concrete START POS")
    if start.target.qualifiers:
        raise ProfileCapabilityError("WAYPOINT START POS cannot contain qualifiers")

    targets = [item for item in program.milestones if item.role == "via"]
    goal = program.goal()
    if goal is not None:
        targets.append(goal)
    for target in targets:
        if target.target.kind not in {"obj", "region", "portal", "level"}:
            raise ProfileCapabilityError(
                "WAYPOINT supports only grounded OBJ/REGION/PORTAL/LEVEL targets, "
                f"got {target.target.kind.upper()}"
            )
        if target.target.selector != "id" or len(target.target.values) != 1:
            raise ProfileCapabilityError("WAYPOINT targets must contain exactly one grounded ASG id")
        if target.target.qualifiers:
            raise ProfileCapabilityError("WAYPOINT targets cannot contain grounding qualifiers")
