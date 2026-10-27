"""Grounded semantic checks for AVOID constraints."""

from __future__ import annotations

from dataclasses import dataclass


from reconav.scene.asg import ASG
from reconav.solver.dsl_ast import AvoidSpec, DSLProgram
from reconav.solver.target_grounding import ground_target, target_from_avoid


class AvoidConsistencyError(ValueError):
    """Raised when AVOID constraints make required milestones impossible."""


@dataclass(frozen=True)
class GroundedAvoid:
    spec: AvoidSpec
    object_ids: frozenset[int]
    region_ids: frozenset[int]
    portal_ids: frozenset[int]
    level_ids: frozenset[int]


def _ground_avoid(spec: AvoidSpec, asg: ASG, program: DSLProgram) -> GroundedAvoid:
    grounded = ground_target(asg, target_from_avoid(spec), spec.source.number, start_target=program.original_start_target or program.start().target)
    if grounded.kind == "obj":
        return GroundedAvoid(spec, grounded.ids, frozenset(), frozenset(), frozenset())
    if grounded.kind == "region":
        return GroundedAvoid(spec, frozenset(), grounded.ids, frozenset(), frozenset())
    if grounded.kind == "portal":
        return GroundedAvoid(spec, frozenset(), frozenset(), grounded.ids, frozenset())
    if grounded.kind == "level":
        regions = frozenset(
            region.index for region in asg.regions if region.level in grounded.ids
        )
        return GroundedAvoid(spec, frozenset(), regions, frozenset(), grounded.ids)
    raise ValueError(f"unsupported avoid target kind: {grounded.kind}")
