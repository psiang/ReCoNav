"""Runtime grounding for active AVOID constraints during route search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from reconav.scene.asg import ASG
from reconav.solver.dsl_ast import AvoidSpec, DSLProgram
from reconav.solver.order_runtime import Progress
from reconav.solver.target_grounding import ground_target, target_from_avoid
from reconav.solver.temporal_semantics import transition_scope_active


@dataclass(frozen=True)
class GroundedAvoid:
    spec: AvoidSpec
    object_ids: frozenset[int]
    region_ids: frozenset[int]
    portal_ids: frozenset[int]
    level_ids: frozenset[int]


@dataclass(frozen=True)
class ActiveAvoid:
    object_ids: frozenset[int]
    region_ids: frozenset[int]
    portal_ids: frozenset[int]
    level_ids: frozenset[int]


class AvoidRuntime:
    """Expose currently active AVOID entities for a search prefix."""

    def __init__(self, program: DSLProgram, asg: ASG) -> None:
        self.asg = asg
        self.avoids = tuple(_ground_avoid(spec, asg, program) for spec in program.avoid_specs)

    def active_for_transition(self, progress: Progress, target_names: Iterable[str] = ()) -> ActiveAvoid:
        targets = set(target_names)
        object_ids: set[int] = set()
        region_ids: set[int] = set()
        portal_ids: set[int] = set()
        level_ids: set[int] = set()
        for avoid in self.avoids:
            if not transition_scope_active(
                avoid.spec.after,
                avoid.spec.until,
                progress,
                targets,
                include_until_transition=True,
            ):
                continue
            object_ids.update(avoid.object_ids)
            region_ids.update(avoid.region_ids)
            portal_ids.update(avoid.portal_ids)
            level_ids.update(avoid.level_ids)
        region_ids.update(
            region.index for region in self.asg.regions if region.level in level_ids
        )
        return ActiveAvoid(
            frozenset(object_ids),
            frozenset(region_ids),
            frozenset(portal_ids),
            frozenset(level_ids),
        )

    def objects_for_region(self, active: ActiveAvoid, region_index: int) -> set[int]:
        out: set[int] = set()
        for obj_id in active.object_ids:
            obj = self.asg.get_object(obj_id)
            if obj is not None and (
                int(region_index) == int(obj.region_index)
                or int(region_index) in obj.body_band_footprint_2d_by_region
            ):
                out.add(obj_id)
        return out


def _ground_avoid(spec: AvoidSpec, asg: ASG, program: DSLProgram) -> GroundedAvoid:
    grounded = ground_target(asg, target_from_avoid(spec), spec.source.number, start_target=program.original_start_target or program.start().target)
    if grounded.kind == "obj":
        return GroundedAvoid(spec, grounded.ids, frozenset(), frozenset(), frozenset())
    if grounded.kind == "region":
        return GroundedAvoid(spec, frozenset(), grounded.ids, frozenset(), frozenset())
    if grounded.kind == "portal":
        return GroundedAvoid(spec, frozenset(), frozenset(), grounded.ids, frozenset())
    if grounded.kind == "level":
        return GroundedAvoid(spec, frozenset(), frozenset(), frozenset(), grounded.ids)
    raise ValueError(f"unsupported avoid target kind: {grounded.kind}")
