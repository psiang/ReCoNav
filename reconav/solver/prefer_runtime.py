"""Runtime state for soft PREFER constraints during route search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from reconav.scene.asg import ASG
from reconav.scene.geometry import Point3
from reconav.solver.dsl_ast import (
    DSLProgram,
    OrderConstraint,
    SoftAvoidSpec,
    SoftCountSpec,
    SoftViaSpec,
)
from reconav.solver.order_runtime import Progress
from reconav.solver.route_events import object_encounter_entries
from reconav.solver.target_grounding import ground_target
from reconav.solver.temporal_semantics import (
    minimum_soft_order_penalty,
    order_satisfied_over_occurrences,
    transition_scope_active,
)

ObservedEntity = tuple[str, int]


@dataclass(frozen=True)
class PreferCountCell:
    status: str
    mask: int = 0


PreferCountState = tuple[PreferCountCell, ...]


@dataclass(frozen=True)
class GroundedPreferAvoid:
    spec: SoftAvoidSpec
    kind: str
    ids: frozenset[int]


@dataclass(frozen=True)
class GroundedPreferVia:
    spec: SoftViaSpec
    index: int
    event_name: str
    kind: str
    ids: frozenset[int]


@dataclass(frozen=True)
class GroundedPreferCount:
    spec: SoftCountSpec
    kind: str
    ids: frozenset[int]
    bit_by_id: dict[int, int]


@dataclass(frozen=True)
class PreferState:
    satisfied_via: frozenset[int]
    violated_avoid: frozenset[int]
    count_state: PreferCountState
    events: tuple[str, ...]


@dataclass(frozen=True)
class PreferStep:
    state: PreferState
    penalty: float


class PreferRuntime:
    """Track soft preference witnesses and penalties."""

    def __init__(self, program: DSLProgram, asg: ASG) -> None:
        self.asg = asg
        self.avoids = tuple(_ground_avoid(spec, asg, program) for spec in program.soft_avoid_specs)
        self.vias = tuple(
            _ground_via(i, spec, asg, program) for i, spec in enumerate(program.soft_via_specs)
        )
        self.counts = tuple(_ground_count(spec, asg, program) for spec in program.soft_count_specs)
        self.orders = tuple(program.soft_order_specs)
        self.hard_event_names = frozenset(program.milestone_by_name())
        self.conditional_hard_orders = tuple(
            constraint
            for constraint in program.order_constraints
            if _constraint_mentions_soft(constraint, {via.event_name for via in self.vias})
        )
        self.object_ids_by_region: dict[int, set[int]] = {}
        for via in self.vias:
            if via.kind == "obj":
                for obj_id in via.ids:
                    obj = asg.get_object(obj_id)
                    if obj is not None:
                        for region_index in {
                            int(obj.region_index),
                            *obj.body_band_footprint_2d_by_region,
                        }:
                            self.object_ids_by_region.setdefault(region_index, set()).add(obj_id)
        for avoid in self.avoids:
            if avoid.kind != "obj":
                continue
            for obj_id in avoid.ids:
                obj = asg.get_object(obj_id)
                if obj is not None:
                    for region_index in {
                        int(obj.region_index),
                        *obj.body_band_footprint_2d_by_region,
                    }:
                        self.object_ids_by_region.setdefault(region_index, set()).add(obj_id)
        for count in self.counts:
            if count.kind != "obj":
                continue
            for obj_id in count.ids:
                obj = asg.get_object(obj_id)
                if obj is not None:
                    for region_index in {
                        int(obj.region_index),
                        *obj.body_band_footprint_2d_by_region,
                    }:
                        self.object_ids_by_region.setdefault(region_index, set()).add(obj_id)

    def initial_state(self, progress: Progress, start_region: int) -> PreferState:
        count_state = tuple(
            PreferCountCell("SATURATED_OK")
            if count.spec.op == "at_least" and count.spec.count == 0
            else PreferCountCell("EXACT_SET")
            for count in self.counts
        )
        state = PreferState(frozenset(), frozenset(), count_state, tuple(progress))
        region = self.asg.get_region(start_region)
        entities = [("region", start_region)]
        if region is not None:
            entities.append(("level", region.level))
        step = self.observe(state, progress, set(progress), entities)
        return step.state

    def observe(
        self,
        state: PreferState,
        progress: Progress,
        target_names: Iterable[str],
        entities: Iterable[ObservedEntity],
    ) -> PreferStep:
        event_names = tuple(target_names)
        targets = set(event_names)
        entity_list = tuple(entities)
        events = _append_events(state.events, event_names)
        satisfied_via = set(state.satisfied_via)
        for via in self.vias:
            if any(kind == via.kind and index in via.ids for kind, index in entity_list):
                candidate_events = _append_soft_event(
                    events,
                    via.event_name,
                    self.hard_event_names,
                )
                if self._conditional_hard_orders_prefix_ok(candidate_events):
                    satisfied_via.add(via.index)
                    events = candidate_events

        violated_avoid = set(state.violated_avoid)
        penalty = 0.0
        for i, avoid in enumerate(self.avoids):
            if i in violated_avoid or not _active(
                avoid.spec,
                progress,
                targets,
                include_until_transition=False,
            ):
                continue
            if any(kind == avoid.kind and index in avoid.ids for kind, index in entity_list):
                violated_avoid.add(i)
                penalty += 1.0

        count_state, count_penalty = self._observe_counts(
            state.count_state,
            progress,
            targets,
            entity_list,
            include_until_transition=False,
        )
        next_state = PreferState(frozenset(satisfied_via), frozenset(violated_avoid), count_state, events)
        if not self._conditional_hard_orders_prefix_ok(next_state.events):
            return PreferStep(state, float("inf"))
        return PreferStep(next_state, penalty + count_penalty)

    def final_penalty(self, state: PreferState) -> float:
        penalty = 0.0
        for via in self.vias:
            if via.index not in state.satisfied_via:
                penalty += 1.0
        penalty += float(
            minimum_soft_order_penalty(
                [
                    {
                        "kind": order.constraint.kind,
                        "first": order.constraint.first,
                        "second": order.constraint.second,
                        "names": list(order.constraint.names),
                    }
                    for order in self.orders
                ],
                state.events,
                {via.event_name for via in self.vias},
            )["penalty"]
        )
        for count, cell in zip(self.counts, state.count_state):
            if cell.status in {"CLOSED", "SATURATED_OK", "PAID_VIOLATION"}:
                continue
            if not _count_satisfied(
                count.spec.op,
                _popcount(cell.mask),
                count.spec.count,
            ):
                penalty += 1.0
        if not self._conditional_hard_orders_ok(state.events):
            return float("inf")
        return penalty

    def outcome_vector(self, state: PreferState) -> tuple[tuple[str, int, int], ...]:
        """Return one independently auditable binary outcome per soft clause."""

        outcomes: list[tuple[str, int, int]] = []
        outcomes.extend(
            ("avoid", index, int(index in state.violated_avoid))
            for index in range(len(self.avoids))
        )
        outcomes.extend(
            ("via", via.index, int(via.index not in state.satisfied_via))
            for via in self.vias
        )
        for index, (count, cell) in enumerate(zip(self.counts, state.count_state)):
            if cell.status in {"SATURATED_OK", "CLOSED"}:
                violated = False
            elif cell.status == "PAID_VIOLATION":
                violated = True
            else:
                violated = not _count_satisfied(
                    count.spec.op,
                    _popcount(cell.mask),
                    count.spec.count,
                )
            outcomes.append(("count", index, int(violated)))
        outcomes.extend(
            ("order", index, int(not _order_satisfied(order.constraint, state.events)))
            for index, order in enumerate(self.orders)
        )
        return tuple(outcomes)

    def satisfies_requirements(
        self,
        state: PreferState,
        requirements: tuple[tuple[str, int, int], ...],
    ) -> bool:
        actual = {
            (kind, index): outcome
            for kind, index, outcome in self.outcome_vector(state)
        }
        return all(actual.get((kind, index)) == outcome for kind, index, outcome in requirements)

    def observe_path(
        self,
        state: PreferState,
        progress: Progress,
        target_names: Iterable[str],
        points: list[Point3],
        region_index: int,
        *,
        region_indices: Sequence[int] | None = None,
        excluded_entities: Iterable[ObservedEntity] = (),
    ) -> PreferStep:
        if len(points) < 2:
            return PreferStep(state, 0.0)
        targets = set(target_names)
        excluded = set(excluded_entities)
        route_regions = (
            [int(region_index)] * len(points)
            if region_indices is None
            else [int(value) for value in region_indices]
        )
        if len(route_regions) != len(points):
            raise ValueError("route region count must match point count")
        candidate_ids = set().union(
            *(self.object_ids_by_region.get(value, set()) for value in route_regions)
        )
        entries = object_encounter_entries(
            self.asg,
            candidate_ids,
            points,
            route_regions,
            excluded_terminal_object_ids=(
                entity_id
                for kind, entity_id in excluded
                if kind == "obj"
            ),
        )
        encountered = tuple(
            ("obj", item.object_id)
            for item in entries
        )

        events = state.events
        satisfied_via = set(state.satisfied_via)
        for kind, object_id in encountered:
            for via in self.vias:
                if via.kind != kind or object_id not in via.ids:
                    continue
                candidate_events = _insert_soft_event_before_targets(
                    events,
                    via.event_name,
                    self.hard_event_names,
                    targets,
                )
                if self._conditional_hard_orders_prefix_ok(candidate_events):
                    satisfied_via.add(via.index)
                    events = candidate_events

        violated = set(state.violated_avoid)
        penalty = 0.0
        for i, avoid in enumerate(self.avoids):
            if i in violated or avoid.kind != "obj" or not _active(
                avoid.spec,
                progress,
                targets,
                include_until_transition=True,
            ):
                continue
            if any(kind == "obj" and obj_id in avoid.ids for kind, obj_id in encountered):
                violated.add(i)
                penalty += 1.0
        count_state, count_penalty = self._observe_counts(
            state.count_state,
            progress,
            targets,
            encountered,
            include_until_transition=True,
        )
        if (
            satisfied_via == set(state.satisfied_via)
            and not violated - state.violated_avoid
            and count_state == state.count_state
        ):
            return PreferStep(state, 0.0)
        return PreferStep(
            PreferState(
                frozenset(satisfied_via),
                frozenset(violated),
                count_state,
                events,
            ),
            penalty + count_penalty,
        )

    def optional_object_targets_in_region(
        self,
        state: PreferState,
        progress: Progress,
        region_index: int,
    ) -> list[int]:
        candidates = self.object_ids_by_region.get(region_index, set())
        if not candidates:
            return []
        out: set[int] = set()
        for via in self.vias:
            if via.kind == "obj" and via.index not in state.satisfied_via:
                out.update(via.ids & candidates)
        for count, cell in zip(self.counts, state.count_state):
            if (
                count.kind != "obj"
                or count.spec.op == "at_most"
                or cell.status != "EXACT_SET"
            ):
                continue
            if not _active(
                count.spec,
                progress,
                set(),
                include_until_transition=False,
            ):
                continue
            seen = {
                entity_id
                for entity_id, bit in count.bit_by_id.items()
                if cell.mask & (1 << bit)
            }
            if len(seen) >= count.spec.count:
                continue
            out.update((count.ids - seen) & candidates)
        return sorted(out)

    def _observe_counts(
        self,
        state: PreferCountState,
        progress: Progress,
        targets: set[str],
        entities: tuple[ObservedEntity, ...],
        *,
        include_until_transition: bool,
    ) -> tuple[PreferCountState, float]:
        if not self.counts:
            return state, 0.0
        next_cells = list(state)
        changed = False
        penalty = 0.0
        for i, (count, cell) in enumerate(zip(self.counts, state)):
            if cell.status == "CLOSED":
                continue
            active = _active(
                count.spec,
                progress,
                targets,
                include_until_transition=include_until_transition,
            )
            following = cell
            if active and cell.status == "EXACT_SET":
                mask = cell.mask
                for kind, entity_id in set(entities):
                    bit = count.bit_by_id.get(entity_id) if kind == count.kind else None
                    if bit is not None:
                        mask |= 1 << bit
                n = _popcount(mask)
                if count.spec.op in {"at_most", "exactly"} and n > count.spec.count:
                    following = PreferCountCell("PAID_VIOLATION")
                    penalty += 1.0
                elif count.spec.op == "at_least" and n >= count.spec.count:
                    following = PreferCountCell("SATURATED_OK")
                else:
                    following = PreferCountCell("EXACT_SET", mask)
            if count.spec.until in targets:
                if following.status == "EXACT_SET" and not _count_satisfied(
                    count.spec.op,
                    _popcount(following.mask),
                    count.spec.count,
                ):
                    penalty += 1.0
                    following = PreferCountCell("PAID_VIOLATION")
                elif following.status != "PAID_VIOLATION":
                    following = PreferCountCell("CLOSED")
            if following != cell:
                next_cells[i] = following
                changed = True
        return (tuple(next_cells) if changed else state), penalty

    def _conditional_hard_orders_ok(self, events: tuple[str, ...]) -> bool:
        for constraint in self.conditional_hard_orders:
            refs = _constraint_refs(constraint)
            if all(ref in events for ref in refs) and not _order_satisfied(constraint, events):
                return False
        return True

    def _conditional_hard_orders_prefix_ok(self, events: tuple[str, ...]) -> bool:
        """Reject a soft-VIA occurrence only when its current prefix is impossible.

        A preferred target may be traversed more than once.  The first spatial
        encounter is therefore not forced to become the semantic occurrence when
        a hard order requires a later encounter.
        """
        seen = set(events)
        for constraint in self.conditional_hard_orders:
            a, b = constraint.first, constraint.second
            if constraint.kind == "before":
                if b in seen and a not in seen:
                    return False
                if a in seen and b in seen and events.index(a) >= events.index(b):
                    return False
            elif constraint.kind == "next":
                if b in seen and a not in seen:
                    return False
                if a in seen:
                    first = events.index(a)
                    if first + 1 < len(events) and events[first + 1] != b:
                        return False
                    if b in seen and events.index(b) != first + 1:
                        return False
            elif constraint.kind in {"between_only", "between_without"}:
                if not _between_prefix_ok(events, constraint):
                    return False
            else:
                raise ValueError(f"unsupported order constraint kind: {constraint.kind}")
        return True


def _ground_avoid(spec: SoftAvoidSpec, asg: ASG, program: DSLProgram) -> GroundedPreferAvoid:
    grounded = ground_target(asg, spec.target, spec.source.number, start_target=program.original_start_target or program.start().target)
    return GroundedPreferAvoid(spec, grounded.kind, grounded.ids)


def _ground_via(index: int, spec: SoftViaSpec, asg: ASG, program: DSLProgram) -> GroundedPreferVia:
    grounded = ground_target(asg, spec.target, spec.source.number, start_target=program.original_start_target or program.start().target)
    name = spec.name if spec.name is not None else f"_prefer_{index}"
    return GroundedPreferVia(spec, index, name, grounded.kind, grounded.ids)


def _ground_count(spec: SoftCountSpec, asg: ASG, program: DSLProgram) -> GroundedPreferCount:
    grounded = ground_target(asg, spec.target, spec.source.number, start_target=program.original_start_target or program.start().target)
    ordered = tuple(sorted(grounded.ids))
    return GroundedPreferCount(
        spec,
        grounded.kind,
        grounded.ids,
        {entity_id: index for index, entity_id in enumerate(ordered)},
    )


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _active(
    spec,
    progress: Progress,
    target_names: set[str],
    *,
    include_until_transition: bool,
) -> bool:
    return transition_scope_active(
        spec.after,
        spec.until,
        progress,
        target_names,
        include_until_transition=include_until_transition,
    )


def _append_events(events: tuple[str, ...], names: Iterable[str]) -> tuple[str, ...]:
    out = list(events)
    for name in names:
        if name not in out:
            out.append(name)
    return tuple(out)


def _append_soft_event(
    events: tuple[str, ...],
    name: str,
    hard_event_names: frozenset[str],
) -> tuple[str, ...]:
    last_hard = max(
        (index for index, event in enumerate(events) if event in hard_event_names),
        default=-1,
    )
    if name in events[last_hard + 1 :]:
        return events
    return (*events, name)


def _insert_soft_event_before_targets(
    events: tuple[str, ...],
    name: str,
    hard_event_names: frozenset[str],
    target_names: set[str],
) -> tuple[str, ...]:
    """Place an incidental path encounter before the transition endpoint."""

    insertion = min(
        (index for index, event in enumerate(events) if event in target_names),
        default=len(events),
    )
    prefix = events[:insertion]
    updated = _append_soft_event(prefix, name, hard_event_names)
    return (*updated, *events[insertion:])


def _constraint_refs(constraint: OrderConstraint) -> tuple[str, ...]:
    return (constraint.first, constraint.second, *constraint.names)


def _constraint_mentions_soft(constraint: OrderConstraint, soft_names: set[str]) -> bool:
    return any(name in soft_names for name in _constraint_refs(constraint))


def _order_satisfied(constraint: OrderConstraint, events: tuple[str, ...]) -> bool:
    satisfied, _witness = order_satisfied_over_occurrences(
        constraint.kind,
        constraint.first,
        constraint.second,
        constraint.names,
        events,
    )
    return satisfied


def _between_prefix_ok(events: tuple[str, ...], constraint: OrderConstraint) -> bool:
    seen = set(events)
    a, b = constraint.first, constraint.second
    names = set(constraint.names)
    if b in seen and a not in seen:
        return False
    if a not in seen:
        return not any(name in seen for name in names)
    first = events.index(a)
    if constraint.kind == "between_only" and any(
        name in seen and events.index(name) < first for name in names
    ):
        return False
    end = events.index(b) if b in seen else len(events)
    if end <= first:
        return False
    middle = events[first + 1 : end]
    if constraint.kind == "between_only":
        return all(name in names for name in middle) and (
            b not in seen or names.issubset(middle)
        )
    return all(name not in names for name in middle)


def _count_satisfied(op: str, actual: int, expected: int) -> bool:
    if op == "at_least":
        return actual >= expected
    if op == "at_most":
        return actual <= expected
    if op == "exactly":
        return actual == expected
    raise ValueError(f"unsupported count op: {op}")
