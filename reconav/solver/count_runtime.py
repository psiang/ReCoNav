"""Compact exact runtime state for hard COUNT constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from reconav.scene.asg import ASG
from reconav.scene.geometry import Point3
from reconav.solver.dsl_ast import CountSpec, DSLProgram
from reconav.solver.order_runtime import Progress
from reconav.solver.route_events import object_ids_encountered
from reconav.solver.target_grounding import ground_target
from reconav.solver.temporal_semantics import transition_scope_active

ObservedEntity = tuple[str, int]


@dataclass(frozen=True)
class CountCell:
    """One exact COUNT key cell; settled cells deliberately discard ID sets."""

    status: str
    mask: int = 0


CountState = tuple[CountCell, ...]


@dataclass(frozen=True)
class GroundedCount:
    spec: CountSpec
    kind: str
    ids: frozenset[int]
    bit_by_id: dict[int, int]


@dataclass(frozen=True)
class CountObservationDelta:
    constraint_index: int
    entity_kind: str
    entity_id: int


class CountRuntime:
    """Track exact distinct entities without retaining settled ID sets."""

    def __init__(self, program: DSLProgram, asg: ASG) -> None:
        self.asg = asg
        self.counts = tuple(_ground_count(spec, asg, program) for spec in program.count_specs)
        self.object_ids_by_region: dict[int, set[int]] = {}
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

    def initial_state(self) -> CountState:
        return tuple(
            CountCell("SATURATED_OK")
            if count.spec.op == "at_least" and count.spec.count == 0
            else CountCell("EXACT_SET")
            for count in self.counts
        )

    def observe(
        self,
        state: CountState,
        progress: Progress,
        target_names: Iterable[str],
        entities: Iterable[ObservedEntity],
        *,
        include_until_transition: bool = False,
    ) -> CountState | None:
        """Consume one ordered event group and settle scopes ending here."""

        if not self.counts:
            return state
        targets = set(target_names)
        entity_list = tuple(set(entities))
        cells = list(state)
        changed = False
        for index, (count, cell) in enumerate(zip(self.counts, state)):
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
                for kind, entity_id in entity_list:
                    bit = count.bit_by_id.get(entity_id) if kind == count.kind else None
                    if bit is not None:
                        mask |= 1 << bit
                n = _popcount(mask)
                if count.spec.op in {"at_most", "exactly"} and n > count.spec.count:
                    return None
                if count.spec.op == "at_least" and n >= count.spec.count:
                    following = CountCell("SATURATED_OK")
                else:
                    following = CountCell("EXACT_SET", mask)
            if count.spec.until in targets:
                if following.status == "SATURATED_OK":
                    following = CountCell("CLOSED")
                else:
                    n = _popcount(following.mask)
                    if not _satisfied(count.spec.op, n, count.spec.count):
                        return None
                    following = CountCell("CLOSED")
            if following != cell:
                cells[index] = following
                changed = True
        return tuple(cells) if changed else state

    def accepting(self, state: CountState) -> bool:
        for count, cell in zip(self.counts, state):
            if cell.status in {"CLOSED", "SATURATED_OK"}:
                continue
            if not _satisfied(count.spec.op, _popcount(cell.mask), count.spec.count):
                return False
        return True

    def observation_deltas_for_event(
        self,
        state: CountState,
        progress: Progress,
        target_names: Iterable[str],
        entities: Iterable[ObservedEntity],
        *,
        include_until_transition: bool = False,
    ) -> tuple[CountObservationDelta, ...]:
        """Return real observations before state compression discards IDs."""

        targets = set(target_names)
        entity_list = tuple(set(entities))
        output: list[CountObservationDelta] = []
        for index, (count, cell) in enumerate(zip(self.counts, state)):
            if cell.status != "EXACT_SET" or not _active(
                count.spec,
                progress,
                targets,
                include_until_transition=include_until_transition,
            ):
                continue
            for kind, entity_id in entity_list:
                bit = count.bit_by_id.get(entity_id) if kind == count.kind else None
                if bit is not None and not (cell.mask & (1 << bit)):
                    output.append(CountObservationDelta(index, kind, entity_id))
        return tuple(sorted(output, key=lambda item: (item.constraint_index, item.entity_id)))

    def observation_deltas(
        self,
        before: CountState,
        after: CountState,
    ) -> tuple[CountObservationDelta, ...]:
        """Compatibility helper for transitions that retain both masks."""

        output: list[CountObservationDelta] = []
        for index, (count, old, new) in enumerate(zip(self.counts, before, after)):
            if old.status != "EXACT_SET" or new.status != "EXACT_SET":
                continue
            added = new.mask & ~old.mask
            for entity_id, bit in count.bit_by_id.items():
                if added & (1 << bit):
                    output.append(CountObservationDelta(index, count.kind, entity_id))
        return tuple(output)

    def count_object_targets_in_region(
        self,
        state: CountState,
        progress: Progress,
        region_index: int,
    ) -> list[int]:
        candidates = self.object_ids_by_region.get(region_index, set())
        if not candidates:
            return []
        out: set[int] = set()
        for count, cell in zip(self.counts, state):
            if (
                count.kind != "obj"
                or cell.status != "EXACT_SET"
                or count.spec.after not in progress
                or count.spec.until in progress
                or count.spec.op == "at_most"
            ):
                continue
            seen = {
                entity_id
                for entity_id, bit in count.bit_by_id.items()
                if cell.mask & (1 << bit)
            }
            if len(seen) < count.spec.count:
                out.update((count.ids - seen) & candidates)
        return sorted(out)

    def observe_path(
        self,
        state: CountState,
        progress: Progress,
        target_names: Iterable[str],
        points: list[Point3],
        region_index: int,
        *,
        region_indices: Sequence[int] | None = None,
        excluded_entities: Iterable[ObservedEntity] = (),
    ) -> CountState | None:
        if len(points) < 2:
            return self.observe(state, progress, target_names, ())
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
        encountered = [
            ("obj", object_id)
            for object_id in sorted(
                object_ids_encountered(
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
            )
        ]
        return self.observe(
            state,
            progress,
            target_names,
            encountered,
            include_until_transition=True,
        )


def _ground_count(spec: CountSpec, asg: ASG, program: DSLProgram) -> GroundedCount:
    grounded = ground_target(
        asg,
        spec.target,
        spec.source.number,
        start_target=program.original_start_target or program.start().target,
    )
    if grounded.kind not in {"obj", "region", "portal", "level"}:
        raise ValueError(f"unsupported count target kind: {grounded.kind}")
    ordered = tuple(sorted(grounded.ids))
    return GroundedCount(
        spec,
        grounded.kind,
        grounded.ids,
        {entity_id: index for index, entity_id in enumerate(ordered)},
    )


def _satisfied(op: str, observed: int, required: int) -> bool:
    if op == "at_least":
        return observed >= required
    if op == "at_most":
        return observed <= required
    if op == "exactly":
        return observed == required
    raise ValueError(f"unsupported COUNT operator: {op}")


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _active(
    spec: CountSpec,
    progress: Progress,
    target_names: set[str],
    *,
    include_until_transition: bool = False,
) -> bool:
    return transition_scope_active(
        spec.after,
        spec.until,
        progress,
        target_names,
        include_until_transition=include_until_transition,
    )
