"""Milestone progress runtime for route-level graph search."""

from __future__ import annotations

from dataclasses import dataclass, field

from reconav.scene.asg import ASG
from reconav.solver.dsl_ast import DSLProgram, MilestoneSpec, OrderConstraint
from reconav.solver.target_grounding import ground_target

Progress = tuple[str, ...]
ProgressOption = tuple[Progress, tuple["ObservedEvent", ...]]


@dataclass(frozen=True)
class ObjectMilestone:
    name: str
    object_index: int


@dataclass(frozen=True)
class ObservedEvent:
    name: str
    target_kind: str
    resolved_index: int


@dataclass
class OrderRuntime:
    """Ground DSL milestones to ASG events and validate witness prefixes."""

    program: DSLProgram
    asg: ASG
    constraints: tuple[OrderConstraint, ...] = field(init=False)
    names: tuple[str, ...] = field(init=False)
    object_by_region: dict[int, list[ObjectMilestone]] = field(init=False, default_factory=dict)
    region_events: dict[int, list[ObservedEvent]] = field(init=False, default_factory=dict)
    portal_events: dict[int, list[ObservedEvent]] = field(init=False, default_factory=dict)
    level_events: dict[int, list[ObservedEvent]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.names = tuple(m.name for m in self.program.milestones)
        hard_names = set(self.names)
        self.constraints = tuple(
            constraint
            for constraint in self.program.order_constraints
            if all(name in hard_names for name in (constraint.first, constraint.second, *constraint.names))
        )
        for milestone in self.program.milestones:
            self._ground_milestone(milestone)

    def initial_options(self, start_region: int) -> list[Progress]:
        return [progress for progress, _hits in self.initial_options_with_hits(start_region)]

    def initial_options_with_hits(self, start_region: int) -> list[ProgressOption]:
        progress = self._observe_one((), "START")
        if progress is None:
            raise ValueError("START violates order constraints")
        return self.observe_region_with_hits(progress, start_region)

    def observe_object(self, progress: Progress, milestone_name: str) -> Progress | None:
        return self._observe_one(progress, milestone_name)

    def observe_portal_and_region(self, progress: Progress, portal_index: int, region_index: int) -> list[Progress]:
        return [
            option
            for option, _hits in self.observe_portal_and_region_with_hits(progress, portal_index, region_index)
        ]

    def observe_portal_and_region_with_hits(
        self, progress: Progress, portal_index: int, region_index: int
    ) -> list[ProgressOption]:
        options = self._observe_optional_events_with_hits(progress, self.portal_events.get(portal_index, []))
        out: list[ProgressOption] = []
        for option, portal_hits in options:
            for region_option, region_hits in self.observe_region_with_hits(option, region_index):
                out.append((region_option, portal_hits + region_hits))
        return out

    def observe_region(self, progress: Progress, region_index: int) -> list[Progress]:
        return [option for option, _hits in self.observe_region_with_hits(progress, region_index)]

    def observe_region_with_hits(self, progress: Progress, region_index: int) -> list[ProgressOption]:
        region = self.asg.get_region(region_index)
        level_events = self.level_events.get(region.level, []) if region is not None else []
        return self._observe_optional_events_with_hits(
            progress,
            [*level_events, *self.region_events.get(region_index, [])],
        )

    def observe_goal(self, progress: Progress) -> Progress | None:
        return self._observe_one(progress, "GOAL")

    def can_observe(self, progress: Progress, milestone_name: str) -> bool:
        return self._observe_one(progress, milestone_name) is not None

    def accepting(self, progress: Progress) -> bool:
        return bool(progress) and progress[-1] == "GOAL" and set(self.names).issubset(progress)

    def object_milestones_in_region(self, region_index: int) -> list[ObjectMilestone]:
        return self.object_by_region.get(region_index, [])

    def _ground_milestone(self, milestone: MilestoneSpec) -> None:
        if milestone.name in {"START", "GOAL"}:
            return
        grounded = ground_target(self.asg, milestone.target, milestone.source.number, start_target=self.program.original_start_target or self.program.start().target)
        for index in sorted(grounded.ids):
            self._ground_id_milestone(milestone, index)

    def _ground_id_milestone(self, milestone: MilestoneSpec, index: int) -> None:
        kind = milestone.target.kind
        if kind == "obj":
            obj = self.asg.get_object(index)
            if obj is None:
                raise ValueError(f"line {milestone.source.number}: object {index} not found")
            self.object_by_region.setdefault(obj.region_index, []).append(ObjectMilestone(milestone.name, index))
        elif kind == "region":
            if self.asg.get_region(index) is None:
                raise ValueError(f"line {milestone.source.number}: region {index} not found")
            self.region_events.setdefault(index, []).append(ObservedEvent(milestone.name, "region", index))
        elif kind == "portal":
            if self.asg.get_portal(index) is None:
                raise ValueError(f"line {milestone.source.number}: portal {index} not found")
            self.portal_events.setdefault(index, []).append(ObservedEvent(milestone.name, "portal", index))
        elif kind == "level":
            if self.asg.get_level(index) is None:
                raise ValueError(f"line {milestone.source.number}: level {index} not found")
            self.level_events.setdefault(index, []).append(ObservedEvent(milestone.name, "level", index))
        else:
            raise ValueError(f"unsupported milestone target kind: {kind}")

    def _observe_optional_events_with_hits(self, progress: Progress, events: list[ObservedEvent]) -> list[ProgressOption]:
        options: list[ProgressOption] = [(progress, ())]
        for event in events:
            next_options = list(options)
            for option, hits in options:
                if event.name in option:
                    continue
                observed = self._observe_one(option, event.name)
                candidate = (observed, hits + (event,)) if observed is not None else None
                if candidate is not None and candidate not in next_options:
                    next_options.append(candidate)
            options = next_options
        return options

    def _observe_one(self, progress: Progress, name: str) -> Progress | None:
        if name in progress:
            return progress
        candidate = progress + (name,)
        return candidate if self._prefix_ok(candidate) else None

    def _prefix_ok(self, progress: Progress) -> bool:
        seen = set(progress)
        for constraint in self.constraints:
            a, b = constraint.first, constraint.second
            if constraint.kind == "before":
                if b in seen and a not in seen:
                    return False
                if a in seen and b in seen and progress.index(a) >= progress.index(b):
                    return False
            elif constraint.kind == "next":
                if b in seen and a not in seen:
                    return False
                if a in seen:
                    ia = progress.index(a)
                    if ia + 1 < len(progress) and progress[ia + 1] != b:
                        return False
                    if b in seen and progress.index(b) != ia + 1:
                        return False
            elif constraint.kind == "between_only":
                if not self._between_prefix_ok(progress, a, b, set(constraint.names), only=True):
                    return False
            elif constraint.kind == "between_without":
                if not self._between_prefix_ok(progress, a, b, set(constraint.names), only=False):
                    return False
            else:
                raise ValueError(f"unsupported order constraint kind: {constraint.kind}")
        return True

    @staticmethod
    def _between_prefix_ok(progress: Progress, a: str, b: str, names: set[str], *, only: bool) -> bool:
        seen = set(progress)
        if b in seen and a not in seen:
            return False
        if a not in seen:
            # WITHOUT excludes names only inside the open a..b interval.
            # Seeing such a name before a is allowed (and may be required by
            # the rest of the task). ONLY names, by contrast, must occur inside.
            return not only or not any(name in seen for name in names)
        ia = progress.index(a)
        if only and any(name in seen and progress.index(name) < ia for name in names):
            return False
        end = progress.index(b) if b in seen else len(progress)
        if end <= ia:
            return False
        middle = progress[ia + 1:end]
        if only:
            return all(name in names for name in middle) and (b not in seen or names.issubset(middle))
        return all(name not in names for name in middle)
