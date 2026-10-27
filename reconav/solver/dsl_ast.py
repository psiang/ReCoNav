"""Typed AST for the navigation DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TargetKind = Literal["obj", "region", "portal", "level", "pos"]
SelectorKind = Literal["id", "category", "any", "coord"]
MilestoneRole = Literal["start", "goal", "via"]
OrderKind = Literal["before", "next", "between_only", "between_without"]
AvoidKind = Literal["obj", "region", "portal", "level"]
QualifierKind = Literal[
    "in_region",
    "in_level",
    "between_regions",
    "near",
    "nearest_to",
    "farthest_from",
]
CountOp = Literal["at_least", "at_most", "exactly"]
SoftConstraintKind = Literal["avoid", "count", "via", "order"]


@dataclass(frozen=True)
class SourceLine:
    number: int
    text: str


@dataclass(frozen=True)
class TargetSelector:
    kind: TargetKind
    selector: SelectorKind
    values: tuple[int | str, ...]
    qualifiers: tuple[TargetQualifier, ...] = ()


@dataclass(frozen=True)
class TargetQualifier:
    kind: QualifierKind
    target: TargetSelector
    second_target: TargetSelector | None = None
    reference_mode: Literal["unique", "exists", "start"] = "unique"
    # NEAREST/FARTHEST compare the ordinary-filtered noun candidates on the
    # reference's level.  This is part of the distance operator, not a second
    # independent IN LEVEL qualifier.
    comparison_domain: Literal["reference_level"] | None = None

    def __post_init__(self) -> None:
        extreme = self.kind in {"nearest_to", "farthest_from"}
        if extreme and self.comparison_domain is None:
            object.__setattr__(self, "comparison_domain", "reference_level")
        elif not extreme and self.comparison_domain is not None:
            raise ValueError(
                f"{self.kind} does not accept comparison_domain"
            )


@dataclass(frozen=True)
class MilestoneSpec:
    name: str
    role: MilestoneRole
    target: TargetSelector
    source: SourceLine
    anonymous: bool = False


@dataclass(frozen=True)
class OrderConstraint:
    kind: OrderKind
    first: str
    second: str
    names: tuple[str, ...]
    source: SourceLine


@dataclass(frozen=True)
class AvoidSpec:
    kind: AvoidKind
    selector: SelectorKind
    values: tuple[int | str, ...]
    qualifiers: tuple[TargetQualifier, ...]
    after: str
    until: str
    source: SourceLine


@dataclass(frozen=True)
class CountSpec:
    target: TargetSelector
    op: CountOp
    count: int
    after: str
    until: str
    source: SourceLine


@dataclass(frozen=True)
class SoftViaSpec:
    name: str | None
    target: TargetSelector
    source: SourceLine


@dataclass(frozen=True)
class SoftAvoidSpec:
    target: TargetSelector
    after: str
    until: str
    source: SourceLine


@dataclass(frozen=True)
class SoftCountSpec:
    target: TargetSelector
    op: CountOp
    count: int
    after: str
    until: str
    source: SourceLine


@dataclass(frozen=True)
class SoftOrderSpec:
    constraint: OrderConstraint
    source: SourceLine


@dataclass
class DSLProgram:
    milestones: list[MilestoneSpec] = field(default_factory=list)
    order_constraints: list[OrderConstraint] = field(default_factory=list)
    avoid_specs: list[AvoidSpec] = field(default_factory=list)
    count_specs: list[CountSpec] = field(default_factory=list)
    soft_via_specs: list[SoftViaSpec] = field(default_factory=list)
    soft_avoid_specs: list[SoftAvoidSpec] = field(default_factory=list)
    soft_count_specs: list[SoftCountSpec] = field(default_factory=list)
    soft_order_specs: list[SoftOrderSpec] = field(default_factory=list)
    # Internal query context; never supplied by an LLM or changed at handoff.
    original_start_target: TargetSelector | None = None
    query_object_ids: tuple[int, ...] = ()

    def milestone_by_name(self) -> dict[str, MilestoneSpec]:
        return {m.name: m for m in self.milestones}

    def soft_event_by_name(self) -> dict[str, SoftViaSpec]:
        return {s.name: s for s in self.soft_via_specs if s.name is not None}

    def has_soft_constraints(self) -> bool:
        return bool(self.soft_via_specs or self.soft_avoid_specs or self.soft_count_specs or self.soft_order_specs)

    def has_route_structure_constraints(self) -> bool:
        return any(m.role == "via" for m in self.milestones) or bool(self.order_constraints)

    def start(self) -> MilestoneSpec | None:
        return next((m for m in self.milestones if m.role == "start"), None)

    def goal(self) -> MilestoneSpec | None:
        return next((m for m in self.milestones if m.role == "goal"), None)
