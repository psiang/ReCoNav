"""Scene-independent instruction contract schema."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


SourceText = Annotated[str, Field(min_length=1)]
CountValue = Annotated[int, Field(ge=0)]
WidthMeters = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class NominalReference(StrictModel):
    kind: Literal["nominal"]
    quantifier: Literal["unique", "exists"]
    source_text: SourceText
    head_text: SourceText
    qualifiers: list["Qualifier"] | None = None


class StartReference(StrictModel):
    kind: Literal["start"]
    source_text: SourceText


ResolvedReference = Annotated[
    Union[NominalReference, StartReference],
    Field(discriminator="kind"),
]


class CoreferenceReference(StrictModel):
    kind: Literal["coreference"]
    source_text: SourceText
    antecedent: ResolvedReference


QualifierReference = Annotated[
    Union[NominalReference, CoreferenceReference, StartReference],
    Field(discriminator="kind"),
]


class Qualifier(StrictModel):
    source_text: SourceText
    relation: Literal["in", "near", "nearest_to", "farthest_from", "between"]
    references: list[QualifierReference]

    @model_validator(mode="after")
    def validate_reference_mode(self):
        for reference in self.references:
            value = reference.antecedent if reference.kind == "coreference" else reference
            if getattr(value, "quantifier", "unique") == "exists" and self.relation != "near":
                raise ValueError("existential class references require NEAR")
            if value.kind == "start" and self.relation not in {"nearest_to", "farthest_from"}:
                raise ValueError("START is a fixed distance reference for nearest/farthest only")
        return self


class NominalReferent(StrictModel):
    kind: Literal["nominal"]
    source_text: SourceText
    head_text: SourceText
    qualifiers: list[Qualifier] | None = None


ResolvedReferent = Annotated[
    Union[NominalReferent, StartReference],
    Field(discriminator="kind"),
]


class CoreferenceReferent(StrictModel):
    kind: Literal["coreference"]
    source_text: SourceText
    antecedent: ResolvedReferent


Referent = Annotated[
    Union[NominalReferent, CoreferenceReferent, StartReference],
    Field(discriminator="kind"),
]


class ViaEvent(StrictModel):
    source_text: SourceText
    target: Referent
    preference_source_text: SourceText | None = None


class GoalEvent(StrictModel):
    source_text: SourceText
    target: Referent


class RouteSemantics(StrictModel):
    via_events: list[ViaEvent]
    goal: GoalEvent


class EventReference(StrictModel):
    kind: Literal["event"]
    source_text: SourceText
    target_event_source_text: SourceText


EventBoundary = Annotated[
    Union[EventReference, StartReference],
    Field(discriminator="kind"),
]


class PrecedenceOrder(StrictModel):
    source_text: SourceText
    earlier: EventBoundary
    later: EventBoundary
    preference_source_text: SourceText | None = None


class ConsecutiveOrder(StrictModel):
    source_text: SourceText
    first: EventBoundary
    second: EventBoundary
    preference_source_text: SourceText | None = None


class BetweenOnlyOrder(StrictModel):
    source_text: SourceText
    start: EventBoundary
    end: EventBoundary
    allowed: list[EventReference]
    preference_source_text: SourceText | None = None


class BetweenWithoutOrder(StrictModel):
    source_text: SourceText
    start: EventBoundary
    end: EventBoundary
    forbidden: list[EventReference]
    preference_source_text: SourceText | None = None


class OrderSemantics(StrictModel):
    precedence: list[PrecedenceOrder]
    consecutive: list[ConsecutiveOrder]
    between_only: list[BetweenOnlyOrder]
    between_without: list[BetweenWithoutOrder]


class Scope(StrictModel):
    source_text: SourceText
    after: EventBoundary | None = None
    until: EventBoundary | None = None


class AvoidConstraint(StrictModel):
    source_text: SourceText
    target: Referent
    scope: Scope | None = None
    preference_source_text: SourceText | None = None


class CountConstraint(StrictModel):
    source_text: SourceText
    target: Referent
    comparison: Literal["at_least", "at_most", "exactly"]
    value: CountValue
    scope: Scope | None = None
    preference_source_text: SourceText | None = None


class CarryConstraint(StrictModel):
    source_text: SourceText
    payload: Referent
    scope: Scope | None = None


class ExplicitRobotWidthConstraint(StrictModel):
    source_text: SourceText
    value_m: WidthMeters
    scope: Scope | None = None


class ReferenceRobotWidthConstraint(StrictModel):
    source_text: SourceText
    reference: Referent
    scope: Scope | None = None


RobotWidthConstraint = Union[
    ExplicitRobotWidthConstraint,
    ReferenceRobotWidthConstraint,
]


class ConstraintSemantics(StrictModel):
    avoid: list[AvoidConstraint]
    count: list[CountConstraint]
    carry: list[CarryConstraint]
    robot_width: list[RobotWidthConstraint]


class SceneIndependentSemanticContract(StrictModel):
    """Navigation meaning before any scene realization or grounding."""

    route: RouteSemantics
    orders: OrderSemantics
    constraints: ConstraintSemantics


SceneIndependentSemanticContract.model_rebuild()
