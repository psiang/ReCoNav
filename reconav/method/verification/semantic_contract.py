"""Validate representation invariants of scene-independent instruction contracts."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from math import isfinite
from typing import Any, Literal

from pydantic import ValidationError

from reconav.method.semantic.contracts import (
    BetweenOnlyOrder,
    BetweenWithoutOrder,
    CarryConstraint,
    ConsecutiveOrder,
    CoreferenceReference,
    EventBoundary,
    EventReference,
    ExplicitRobotWidthConstraint,
    NominalReference,
    Qualifier,
    ReferenceRobotWidthConstraint,
    Referent,
    SceneIndependentSemanticContract,
    Scope,
    StrictModel,
)
from reconav.method.semantic.source_text import lexical_occurrences

NONBLOCKING_DIAGNOSTIC_CODES = frozenset({
    "SOURCE_TEXT_NOT_IN_INSTRUCTION",
    "HEAD_NOT_IN_REFERENT_SOURCE",
    "PREFERENCE_SOURCE_NOT_IN_INSTRUCTION",
    "QUALIFIER_NOT_IN_REFERENT_SOURCE",
    "QUALIFIER_ARGUMENT_NOT_IN_QUALIFIER_SOURCE",
    "TARGET_NOT_IN_OWNER_SOURCE",
    "PAYLOAD_NOT_IN_OWNER_SOURCE",
    "DUPLICATE_EVENT_OCCURRENCE",
})

DiagnosticCode = Literal[
    "SCHEMA_INVALID",
    "MISSING_REQUIRED_FIELD",
    "EXTRA_FIELD",
    "INVALID_ENUM",
    "INVALID_VALUE_TYPE",
    "INVALID_WIDTH_VALUE",
    "SOURCE_TEXT_NOT_IN_INSTRUCTION",
    "HEAD_NOT_IN_REFERENT_SOURCE",
    "PREFERENCE_SOURCE_NOT_IN_INSTRUCTION",
    "QUALIFIER_NOT_IN_REFERENT_SOURCE",
    "QUALIFIER_ARGUMENT_NOT_IN_QUALIFIER_SOURCE",
    "TARGET_NOT_IN_OWNER_SOURCE",
    "PAYLOAD_NOT_IN_OWNER_SOURCE",
    "EVENT_REFERENCE_TARGET_NOT_DECLARED",
    "AMBIGUOUS_EVENT_REFERENCE",
    "DUPLICATE_EVENT_OCCURRENCE",
    "NONPREFERENCE_MARKER_USED_AS_PREFERENCE",
    "COUNT_DERIVED_VIA",
    "NEAR_INDEFINITE_REFERENCE_NOT_EXISTENTIAL",
    "NEAR_DEFINITE_REFERENCE_NOT_UNIQUE",
    "INVALID_QUALIFIER_ARITY",
    "SELF_ORDER",
    "ORDER_CYCLE",
    "CONSECUTIVE_INTERVENING_EVENT",
    "INVALID_BETWEEN_BOUNDARY",
    "BETWEEN_ALLOWED_CONTAINS_BOUNDARY",
    "BETWEEN_FORBIDDEN_CONTAINS_BOUNDARY",
    "DUPLICATE_BETWEEN_ALLOWED_EVENT",
    "DUPLICATE_BETWEEN_FORBIDDEN_EVENT",
]


class AssuranceDiagnostic(StrictModel):
    code: DiagnosticCode
    path: str
    message: str
    determination: Literal["PROVED_INVALID", "UNKNOWN"]


class SourceOffset(StrictModel):
    path: str
    source_text: str
    start: int
    end: int


class DerivedItem(StrictModel):
    handle: str
    path: str
    family: str
    role: str
    source_text: str
    start: int | None = None
    end: int | None = None
    modality: Literal["HARD", "SOFT"]
    weight: int | None = None


class DerivedBookkeeping(StrictModel):
    events: list[DerivedItem]
    referents: list[DerivedItem]
    constraints: list[DerivedItem]
    source_offsets: list[SourceOffset]


class SemanticAssuranceResult(StrictModel):
    status: Literal["PROVED_VALID", "PROVED_INVALID", "UNKNOWN"]
    passed: bool
    accepted: bool
    diagnostics: list[AssuranceDiagnostic]
    derived: DerivedBookkeeping


_EMPTY_DERIVED = DerivedBookkeeping(
    events=[], referents=[], constraints=[], source_offsets=[]
)


def audit_semantic_contract(
    instruction: str,
    candidate: Any,
) -> SemanticAssuranceResult:
    """Check only fixed representation invariants; never infer missing semantics."""
    try:
        contract = (
            candidate
            if isinstance(candidate, SceneIndependentSemanticContract)
            else SceneIndependentSemanticContract.model_validate(candidate)
        )
    except ValidationError as exc:
        diagnostics = _schema_diagnostics(exc)
        return SemanticAssuranceResult(
            status="PROVED_INVALID",
            passed=False,
            accepted=False,
            diagnostics=diagnostics,
            derived=_EMPTY_DERIVED,
        )

    checker = _Checker(instruction, contract)
    checker.run()
    diagnostics = _deduplicate_diagnostics(checker.diagnostics)
    invalid = any(item.determination == "PROVED_INVALID" for item in diagnostics)
    unknown = any(item.determination == "UNKNOWN" for item in diagnostics)
    status = "PROVED_INVALID" if invalid else "UNKNOWN" if unknown else "PROVED_VALID"
    return SemanticAssuranceResult(
        status=status,
        passed=status == "PROVED_VALID",
        accepted=not invalid,
        diagnostics=diagnostics,
        derived=checker.derived(),
    )


class _Checker:
    def __init__(
        self,
        instruction: str,
        contract: SceneIndependentSemanticContract,
    ) -> None:
        self.instruction = instruction
        self.contract = contract
        self.diagnostics: list[AssuranceDiagnostic] = []
        self.offsets: list[SourceOffset] = []
        self.event_items: list[DerivedItem] = []
        self.referent_items: list[DerivedItem] = []
        self.constraint_items: list[DerivedItem] = []
        self._referent_index = 0
        self._constraint_index = 0
        events = [*contract.route.via_events, contract.route.goal]
        self.declared_event_counts = Counter(event.source_text for event in events)
        self.declared_event_sources = set(self.declared_event_counts)

    def run(self) -> None:
        self._check_events()
        self._check_orders()
        self._check_constraints()
        self._check_event_occurrence_uniqueness()
        self._check_count_via_separation()
        self._check_temporal_consistency()

    def derived(self) -> DerivedBookkeeping:
        return DerivedBookkeeping(
            events=self.event_items,
            referents=self.referent_items,
            constraints=self.constraint_items,
            source_offsets=sorted(self.offsets, key=lambda item: (item.start, item.path)),
        )

    def _check_events(self) -> None:
        events = [
            (f"$.route.via_events[{index}]", "VIA", event)
            for index, event in enumerate(self.contract.route.via_events)
        ]
        events.append(("$.route.goal", "GOAL", self.contract.route.goal))
        for index, (path, role, event) in enumerate(events):
            span = self._source(event.source_text, f"{path}.source_text")
            self._check_referent(
                event.target,
                f"{path}.target",
                owner_source=event.source_text,
                owner_span=span,
                containment_code="TARGET_NOT_IN_OWNER_SOURCE",
            )
            preference = getattr(event, "preference_source_text", None)
            self._preference(preference, f"{path}.preference_source_text")
            self.event_items.append(
                self._derived_item(
                    handle=f"e{index}",
                    path=path,
                    family=role,
                    role="route_event",
                    source_text=event.source_text,
                    span=span,
                    preferred=preference is not None,
                )
            )

    def _check_orders(self) -> None:
        families: list[tuple[str, str, list[Any]]] = [
            ("precedence", "PRECEDENCE", self.contract.orders.precedence),
            ("consecutive", "CONSECUTIVE", self.contract.orders.consecutive),
            ("between_only", "BETWEEN_ONLY", self.contract.orders.between_only),
            ("between_without", "BETWEEN_WITHOUT", self.contract.orders.between_without),
        ]
        for field, family, orders in families:
            for index, order in enumerate(orders):
                path = f"$.orders.{field}[{index}]"
                span = self._source(order.source_text, f"{path}.source_text")
                preferred = order.preference_source_text is not None
                self._preference(
                    order.preference_source_text,
                    f"{path}.preference_source_text",
                )
                if isinstance(order, ConsecutiveOrder):
                    self._check_boundary(order.first, f"{path}.first")
                    self._check_boundary(order.second, f"{path}.second")
                else:
                    first = order.start if isinstance(order, (BetweenOnlyOrder, BetweenWithoutOrder)) else order.earlier
                    second = order.end if isinstance(order, (BetweenOnlyOrder, BetweenWithoutOrder)) else order.later
                    first_name = "start" if isinstance(order, (BetweenOnlyOrder, BetweenWithoutOrder)) else "earlier"
                    second_name = "end" if isinstance(order, (BetweenOnlyOrder, BetweenWithoutOrder)) else "later"
                    self._check_boundary(first, f"{path}.{first_name}")
                    self._check_boundary(second, f"{path}.{second_name}")
                if isinstance(order, BetweenOnlyOrder):
                    self._check_between_members(
                        order.start,
                        order.end,
                        order.allowed,
                        f"{path}.allowed",
                        "allowed",
                    )
                elif isinstance(order, BetweenWithoutOrder):
                    self._check_between_members(
                        order.start,
                        order.end,
                        order.forbidden,
                        f"{path}.forbidden",
                        "forbidden",
                    )
                self._add_constraint_item(path, family, order.source_text, span, preferred)

    def _check_constraints(self) -> None:
        groups: list[tuple[str, str, list[Any]]] = [
            ("avoid", "AVOID", self.contract.constraints.avoid),
            ("count", "COUNT", self.contract.constraints.count),
            ("carry", "CARRY", self.contract.constraints.carry),
            ("robot_width", "WIDTH", self.contract.constraints.robot_width),
        ]
        for field, family, constraints in groups:
            for index, constraint in enumerate(constraints):
                path = f"$.constraints.{field}[{index}]"
                span = self._source(constraint.source_text, f"{path}.source_text")
                preferred = getattr(constraint, "preference_source_text", None) is not None
                self._preference(
                    getattr(constraint, "preference_source_text", None),
                    f"{path}.preference_source_text",
                )
                if isinstance(constraint, CarryConstraint):
                    self._check_referent(
                        constraint.payload,
                        f"{path}.payload",
                        constraint.source_text,
                        span,
                        "PAYLOAD_NOT_IN_OWNER_SOURCE",
                    )
                elif isinstance(constraint, ReferenceRobotWidthConstraint):
                    self._check_referent(
                        constraint.reference,
                        f"{path}.reference",
                        constraint.source_text,
                        span,
                        "TARGET_NOT_IN_OWNER_SOURCE",
                    )
                elif isinstance(constraint, ExplicitRobotWidthConstraint):
                    if not isfinite(constraint.value_m) or constraint.value_m <= 0:
                        self._invalid(
                            "INVALID_WIDTH_VALUE",
                            f"{path}.value_m",
                            "explicit robot width must be finite and greater than zero",
                        )
                else:
                    self._check_referent(
                        constraint.target,
                        f"{path}.target",
                        constraint.source_text,
                        span,
                        "TARGET_NOT_IN_OWNER_SOURCE",
                    )
                if constraint.scope is not None:
                    self._check_scope(constraint.scope, f"{path}.scope")
                self._add_constraint_item(path, family, constraint.source_text, span, preferred)

    def _check_referent(
        self,
        referent: Referent | NominalReference | CoreferenceReference,
        path: str,
        owner_source: str,
        owner_span: tuple[int, int] | None,
        containment_code: DiagnosticCode,
        check_owner_containment: bool = True,
    ) -> None:
        if check_owner_containment and not lexical_occurrences(
            owner_source, referent.source_text
        ):
            self._invalid(
                containment_code,
                f"{path}.source_text",
                "referent source is not contained in its owning semantic clause",
            )
        span = self._source(
            referent.source_text,
            f"{path}.source_text",
            parent_span=owner_span,
        )
        if referent.kind == "nominal":
            if not lexical_occurrences(referent.source_text, referent.head_text):
                self._invalid(
                    "HEAD_NOT_IN_REFERENT_SOURCE",
                    f"{path}.head_text",
                    "head_text is not contained in referent source_text",
                )
            else:
                self._source(
                    referent.head_text,
                    f"{path}.head_text",
                    parent_span=span,
                )
            for index, qualifier in enumerate(getattr(referent, "qualifiers", None) or []):
                self._check_qualifier(
                    qualifier,
                    f"{path}.qualifiers[{index}]",
                    referent.source_text,
                    span,
                )
        elif referent.kind == "coreference":
            self._check_referent(
                referent.antecedent,
                f"{path}.antecedent",
                self.instruction,
                None,
                "SOURCE_TEXT_NOT_IN_INSTRUCTION",
                check_owner_containment=False,
            )
        self.referent_items.append(
            self._derived_item(
                handle=f"r{self._referent_index}",
                path=path,
                family="REFERENT",
                role=_referent_role(referent),
                source_text=referent.source_text,
                span=span,
                preferred=False,
            )
        )
        self._referent_index += 1

    def _check_qualifier(
        self,
        qualifier: Qualifier,
        path: str,
        referent_source: str,
        referent_span: tuple[int, int] | None,
    ) -> None:
        if not lexical_occurrences(referent_source, qualifier.source_text):
            self._invalid(
                "QUALIFIER_NOT_IN_REFERENT_SOURCE",
                f"{path}.source_text",
                "qualifier source is not contained in its parent referent source",
            )
        span = self._source(
            qualifier.source_text,
            f"{path}.source_text",
            parent_span=referent_span,
        )
        expected = 2 if qualifier.relation == "between" else 1
        if len(qualifier.references) != expected:
            self._invalid(
                "INVALID_QUALIFIER_ARITY",
                f"{path}.references",
                f"{qualifier.relation} requires exactly {expected} reference(s)",
            )
        for index, reference in enumerate(qualifier.references):
            if not lexical_occurrences(qualifier.source_text, reference.source_text):
                self._invalid(
                    "QUALIFIER_ARGUMENT_NOT_IN_QUALIFIER_SOURCE",
                    f"{path}.references[{index}].source_text",
                    "qualifier argument source is not contained in qualifier source",
                )
            self._check_referent(
                reference,
                f"{path}.references[{index}]",
                qualifier.source_text,
                span,
                "QUALIFIER_ARGUMENT_NOT_IN_QUALIFIER_SOURCE",
            )
            if qualifier.relation == "near":
                current = reference
                while getattr(current, "kind", None) == "coreference":
                    current = current.antecedent
                if getattr(current, "kind", None) == "nominal":
                    words = str(current.source_text).strip().lower().split()
                    first = words[0].strip(".,;:!?()[]{}\"'") if words else ""
                    quantifier = str(current.quantifier)
                    if first in {"a", "an"} and quantifier != "exists":
                        self._invalid(
                            "NEAR_INDEFINITE_REFERENCE_NOT_EXISTENTIAL",
                            f"{path}.references[{index}].quantifier",
                            (
                                "ordinary indefinite NEAR reference must be existential "
                                f"for benchmark semantics: {current.source_text!r}"
                            ),
                        )
                    if (
                        first in {"the", "this", "that", "these", "those"}
                        and quantifier != "unique"
                    ):
                        self._invalid(
                            "NEAR_DEFINITE_REFERENCE_NOT_UNIQUE",
                            f"{path}.references[{index}].quantifier",
                            (
                                "definite/demonstrative NEAR reference must be unique "
                                f"for benchmark semantics: {current.source_text!r}"
                            ),
                        )

    def _check_scope(self, scope: Scope, path: str) -> None:
        self._source(scope.source_text, f"{path}.source_text")
        for field in ("after", "until"):
            boundary = getattr(scope, field)
            if boundary is not None:
                self._check_boundary(boundary, f"{path}.{field}")

    def _check_boundary(
        self,
        boundary: EventBoundary,
        path: str,
    ) -> None:
        span = self._source(boundary.source_text, f"{path}.source_text")
        if boundary.kind == "start":
            self.referent_items.append(
                self._derived_item(
                    handle=f"r{self._referent_index}",
                    path=path,
                    family="REFERENT",
                    role="start",
                    source_text=boundary.source_text,
                    span=span,
                    preferred=False,
                )
            )
            self._referent_index += 1
            return
        target = boundary.target_event_source_text
        count = self.declared_event_counts.get(target, 0)
        if count == 0:
            self._invalid(
                "EVENT_REFERENCE_TARGET_NOT_DECLARED",
                f"{path}.target_event_source_text",
                "event reference target does not exactly match a declared event source_text",
            )
        elif count > 1:
            self._invalid(
                "AMBIGUOUS_EVENT_REFERENCE",
                f"{path}.target_event_source_text",
                "event reference target matches more than one declared event",
            )

    def _check_between_members(
        self,
        start: EventBoundary,
        end: EventBoundary,
        members: list[EventReference],
        path: str,
        role: Literal["allowed", "forbidden"],
    ) -> None:
        start_key, end_key = _boundary_key(start), _boundary_key(end)
        if start_key == end_key:
            self._invalid(
                "INVALID_BETWEEN_BOUNDARY",
                path,
                "between start and end must identify distinct boundaries",
            )
        keys = [_boundary_key(member) for member in members]
        duplicate_code: DiagnosticCode = (
            "DUPLICATE_BETWEEN_ALLOWED_EVENT"
            if role == "allowed"
            else "DUPLICATE_BETWEEN_FORBIDDEN_EVENT"
        )
        boundary_code: DiagnosticCode = (
            "BETWEEN_ALLOWED_CONTAINS_BOUNDARY"
            if role == "allowed"
            else "BETWEEN_FORBIDDEN_CONTAINS_BOUNDARY"
        )
        if len(set(keys)) != len(keys):
            self._invalid(duplicate_code, path, f"between {role} events contain duplicates")
        for index, member in enumerate(members):
            self._check_boundary(member, f"{path}[{index}]")
            if _boundary_key(member) in {start_key, end_key}:
                self._invalid(
                    boundary_code,
                    f"{path}[{index}]",
                    f"between {role} events must not include either boundary",
                )

    def _check_event_occurrence_uniqueness(self) -> None:
        for source_text, declarations in self.declared_event_counts.items():
            occurrences = len(lexical_occurrences(self.instruction, source_text))
            if declarations > occurrences:
                self._invalid(
                    "DUPLICATE_EVENT_OCCURRENCE",
                    "$.route",
                    f"{declarations} events reuse only {occurrences} source occurrence(s): {source_text!r}",
                )

    @staticmethod
    def _referent_head_text(referent: Any) -> str | None:
        current = referent
        while getattr(current, "kind", None) == "coreference":
            current = current.antecedent
        if getattr(current, "kind", None) != "nominal":
            return None
        return " ".join(str(current.head_text).lower().split())

    def _check_count_via_separation(self) -> None:
        """Reject a VIA materialized only from the same aggregate COUNT phrase."""
        for via_index, event in enumerate(self.contract.route.via_events):
            via_head = self._referent_head_text(event.target)
            if via_head is None:
                continue
            for count_index, constraint in enumerate(self.contract.constraints.count):
                count_head = self._referent_head_text(constraint.target)
                if count_head != via_head:
                    continue
                if not lexical_occurrences(constraint.source_text, event.source_text):
                    continue
                self._invalid(
                    "COUNT_DERIVED_VIA",
                    f"$.route.via_events[{via_index}]",
                    (
                        "route VIA is contained in the same aggregate COUNT clause "
                        f"$.constraints.count[{count_index}] for head {via_head!r}; "
                        "COUNT encounters are anonymous unless independently individuated"
                    ),
                )

    def _check_temporal_consistency(self) -> None:
        hard_edges: set[tuple[str, str]] = set()
        consecutive: list[tuple[str, str, str]] = []

        for index, order in enumerate(self.contract.orders.precedence):
            self._edge(
                order.earlier,
                order.later,
                f"$.orders.precedence[{index}]",
                hard_edges if order.preference_source_text is None else None,
            )
        for index, order in enumerate(self.contract.orders.consecutive):
            path = f"$.orders.consecutive[{index}]"
            edge = self._edge(
                order.first,
                order.second,
                path,
                hard_edges if order.preference_source_text is None else None,
            )
            if edge is not None and order.preference_source_text is None:
                consecutive.append((*edge, path))
        for field, orders in (
            ("between_only", self.contract.orders.between_only),
            ("between_without", self.contract.orders.between_without),
        ):
            for index, order in enumerate(orders):
                self._edge(
                    order.start,
                    order.end,
                    f"$.orders.{field}[{index}]",
                    hard_edges if order.preference_source_text is None else None,
                )
        for path, constraint in self._scoped_constraints():
            scope = constraint.scope
            if scope is not None and scope.after is not None and scope.until is not None:
                self._edge(scope.after, scope.until, f"{path}.scope", hard_edges)

        if _has_cycle(hard_edges):
            self._invalid("ORDER_CYCLE", "$.orders", "hard semantic order theory contains a cycle")
        for first, second, path in consecutive:
            for event in self.declared_event_sources:
                if event in {first, second}:
                    continue
                if _reachable(first, event, hard_edges) and _reachable(event, second, hard_edges):
                    self._invalid(
                        "CONSECUTIVE_INTERVENING_EVENT",
                        path,
                        f"the contract proves that {event!r} intervenes between consecutive boundaries",
                    )
                    break

    def _edge(
        self,
        first: EventBoundary,
        second: EventBoundary,
        path: str,
        destination: set[tuple[str, str]] | None,
    ) -> tuple[str, str] | None:
        left, right = _boundary_key(first), _boundary_key(second)
        if left == right:
            self._invalid("SELF_ORDER", path, "order boundaries resolve to the same event")
            return None
        if not self._boundary_resolves(first) or not self._boundary_resolves(second):
            return None
        if destination is not None:
            destination.add((left, right))
        return left, right

    def _boundary_resolves(self, boundary: EventBoundary) -> bool:
        if boundary.kind == "start":
            return True
        return self.declared_event_counts.get(boundary.target_event_source_text, 0) == 1

    def _scoped_constraints(self) -> Iterable[tuple[str, Any]]:
        for field in ("avoid", "count", "carry", "robot_width"):
            for index, constraint in enumerate(getattr(self.contract.constraints, field)):
                yield f"$.constraints.{field}[{index}]", constraint

    def _source(
        self,
        source_text: str,
        path: str,
        parent_span: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
        occurrences = lexical_occurrences(self.instruction, source_text)
        if not occurrences:
            self._invalid(
                "SOURCE_TEXT_NOT_IN_INSTRUCTION",
                path,
                "exact source text does not occur as a lexical span in the instruction",
            )
            return None
        if len(occurrences) > 1:
            return None
        if parent_span is not None:
            contextual = [
                span
                for span in occurrences
                if parent_span[0] <= span[0] and span[1] <= parent_span[1]
            ]
            if not contextual:
                return None
            occurrences = contextual
        start, end = occurrences[0]
        self.offsets.append(
            SourceOffset(path=path, source_text=source_text, start=start, end=end)
        )
        return start, end

    def _preference(self, source_text: str | None, path: str) -> None:
        if source_text is None:
            return
        if not lexical_occurrences(self.instruction, source_text):
            self._invalid(
                "PREFERENCE_SOURCE_NOT_IN_INSTRUCTION",
                path,
                "preference source text does not occur in the instruction",
            )
            return
        normalized = " ".join(source_text.lower().strip().split())
        if normalized in {
            "directly",
            "straight",
            "straight to",
            "then",
            "next",
            "on your way",
        }:
            self._invalid(
                "NONPREFERENCE_MARKER_USED_AS_PREFERENCE",
                path,
                f"{source_text!r} is a hard movement/sequencing modifier, not soft preference evidence",
            )
        self._source(source_text, path)

    def _derived_item(
        self,
        *,
        handle: str,
        path: str,
        family: str,
        role: str,
        source_text: str,
        span: tuple[int, int] | None,
        preferred: bool,
    ) -> DerivedItem:
        return DerivedItem(
            handle=handle,
            path=path,
            family=family,
            role=role,
            source_text=source_text,
            start=span[0] if span is not None else None,
            end=span[1] if span is not None else None,
            modality="SOFT" if preferred else "HARD",
            weight=1 if preferred else None,
        )

    def _add_constraint_item(
        self,
        path: str,
        family: str,
        source_text: str,
        span: tuple[int, int] | None,
        preferred: bool,
    ) -> None:
        self.constraint_items.append(
            self._derived_item(
                handle=f"c{self._constraint_index}",
                path=path,
                family=family,
                role="constraint",
                source_text=source_text,
                span=span,
                preferred=preferred,
            )
        )
        self._constraint_index += 1

    def _invalid(self, code: DiagnosticCode, path: str, message: str) -> None:
        # Provenance/local-span anomalies are recorded but do not block any
        # method. Earlier experiments found no benefit from sending warning-only
        # cases through learned recovery; all methods therefore continue on the
        # same non-blocking policy. Structural/temporal contradictions remain
        # PROVED_INVALID.
        determination = (
            "UNKNOWN"
            if code in NONBLOCKING_DIAGNOSTIC_CODES
            else "PROVED_INVALID"
        )
        self.diagnostics.append(
            AssuranceDiagnostic(
                code=code,
                path=path,
                message=message,
                determination=determination,
            )
        )

def _schema_diagnostics(exc: ValidationError) -> list[AssuranceDiagnostic]:
    diagnostics = []
    for error in exc.errors():
        error_type = str(error.get("type", ""))
        location = tuple(error.get("loc", ()))
        if error_type == "missing":
            code: DiagnosticCode = "MISSING_REQUIRED_FIELD"
        elif error_type == "extra_forbidden":
            code = "EXTRA_FIELD"
        elif error_type == "literal_error":
            code = "INVALID_ENUM"
        elif "greater_than" in error_type or "finite" in error_type:
            code = "INVALID_WIDTH_VALUE" if "robot_width" in location else "SCHEMA_INVALID"
        elif error_type.endswith(("_type", "_parsing")):
            code = "INVALID_VALUE_TYPE"
        else:
            code = "SCHEMA_INVALID"
        diagnostics.append(
            AssuranceDiagnostic(
                code=code,
                path=_validation_path(location),
                message=str(error.get("msg", "schema validation failed")),
                determination="PROVED_INVALID",
            )
        )
    return _deduplicate_diagnostics(diagnostics)


def _validation_path(location: tuple[Any, ...]) -> str:
    path = "$"
    for item in location:
        path += f"[{item}]" if isinstance(item, int) else f".{item}"
    return path


def _boundary_key(boundary: EventBoundary) -> str:
    return "START" if boundary.kind == "start" else boundary.target_event_source_text


def _referent_role(referent: Any) -> str:
    return str(referent.kind)


def _reachable(first: str, second: str, edges: set[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
    queue, seen = deque([first]), {first}
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if nxt == second:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _has_cycle(edges: set[tuple[str, str]]) -> bool:
    return any(left == right or _reachable(right, left, edges) for left, right in edges)


def _deduplicate_diagnostics(
    diagnostics: list[AssuranceDiagnostic],
) -> list[AssuranceDiagnostic]:
    unique = {
        (item.code, item.path, item.message, item.determination): item
        for item in diagnostics
    }
    return [unique[key] for key in sorted(unique)]
