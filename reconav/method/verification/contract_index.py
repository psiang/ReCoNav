"""Deterministic occurrence identity and reference graph for a canonical Contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reconav.method.semantic.contracts import SceneIndependentSemanticContract
from reconav.method.semantic.source_text import lexical_occurrences


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ContractNode(_StrictModel):
    handle: str
    kind: Literal["EVENT", "REFERENT", "CONSTRAINT"]
    path: str
    role: str
    source_text: str
    source_offsets: list[tuple[int, int]]


class ContractEdge(_StrictModel):
    kind: Literal["OWNS", "COREFERENCE", "EVENT_REFERENCE"]
    from_handle: str
    to_handle: str | None
    path: str
    target_source_text: str | None = None
    target_source_offsets: list[tuple[int, int]] = Field(default_factory=list)
    resolution: Literal["DIRECT", "UNIQUE_TEXT", "DANGLING", "AMBIGUOUS"]


class ContractIndex(_StrictModel):
    nodes: list[ContractNode]
    edges: list[ContractEdge]

    def handle_for_path(self, path: str) -> str | None:
        return next((item.handle for item in self.nodes if item.path == path), None)


def build_contract_index(
    instruction: str,
    contract: SceneIndependentSemanticContract,
) -> ContractIndex:
    """Assign stable R/E/C handles without treating source text as identity."""
    nodes: list[ContractNode] = []
    edges: list[ContractEdge] = []
    event_by_source: dict[str, list[str]] = defaultdict(list)

    def node(
        kind: Literal["EVENT", "REFERENT", "CONSTRAINT"],
        path: str,
        role: str,
        source_text: str,
    ) -> str:
        prefix = {"EVENT": "E", "REFERENT": "R", "CONSTRAINT": "C"}[kind]
        handle = f"{prefix}{sum(item.kind == kind for item in nodes)}"
        nodes.append(
            ContractNode(
                handle=handle,
                kind=kind,
                path=path,
                role=role,
                source_text=source_text,
                source_offsets=list(lexical_occurrences(instruction, source_text)),
            )
        )
        return handle

    route_events: list[tuple[str, str, Any]] = [
        (f"$.route.via_events[{index}]", "VIA", event)
        for index, event in enumerate(contract.route.via_events)
    ]
    route_events.append(("$.route.goal", "GOAL", contract.route.goal))
    event_handles: dict[str, str] = {}
    for path, role, event in route_events:
        handle = node("EVENT", path, role, event.source_text)
        event_handles[path] = handle
        event_by_source[event.source_text].append(handle)

    def referent(value: Any, path: str, owner: str, role: str) -> str:
        handle = node("REFERENT", path, role, value.source_text)
        edges.append(
            ContractEdge(
                kind="OWNS",
                from_handle=owner,
                to_handle=handle,
                path=path,
                resolution="DIRECT",
            )
        )
        if value.kind == "coreference":
            antecedent = referent(
                value.antecedent, f"{path}.antecedent", handle, "ANTECEDENT"
            )
            edges.append(
                ContractEdge(
                    kind="COREFERENCE",
                    from_handle=handle,
                    to_handle=antecedent,
                    path=f"{path}.antecedent",
                    resolution="DIRECT",
                )
            )
            return handle
        for q_index, qualifier in enumerate(getattr(value, "qualifiers", None) or []):
            for r_index, reference in enumerate(qualifier.references):
                ref_path = f"{path}.qualifiers[{q_index}].references[{r_index}]"
                referent(reference, ref_path, handle, "QUALIFIER_REFERENCE")
        return handle

    for path, _role, event in route_events:
        referent(event.target, f"{path}.target", event_handles[path], "EVENT_TARGET")

    constraint_groups = (
        ("precedence", contract.orders.precedence),
        ("consecutive", contract.orders.consecutive),
        ("between_only", contract.orders.between_only),
        ("between_without", contract.orders.between_without),
        ("avoid", contract.constraints.avoid),
        ("count", contract.constraints.count),
        ("carry", contract.constraints.carry),
        ("robot_width", contract.constraints.robot_width),
    )
    constraint_handles: list[tuple[str, Any, str]] = []
    for field, values in constraint_groups:
        section = (
            "orders"
            if field in {"precedence", "consecutive", "between_only", "between_without"}
            else "constraints"
        )
        for index, value in enumerate(values):
            path = f"$.{section}.{field}[{index}]"
            handle = node("CONSTRAINT", path, field.upper(), value.source_text)
            constraint_handles.append((path, value, handle))
            target = getattr(value, "target", None)
            if target is not None:
                referent(target, f"{path}.target", handle, "CONSTRAINT_TARGET")
            payload = getattr(value, "payload", None)
            if payload is not None:
                referent(payload, f"{path}.payload", handle, "CARRY_PAYLOAD")
            reference = getattr(value, "reference", None)
            if reference is not None:
                referent(reference, f"{path}.reference", handle, "WIDTH_REFERENCE")

    def boundary(value: Any, path: str, owner: str) -> None:
        if value is None or value.kind == "start":
            return
        candidates = event_by_source.get(value.target_event_source_text, [])
        resolution = (
            "UNIQUE_TEXT"
            if len(candidates) == 1
            else "DANGLING"
            if not candidates
            else "AMBIGUOUS"
        )
        edges.append(
            ContractEdge(
                kind="EVENT_REFERENCE",
                from_handle=owner,
                to_handle=candidates[0] if len(candidates) == 1 else None,
                path=path,
                target_source_text=value.target_event_source_text,
                target_source_offsets=list(
                    lexical_occurrences(instruction, value.target_event_source_text)
                ),
                resolution=resolution,
            )
        )

    for path, value, handle in constraint_handles:
        for field in ("earlier", "later", "first", "second", "start", "end"):
            if hasattr(value, field):
                boundary(getattr(value, field), f"{path}.{field}", handle)
        for field in ("allowed", "forbidden"):
            for index, item in enumerate(getattr(value, field, ())):
                boundary(item, f"{path}.{field}[{index}]", handle)
        scope = getattr(value, "scope", None)
        if scope is not None:
            boundary(scope.after, f"{path}.scope.after", handle)
            boundary(scope.until, f"{path}.scope.until", handle)
    return ContractIndex(nodes=nodes, edges=edges)
