"""Gold-free projection and lowering into the repository's navigation DSL."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reconav.scene.asg import ASG
from reconav.solver.dsl_ast import DSLProgram
from reconav.solver.dsl_parser import parse_dsl
from reconav.method.grounding.grounded_contract import SceneGroundedContract
from reconav.method.grounding.label_grounding import collect_nominal_commitments
from reconav.method.grounding.scene_binding import binding_for_referent
from reconav.method.semantic.contracts import SceneIndependentSemanticContract


@dataclass(frozen=True)
class LoweringResult:
    specification: dict[str, Any]
    grounded_constraints: dict[str, Any]
    dsl: str
    parsed_program: DSLProgram
    carry_elaborations: tuple[CarryElaboration, ...]


@dataclass(frozen=True)
class CarryElaboration:
    carry_index: int
    acquisition_name: str
    acquisition_path: str
    until: str
    derived_acquisition: bool


@dataclass(frozen=True)
class EventNames:
    by_source: dict[str, tuple[str, ...]]
    route_names: tuple[str, ...]

    def resolve(self, boundary: Any) -> str:
        if boundary is None:
            raise ValueError("missing event boundary")
        if boundary.kind == "start":
            return "START"
        names = self.by_source.get(boundary.target_event_source_text, ())
        if len(names) != 1:
            raise ValueError(
                "event reference is missing or ambiguous: "
                f"{boundary.target_event_source_text!r} -> {list(names)}"
            )
        return names[0]


def lower_grounded_contract(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    robot_position: Mapping[str, Any],
    asg: ASG,
) -> LoweringResult:
    """Project predictions and emit concrete-ID DSL without benchmark gold."""
    if scene.semantic_contract != contract.model_dump(mode="json"):
        raise ValueError("scene overlay does not preserve the supplied semantic contract")
    if contract.constraints.robot_width:
        raise ValueError("robot-width constraints are not part of the current DSL")
    events = event_names(contract)
    carries = _carry_elaborations(contract, scene, events, require_complete=True)
    specification = canonical_specification(contract, scene, events, carries)
    grounded = grounded_constraints(contract, scene, events, asg, carries)
    dsl = emit_dsl(contract, scene, events, robot_position, asg, carries)
    program = parse_dsl(dsl)
    return LoweringResult(specification, grounded, dsl, program, carries)


def event_names(contract: SceneIndependentSemanticContract) -> EventNames:
    by_source: dict[str, list[str]] = {}
    route_names: list[str] = []
    hard_index = soft_index = 0
    for event in contract.route.via_events:
        if event.preference_source_text is None:
            hard_index += 1
            name = f"v{hard_index}"
        else:
            soft_index += 1
            name = f"p{soft_index}"
        route_names.append(name)
        by_source.setdefault(event.source_text, []).append(name)
    by_source.setdefault(contract.route.goal.source_text, []).append("GOAL")
    return EventNames(
        by_source={key: tuple(value) for key, value in by_source.items()},
        route_names=tuple(route_names),
    )


def canonical_specification(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract | None,
    events: EventNames | None = None,
    carries: tuple[CarryElaboration, ...] | None = None,
) -> dict[str, Any]:
    events = events or event_names(contract)
    carries = carries or _carry_elaborations(
        contract, scene, events, require_complete=False
    )
    specification: dict[str, Any] = {
        "goal": _semantic_target(contract, scene, contract.route.goal.target, "$.route.goal.target"),
        "via": [],
        "order": [],
        "avoid": [],
        "count": [],
        "carry": [],
        "prefer": [],
        "prefer_order": [],
    }
    for index, event in enumerate(contract.route.via_events):
        target = _semantic_target(
            contract, scene, event.target, f"$.route.via_events[{index}].target"
        )
        name = events.route_names[index]
        if event.preference_source_text is None:
            specification["via"].append({**target, "name": name})
        else:
            specification["prefer"].append(
                {**target, "preference": "via", "name": name}
            )
    _append_orders(specification, contract, events)
    _append_symbolic_constraints(specification, contract, scene, events)
    _append_carry_specification(specification, contract, scene, events, carries)
    return specification


def stage1_specification(contract: SceneIndependentSemanticContract) -> dict[str, Any]:
    """Preserve a blocked Stage-1 interpretation without inventing scene bindings."""
    return canonical_specification(contract, None)


def grounded_constraints(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    events: EventNames,
    asg: ASG,
    carries: tuple[CarryElaboration, ...] | None = None,
) -> dict[str, Any]:
    carries = carries or _carry_elaborations(
        contract, scene, events, require_complete=True
    )
    grounded: dict[str, Any] = {
        "goal": _grounded_target(contract, scene, "$.route.goal.target"),
        "via": [],
        "order": [],
        "avoid": [],
        "count": [],
        "carry": [],
        "prefer": [],
        "prefer_order": [],
    }
    for index, event in enumerate(contract.route.via_events):
        target = _grounded_target(contract, scene, f"$.route.via_events[{index}].target")
        name = events.route_names[index]
        if event.preference_source_text is None:
            grounded["via"].append({**target, "name": name})
        else:
            grounded["prefer"].append(
                {**target, "preference": "via", "name": name}
            )
    _append_orders(grounded, contract, events)
    for index, constraint in enumerate(contract.constraints.avoid):
        item = {
            **_grounded_target(contract, scene, f"$.constraints.avoid[{index}].target"),
            **_scope(constraint.scope, events),
        }
        if constraint.preference_source_text is None:
            grounded["avoid"].append(item)
        else:
            grounded["prefer"].append({**item, "preference": "avoid"})
    for index, constraint in enumerate(contract.constraints.count):
        item = {
            **_grounded_target(contract, scene, f"$.constraints.count[{index}].target"),
            "op": constraint.comparison,
            "count": constraint.value,
            **_scope(constraint.scope, events),
        }
        if constraint.preference_source_text is None:
            grounded["count"].append(item)
        else:
            grounded["prefer"].append({**item, "preference": "count"})
    for carry in carries:
        payload = _grounded_target(contract, scene, carry.acquisition_path)
        grounded["carry"].append({**payload, "until": carry.until})
        if carry.derived_acquisition:
            grounded["via"].append(
                {
                    **payload,
                    "name": carry.acquisition_name,
                    "derived_from": "carry",
                    "carry_index": carry.carry_index,
                }
            )
    return grounded


def emit_dsl(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    events: EventNames,
    robot_position: Mapping[str, Any],
    asg: ASG,
    carries: tuple[CarryElaboration, ...] | None = None,
) -> str:
    carries = carries or _carry_elaborations(
        contract, scene, events, require_complete=True
    )
    try:
        start = (
            float(robot_position["x"]),
            float(robot_position["floor_height"]),
            float(robot_position["y"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("task robot_position requires numeric x, floor_height, and y") from exc
    lines = ["START POS " + " ".join(_number(value) for value in start)]
    for index, event in enumerate(contract.route.via_events):
        if event.preference_source_text is not None:
            continue
        target = _grounded_target(contract, scene, f"$.route.via_events[{index}].target")
        lines.append(f"{events.route_names[index]}: VIA {_dsl_target(target)}")
    for carry in carries:
        if carry.derived_acquisition:
            target = _grounded_target(contract, scene, carry.acquisition_path)
            lines.append(f"{carry.acquisition_name}: VIA {_dsl_target(target)}")
    lines.append("GOAL " + _dsl_target(_grounded_target(contract, scene, "$.route.goal.target"), unique=True))

    for index, event in enumerate(contract.route.via_events):
        if event.preference_source_text is None:
            continue
        target = _grounded_target(contract, scene, f"$.route.via_events[{index}].target")
        lines.append(
            f"{events.route_names[index]}: PREFER VIA {_dsl_target(target)}"
        )
    lines.extend(_order_lines(contract, events))
    for index, constraint in enumerate(contract.constraints.avoid):
        target = _grounded_target(contract, scene, f"$.constraints.avoid[{index}].target")
        if not target["grounded_ids"]:
            continue
        scope = _scope(constraint.scope, events)
        prefix = "AVOID" if constraint.preference_source_text is None else "PREFER AVOID"
        lines.append(
            f"{prefix} {_dsl_target(target)} AFTER {scope['after']} UNTIL {scope['until']}"
        )
    for index, constraint in enumerate(contract.constraints.count):
        target = _grounded_target(contract, scene, f"$.constraints.count[{index}].target")
        if not target["grounded_ids"]:
            if constraint.preference_source_text is None and not _empty_count_satisfied(
                constraint.comparison, constraint.value
            ):
                raise ValueError(
                    "hard COUNT is violated by an empty deterministic candidate set"
                )
            continue
        scope = _scope(constraint.scope, events)
        prefix = "COUNT" if constraint.preference_source_text is None else "PREFER COUNT"
        op = {"at_least": "AT LEAST", "at_most": "AT MOST", "exactly": "EXACTLY"}[
            constraint.comparison
        ]
        lines.append(
            f"{prefix} {_dsl_target(target)} {op} {constraint.value} "
            f"AFTER {scope['after']} UNTIL {scope['until']}"
        )
    return "\n".join(lines) + "\n"


def _append_orders(
    output: dict[str, Any],
    contract: SceneIndependentSemanticContract,
    events: EventNames,
) -> None:
    families = (
        ("precedence", "before", "earlier", "later", None),
        ("consecutive", "next", "first", "second", None),
        ("between_only", "between_only", "start", "end", "allowed"),
        ("between_without", "between_without", "start", "end", "forbidden"),
    )
    for field, kind, first_key, second_key, names_key in families:
        for order in getattr(contract.orders, field):
            item = {
                "kind": kind,
                "first": events.resolve(getattr(order, first_key)),
                "second": events.resolve(getattr(order, second_key)),
                "names": [events.resolve(value) for value in getattr(order, names_key)]
                if names_key
                else [],
            }
            soft = order.preference_source_text is not None
            if not soft and kind == "before" and (
                item["first"] == "START" or item["second"] == "GOAL"
            ):
                continue
            if soft:
                output["prefer_order"].append(item)
            else:
                output["order"].append(item)


def _append_symbolic_constraints(
    specification: dict[str, Any],
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract | None,
    events: EventNames,
) -> None:
    for index, constraint in enumerate(contract.constraints.avoid):
        item = {
            **_semantic_target(
                contract, scene, constraint.target, f"$.constraints.avoid[{index}].target"
            ),
            **_scope(constraint.scope, events),
        }
        if constraint.preference_source_text is None:
            specification["avoid"].append(item)
        else:
            specification["prefer"].append({**item, "preference": "avoid"})
    for index, constraint in enumerate(contract.constraints.count):
        item = {
            **_semantic_target(
                contract, scene, constraint.target, f"$.constraints.count[{index}].target"
            ),
            "op": constraint.comparison,
            "count": constraint.value,
            **_scope(constraint.scope, events),
        }
        if constraint.preference_source_text is None:
            specification["count"].append(item)
        else:
            specification["prefer"].append({**item, "preference": "count"})


def _append_carry_specification(
    specification: dict[str, Any],
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract | None,
    events: EventNames,
    carries: tuple[CarryElaboration, ...],
) -> None:
    by_index = {item.carry_index: item for item in carries}
    for index, constraint in enumerate(contract.constraints.carry):
        elaboration = by_index.get(index)
        until = elaboration.until if elaboration is not None else _carry_until(
            constraint, events, require_complete=False
        )
        specification["carry"].append(
            {
                **_semantic_target(
                    contract,
                    scene,
                    constraint.payload,
                    f"$.constraints.carry[{index}].payload",
                ),
                "until": until,
            }
        )
        if elaboration is None or not elaboration.derived_acquisition:
            continue
        specification["via"].append(
            {
                **_semantic_target(
                    contract,
                    scene,
                    constraint.payload,
                    elaboration.acquisition_path,
                ),
                "name": elaboration.acquisition_name,
                "derived_from": "carry",
                "carry_index": index,
            }
        )


def _carry_elaborations(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract | None,
    events: EventNames,
    *,
    require_complete: bool,
) -> tuple[CarryElaboration, ...]:
    next_hard_index = 1 + sum(
        event.preference_source_text is None for event in contract.route.via_events
    )
    result: list[CarryElaboration] = []
    for index, constraint in enumerate(contract.constraints.carry):
        until = _carry_until(
            constraint, events, require_complete=require_complete
        )
        if until is None:
            continue
        if until != "GOAL" and not until.startswith("v"):
            raise ValueError("CARRY endpoint must resolve to a hard route event")

        payload_path = f"$.constraints.carry[{index}].payload"
        matching = [
            (via_index, events.route_names[via_index])
            for via_index, event in enumerate(contract.route.via_events)
            if event.preference_source_text is None
            and _same_referent_target(
                contract,
                scene,
                constraint.payload,
                payload_path,
                event.target,
                f"$.route.via_events[{via_index}].target",
            )
        ]
        after = None
        if constraint.scope is not None and constraint.scope.after is not None:
            resolved = events.resolve(constraint.scope.after)
            if resolved != "START":
                after = resolved
        if after is not None:
            selected = [item for item in matching if item[1] == after]
            if len(selected) != 1:
                raise ValueError(
                    "CARRY acquisition boundary does not match a hard VIA for its payload"
                )
        else:
            selected = []

        if selected:
            via_index, acquisition_name = selected[0]
            acquisition_path = f"$.route.via_events[{via_index}].target"
            derived = False
        else:
            acquisition_name = f"v{next_hard_index}"
            next_hard_index += 1
            acquisition_path = payload_path
            derived = True
        if acquisition_name == until:
            raise ValueError("CARRY acquisition and endpoint must be distinct events")
        result.append(
            CarryElaboration(
                carry_index=index,
                acquisition_name=acquisition_name,
                acquisition_path=acquisition_path,
                until=until,
                derived_acquisition=derived,
            )
        )
    return tuple(result)


def _carry_until(
    constraint: Any,
    events: EventNames,
    *,
    require_complete: bool,
) -> str | None:
    boundary = constraint.scope.until if constraint.scope is not None else None
    if boundary is None:
        if require_complete:
            raise ValueError("CARRY requires an explicit semantic endpoint")
        return None
    until = events.resolve(boundary)
    if until == "START":
        raise ValueError("CARRY endpoint cannot be navigation START")
    return until


def _same_referent_target(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract | None,
    left: Any,
    left_path: str,
    right: Any,
    right_path: str,
) -> bool:
    if scene is not None:
        return _grounded_target(contract, scene, left_path) == _grounded_target(
            contract, scene, right_path
        )
    return _json_key(_semantic_target(contract, None, left, left_path)) == _json_key(
        _semantic_target(contract, None, right, right_path)
    )


def _order_lines(contract: SceneIndependentSemanticContract, events: EventNames) -> list[str]:
    projected = {"order": [], "prefer_order": []}
    _append_orders(projected, contract, events)
    lines = [_order_line(item, soft=False) for item in projected["order"]]
    lines.extend(_order_line(item, soft=True) for item in projected["prefer_order"])
    return lines


def _order_line(item: Mapping[str, Any], *, soft: bool) -> str:
    if item["kind"] in {"before", "next"}:
        expression = f"{str(item['kind']).upper()}({item['first']}, {item['second']})"
    else:
        mode = "ONLY" if item["kind"] == "between_only" else "WITHOUT"
        names = ", ".join(item["names"])
        expression = f"BETWEEN({item['first']}, {item['second']}, {mode}({names}))"
    return f"PREFER ORDER {expression}" if soft else f"ORDER {expression}"


def _scope(scope: Any, events: EventNames) -> dict[str, str]:
    return {
        "after": events.resolve(scope.after) if scope is not None and scope.after is not None else "START",
        "until": events.resolve(scope.until) if scope is not None and scope.until is not None else "GOAL",
    }


def _semantic_target(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract | None,
    node: Any,
    path: str,
) -> dict[str, Any]:
    if node.kind == "coreference":
        return _semantic_target(contract, scene, node.antecedent, f"{path}.antecedent")
    if node.kind == "start":
        return {"kind": "start", "selector": "event", "values": ["START"], "qualifiers": []}
    if scene is None:
        commitments = {
            item.path: item for item in collect_nominal_commitments(contract).commitments
        }
        try:
            commitment = commitments[path]
        except KeyError as exc:
            raise ValueError(f"missing Stage-1 nominal occurrence at {path}") from exc
        semantic_kind = (
            next(iter(commitment.possible_kinds))
            if len(commitment.possible_kinds) == 1
            else None
        )
        kind = {
            "object": "obj",
            "region": "region",
            "portal": "portal",
            "level": "level",
            None: "unknown",
        }[semantic_kind]
        label = commitment.head_text
        selector = "category"
    else:
        binding = binding_for_referent(contract, scene, path)
        kind = {
            "object": "obj",
            "region": "region",
            "portal": "portal",
            "level": "level",
        }[binding.asg_kind]
        label = binding.asg_label
        selector = (
            "any"
            if binding.asg_label == "any"
            else "category"
        )
    target: dict[str, Any] = {
        "kind": kind,
        "selector": selector,
        "values": [] if selector == "any" else [label],
        "qualifiers": [],
    }
    for index, qualifier in enumerate(getattr(node, "qualifiers", None) or []):
        q_path = f"{path}.qualifiers[{index}]"
        references = [
            _semantic_target(contract, scene, ref, f"{q_path}.references[{ref_index}]")
            for ref_index, ref in enumerate(qualifier.references)
        ]
        if qualifier.relation == "between":
            target["qualifiers"].append({"kind": "between_regions", "regions": references})
        else:
            qualifier_kind = None
            if qualifier.relation == "in":
                qualifier_kind = f"in_{references[0]['kind']}"
            elif qualifier.relation in {"near", "nearest_to", "farthest_from"}:
                qualifier_kind = qualifier.relation
            item: dict[str, Any] = {
                "kind": qualifier_kind,
                "target": references[0],
            }
            if qualifier.relation in {"near", "nearest_to", "farthest_from"}:
                ref_node = qualifier.references[0]
                while ref_node.kind == "coreference":
                    ref_node = ref_node.antecedent
                item["reference_mode"] = "start" if ref_node.kind == "start" else getattr(ref_node, "quantifier", "unique")
                if qualifier.relation in {"nearest_to", "farthest_from"}:
                    item["comparison_domain"] = {
                        "kind": "reference_level"
                    }
            if qualifier_kind is None:
                raise ValueError(
                    f"unsupported qualifier in current DSL: {qualifier.relation}"
                )
            target["qualifiers"].append(item)
    return target


def _grounded_target(
    contract: SceneIndependentSemanticContract,
    scene: SceneGroundedContract,
    path: str,
) -> dict[str, Any]:
    binding = binding_for_referent(contract, scene, path)
    return {
        "kind": {"object": "obj", "region": "region", "portal": "portal", "level": "level"}[
            binding.asg_kind
        ],
        "grounded_ids": list(binding.grounded_ids),
    }


def _dsl_target(target: Mapping[str, Any], *, unique: bool = False) -> str:
    ids = [int(value) for value in target.get("grounded_ids", [])]
    if not ids:
        raise ValueError("executable target has no grounded IDs")
    if unique and len(ids) != 1:
        raise ValueError(f"executable target is ambiguous: candidates {ids}")
    return f"{str(target['kind']).upper()} " + " ".join(str(value) for value in ids)


def _empty_count_satisfied(comparison: str, value: int) -> bool:
    if comparison == "exactly":
        return value == 0
    if comparison == "at_most":
        return value >= 0
    if comparison == "at_least":
        return value <= 0
    raise ValueError(f"unsupported COUNT comparison: {comparison}")


def _number(value: float) -> str:
    return format(float(value), ".12g")


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
