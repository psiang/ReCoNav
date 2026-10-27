from __future__ import annotations

import copy
import itertools
import json
import math
from collections import Counter
from typing import Any

from reconav.benchmark.carry_projection import (
    acquisition_only_spec_names,
)
from reconav.benchmark.route_evaluator import (
    START_POSITION_TOLERANCE_M,
    evaluate_grounded_route,
    route_start_distance,
)
from reconav.benchmark.semantic_metrics import (
    grounding_id_score,
    semantic_field_scores,
    semantic_specification_score,
)
from reconav.scene.asg import ASG
from reconav.scene.geometry import Point2
from reconav.scene.physical import ENDPOINT_OBJECT_CLEARANCE_M, within_semantic_clearance
from reconav.solver.route_events import object_encounter_entries
from reconav.method.lowering.dsl import stage1_specification
from reconav.method.semantic.contracts import SceneIndependentSemanticContract
from reconav.scene.navmesh import (
    distance_to_polygons,
    point_in_region,
    point_to_segment_distance,
)

PAPER_METRICS = (
    "semantic_spec_f1",
    "g_id_em",
    "route_found",
    "hard_ns",
    "preference_satisfaction",
    "full_is",
)
COMPONENTS = (
    "goal",
    "via",
    "order",
    "avoid",
    "count",
    "carry",
    "width",
    "prefer",
    "prefer_order",
)
HARD_COMPONENTS = ("goal", "via_order", "avoid", "count", "width")


class RouteRuntimePreferenceMismatchError(RuntimeError):
    """Raised when route replay disagrees with comparable runtime evidence."""


def evaluate_prediction(
    task: dict[str, Any],
    prediction: dict[str, Any],
    asg: ASG,
) -> dict[str, Any]:
    # Score hard requirements and explicit preference satisfaction by replay.
    predicted = prediction.get("predicted") if isinstance(prediction.get("predicted"), dict) else {}
    specification = predicted.get("specification")
    gold_spec, semantic_gold_grounded = semantic_gold_artifacts(task)
    pred_spec, semantic_predicted_grounded = semantic_prediction_artifacts(
        prediction, asg
    )
    gold_ownership_spec, _ = semantic_gold_artifacts(task, sort_families=False)
    pred_ownership_spec, _ = semantic_prediction_artifacts(
        prediction, asg, sort_families=False
    )
    semantic_contract_score = None
    semantic_contract_error = None
    raw_semantic_contract = prediction.get("semantic_contract")
    if isinstance(raw_semantic_contract, dict):
        try:
            semantic_contract = SceneIndependentSemanticContract.model_validate(
                raw_semantic_contract
            )
            semantic_contract_spec = canonical_spec(
                stage1_specification(semantic_contract)
            )
            semantic_contract_score = semantic_specification_score(
                task,
                gold_spec,
                semantic_contract_spec,
                asg,
                gold_grounded=semantic_gold_grounded,
                predicted_grounded={},
            )
        except Exception as exc:
            semantic_contract_error = f"{type(exc).__name__}: {exc}"
    literal_spec_score = specification_score(gold_spec, pred_spec)
    semantic_spec_score = semantic_specification_score(
        task,
        gold_spec,
        pred_spec,
        asg,
        gold_grounded=semantic_gold_grounded,
        predicted_grounded=semantic_predicted_grounded,
    )
    native_spec = canonical_spec(specification) if isinstance(specification, dict) else None

    route = prediction.get("route") if isinstance(prediction.get("route"), dict) else {}
    gold_grounded = grounded_route_gold(task, route, asg)
    g_id_score = grounding_id_score(
        task,
        gold_spec,
        pred_spec,
        semantic_predicted_grounded,
        asg,
        semantically_complete=bool(semantic_spec_score["semantically_complete"]),
        gold_ownership_spec=gold_ownership_spec,
        predicted_ownership_spec=pred_ownership_spec,
        gold_grounded_override=semantic_gold_grounded,
    )

    route = prediction.get("route") if isinstance(prediction.get("route"), dict) else {}
    replay_route, route_errors = replayable_route(
        route,
        gold_grounded,
        asg,
        task.get("robot_position", {}),
    )
    route_found = native_route_found(prediction, route)
    replay_valid = route_found and not route_errors
    replay = evaluate_grounded_route(asg, gold_grounded, replay_route, task.get("robot_position", {}))
    implicit_replay_route = {
        **replay_route,
        "milestones": [
            item
            for item in replay_route.get("milestones", [])
            if item.get("role") != "prefer"
        ],
    }
    implicit_preference_cost = evaluate_grounded_route(
        asg,
        gold_grounded,
        implicit_replay_route,
        task.get("robot_position", {}),
    ).get("preference_cost")
    grouped = _constraint_groups(replay.get("constraints", []))
    hard_components = {
        "goal": replay_valid and grouped["goal"],
        "via_order": replay_valid and grouped["via"] and grouped["order"],
        "avoid": replay_valid and grouped["avoid"],
        "count": replay_valid and grouped["count"],
        "width": replay_valid and grouped["width"],
    }
    hard_ns = all(hard_components.values())

    preference_task = bool(gold_grounded["prefer"] or gold_grounded["prefer_order"])
    soft_cost = replay.get("preference_cost") if replay_valid else None
    preference_summary = _preference_clause_summary(
        replay.get("constraints", []),
        gold_grounded,
        replay_valid=replay_valid,
    )
    preference_satisfaction = preference_summary["satisfaction"]
    preference_violations = preference_summary["violations"]
    preference_success = (
        preference_violations == 0 if preference_task else True
    )
    full_is = hard_ns and bool(preference_success)
    runtime_cost = None
    runtime_match = None
    if (
        replay_valid
        and preference_task
        and _same_preference_semantics(
            predicted.get("grounded_constraints", {}), gold_grounded
        )
        and isinstance(route.get("optimal_soft_cost"), (int, float))
    ):
        runtime_cost = float(route["optimal_soft_cost"])
        runtime_match = (
            soft_cost is not None
            and abs(float(soft_cost) - runtime_cost) <= 1e-9
        )
        if not runtime_match:
            raise RouteRuntimePreferenceMismatchError(
                "ROUTE_RUNTIME_PREFERENCE_MISMATCH: evaluator cost "
                f"{soft_cost} != runtime cost {runtime_cost}"
            )
    execution_value = prediction.get("stage_verification", prediction.get("execution"))
    execution = execution_value if isinstance(execution_value, dict) else {}
    semantic_metrics_applicable = bool(
        (prediction.get("metric_applicability") or {}).get(
            "semantic_contract", True
        )
    )
    violations = [
        item for item in replay.get("constraints", [])
        if not item.get("satisfied")
        and item.get("component") not in {"prefer", "prefer_order", "preference_optimality"}
    ]
    commitment = (
        prediction.get("identity_commitment")
        if isinstance(prediction.get("identity_commitment"), dict)
        else {}
    )
    certified_commitment = commitment.get("certified") is True
    fallback_used = commitment.get("source") == "fallback"
    return {
        "semantic_spec_f1": (
            semantic_spec_score["f1"] if semantic_metrics_applicable else None
        ),
        "g_id_em": (
            g_id_score["exact_match"] if semantic_metrics_applicable else None
        ),
        "route_found": route_found,
        "hard_ns": hard_ns,
        "preference_satisfaction": preference_satisfaction,
        "preference_violations": preference_violations,
        "preference_satisfied": preference_summary["satisfied"],
        "preference_total": preference_summary["total"],
        "full_is": full_is,
        "certified_full_is": bool(full_is and certified_commitment),
        "certified_commitment": certified_commitment,
        "fallback_used": fallback_used,
        "semantic_contract_f1": (
            semantic_contract_score["f1"]
            if isinstance(semantic_contract_score, dict)
            else None
        ),
        "soft_cost": soft_cost,
        "runtime_soft_cost": runtime_cost,
        "runtime_evaluator_soft_cost": soft_cost if runtime_cost is not None else None,
        "runtime_preference_cost_match": runtime_match,
        "explicit_implicit_preference_cost_match": (
            soft_cost == implicit_preference_cost
        ),
        "hard_components": hard_components,
        "diagnostics": {
            "native_spec_em": native_spec == gold_spec if native_spec is not None else None,
            "semantic_specification": semantic_spec_score,
            "semantic_families": semantic_spec_score["families"],
            "semantic_fields": semantic_field_scores(gold_spec, pred_spec),
            "semantic_contract_only": semantic_contract_score,
            "semantic_contract_only_error": semantic_contract_error,
            "identity_commitment": commitment,
            "literal_specification": literal_spec_score,
            "literal_constraint_scores": constraint_scores(gold_spec, pred_spec),
            "preference_task": preference_task,
            "g_id": g_id_score,
            "route_replay_valid": replay_valid,
            "route_errors": route_errors,
            "violations": violations,
            "route_replay": replay.get("constraints", []),
            "implicit_optional_via_preference_cost": implicit_preference_cost,
            "json_valid": _bool_or_none(execution.get("json_valid")),
            "schema_valid": _bool_or_none(execution.get("schema_valid")),
            "dsl_valid": _bool_or_none(execution.get("dsl_valid")),
            "solver_success": _bool_or_none(execution.get("solver_success")),
            "failure_category": _failure_category(route_found, replay_valid, hard_components, preference_success),
            "failure": prediction.get("failure"),
        },
    }


def _preference_clause_summary(
    results: list[dict[str, Any]],
    grounded: dict[str, Any],
    *,
    replay_valid: bool,
) -> dict[str, int | float | None]:
    """Score explicit gold preferences directly on the executed route.

    Every explicit PREFER clause has unit metric weight. This metric does not
    ask whether the observed violation count is globally optimal.
    """
    total = len(grounded.get("prefer", [])) + len(
        grounded.get("prefer_order", [])
    )
    if total == 0:
        return {
            "total": 0,
            "satisfied": 0,
            "violations": None,
            "satisfaction": None,
        }
    if not replay_valid:
        return {
            "total": total,
            "satisfied": 0,
            "violations": total,
            "satisfaction": 0.0,
        }
    preference_results = [
        item
        for item in results
        if item.get("component") in {"prefer", "prefer_order"}
    ]
    if len(preference_results) != total:
        raise ValueError(
            "preference replay cardinality mismatch: "
            f"expected {total}, got {len(preference_results)}"
        )
    satisfied = sum(item.get("satisfied") is True for item in preference_results)
    violations = total - satisfied
    return {
        "total": total,
        "satisfied": satisfied,
        "violations": violations,
        "satisfaction": satisfied / total,
    }


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    incomplete = [item for item in predictions
                  if any(not isinstance(item.get("metrics", {}).get(key), (int, float))
                         for key in ("hard_ns", "full_is", "route_found"))
                  or item.get("evaluation_status", "evaluated") != "evaluated"
                  or item.get("infrastructure_errors")]
    summary: dict[str, Any] = {"total": len(predictions), "valid": not incomplete,
                               "incomplete_or_infrastructure_error": len(incomplete)}
    for key in PAPER_METRICS:
        values = [item.get("metrics", {}).get(key) for item in predictions]
        applicable = [value for value in values if isinstance(value, (int, float))]
        total = sum(float(value) for value in applicable)
        summary[key] = {
            "sum": total,
            "applicable": len(applicable),
            "rate": total / len(applicable) if applicable and not incomplete else None,
        }
    family_names = ("goal", "visit", "order", "avoid", "count", "width", "preference")
    field_names = ("target", "operator_value", "scope", "modality")
    summary["preference_violations"] = _metric_summary([
        item.get("metrics", {}).get("preference_violations")
        for item in predictions
    ])
    summary["diagnostics"] = {
        "semantic_families": {
            name: _diagnostic_f1_summary(predictions, "semantic_families", name)
            for name in family_names
        },
        "semantic_fields": {
            name: _diagnostic_f1_summary(predictions, "semantic_fields", name)
            for name in field_names
        },
        "literal_spec_f1": _metric_summary([
            _diagnostics(item).get("literal_specification", {}).get("f1")
            for item in predictions
        ]),
        "semantic_contract_f1": _metric_summary([
            item.get("metrics", {}).get("semantic_contract_f1")
            for item in predictions
        ]),
        "certified_full_is": _metric_summary([
            item.get("metrics", {}).get("certified_full_is")
            for item in predictions
        ]),
        "fallback_rate": _metric_summary([
            item.get("metrics", {}).get("fallback_used")
            for item in predictions
        ]),
        "g_id": _g_id_summary(predictions),
    }
    return summary


def _diagnostic_f1_summary(
    predictions: list[dict[str, Any]], group: str, name: str
) -> dict[str, float | int | None]:
    values = []
    for prediction in predictions:
        score = _diagnostics(prediction).get(group, {}).get(name, {})
        if int(score.get("gold", 0)) or int(score.get("predicted", 0)):
            values.append(score.get("f1"))
    return _metric_summary(values)


def _g_id_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_diagnostics(item).get("g_id", {}) for item in predictions]
    required = sum(int(item.get("required_slots", 0)) for item in scores)
    aligned = sum(int(item.get("aligned_slots", 0)) for item in scores)
    correct = sum(int(item.get("correct_slots", 0)) for item in scores)
    operational = sum(len(item.get("operationally_supported_slots", [])) for item in scores)
    return {
        "required_slots": required,
        "aligned_slots": aligned,
        "correct_slots": correct,
        "direct_correct_slots": correct - operational,
        "operationally_supported_slots": operational,
        "alignment_coverage": aligned / required if required else 1.0,
        "aligned_id_accuracy": correct / aligned if aligned else 0.0,
        "conditional_exact_match": _metric_summary([
            item.get("conditional_exact_match") for item in scores
        ]),
    }


def _diagnostics(prediction: dict[str, Any]) -> dict[str, Any]:
    metrics = prediction.get("metrics") if isinstance(prediction.get("metrics"), dict) else {}
    value = metrics.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _metric_summary(values: list[Any]) -> dict[str, float | int | None]:
    applicable = [float(value) for value in values if isinstance(value, (int, float))]
    total = sum(applicable)
    return {
        "sum": total,
        "applicable": len(applicable),
        "rate": total / len(applicable) if applicable else None,
    }


def canonical_spec(value: Any, *, sort_families: bool = True) -> dict[str, Any]:
    normalized = _normalize_contract(value)
    normalized = _rename_events(normalized)
    for target in [
        normalized["goal"],
        *normalized["via"],
        *normalized["avoid"],
        *normalized["count"],
        *normalized["carry"],
        *normalized["prefer"],
    ]:
        _clean_target(target)
    for item in normalized["prefer"]:
        item["weight"] = 1.0
        if item.get("preference") == "via":
            item.pop("after", None)
            item.pop("until", None)
    for item in normalized["order"]:
        item.setdefault("names", [])
    for item in normalized["prefer_order"]:
        item["weight"] = 1.0
        item.setdefault("names", [])
    if sort_families:
        for family in (
            "order",
            "avoid",
            "count",
            "carry",
            "width",
            "prefer",
            "prefer_order",
        ):
            normalized[family] = sorted(normalized[family], key=_json_key)
    return normalized


def adapted_specification(
    prediction: dict[str, Any], asg: ASG, *, sort_families: bool = True
) -> dict[str, Any]:
    predicted = prediction.get("predicted") if isinstance(prediction.get("predicted"), dict) else {}
    native = predicted.get("specification")
    if isinstance(native, dict):
        return canonical_spec(native, sort_families=sort_families)

    native_program = _referent_program_specification(prediction.get("native_output"))
    if native_program is not None:
        return canonical_spec(native_program, sort_families=sort_families)

    grounded = predicted.get("grounded_constraints") or predicted.get("constraints")
    source = grounded if isinstance(grounded, dict) else {}
    vias = _items(source, "via")
    abstract: dict[str, Any] = {
        "goal": _abstract_target(source.get("goal"), asg),
        "via": [
            {**_abstract_target(item, asg), "name": f"v{index}"}
            for index, item in enumerate(vias, start=1)
        ],
        "order": [_abstract_order(item) for item in _items(source, "order") if item.get("kind") != "chain"],
        "avoid": [_abstract_scoped(item, asg) for item in _items(source, "avoid")],
        "count": [_abstract_count(item, asg) for item in _items(source, "count")],
        "width": [_abstract_width(item) for item in _items(source, "width")],
        "prefer": [_abstract_preference(item, asg, index) for index, item in enumerate(_items(source, "prefer"), start=1)],
        "prefer_order": [_abstract_order(item) | {"weight": 1.0} for item in _items(source, "prefer_order")],
    }
    if any(item.get("kind") == "chain" for item in _items(source, "order")):
        abstract["order"] = [
            {"kind": "before", "first": f"v{index}", "second": f"v{index + 1}", "names": []}
            for index in range(1, len(vias))
        ]
    return canonical_spec(abstract, sort_families=sort_families)


def semantic_gold_artifacts(
    task: dict[str, Any], *, sort_families: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the one supported current benchmark schema, failing if incomplete."""

    specification, grounded = _current_task_artifacts(task)
    return canonical_spec(
        specification,
        sort_families=sort_families,
    ), grounded


def semantic_prediction_artifacts(
    prediction: dict[str, Any],
    asg: ASG,
    *,
    sort_families: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove only deterministically marked CARRY execution artifacts."""
    predicted = (
        prediction.get("predicted")
        if isinstance(prediction.get("predicted"), dict)
        else {}
    )
    native = predicted.get("specification")
    grounded_source = predicted.get("grounded_constraints") or predicted.get(
        "constraints"
    )
    grounded = (
        copy.deepcopy(grounded_source) if isinstance(grounded_source, dict) else {}
    )
    if not isinstance(native, dict):
        return (
            adapted_specification(prediction, asg, sort_families=sort_families),
            grounded,
        )

    specification = copy.deepcopy(native)
    acquisition_only = acquisition_only_spec_names(specification)
    semantic_vias = _items(specification, "via")
    via_keep = [
        item.get("derived_from") != "carry"
        and str(item.get("name", "")) not in acquisition_only
        for item in semantic_vias
    ]
    specification["via"] = [
        item for item, include in zip(semantic_vias, via_keep) if include
    ]
    grounded_vias = _items(grounded, "via")
    grounded["via"] = [
        item
        for index, item in enumerate(grounded_vias)
        if index >= len(via_keep) or via_keep[index]
    ]
    for family in ("width",):
        semantic_items = _items(specification, family)
        keep = [item.get("derived_from") != "carry" for item in semantic_items]
        specification[family] = [
            item for item, include in zip(semantic_items, keep) if include
        ]
        grounded_items = _items(grounded, family)
        grounded[family] = [
            item
            for index, item in enumerate(grounded_items)
            if index >= len(keep) or keep[index]
        ]
    return canonical_spec(specification, sort_families=sort_families), grounded


def _referent_program_specification(value: Any) -> dict[str, Any] | None:
    """Recover S from a validated SRER/program artifact even when G or E failed."""
    native = value if isinstance(value, dict) else {}
    lifted = native.get("referent_lifting", native.get("srer"))
    weighted_program = native.get("weighted_temporal_program", native.get("program"))
    srer = lifted if isinstance(lifted, dict) else None
    program = weighted_program if isinstance(weighted_program, dict) else None
    if srer is None or program is None:
        return None
    atoms = {
        str(item.get("symbol")): item
        for item in srer.get("atoms", [])
        if isinstance(item, dict) and isinstance(item.get("target"), dict)
    }
    goal_symbol = str(program.get("goal", ""))
    visits = [str(item) for item in program.get("visits", [])]
    if goal_symbol not in atoms or any(symbol not in atoms for symbol in visits):
        return None
    via_symbols = [symbol for symbol in visits if symbol != goal_symbol]
    event_names = {symbol: f"v{index}" for index, symbol in enumerate(via_symbols, start=1)}
    event_names[goal_symbol] = "GOAL"
    preference_visits = [
        item for item in program.get("rules", [])
        if isinstance(item, dict) and item.get("type") == "prefer_visit"
    ]
    for index, item in enumerate(preference_visits, start=1):
        event_names[str(item.get("target"))] = f"p{index}"

    specification = {
        "goal": copy.deepcopy(atoms[goal_symbol]["target"]),
        "via": [
            {**copy.deepcopy(atoms[symbol]["target"]), "name": f"v{index}"}
            for index, symbol in enumerate(via_symbols, start=1)
        ],
        "order": [],
        "avoid": [],
        "count": [],
        "width": [],
        "prefer": [],
        "prefer_order": [],
    }
    preference_index = 0
    try:
        for rule in program.get("rules", []):
            if not isinstance(rule, dict):
                return None
            kind = str(rule.get("type", ""))
            if kind in {"before", "next", "between_only", "between_without"}:
                item = {
                    "kind": kind,
                    "first": _program_event(rule["first"], event_names),
                    "second": _program_event(rule["second"], event_names),
                    "names": [_program_event(name, event_names) for name in rule.get("events", [])],
                }
                if item["first"] != "START" and item["second"] != "GOAL":
                    specification["order"].append(item)
            elif kind in {"avoid", "count"}:
                target = str(rule["target"])
                item = {
                    **copy.deepcopy(atoms[target]["target"]),
                    "after": _program_event(rule["after"], event_names),
                    "until": _program_event(rule["until"], event_names),
                }
                if kind == "count":
                    item.update(op=rule["op"], count=rule["count"])
                specification[kind].append(item)
            elif kind == "width_numeric":
                specification["width"].append({
                    "kind": "numeric",
                    "width_m": rule["width_m"],
                    "after": _program_event(rule["after"], event_names),
                    "until": _program_event(rule["until"], event_names),
                })
            elif kind == "width_from_visit":
                specification["width"].append({
                    "kind": "from_milestone",
                    "milestone": _program_event(rule["source"], event_names),
                    "after": _program_event(rule["after"], event_names),
                    "until": _program_event(rule["until"], event_names),
                })
            elif kind == "prefer_visit":
                preference_index += 1
                specification["prefer"].append({
                    **copy.deepcopy(atoms[str(rule["target"])]["target"]),
                    "preference": "via",
                    "weight": 1.0,
                    "name": f"p{preference_index}",
                })
            elif kind in {"prefer_avoid", "prefer_count"}:
                preference = "avoid" if kind == "prefer_avoid" else "count"
                item = {
                    **copy.deepcopy(atoms[str(rule["target"])]["target"]),
                    "preference": preference,
                    "weight": 1.0,
                    "after": _program_event(rule["after"], event_names),
                    "until": _program_event(rule["until"], event_names),
                }
                if preference == "count":
                    item.update(op=rule["op"], count=rule["count"])
                specification["prefer"].append(item)
            elif kind == "prefer_order":
                specification["prefer_order"].append({
                    "kind": "before",
                    "first": _program_event(rule["first"], event_names),
                    "second": _program_event(rule["second"], event_names),
                    "names": [],
                    "weight": 1.0,
                })
    except (KeyError, TypeError, ValueError):
        return None
    return specification


def _program_event(value: Any, events: dict[str, str]) -> str:
    text = str(value)
    if text.upper() in {"START", "GOAL"}:
        return text.upper()
    if text not in events:
        raise ValueError(f"Unknown program event: {text}")
    return events[text]


def specification_score(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float | int]:
    gold_items = _spec_items(gold)
    predicted_items = _spec_items(predicted)
    true_positive = sum((gold_items & predicted_items).values())
    predicted_count = sum(predicted_items.values())
    gold_count = sum(gold_items.values())
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    return {
        "tp": true_positive,
        "fp": predicted_count - true_positive,
        "fn": gold_count - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def native_route_found(prediction: dict[str, Any], route: dict[str, Any]) -> bool:
    waypoints = route.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        return False
    for point in waypoints:
        if not isinstance(point, dict):
            return False
        try:
            coordinates = (float(point["x"]), float(point.get("y", 0.0)), float(point["z"]))
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in coordinates):
            return False
    return True


def grounded_gold(task: dict[str, Any]) -> dict[str, Any]:
    _specification, grounded = _current_task_artifacts(task)
    return grounded


def grounded_route_gold(task: dict[str, Any], route: dict[str, Any], asg: ASG) -> dict[str, Any]:
    """Replay locally qualified identities without making a soft winner hard.

    Each alternative is replayed with a single concrete identity, so event
    identities and AFTER/UNTIL scopes are reconstructed consistently. A route
    satisfies an ambiguous instruction when a legal identity assignment meets
    its hard requirements; preferences then score that assignment. Annotation
    selected IDs remain available to the separate identity diagnostic.
    """
    base = grounded_gold(task)
    proof = task.get("uniqueness_proof", {})
    ids = proof.get("qualified_candidate_ids")
    target = proof.get("identity_target", {})
    if not isinstance(ids, list) or len(ids) < 2 or not route.get("waypoints"):
        return base
    if target.get("role") not in {"goal", "via"}:
        raise ValueError("ambiguous benchmark identity target must be GOAL or VIA")
    best, best_rank = base, None
    for entity_id in sorted(set(map(int, ids))):
        gold = copy.deepcopy(base)
        if target["role"] == "goal":
            binding = gold["goal"]
        else:
            matches = [item for item in gold["via"] if item.get("name") == target.get("name")]
            matches += [item for item in gold["prefer"] if item.get("preference") == "via" and item.get("name") == target.get("name")]
            if len(matches) != 1:
                raise ValueError("ambiguous benchmark VIA must identify exactly one event")
            binding = matches[0]
        binding["grounded_ids"] = [entity_id]
        replay, errors = replayable_route(route, gold, asg, task.get("robot_position", {}))
        check = evaluate_grounded_route(asg, gold, replay, task.get("robot_position", {}))
        hard = not errors and all(item.get("satisfied") for item in check.get("constraints", [])
                                  if item.get("component") not in {"prefer", "prefer_order", "preference_optimality"})
        cost = check.get("preference_cost")
        rank = (hard, -float(cost) if isinstance(cost, (int, float)) else -math.inf)
        if best_rank is None or rank > best_rank:
            best, best_rank = gold, rank
    return best


def _current_task_artifacts(
    task: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert task annotations into evaluator reference artifacts."""

    symbolic = task.get("symbolic_ir")
    if not isinstance(symbolic, dict):
        raise ValueError("current benchmark task omits symbolic_ir")
    goal = symbolic.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("current benchmark task omits symbolic_ir.goal")
    for key in ("kind", "head", "selected_id", "qualifiers"):
        if key not in goal:
            raise ValueError(f"current benchmark goal omits {key}")

    specification: dict[str, Any] = {
        "goal": _current_symbolic_target(goal),
        "via": [],
        "order": copy.deepcopy(symbolic.get("orders", [])),
        "avoid": [],
        "count": [],
        "prefer": [],
        "prefer_order": [],
    }
    grounded: dict[str, Any] = {
        "goal": {
            "kind": str(goal["kind"]),
            "grounded_ids": [int(goal["selected_id"])],
        },
        "via": [],
        "order": copy.deepcopy(symbolic.get("orders", [])),
        "avoid": [],
        "count": [],
        "prefer": [],
        "prefer_order": [],
    }
    vias = symbolic.get("vias")
    constraints = symbolic.get("constraints")
    orders = symbolic.get("orders")
    if not isinstance(vias, list) or not isinstance(constraints, list) or not isinstance(orders, list):
        raise ValueError("current benchmark symbolic_ir arrays are missing")
    for index, via in enumerate(vias, start=1):
        if not isinstance(via, dict) or not {"kind", "id", "name"} <= set(via):
            raise ValueError(f"current benchmark VIA {index} is incomplete")
        target = {
            "kind": str(via["kind"]),
            "selector": "category",
            "values": [str(via.get("label", via.get("phrase", "")))],
            "qualifiers": _current_qualifiers(via.get("qualifiers", [])),
            "name": str(via["name"]),
        }
        specification["via"].append(target)
        grounded["via"].append(
            {
                "kind": str(via["kind"]),
                "grounded_ids": [int(via["id"])],
                "name": str(via["name"]),
            }
        )
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            raise ValueError(f"current benchmark constraint {index} is not an object")
        required = {"kind", "entity_kind", "entity_ids", "scope"}
        if not required <= set(constraint):
            raise ValueError(
                f"current benchmark constraint {index} omits "
                f"{sorted(required - set(constraint))}"
            )
        kind = str(constraint["kind"])
        scope = constraint["scope"]
        if not isinstance(scope, dict) or not {"after", "until"} <= set(scope):
            raise ValueError(f"current benchmark constraint {index} has no complete scope")
        target = {
            "kind": str(constraint["entity_kind"]),
            "selector": "category",
            "values": [
                str(constraint.get("head", constraint.get("phrase", "")))
            ],
            "qualifiers": _current_qualifiers(constraint.get("qualifiers", [])),
        }
        binding = {
            "kind": str(constraint["entity_kind"]),
            "grounded_ids": [int(value) for value in constraint["entity_ids"]],
        }
        scoped = {
            **target,
            "after": str(scope["after"]),
            "until": str(scope["until"]),
        }
        grounded_scoped = {
            **binding,
            "after": str(scope["after"]),
            "until": str(scope["until"]),
        }
        if kind == "avoid":
            specification["avoid"].append(scoped)
            grounded["avoid"].append(grounded_scoped)
        elif kind == "count":
            count_fields = {
                "op": str(constraint["op"]),
                "count": int(constraint["count"]),
            }
            specification["count"].append({**scoped, **count_fields})
            grounded["count"].append({**grounded_scoped, **count_fields})
        elif kind == "prefer_avoid":
            specification["prefer"].append(
                {**scoped, "preference": "avoid", "weight": 1.0}
            )
            grounded["prefer"].append(
                {**grounded_scoped, "preference": "avoid", "weight": 1.0}
            )
        elif kind == "prefer_count":
            count_fields = {
                "op": str(constraint["op"]),
                "count": int(constraint["count"]),
            }
            specification["prefer"].append(
                {
                    **scoped,
                    **count_fields,
                    "preference": "count",
                    "weight": 1.0,
                }
            )
            grounded["prefer"].append(
                {
                    **grounded_scoped,
                    **count_fields,
                    "preference": "count",
                    "weight": 1.0,
                }
            )
        elif kind == "prefer_via":
            name = str(constraint.get("name") or f"p{index + 1}")
            specification["prefer"].append(
                {**target, "preference": "via", "weight": 1.0, "name": name}
            )
            grounded["prefer"].append(
                {**binding, "preference": "via", "weight": 1.0, "name": name}
            )
        elif kind == "prefer_order":
            order = constraint.get("order")
            if not isinstance(order, dict):
                raise ValueError(f"current benchmark PREFER ORDER {index} omits order")
            specification["prefer_order"].append({**copy.deepcopy(order), "weight": 1.0})
            grounded["prefer_order"].append({**copy.deepcopy(order), "weight": 1.0})
        else:
            raise ValueError(f"unsupported current benchmark constraint kind {kind!r}")
    return specification, grounded


def _current_symbolic_target(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(item["kind"]),
        "selector": "category",
        "values": [str(item["head"])],
        "qualifiers": _current_qualifiers(item.get("qualifiers", [])),
    }


def _current_qualifiers(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError("current benchmark qualifier field is not a list")
    result: list[dict[str, Any]] = []
    for qualifier in values:
        if not isinstance(qualifier, dict) or "kind" not in qualifier:
            raise ValueError("current benchmark qualifier is incomplete")
        kind = str(qualifier["kind"])
        if kind == "in_region":
            target = {
                "kind": "region",
                "selector": "category",
                "values": [str(qualifier.get("label", ""))],
                "qualifiers": [],
            }
        elif kind == "in_level":
            target = {
                "kind": "level",
                "selector": "id",
                "values": [str(qualifier["id"])],
                "qualifiers": [],
            }
        elif kind in {"near", "nearest_to", "farthest_from"}:
            reference = qualifier.get("reference")
            if not isinstance(reference, dict):
                raise ValueError(f"{kind} qualifier omits reference")
            target = {
                "kind": str(reference["kind"]),
                "selector": "category",
                "values": [str(reference.get("head", reference.get("label", "")))],
                "qualifiers": _current_qualifiers(reference.get("qualifiers", [])),
            }
        else:
            raise ValueError(f"unsupported current benchmark qualifier {kind!r}")
        item = {"kind": kind, "target": target}
        if kind in {"near", "nearest_to", "farthest_from"}:
            reference = qualifier["reference"]
            mode = "start" if reference["kind"] == "start" else reference.get("quantifier", "unique")
            item["reference_mode"] = mode
            if mode == "start":
                item["target"] = {"kind": "start", "selector": "event", "values": ["START"], "qualifiers": []}
        result.append(item)
    return result


def replayable_route(
    route: dict[str, Any],
    gold: dict[str, Any],
    asg: ASG,
    robot_position: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    raw_points = route.get("waypoints")
    solver_regions = route.get("solver_region_witness")
    use_solver_witness = (
        route.get("route_source") == "dsl_solver"
        and isinstance(raw_points, list)
        and isinstance(solver_regions, list)
        and len(raw_points) == len(solver_regions)
    )
    points = (
        [
            {
                "x": float(point["x"]),
                "y": float(point.get("y", 0.0)),
                "z": float(point["z"]),
            }
            for point in raw_points
            if isinstance(point, dict)
        ]
        if use_solver_witness
        else _dense_waypoints(raw_points)
    )
    if not points:
        return {"waypoints": [], "regions": [], "portals": [], "milestones": []}, ["route is empty or has invalid waypoints"]
    regions = (
        [int(value) for value in solver_regions]
        if use_solver_witness
        else [_region_at(asg, point) for point in points]
    )
    if not use_solver_witness:
        _fill_missing_regions(regions)
    elif any(asg.get_region(region) is None for region in regions):
        return {
            "waypoints": points,
            "regions": regions,
            "portals": [],
            "milestones": [],
        }, ["solver region witness references a missing ASG region"]
    if not any(region is not None for region in regions):
        return {"waypoints": points, "regions": [], "portals": [], "milestones": []}, ["route is outside all ASG regions"]

    portals: list[int] = []
    portal_events: list[tuple[int, int]] = []
    errors: list[str] = []
    start_distance = route_start_distance({"waypoints": points}, robot_position)
    if start_distance is not None and start_distance > START_POSITION_TOLERANCE_M:
        errors.append(
            f"route starts {start_distance:.3f}m from task start "
            f"(tolerance {START_POSITION_TOLERANCE_M:.3f}m)"
        )
    for index, (first, second) in enumerate(
        zip(regions, regions[1:]),  # noqa: RUF007 - Python 3.9 support
        start=1,
    ):
        if first is None or second is None or first == second:
            continue
        matches = [
            asg.get_portal(edge.portal_index)
            for edge in asg.topology
            if {edge.region_a, edge.region_b} == {first, second}
        ]
        matches = [portal for portal in matches if portal is not None]
        if not matches:
            errors.append(f"regions {first} and {second} are not adjacent")
            continue
        before = Point2(points[index - 1]["x"], points[index - 1]["z"])
        after = Point2(points[index]["x"], points[index]["z"])
        portal = min(
            matches,
            key=lambda item: point_to_segment_distance(before, item.anchor_a, item.anchor_b)
            + point_to_segment_distance(after, item.anchor_a, item.anchor_b),
        )
        portals.append(portal.index)
        portal_events.append((index, portal.index))

    start_milestone = {
        "name": "START",
        "role": "start",
        "target_kind": "pos",
        "resolved_index": regions[0],
        "waypoint_index": 0,
        "event_seq": 0,
    }
    route_events = _route_target_events(route, points, regions, portal_events, asg)
    via_occurrences: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in gold.get("via", []):
        event = _matching_route_event(item, str(item.get("name")), route_events)
        occurrences = _target_hit_occurrences(
            item, points, regions, portal_events, asg
        )
        if event is not None:
            preferred = {
                "index": int(event["index"]),
                "segment_fraction": float(event.get("segment_fraction", 1.0)),
                "event_seq": event.get("event_seq"),
                "preferred": True,
            }
            occurrences = [
                preferred,
                *[
                    occurrence
                    for occurrence in occurrences
                    if _occurrence_key(occurrence) != _occurrence_key(preferred)
                ],
            ]
        via_occurrences.append((item, occurrences))
    goal = gold.get("goal", {})
    goal_milestone = None
    if _target_hits_point(goal, points[-1], regions[-1], asg) or (
        goal.get("kind") == "portal" and any(portal in set(goal.get("grounded_ids", [])) and index >= len(points) - 3 for index, portal in portal_events)
    ):
        event = _matching_route_event(goal, "GOAL", route_events)
        goal_milestone = _milestone(
            goal,
            "GOAL",
            "goal",
            len(points) - 1,
            event_seq=event.get("event_seq") if event is not None else None,
        )

    hard_milestones = _select_hard_via_occurrences(
        route=route,
        points=points,
        regions=regions,
        portals=portals,
        start_milestone=start_milestone,
        goal_milestone=goal_milestone,
        via_occurrences=via_occurrences,
        gold=gold,
        asg=asg,
        robot_position=robot_position,
    )
    milestones = list(hard_milestones)
    for item in gold.get("prefer", []):
        if item.get("preference") != "via" or not item.get("name"):
            continue
        event = _matching_route_event(item, str(item["name"]), route_events)
        occurrence = (
            {
                "index": int(event["index"]),
                "segment_fraction": float(event.get("segment_fraction", 1.0)),
            }
            if event is not None
            else next(
                iter(
                    _target_hit_occurrences(
                        item, points, regions, portal_events, asg
                    )
                ),
                None,
            )
        )
        if occurrence is not None:
            milestones.append(_milestone(
                item,
                str(item["name"]),
                "prefer",
                int(occurrence["index"]),
                event_seq=event.get("event_seq") if event is not None else None,
                segment_fraction=float(
                    occurrence.get("segment_fraction", 1.0)
                ),
            ))
    _number_replayed_milestones(milestones)
    return {
        **route,
        "waypoints": points,
        "regions": [int(region) if region is not None else -1 for region in regions],
        "portals": portals,
        "milestones": milestones,
    }, errors


def stored_route_replay_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Map stored geometry and bounded occurrence hints into independent replay.

    Hints do not assert satisfaction: ``replayable_route`` checks each target
    against route geometry and evaluates the full contract at its occurrence.
    """
    points = []
    for point in payload.get("waypoints", []):
        if isinstance(point, dict):
            points.append({"x": float(point["x"]), "y": float(point.get("y", 0.0)), "z": float(point["z"])})
        elif isinstance(point, (list, tuple)) and len(point) == 3:
            points.append(dict(zip(("x", "y", "z"), map(float, point))))
        else:
            raise ValueError("stored route has an invalid waypoint")
    hints = []
    for hit in payload.get("milestone_hits", []):
        if not isinstance(hit, dict):
            continue
        if hit.get("name") in {"START", "GOAL"}:
            # These boundaries are independently fixed to the first/last
            # route point, and must not become fallback hints for a VIA.
            continue
        index, sequence = hit.get("waypoint_index"), hit.get("event_seq")
        entity_id = hit.get("resolved_index", hit.get("id"))
        if (
            type(index) is not int or not 0 <= index < len(points)
            or type(sequence) is not int or sequence < 0
            or type(entity_id) is not int
        ):
            continue
        hints.append({
            "name": str(hit.get("name", "")),
            "target_kind": hit.get("target_kind", hit.get("kind")),
            "resolved_index": entity_id, "waypoint_index": index,
            "event_seq": sequence,
        })
    ordered = sorted(hints, key=lambda hit: hit["event_seq"])
    if any(
        first["event_seq"] == second["event_seq"]
        or first["waypoint_index"] > second["waypoint_index"]
        for first, second in zip(ordered, ordered[1:])
    ):
        hints = []
    return {
        "waypoints": points, "route_source": "dsl_solver",
        "solver_region_witness": [int(value) for value in payload.get("traversed_region_ids", [])],
        "execution_trace_complete": False, "milestones": hints,
    }


def _route_physically_valid(asg, points, regions, portal_ids) -> bool:
    from shapely.geometry import Point
    from reconav.scene.navigation import get_geometry_graph, segment_has_physical_clearance

    if not points or len(points) != len(regions) or any(
        not math.isfinite(value) for point in points for value in point.values()
    ):
        return False
    if len(portal_ids) != sum(first != second for first, second in zip(regions, regions[1:])):
        return False
    physical = get_geometry_graph(asg)
    tolerance = max(1e-5, float(physical.config.numeric_tolerance_m))
    for point, region in zip(points, regions):
        if (
            abs(point["y"] - physical.floor_height(region)) > tolerance
            or physical.locate_local_component(region, Point(point["x"], point["z"])) is None
        ):
            return False
    portals = iter(portal_ids)
    for first, second, region_a, region_b in zip(points, points[1:], regions, regions[1:]):
        a, b = (first["x"], first["z"]), (second["x"], second["z"])
        if region_a == region_b:
            if not segment_has_physical_clearance(physical, region_a, a, b):
                return False
        else:
            portal = next(portals)
            if not any(
                math.dist(a, x) <= tolerance and math.dist(b, y) <= tolerance
                for x, y in physical.portal_crossings(portal, region_a, region_b)
            ):
                return False
    return True


def stored_dsl_witness_report(asg: ASG, dsl: str, payload: Any) -> dict[str, Any] | None:
    """Replay an exact program on a physically checked stored witness."""
    from reconav.scene.geometry import Point3
    from reconav.solver.cross_region import CrossRegionPathResult, RouteMilestoneHit
    from reconav.solver.dsl_parser import parse_dsl
    from reconav.solver.replay import milestone_occurrence_errors, replay_solution

    if not isinstance(payload, dict):
        return None
    try:
        program = parse_dsl(dsl)
        # Current benchmark starts are exact positions. Other start selectors
        # need an independent membership check and retain the solver fallback.
        if program.start().target.kind != "pos":
            return None
        roles = {item.name: item.role for item in program.milestones}
        required_names = set(roles)
        roles.update({
            item.name or f"_prefer_{index}": "prefer"
            for index, item in enumerate(program.soft_via_specs)
        })
        route = stored_route_replay_input(payload)
        points, regions = route["waypoints"], route["solver_region_witness"]
        # Removing an optional VIA changes the event universe. Its old named
        # occurrence must not affect NEXT/BETWEEN or scoped replay afterward.
        hits = [hit for hit in payload.get("milestone_hits", []) if hit.get("name") in roles]
        if not required_names <= {hit.get("name") for hit in hits}:
            return None
        portals = list(payload.get("portal_ids", []))
        if milestone_occurrence_errors(asg, {
            "waypoints": points, "regions": regions, "portals": portals,
            "milestones": hits,
        }):
            return None
        waypoints = [Point3(point["x"], point["y"], point["z"]) for point in points]
        result = CrossRegionPathResult(
            waypoints=waypoints, traversed_region_ids=regions, portal_indices=portals,
            milestone_hits=[RouteMilestoneHit(
                name=hit["name"], role=roles[hit["name"]],
                target_kind=hit.get("target_kind", hit.get("kind")),
                target_selector="ids", target_value=hit.get("resolved_index", hit.get("id")),
                resolved_index=hit.get("resolved_index", hit.get("id")),
                point=waypoints[int(hit["waypoint_index"])],
                waypoint_index=int(hit["waypoint_index"]), event_seq=int(hit["event_seq"]),
            ) for hit in hits],
        )
        report = replay_solution(program, asg, result)
        return report if _route_physically_valid(asg, points, regions, portals) else None
    except Exception:  # noqa: BLE001 - inconclusive evidence retains full solver fallback
        return None


def _same_preference_semantics(
    predicted: dict[str, Any],
    gold: dict[str, Any],
) -> bool:
    def signature(source: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        preferences = _items(source, "prefer")
        names = {
            str(item["name"]): f"P{index}"
            for index, item in enumerate(preferences)
            if item.get("name") is not None
        }
        normalized_preferences = []
        for item in preferences:
            normalized = {
                key: value
                for key, value in item.items()
                if key not in {"name", "selector", "values"}
            }
            normalized_preferences.append(_json_key(normalized))
        normalized_orders = []
        for item in _items(source, "prefer_order"):
            normalized = copy.deepcopy(item)
            for field in ("first", "second"):
                normalized[field] = names.get(
                    str(normalized.get(field)), normalized.get(field)
                )
            normalized["names"] = [
                names.get(str(name), name)
                for name in normalized.get("names", [])
            ]
            normalized_orders.append(_json_key(normalized))
        return (
            tuple(sorted(normalized_preferences)),
            tuple(sorted(normalized_orders)),
        )

    return signature(predicted) == signature(gold)


def constraint_scores(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    scores = {}
    for family in COMPONENTS:
        gold_items = Counter(_json_key(item) for item in _family_items(gold, family))
        pred_items = Counter(_json_key(item) for item in _family_items(predicted, family))
        tp = sum((gold_items & pred_items).values())
        fp, fn = sum(pred_items.values()) - tp, sum(gold_items.values()) - tp
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        scores[family] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return scores


def _normalize_contract(value: Any) -> dict[str, Any]:
    source = copy.deepcopy(value) if isinstance(value, dict) else {}
    arrays = {
        key: source.get(key) if isinstance(source.get(key), list) else []
        for key in COMPONENTS[1:]
    }
    via = [
        {**_normalize_target(item), "name": _spec_event(item.get("name", f"v{index}"))}
        for index, item in enumerate(arrays["via"], start=1)
        if isinstance(item, dict)
    ]
    order = [
        normalized
        for item in arrays["order"]
        if isinstance(item, dict)
        for normalized in [_normalize_order(item)]
        if not _tautological_order(normalized)
    ]
    return {
        "goal": _normalize_target(source.get("goal")),
        "via": via,
        "order": order,
        "avoid": [_normalize_scoped(item) for item in arrays["avoid"] if isinstance(item, dict)],
        "count": [_normalize_count(item) for item in arrays["count"] if isinstance(item, dict)],
        "carry": [_normalize_carry(item) for item in arrays["carry"] if isinstance(item, dict)],
        "width": [_normalize_width(item) for item in arrays["width"] if isinstance(item, dict)],
        "prefer": [_normalize_preference(item) for item in arrays["prefer"] if isinstance(item, dict)],
        "prefer_order": [
            {**_normalize_order(item), "weight": _float(item.get("weight"), 1.0)}
            for item in arrays["prefer_order"]
            if isinstance(item, dict)
        ],
    }


def _normalize_target(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    target = {
        "kind": str(source.get("kind", "obj")).strip().lower(),
        "selector": str(source.get("selector", "category")).strip().lower(),
        "values": [
            " ".join(str(item).strip().lower().replace("_", " ").split())
            for item in source.get("values", [])
        ] if isinstance(source.get("values"), list) else [],
        "qualifiers": [
            _normalize_qualifier(item)
            for item in source.get("qualifiers", [])
            if isinstance(item, dict)
        ],
    }
    if source.get("text"):
        target["text"] = str(source["text"]).strip()
    return target


def _normalize_qualifier(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind", "")).strip().lower()
    if kind == "between_regions":
        return {
            "kind": kind,
            "regions": [_normalize_target(value) for value in item.get("regions", [])[:2]],
        }
    target = item.get("target", "START")
    out = {"kind": kind, "target": _normalize_target(target) if isinstance(target, dict) else target}
    if kind in {"near", "nearest_to", "farthest_from"}:
        out["reference_mode"] = str(item.get("reference_mode", "start" if target == "START" or (isinstance(target, dict) and target.get("kind") == "start") else "unique"))
    if kind == "near":
        out["distance_m"] = _float(item.get("distance_m"), 2.0)
    return out


def _normalize_scoped(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **_normalize_target(item),
        "after": _spec_event(item.get("after", "START")),
        "until": _spec_event(item.get("until", "GOAL")),
    }


def _normalize_count(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **_normalize_scoped(item),
        "op": str(item.get("op", "")).strip().lower().replace(" ", "_"),
        "count": _integer(item.get("count"), 0),
    }


def _normalize_carry(item: dict[str, Any]) -> dict[str, Any]:
    until = item.get("until")
    return {
        **_normalize_target(item),
        "until": _spec_event(until) if until is not None else None,
    }


def _normalize_width(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind", "numeric")).strip().lower()
    out = {
        "kind": kind,
        "after": _spec_event(item.get("after", "START")),
        "until": _spec_event(item.get("until", "GOAL")),
    }
    if kind == "from_milestone":
        out["milestone"] = _spec_event(item.get("milestone", item.get("source", "")))
    else:
        out["width_m"] = _float(item.get("width_m", item.get("grounded_width_m")), 0.0)
    return out


def _normalize_preference(item: dict[str, Any]) -> dict[str, Any]:
    out = {
        **_normalize_target(item),
        "preference": str(item.get("preference", "")).strip().lower(),
        "weight": _float(item.get("weight"), 1.0),
    }
    for key in ("name", "op", "count", "after", "until"):
        if key in item:
            out[key] = item[key]
    for key in ("name", "after", "until"):
        if key in out:
            out[key] = _spec_event(out[key])
    if "count" in out:
        out["count"] = _integer(out["count"], 0)
    return out


def _normalize_order(item: dict[str, Any]) -> dict[str, Any]:
    out = {
        "kind": str(item.get("kind", item.get("op", ""))).strip().lower(),
        "first": _spec_event(item.get("first", item.get("a", "START"))),
        "second": _spec_event(item.get("second", item.get("b", "GOAL"))),
    }
    if "names" in item:
        out["names"] = [_spec_event(value) for value in item.get("names", [])]
    return out


def _spec_event(value: Any) -> str:
    text = str(value).strip()
    return text.upper() if text.upper() in {"START", "GOAL"} else text.lower()


def _tautological_order(item: dict[str, Any]) -> bool:
    return item.get("kind") == "before" and (
        item.get("first") == "START" or item.get("second") == "GOAL"
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rename_events(spec: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    mapping = {"START": "START", "GOAL": "GOAL"}
    for index, item in enumerate(out["via"], start=1):
        mapping[str(item.get("name", f"v{index}"))] = f"v{index}"
        item["name"] = f"v{index}"
    preference_index = 0
    for item in out["prefer"]:
        if item.get("name"):
            preference_index += 1
            mapping[str(item["name"])] = f"p{preference_index}"
            item["name"] = f"p{preference_index}"
    for family in ("order", "prefer_order"):
        for item in out[family]:
            for key in ("first", "second"):
                item[key] = mapping.get(str(item.get(key)), str(item.get(key)))
            if "names" in item:
                item["names"] = sorted(mapping.get(str(name), str(name)) for name in item["names"])
    for family in ("avoid", "count", "carry", "width", "prefer"):
        for item in out[family]:
            for key in ("after", "until", "milestone"):
                if key in item and item[key] is not None:
                    item[key] = mapping.get(str(item[key]), str(item[key]))
    return out


def _clean_target(target: dict[str, Any]) -> None:
    target.pop("text", None)
    target.pop("selector", None)
    if isinstance(target.get("values"), list):
        target["values"] = sorted(target["values"])
    for qualifier in target.get("qualifiers", []):
        if isinstance(qualifier.get("target"), dict):
            _clean_target(qualifier["target"])
        for region in qualifier.get("regions", []):
            _clean_target(region)
        if isinstance(qualifier.get("regions"), list):
            qualifier["regions"] = sorted(qualifier["regions"], key=_json_key)
    if isinstance(target.get("qualifiers"), list):
        target["qualifiers"] = sorted(target["qualifiers"], key=_json_key)


def _ground_target(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    ids = item.get("grounded_ids", item.get("ids", item.get("id")))
    values = ids if isinstance(ids, list) else [ids]
    return {"kind": str(item.get("kind", "")).lower(), "grounded_ids": sorted(int(value) for value in values if _int_like(value))}


def _ground_scoped(item: dict[str, Any], events: dict[str, str]) -> dict[str, Any]:
    return {**(_ground_target(item) or {}), "after": _event(item.get("after", "START"), events), "until": _event(item.get("until", "GOAL"), events)}


def _ground_count(item: dict[str, Any], events: dict[str, str]) -> dict[str, Any]:
    return {**_ground_scoped(item, events), "op": str(item.get("op", "")).lower(), "count": int(item.get("count", 0))}


def _event(value: Any, events: dict[str, str]) -> str:
    text = str(value)
    if text in events:
        return events[text]
    if _int_like(value):
        return f"v{int(value)}"
    return text.upper() if text.upper() in {"START", "GOAL"} else text


def _constraint_groups(results: list[dict[str, Any]]) -> dict[str, bool]:
    grouped = {key: True for key in ("goal", "via", "order", "avoid", "count", "width")}
    for item in results:
        component = item.get("component")
        if component in grouped:
            grouped[component] = grouped[component] and item.get("satisfied") is True
    return grouped


def _failure_category(
    route_found: bool,
    replay_valid: bool,
    components: dict[str, bool],
    preference_success: bool | None,
) -> str | None:
    if not route_found:
        return "route_not_found"
    if not replay_valid:
        return "route_replay_invalid"
    for component in HARD_COMPONENTS:
        if not components[component]:
            return f"hard_{component}"
    if preference_success is None:
        return "preference_oracle_unavailable"
    return None if preference_success else "preference_suboptimal"


def _dense_waypoints(raw: Any, step_m: float = 0.2) -> list[dict[str, float]]:
    if not isinstance(raw, list):
        return []
    source = []
    for item in raw:
        if not isinstance(item, dict):
            return []
        try:
            point = {"x": float(item["x"]), "y": float(item.get("y", 0.0)), "z": float(item["z"])}
        except (KeyError, TypeError, ValueError):
            return []
        if not all(math.isfinite(value) for value in point.values()):
            return []
        source.append(point)
    if not source:
        return []
    dense = [{**source[0], "source_index": 0}]
    for source_index, (first, second) in enumerate(
        zip(source, source[1:])  # noqa: RUF007 - Python 3.9 support
    ):
        distance = math.hypot(second["x"] - first["x"], second["z"] - first["z"])
        steps = max(1, math.ceil(distance / step_m))
        for index in range(1, steps + 1):
            ratio = index / steps
            dense.append({
                **{key: first[key] + (second[key] - first[key]) * ratio for key in ("x", "y", "z")},
                "source_index": source_index + ratio,
            })
    return dense


ROUTE_REGION_REPLAY_VERSION = "height-aware-level-v1"


def _region_at(asg: ASG, point: dict[str, float]) -> int | None:
    """Find the nearest floor using waypoint foot height, then test its polygons.

    Compare every annotated level using its nearest region floor elevation.
    Equal-distance levels remain unassigned for route gap handling. Within a
    level, polygon order determines the first containing region.
    """
    p = Point2(point["x"], point["z"])
    height = float(point.get("y", 0.0))
    distances: dict[int, float] = {}
    for region in asg.regions:
        distance = abs(height - region.get_height())
        distances[region.level] = min(distances.get(region.level, math.inf), distance)
    if not distances:
        return None
    nearest = min(distances.values())
    levels = [level for level, distance in distances.items()
              if math.isclose(distance, nearest, rel_tol=0.0, abs_tol=1e-8)]
    if len(levels) != 1:
        return None
    return next((region.index for region in asg.regions
                 if region.level == levels[0] and point_in_region(p, region)), None)


def _fill_missing_regions(regions: list[int | None]) -> None:
    previous = None
    for index, value in enumerate(regions):
        if value is None and previous is not None:
            regions[index] = previous
        elif value is not None:
            previous = value
    following = None
    for index in range(len(regions) - 1, -1, -1):
        if regions[index] is None and following is not None:
            regions[index] = following
        elif regions[index] is not None:
            following = regions[index]


def _select_hard_via_occurrences(
    *,
    route: dict[str, Any],
    points: list[dict[str, float]],
    regions: list[int | None],
    portals: list[int],
    start_milestone: dict[str, Any],
    goal_milestone: dict[str, Any] | None,
    via_occurrences: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    gold: dict[str, Any],
    asg: ASG,
    robot_position: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Choose hard VIA occurrences using the same full-route semantics.

    A route may enter one target more than once.  The DSL runtime may decline
    an earlier optional observation and select a later one, so replay must not
    silently force the first geometric hit.  Enumerate the (normally very
    small) entry-event product, prefer hard-valid assignments, and then the
    minimum soft cost.  A supplied solver trace is only a deterministic
    tie-break after its geometry has been validated.
    """

    if not via_occurrences:
        milestones = [start_milestone]
        if goal_milestone is not None:
            milestones.append(goal_milestone)
        _number_replayed_milestones(milestones)
        return milestones

    if any(not occurrences for _item, occurrences in via_occurrences):
        milestones = [start_milestone]
        for item, occurrences in via_occurrences:
            if occurrences:
                occurrence = occurrences[0]
                milestones.append(
                    _milestone(
                        item,
                        str(item.get("name")),
                        "via",
                        int(occurrence["index"]),
                        event_seq=occurrence.get("event_seq"),
                        segment_fraction=float(
                            occurrence.get("segment_fraction", 1.0)
                        ),
                    )
                )
        if goal_milestone is not None:
            milestones.append(goal_milestone)
        _number_replayed_milestones(milestones)
        return milestones

    best: tuple[tuple[Any, ...], list[dict[str, Any]]] | None = None
    choices = [occurrences for _item, occurrences in via_occurrences]
    for selected in itertools.product(*choices):
        milestones = [copy.deepcopy(start_milestone)]
        for (item, _occurrences), occurrence in zip(
            via_occurrences, selected
        ):
            milestones.append(
                _milestone(
                    item,
                    str(item.get("name")),
                    "via",
                    int(occurrence["index"]),
                    event_seq=occurrence.get("event_seq"),
                    segment_fraction=float(
                        occurrence.get("segment_fraction", 1.0)
                    ),
                )
            )
        if goal_milestone is not None:
            milestones.append(copy.deepcopy(goal_milestone))
        _number_replayed_milestones(milestones)
        candidate_route = {
            **route,
            "waypoints": points,
            "regions": [
                int(region) if region is not None else -1 for region in regions
            ],
            "portals": portals,
            "milestones": milestones,
        }
        # Runtime-reported optimality is checked only after occurrence
        # reconstruction.  It must not influence which physical occurrence is
        # considered the semantic witness.
        candidate_route.pop("optimal_soft_cost", None)
        report = evaluate_grounded_route(
            asg, gold, candidate_route, robot_position or {}
        )
        hard_failures = sum(
            not bool(result.get("satisfied"))
            for result in report.get("constraints", [])
            if result.get("component")
            not in {"prefer", "prefer_order", "preference_optimality"}
        )
        preference_cost = report.get("preference_cost")
        soft_rank = (
            float(preference_cost)
            if hard_failures == 0 and isinstance(preference_cost, (int, float))
            else math.inf
        )
        trace_misses = sum(
            any(bool(option.get("preferred")) for option in occurrences)
            and not bool(occurrence.get("preferred"))
            for (_item, occurrences), occurrence in zip(
                via_occurrences, selected
            )
        )
        physical_order = tuple(
            _occurrence_key(occurrence) for occurrence in selected
        )
        rank = (hard_failures, soft_rank, trace_misses, physical_order)
        if best is None or rank < best[0]:
            best = rank, milestones
        if rank[:3] == (0, 0.0, 0):
            break
    assert best is not None
    return best[1]


def _occurrence_key(occurrence: dict[str, Any]) -> tuple[int, float]:
    return (
        int(occurrence["index"]),
        round(float(occurrence.get("segment_fraction", 1.0)), 9),
    )


def _target_hit_occurrences(
    item: dict[str, Any],
    points: list[dict[str, float]],
    regions: list[int | None],
    portals: list[tuple[int, int]],
    asg: ASG,
) -> list[dict[str, Any]]:
    ids = {int(value) for value in item.get("grounded_ids", [])}
    kind = item.get("kind")
    if kind == "portal":
        return [
            {"index": index, "segment_fraction": 1.0, "entity_id": portal}
            for index, portal in portals
            if portal in ids
        ]
    if kind == "region":
        return [
            {"index": index, "segment_fraction": 1.0, "entity_id": region}
            for index, region in enumerate(regions)
            if region in ids and (index == 0 or regions[index - 1] != region)
        ]
    if kind == "level":
        levels = [
            asg.get_region(region).level
            if region is not None and asg.get_region(region) is not None
            else None
            for region in regions
        ]
        return [
            {"index": index, "segment_fraction": 1.0, "entity_id": level}
            for index, level in enumerate(levels)
            if level in ids and (index == 0 or regions[index - 1] != regions[index])
        ]
    if kind == "obj" and all(region is not None for region in regions):
        return [
            {
                "index": occurrence.waypoint_index,
                "segment_fraction": occurrence.segment_fraction,
                "entity_id": occurrence.object_id,
            }
            for occurrence in object_encounter_entries(
                asg,
                ids,
                points,
                [int(region) for region in regions],
                clearance_m=ENDPOINT_OBJECT_CLEARANCE_M,
            )
        ]
    return []


def _route_target_events(
    route: dict[str, Any],
    points: list[dict[str, float]],
    regions: list[int | None],
    portals: list[tuple[int, int]],
    asg: ASG,
) -> list[dict[str, Any]]:
    events = []
    for item in route.get("milestones", []):
        if not isinstance(item, dict) or not _int_like(item.get("resolved_index")) or not _int_like(item.get("waypoint_index")):
            continue
        index = min(
            range(len(points)),
            key=lambda pos: abs(
                float(points[pos].get("source_index", pos))
                - int(item["waypoint_index"])
            ),
        )
        target = {"kind": item.get("target_kind"), "grounded_ids": [int(item["resolved_index"])]}
        portal_hit = target["kind"] == "portal" and any(
            event_index == index and portal_id == int(item["resolved_index"])
            for event_index, portal_id in portals
        )
        target_hit = (
            portal_hit if target["kind"] == "portal"
            else _target_hits_point(target, points[index], regions[index], asg)
        )
        if target["kind"] in {"region", "level"} and item.get("name") != "GOAL":
            # Match the complete solver's eligible(): non-goal region/level
            # milestones occur initially or on a region-entry transition.
            target_hit = target_hit and (index == 0 or regions[index - 1] != regions[index])
        if target_hit:
            events.append({
                "name": str(item.get("name", "")),
                "kind": target["kind"],
                "id": int(item["resolved_index"]),
                "index": index,
                "segment_fraction": float(item.get("segment_fraction", 1.0)),
                "event_seq": int(item["event_seq"])
                if _int_like(item.get("event_seq"))
                else index,
            })
    if events:
        return events

    cursor = 0
    for item in route.get("targets", []):
        if not isinstance(item, dict) or item.get("kind") not in {"obj", "region", "portal", "level"} or not _int_like(item.get("id")):
            continue
        target = {"kind": item["kind"], "grounded_ids": [int(item["id"])]}
        candidates = []
        if target["kind"] == "portal":
            candidates = [index for index, portal in portals if portal == int(item["id"]) and index >= cursor]
        else:
            candidates = [
                index for index in range(cursor, len(points))
                if _target_hits_point(target, points[index], regions[index], asg)
            ]
        if not candidates:
            continue
        index = candidates[0]
        events.append({
            "name": str(item.get("name", "")),
            "kind": item["kind"],
            "id": int(item["id"]),
            "index": index,
            "event_seq": int(item["event_seq"])
            if _int_like(item.get("event_seq"))
            else index,
        })
        cursor = index
    return events


def _matching_route_event(
    item: dict[str, Any], name: str, events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    ids = {int(value) for value in item.get("grounded_ids", [])}
    named = [event for event in events if event["name"] == name and event["kind"] == item.get("kind") and event["id"] in ids]
    matches = named or [event for event in events if event["kind"] == item.get("kind") and event["id"] in ids]
    return matches[0] if matches else None


def _target_hits_point(item: dict[str, Any], point: dict[str, float], region_id: int | None, asg: ASG) -> bool:
    ids = {int(value) for value in item.get("grounded_ids", [])}
    if item.get("kind") == "region":
        return region_id in ids
    if item.get("kind") == "level":
        region = asg.get_region(region_id) if region_id is not None else None
        return region is not None and region.level in ids
    p = Point2(point["x"], point["z"])
    if item.get("kind") == "obj":
        if region_id is None:
            return False
        for obj_id in ids:
            obj = asg.get_object(obj_id)
            if obj is None:
                continue
            if (
                int(region_id) != int(obj.region_index)
                and int(region_id) not in obj.body_band_footprint_2d_by_region
            ):
                continue
            # Object targets are semantic encounters.  Retained collision
            # footprints define GOAL/VIA (and the shared route-event code uses
            # them for AVOID/COUNT/PREFER); body-band geometry is reserved for
            # scene-mesh physical collision construction.
            footprints = obj.navigation_footprint_2d()
            if footprints and within_semantic_clearance(
                distance_to_polygons(p, footprints),
                ENDPOINT_OBJECT_CLEARANCE_M,
            ):
                return True
    if item.get("kind") == "portal":
        for portal_id in ids:
            portal = asg.get_portal(portal_id)
            if portal is not None and point_to_segment_distance(p, portal.anchor_a, portal.anchor_b) <= 0.75:
                return True
    return False


def _milestone(
    item: dict[str, Any],
    name: str,
    role: str,
    index: int,
    *,
    event_seq: int | None,
    segment_fraction: float = 1.0,
) -> dict[str, Any]:
    ids = [int(value) for value in item.get("grounded_ids", [])]
    return {
        "name": name,
        "role": role,
        "target_kind": item.get("kind"),
        "resolved_index": ids[0],
        "waypoint_index": index,
        "segment_fraction": float(segment_fraction),
        "event_seq": event_seq,
    }


def _number_replayed_milestones(milestones: list[dict[str, Any]]) -> None:
    """Build a comparable logical trace for gold events witnessed on one route."""
    ordered = sorted(
        enumerate(milestones),
        key=lambda pair: (
            int(pair[1]["waypoint_index"]),
            float(pair[1].get("segment_fraction", 1.0)),
            pair[1]["event_seq"] is None,
            int(pair[1]["event_seq"] or 0),
            pair[0],
        ),
    )
    for event_seq, (_index, milestone) in enumerate(ordered):
        milestone["event_seq"] = event_seq


def _items(source: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = source.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _spec_items(specification: dict[str, Any]) -> Counter[str]:
    return Counter(
        _json_key({"family": family, "item": item})
        for family in COMPONENTS
        for item in _family_items(specification, family)
    )


def _family_items(specification: dict[str, Any], family: str) -> list[dict[str, Any]]:
    if family != "goal":
        return _items(specification, family)
    goal = specification.get("goal")
    if not isinstance(goal, dict):
        return []
    if goal.get("selector") == "category" and not goal.get("values"):
        return []
    return [goal]


def _abstract_target(item: Any, asg: ASG) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    if item.get("selector") in {"category", "any"}:
        return {
            key: copy.deepcopy(item[key])
            for key in ("kind", "selector", "values", "qualifiers", "text")
            if key in item
        }
    kind = str(item.get("kind", "")).lower()
    ids = item.get("grounded_ids", item.get("ids", item.get("id")))
    values = ids if isinstance(ids, list) else [ids]
    labels = sorted({
        label for value in values
        if _int_like(value) and (label := _entity_label(asg, kind, int(value)))
    })
    return {"kind": kind, "selector": "category", "values": labels or ["__unknown__"], "qualifiers": []}


def _entity_label(asg: ASG, kind: str, index: int) -> str | None:
    getter = {
        "obj": asg.get_object,
        "region": asg.get_region,
        "portal": asg.get_portal,
        "level": asg.get_level,
    }.get(kind)
    entity = getter(index) if getter is not None else None
    if entity is None:
        return None
    return " ".join(entity.label.strip().lower().replace("_", " ").split())


def _abstract_order(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(item.get("kind", item.get("op", ""))).lower(),
        "first": _abstract_event(item.get("first", item.get("a", "START"))),
        "second": _abstract_event(item.get("second", item.get("b", "GOAL"))),
        "names": sorted(_abstract_event(name) for name in item.get("names", [])),
    }


def _abstract_scoped(item: dict[str, Any], asg: ASG) -> dict[str, Any]:
    return {
        **_abstract_target(item, asg),
        "after": _abstract_event(item.get("after", "START")),
        "until": _abstract_event(item.get("until", "GOAL")),
    }


def _abstract_count(item: dict[str, Any], asg: ASG) -> dict[str, Any]:
    return {
        **_abstract_scoped(item, asg),
        "op": str(item.get("op", "")).lower(),
        "count": int(item.get("count", 0)),
    }


def _abstract_width(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind", "numeric")).lower()
    out = {
        "kind": kind,
        "after": _abstract_event(item.get("after", "START")),
        "until": _abstract_event(item.get("until", "GOAL")),
    }
    if kind == "from_milestone":
        out["milestone"] = _abstract_event(item.get("milestone", item.get("source", "")))
    else:
        out["width_m"] = float(item.get("width_m", item.get("grounded_width_m", 0.0)))
    return out


def _abstract_preference(item: dict[str, Any], asg: ASG, index: int) -> dict[str, Any]:
    out = {
        **_abstract_target(item, asg),
        "preference": str(item.get("preference", "")).lower(),
        "weight": 1.0,
    }
    for key in ("op", "count"):
        if key in item:
            out[key] = int(item[key]) if key == "count" else str(item[key]).lower()
    for key in ("after", "until"):
        if key in item:
            out[key] = _abstract_event(item[key])
    if item.get("name"):
        out["name"] = f"p{index}"
    return out


def _abstract_event(value: Any) -> str:
    if _int_like(value):
        return f"v{int(value)}"
    text = str(value)
    return text.upper() if text.upper() in {"START", "GOAL"} else text


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
