"""Task-semantic specification and referent-binding metrics."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict, deque
from typing import Any

from reconav.scene.asg import ASG
from reconav.scene.geometry import Point2
from reconav.solver.dsl_ast import TargetQualifier, TargetSelector
from reconav.solver.target_grounding import ground_target
from reconav.scene.navmesh import point_in_region

SEMANTIC_FAMILIES = (
    "goal",
    "visit",
    "order",
    "avoid",
    "count",
    "carry",
    "width",
    "preference",
)


def semantic_specification_score(
    task: dict[str, Any],
    gold_spec: dict[str, Any],
    predicted_spec: dict[str, Any],
    asg: ASG,
    *,
    gold_grounded: dict[str, Any] | None = None,
    predicted_grounded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_view = _support_view(
        gold_spec,
        gold_grounded
        if isinstance(gold_grounded, dict)
        else task.get("grounding")
        if isinstance(task.get("grounding"), dict)
        else {},
        task,
        asg,
        derive_bindings=True,
    )
    predicted_view = _support_view(
        predicted_spec,
        predicted_grounded if isinstance(predicted_grounded, dict) else {},
        task,
        asg,
    )
    gold = gold_view["obligations"]
    predicted = predicted_view["obligations"]
    unsatisfiable = _order_unsatisfiable(predicted)
    if not predicted or unsatisfiable:
        overall = _score(0, len(predicted), len(gold))
        families = {
            family: {
                **_score(0, len(_family(predicted, family)), len(_family(gold, family))),
                "explicit_predicted": len(_family(predicted, family)),
                "operational_predicted": 0,
            }
            for family in SEMANTIC_FAMILIES
        }
    else:
        precision_hits = sum(
            _supported(item, predicted_view, gold_view, task, asg)
            for item in predicted
        )
        recall_hits = sum(
            _supported(item, gold_view, predicted_view, task, asg)
            for item in gold
        )
        overall = _bidirectional_score(precision_hits, len(predicted), recall_hits, len(gold))
        families = {}
        for family in SEMANTIC_FAMILIES:
            gold_family = _family(gold, family)
            pred_family = _family(predicted, family)
            precision_hits = sum(
                _supported(item, predicted_view, gold_view, task, asg)
                for item in pred_family
            )
            recall_hits = sum(
                _supported(item, gold_view, predicted_view, task, asg)
                for item in gold_family
            )
            effective_predicted = len(pred_family)
            if not effective_predicted and recall_hits:
                effective_predicted = recall_hits
                precision_hits = recall_hits
            families[family] = _bidirectional_score(
                precision_hits, effective_predicted, recall_hits, len(gold_family)
            )
            families[family]["explicit_predicted"] = len(pred_family)
            families[family]["operational_predicted"] = effective_predicted - len(pred_family)
    return {
        **overall,
        "semantically_complete": overall["recall"] == 1.0,
        "semantically_exact": overall["precision"] == 1.0 and overall["recall"] == 1.0,
        "unsatisfiable_prediction": unsatisfiable,
        "families": families,
        "gold_obligations": len(gold),
        "predicted_obligations": len(predicted),
    }


def grounding_id_score(
    task: dict[str, Any],
    gold_spec: dict[str, Any],
    predicted_spec: dict[str, Any],
    predicted_grounded: dict[str, Any],
    asg: ASG,
    *,
    semantically_complete: bool,
    gold_ownership_spec: dict[str, Any] | None = None,
    predicted_ownership_spec: dict[str, Any] | None = None,
    gold_grounded_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_grounded = (
        gold_grounded_override
        if isinstance(gold_grounded_override, dict)
        else task.get("grounding")
        if isinstance(task.get("grounding"), dict)
        else {}
    )
    gold_records, gold_alignment = _binding_records(
        gold_spec,
        gold_grounded,
        task,
        asg,
        ownership_spec=gold_ownership_spec,
    )
    pred_records, predicted_alignment = _binding_records(
        predicted_spec,
        predicted_grounded,
        task,
        asg,
        ownership_spec=predicted_ownership_spec,
    )
    predicted_view = _support_view(predicted_spec, predicted_grounded, task, asg)
    operational = _operational_grounding_descriptors(
        gold_spec,
        gold_grounded,
        predicted_view,
        task,
        asg,
        ownership_spec=gold_ownership_spec,
    )
    available: dict[str, list[tuple[str, tuple[int, ...]] | None]] = defaultdict(list)
    for descriptor, binding in pred_records:
        available[descriptor].append(binding)

    aligned = correct = 0
    missing: list[str] = []
    wrong: list[dict[str, Any]] = []
    operationally_supported: list[str] = []
    for descriptor, acceptable in gold_records:
        candidates = available.get(descriptor, [])
        if acceptable in candidates:
            aligned += 1
            correct += 1
            candidates.remove(acceptable)
            continue
        if operational[descriptor] > 0:
            operational[descriptor] -= 1
            aligned += 1
            correct += 1
            operationally_supported.append(descriptor)
            continue
        if not candidates:
            missing.append(descriptor)
            continue
        aligned += 1
        binding = candidates[0]
        candidates.remove(binding)
        wrong.append({"slot": descriptor, "predicted": binding, "acceptable": acceptable})
    total = len(gold_records)
    exact = correct == total
    return {
        "exact_match": exact,
        "required_slots": total,
        "aligned_slots": aligned,
        "correct_slots": correct,
        "alignment_coverage": aligned / total if total else 1.0,
        "aligned_id_accuracy": correct / aligned if aligned else 0.0,
        "conditional_exact_match": exact if semantically_complete else None,
        "operationally_supported_slots": operationally_supported,
        "missing_slots": missing,
        "wrong_bindings": wrong,
        "ownership_alignment": {
            "gold": gold_alignment,
            "predicted": predicted_alignment,
        },
    }


def semantic_field_scores(
    gold_spec: dict[str, Any],
    predicted_spec: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    """Diagnostic projections; these do not alter the primary semantic score."""
    gold = _obligations(gold_spec)
    predicted = _obligations(predicted_spec)
    fields = {}
    for name, keys in {
        "target": ("target", "first", "second", "source", "names"),
        "operator_value": ("kind", "operator", "value"),
        "scope": ("after", "until"),
        "modality": ("modality",),
    }.items():
        gold_projection = _field_projection(gold, keys)
        predicted_projection = _field_projection(predicted, keys)
        hits = len(gold_projection & predicted_projection)
        fields[name] = _score(hits, len(predicted_projection), len(gold_projection))
    return fields


def _obligations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    goal = _target(spec.get("goal"))
    vias = [item for item in spec.get("via", []) if isinstance(item, dict)]
    preferences = [item for item in spec.get("prefer", []) if isinstance(item, dict)]
    events: dict[str, Any] = {"START": {"special": "START"}}
    if goal is not None:
        events["GOAL"] = {"special": "GOAL", "target": goal}
    for index, item in enumerate(vias, start=1):
        events[str(item.get("name", f"v{index}"))] = {"target": _target(item)}
    for index, item in enumerate(preferences, start=1):
        if item.get("name"):
            events[str(item["name"])] = {"target": _target(item)}

    out: list[dict[str, Any]] = []
    if goal is not None:
        out.append({"family": "goal", "modality": "hard", "target": goal})
    for item in vias:
        target = _target(item)
        if target is not None:
            out.append({"family": "visit", "modality": "hard", "target": target})
    for item in _items(spec, "order"):
        out.append(_order_obligation(item, events, modality="hard", family="order"))
    for item in _items(spec, "avoid"):
        out.append({
            "family": "avoid", "modality": "hard", "target": _target(item),
            "after": _event(item.get("after", "START"), events),
            "until": _event(item.get("until", "GOAL"), events),
        })
    for item in _items(spec, "count"):
        out.append({
            "family": "count", "modality": "hard", "target": _target(item),
            "operator": str(item.get("op", "")), "value": int(item.get("count", 0)),
            "after": _event(item.get("after", "START"), events),
            "until": _event(item.get("until", "GOAL"), events),
        })
    for item in _items(spec, "carry"):
        until = item.get("until")
        out.append({
            "family": "carry",
            "modality": "hard",
            "target": _target(item),
            "until": _event(until, events) if until is not None else None,
        })
    for item in _items(spec, "width"):
        obligation = {
            "family": "width", "modality": "hard", "kind": str(item.get("kind", "numeric")),
            "after": _event(item.get("after", "START"), events),
            "until": _event(item.get("until", "GOAL"), events),
        }
        if obligation["kind"] == "from_milestone":
            obligation["source"] = _event(item.get("milestone", ""), events)
        else:
            obligation["value"] = round(float(item.get("width_m", 0.0)), 6)
        out.append(obligation)
    for item in preferences:
        preference = str(item.get("preference", ""))
        obligation = {
            "family": "preference", "modality": "soft", "kind": preference,
            "target": _target(item),
        }
        for key in ("op", "count"):
            if key in item:
                obligation["operator" if key == "op" else "value"] = item[key]
        if preference != "via":
            obligation["after"] = _event(item.get("after", "START"), events)
            obligation["until"] = _event(item.get("until", "GOAL"), events)
        out.append(obligation)
    for item in _items(spec, "prefer_order"):
        out.append(_order_obligation(item, events, modality="soft", family="preference"))
    unique = {_key(item): item for item in out if _valid_obligation(item)}
    return [unique[key] for key in sorted(unique)]


def _order_obligation(
    item: dict[str, Any], events: dict[str, Any], *, modality: str, family: str
) -> dict[str, Any]:
    return {
        "family": family,
        "modality": modality,
        "kind": str(item.get("kind", "")),
        "first": _event(item.get("first", "START"), events),
        "second": _event(item.get("second", "GOAL"), events),
        "names": sorted((_event(name, events) for name in item.get("names", [])), key=_key),
    }


def _support_view(
    spec: dict[str, Any],
    grounded: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
    *,
    derive_bindings: bool = False,
) -> dict[str, Any]:
    vias = _items(spec, "via")
    grounded_vias = _items(grounded, "via")
    chain = []
    for index, item in enumerate(vias, start=1):
        explicit = grounded_vias[index - 1] if index <= len(grounded_vias) else None
        chain.append({
            "name": str(item.get("name", f"v{index}")),
            "position": index,
            "role": "via",
            "target": _target(item),
            "binding": _binding(explicit) or (
                _resolve_target(item, task, asg) if derive_bindings else None
            ),
        })
    goal = spec.get("goal") if isinstance(spec.get("goal"), dict) else None
    explicit_goal = grounded.get("goal") if isinstance(grounded.get("goal"), dict) else None
    if goal is not None:
        chain.append({
            "name": "GOAL",
            "position": len(vias) + 1,
            "role": "goal",
            "target": _target(goal),
            "binding": _binding(explicit_goal) or (
                _resolve_target(goal, task, asg) if derive_bindings else None
            ),
        })
    return {
        "spec": spec,
        "grounded": grounded,
        "obligations": _obligations(spec),
        "chain": chain,
        "via_count": len(vias),
    }


def _grounded_target_refinement(
    item: dict[str, Any],
    item_view: dict[str, Any],
    source_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    role = "goal" if item.get("family") == "goal" else "via"
    item_records = _records_for_target(item.get("target"), item_view, role, task, asg)
    source_targets = [
        candidate.get("target")
        for candidate in source_view["obligations"]
        if candidate.get("family") == item.get("family")
        and candidate.get("modality") == item.get("modality")
    ]
    source_records = [
        record
        for target in source_targets
        for record in _records_for_target(target, source_view, role, task, asg)
    ]
    return any(
        first.get("binding") is not None
        and first.get("binding") == second.get("binding")
        for first in item_records
        for second in source_records
    )


def _chain_supports_count(
    count: dict[str, Any],
    chain_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    if count.get("operator") != "at_least" or int(count.get("value", 0)) < 1:
        return False
    domain = _resolve_target(count.get("target"), task, asg)
    if domain is None:
        return False
    for after, until in _scope_windows(count, chain_view, task, asg):
        witnessed: set[int] = set()
        for record in _via_records(chain_view):
            if after < int(record["position"]) < until:
                witnessed.update(_binding_ids_in_domain(record.get("binding"), domain))
        if len(witnessed) >= int(count["value"]):
            return True
    return False


def _chain_supports_preferred_via(
    preference: dict[str, Any],
    chain_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    return any(
        record.get("binding") is not None
        and _record_matches_target(record, preference.get("target"), task, asg)
        for record in _via_records(chain_view)
    )


def _visit_implements_count(
    item: dict[str, Any],
    item_view: dict[str, Any],
    source_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    records = _records_for_target(item.get("target"), item_view, "via", task, asg)
    for count in _family(source_view["obligations"], "count"):
        if not _chain_supports_count(count, item_view, task, asg):
            continue
        domain = _resolve_target(count.get("target"), task, asg)
        if domain is None:
            continue
        for after, until in _scope_windows(count, item_view, task, asg):
            if any(
                after < int(record["position"]) < until
                and bool(_binding_ids_in_domain(record.get("binding"), domain))
                for record in records
            ):
                return True
    return False


def _visit_implements_preferred_via(
    item: dict[str, Any],
    item_view: dict[str, Any],
    source_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    records = _records_for_target(item.get("target"), item_view, "via", task, asg)
    return any(
        preference.get("kind") == "via"
        and preference.get("modality") == "soft"
        and any(
            record.get("binding") is not None
            and _record_matches_target(record, preference.get("target"), task, asg)
            for record in records
        )
        for preference in _family(source_view["obligations"], "preference")
    )


def _visit_implements_carry(
    item: dict[str, Any],
    source_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    target = _target(item.get("target"))
    if target is None:
        return False
    resolved = _resolve_target(target, task, asg)
    for carry in _family(source_view["obligations"], "carry"):
        payload = _target(carry.get("target"))
        if payload == target:
            return True
        if resolved is not None and _resolve_target(payload, task, asg) == resolved:
            return True
    return False


def _order_implements_count(
    item: dict[str, Any],
    item_view: dict[str, Any],
    source_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    if item.get("family") != "order" or item.get("modality") != "hard":
        return False
    first_positions = _event_positions(item.get("first"), item_view, task, asg)
    second_positions = _event_positions(item.get("second"), item_view, task, asg)
    for count in _family(source_view["obligations"], "count"):
        if not _chain_supports_count(count, item_view, task, asg):
            continue
        domain = _resolve_target(count.get("target"), task, asg)
        if domain is None:
            continue
        for after, until in _scope_windows(count, item_view, task, asg):
            for first in first_positions:
                for second in second_positions:
                    if not after <= first < second <= until:
                        continue
                    if _position_matches_domain(first, item_view, domain) or _position_matches_domain(
                        second, item_view, domain
                    ):
                        return True
    return False


def _scope_windows(
    obligation: dict[str, Any],
    chain_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> list[tuple[int, int]]:
    after_positions = _event_positions(obligation.get("after"), chain_view, task, asg)
    until_positions = _event_positions(obligation.get("until"), chain_view, task, asg)
    return [
        (after, until)
        for after in after_positions
        for until in until_positions
        if after < until
    ]


def _event_positions(
    event: Any,
    chain_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> list[int]:
    if isinstance(event, dict) and event.get("special") == "START":
        return [0]
    if isinstance(event, dict) and event.get("special") == "GOAL":
        return [int(chain_view["via_count"]) + 1]
    target = event.get("target") if isinstance(event, dict) else None
    return [
        int(record["position"])
        for record in chain_view["chain"]
        if _record_matches_target(record, target, task, asg)
    ]


def _records_for_target(
    target: Any,
    view: dict[str, Any],
    role: str,
    task: dict[str, Any],
    asg: ASG,
) -> list[dict[str, Any]]:
    return [
        record
        for record in view["chain"]
        if record.get("role") == role and _record_matches_target(record, target, task, asg)
    ]


def _record_matches_target(
    record: dict[str, Any],
    target: Any,
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    normalized = _target(target)
    if normalized is None:
        return False
    if record.get("target") == normalized:
        return True
    domain = _resolve_target(normalized, task, asg)
    binding = record.get("binding")
    return bool(_binding_ids_in_domain(binding, domain)) if domain is not None else False


def _binding_ids_in_domain(
    binding: tuple[str, tuple[int, ...]] | None,
    domain: tuple[str, tuple[int, ...]],
) -> set[int]:
    if binding is None or binding[0] != domain[0]:
        return set()
    ids = set(binding[1])
    return ids if ids and ids.issubset(set(domain[1])) else set()


def _via_records(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in view["chain"] if record.get("role") == "via"]


def _position_matches_domain(
    position: int,
    view: dict[str, Any],
    domain: tuple[str, tuple[int, ...]],
) -> bool:
    return any(
        int(record["position"]) == position
        and bool(_binding_ids_in_domain(record.get("binding"), domain))
        for record in view["chain"]
    )


def _semantic_value_equivalent(
    first: Any,
    second: Any,
    task: dict[str, Any],
    asg: ASG,
) -> bool:
    """Compare semantic structures while allowing equivalent ASG denotations.

    Target expressions that deterministically denote the same ASG set are
    equivalent; all other structure must match recursively.
    """
    if isinstance(first, dict) and isinstance(second, dict):
        if _looks_like_target(first) and _looks_like_target(second):
            left = _resolve_target(first, task, asg)
            right = _resolve_target(second, task, asg)
            if left is not None and right is not None:
                return left == right
            return _target(first) == _target(second)
        if set(first) != set(second):
            return False
        return all(
            _semantic_value_equivalent(first[key], second[key], task, asg)
            for key in first
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _semantic_value_equivalent(left, right, task, asg)
            for left, right in zip(first, second)
        )
    return first == second


def _looks_like_target(value: dict[str, Any]) -> bool:
    return (
        value.get("kind") in {"obj", "object", "region", "portal", "level", "start", "pos"}
        and any(key in value for key in ("selector", "values", "qualifiers"))
    )


def _supported(
    item: dict[str, Any],
    item_view: dict[str, Any],
    source_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> int:
    source = source_view["obligations"]
    if any(
        _semantic_value_equivalent(item, candidate, task, asg)
        for candidate in source
    ):
        return 1
    if item.get("family") in {"goal", "visit"} and _grounded_target_refinement(
        item, item_view, source_view, task, asg
    ):
        return 1
    if item.get("family") == "visit" and _implied_visit(item, source, task, asg):
        return 1
    if item.get("family") == "visit" and (
        _visit_implements_count(item, item_view, source_view, task, asg)
        or _visit_implements_preferred_via(item, item_view, source_view, task, asg)
        or _visit_implements_carry(item, source_view, task, asg)
    ):
        return 1
    if item.get("family") == "count" and _chain_supports_count(item, source_view, task, asg):
        return 1
    if (
        item.get("family") == "preference"
        and item.get("modality") == "soft"
        and item.get("kind") == "via"
        and _chain_supports_preferred_via(item, source_view, task, asg)
    ):
        return 1
    if item.get("kind") in {"before", "next"} and item.get("family") in {"order", "preference"}:
        if _order_supported(item, source):
            return 1
        if _order_implements_count(item, item_view, source_view, task, asg):
            return 1
    return 0


def _order_supported(item: dict[str, Any], source: list[dict[str, Any]]) -> bool:
    orders = [candidate for candidate in source if _compatible_order(item, candidate)]
    first, second = _key(item.get("first")), _key(item.get("second"))
    edges = {(_key(order.get("first")), _key(order.get("second"))) for order in orders}
    start = _key({"special": "START"})
    goals = {
        _key({"special": "GOAL", "target": candidate.get("target")})
        for candidate in source
        if candidate.get("family") == "goal" and candidate.get("modality") == "hard"
    }
    visits = {
        _key({"target": candidate.get("target")})
        for candidate in source
        if candidate.get("family") == "visit" and candidate.get("modality") == "hard"
    }
    visits.update(
        _key({"target": candidate.get("target")})
        for candidate in source
        if candidate.get("family") == "preference"
        and candidate.get("modality") == "soft"
        and candidate.get("kind") == "via"
    )
    visible = goals | visits | {start}
    edges.update((start, event) for event in visible if event != start)
    edges.update((visit, goal) for visit in visits for goal in goals)
    if not _reachable(first, second, edges):
        return False
    if item.get("kind") == "before":
        return True
    if any(
        order.get("kind") == "next"
        and _key(order.get("first")) == first
        and _key(order.get("second")) == second
        for order in orders
    ):
        return True
    return not any(
        event not in {first, second}
        and not _reachable(event, first, edges)
        and not _reachable(second, event, edges)
        for event in visible
    )


def _compatible_order(item: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if candidate.get("kind") not in {"before", "next"}:
        return False
    item_signature = (item.get("family"), item.get("modality"))
    candidate_signature = (candidate.get("family"), candidate.get("modality"))
    return candidate_signature == item_signature or {
        item_signature,
        candidate_signature,
    } == {("order", "hard"), ("preference", "soft")}


def _implied_visit(item: dict[str, Any], source: list[dict[str, Any]], task: dict[str, Any], asg: ASG) -> bool:
    target = item.get("target")
    if not isinstance(target, dict) or target.get("kind") not in {"region", "portal"}:
        return False
    required = _resolve_target(target, task, asg)
    if required is None or not required[1]:
        return False
    destinations = [
        candidate.get("target")
        for candidate in source
        if candidate.get("family") in {"goal", "visit"}
    ]
    start = _start_region(task, asg)
    if start is None:
        return False
    for destination in destinations:
        resolved = _resolve_target(destination, task, asg)
        if resolved is None:
            continue
        regions = _target_regions(resolved, asg)
        if not regions:
            continue
        if target.get("kind") == "region" and all(
            _region_unavoidable(start, region, required[1], asg) for region in regions
        ):
            return True
        if target.get("kind") == "portal" and all(
            _portal_unavoidable(start, region, required[1], asg) for region in regions
        ):
            return True
    return False


def _region_unavoidable(start: int, goal: int, required: tuple[int, ...], asg: ASG) -> bool:
    return all(region == goal or not _region_reachable(start, goal, asg, blocked_region=region) for region in required)


def _portal_unavoidable(start: int, goal: int, required: tuple[int, ...], asg: ASG) -> bool:
    return all(not _region_reachable(start, goal, asg, blocked_portal=portal) for portal in required)


def _region_reachable(
    start: int, goal: int, asg: ASG, *, blocked_region: int | None = None, blocked_portal: int | None = None
) -> bool:
    if start == blocked_region or goal == blocked_region:
        return False
    queue, seen = deque([start]), {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for edge in asg.topology_for_region(current):
            if edge.portal_index == blocked_portal:
                continue
            nxt = edge.region_b if edge.region_a == current else edge.region_a
            if nxt != blocked_region and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _order_unsatisfiable(obligations: list[dict[str, Any]]) -> bool:
    edges = {
        (_key(item.get("first")), _key(item.get("second")))
        for item in obligations
        if item.get("modality") == "hard" and item.get("kind") in {"before", "next"}
        and _key(item.get("first")) != _key(item.get("second"))
    }
    start = _key({"special": "START"})
    goals = {
        _key({"special": "GOAL", "target": item.get("target")})
        for item in obligations
        if item.get("family") == "goal" and item.get("modality") == "hard"
    }
    visits = {
        _key({"target": item.get("target")})
        for item in obligations
        if item.get("family") == "visit" and item.get("modality") == "hard"
    }
    edges.update((start, event) for event in goals | visits)
    edges.update((visit, goal) for visit in visits for goal in goals)
    return any(_reachable(second, first, edges) for first, second in edges)


def _reachable(first: str, second: str, edges: set[tuple[str, str]]) -> bool:
    queue, seen = deque([first]), {first}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if nxt == second:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _operational_grounding_descriptors(
    gold_spec: dict[str, Any],
    gold_grounded: dict[str, Any],
    predicted_view: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
    *,
    ownership_spec: dict[str, Any] | None = None,
) -> Counter[str]:
    supported: Counter[str] = Counter()
    gold_view = _support_view(
        gold_spec,
        gold_grounded,
        task,
        asg,
        derive_bindings=True,
    )

    for count in _family(gold_view["obligations"], "count"):
        if _chain_supports_count(count, predicted_view, task, asg):
            supported[_key(count.get("target"))] += 1

    for preference in _family(gold_view["obligations"], "preference"):
        if (
            preference.get("kind") == "via"
            and _chain_supports_preferred_via(preference, predicted_view, task, asg)
        ):
            supported[_key(preference.get("target"))] += 1

    for family, role in (("goal", "goal"), ("via", "via")):
        spec_items = _target_items(gold_spec, family)
        grounded_items = _target_items(gold_grounded, family)
        owned_items, _ = _owned_grounded_items(
            spec_items,
            _target_items(ownership_spec or gold_spec, family),
            grounded_items,
            family,
        )
        for target, explicit in owned_items:
            binding = _binding(explicit)
            if binding is None or not any(
                record.get("binding") == binding
                for record in predicted_view["chain"]
                if record.get("role") == role
            ):
                continue
            for descriptor, _ in _target_binding_records(target, binding, task, asg):
                supported[descriptor] += 1
    return supported


def _binding_records(
    spec: dict[str, Any],
    grounded: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
    *,
    ownership_spec: dict[str, Any] | None = None,
) -> tuple[
    list[tuple[str, tuple[str, tuple[int, ...]] | None]],
    list[dict[str, Any]],
]:
    records: list[tuple[str, tuple[str, tuple[int, ...]] | None]] = []
    diagnostics: list[dict[str, Any]] = []
    for family in ("goal", "via", "avoid", "count", "carry", "prefer"):
        spec_items = _target_items(spec, family)
        grounded_items = _target_items(grounded, family)
        owned_items, issues = _owned_grounded_items(
            spec_items,
            _target_items(ownership_spec or spec, family),
            grounded_items,
            family,
        )
        diagnostics.extend(issues)
        for item, explicit in owned_items:
            binding = _binding(explicit)
            records.extend(_target_binding_records(item, binding, task, asg))
    return sorted(records, key=lambda item: (item[0], _key(item[1]))), diagnostics


def _owned_grounded_items(
    canonical_items: list[dict[str, Any]],
    ownership_items: list[dict[str, Any]],
    grounded_items: list[dict[str, Any]],
    family: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], list[dict[str, Any]]]:
    """Carry original semantic ownership through independent canonical sorting."""

    owned: dict[str, deque[tuple[int, dict[str, Any] | None]]] = defaultdict(deque)
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(ownership_items):
        key = _ownership_key(family, item)
        grounded = grounded_items[index] if index < len(grounded_items) else None
        owned[key].append((index, grounded))

    for key, occurrences in owned.items():
        if len(occurrences) > 1:
            diagnostics.append({
                "family": family,
                "kind": "duplicate_semantic_key",
                "semantic_key": key,
                "occurrences": len(occurrences),
                "resolution": "stable_occurrence_order",
            })

    aligned: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for index, item in enumerate(canonical_items):
        key = _ownership_key(family, item)
        if owned[key]:
            _source_index, grounded = owned[key].popleft()
        else:
            grounded = None
            diagnostics.append({
                "family": family,
                "kind": "canonical_item_without_owner",
                "canonical_index": index,
                "semantic_key": key,
            })
        aligned.append((item, grounded))

    for key, occurrences in owned.items():
        for source_index, _grounded in occurrences:
            diagnostics.append({
                "family": family,
                "kind": "unconsumed_owned_item",
                "source_index": source_index,
                "semantic_key": key,
            })
    if len(grounded_items) > len(ownership_items):
        diagnostics.append({
            "family": family,
            "kind": "grounding_without_semantic_owner",
            "grounding_indices": list(range(len(ownership_items), len(grounded_items))),
        })
    return aligned, diagnostics


def _ownership_key(family: str, item: dict[str, Any]) -> str:
    semantic: dict[str, Any] = {"family": family, "target": _target(item)}
    for field in (
        "name",
        "op",
        "count",
        "after",
        "until",
        "preference",
        "weight",
    ):
        if field in item:
            semantic[field] = item[field]
    return _key(semantic)


def _target_binding_records(
    target: dict[str, Any], binding: tuple[str, tuple[int, ...]] | None,
    task: dict[str, Any], asg: ASG,
) -> list[tuple[str, tuple[str, tuple[int, ...]] | None]]:
    normalized = _target(target)
    if normalized is None:
        return []
    out = [(_target_descriptor(normalized, task, asg), binding)]
    for qualifier in normalized.get("qualifiers", []):
        nested = qualifier.get("regions", []) if qualifier.get("kind") == "between_regions" else [qualifier.get("target")]
        for anchor in nested:
            if not isinstance(anchor, dict) or anchor.get("kind") in {"start", "pos"}:
                continue
            nested_binding = _derived_binding(anchor, task, asg) if binding is not None else None
            out.extend(_target_binding_records(anchor, nested_binding, task, asg))
    return out


def _target_descriptor(
    target: dict[str, Any],
    task: dict[str, Any],
    asg: ASG,
) -> str:
    """Stable scene-semantic slot key for G-ID alignment."""
    resolved = _resolve_target(target, task, asg)
    if resolved is not None:
        kind, ids = resolved
        return _key({"kind": kind, "denotation_ids": list(ids)})
    return _key(_target(target))


def _derived_binding(target: dict[str, Any], task: dict[str, Any], asg: ASG) -> tuple[str, tuple[int, ...]] | None:
    resolved = _resolve_target(target, task, asg)
    return (resolved[0], resolved[1]) if resolved is not None else None


def _resolve_target(target: Any, task: dict[str, Any], asg: ASG) -> tuple[str, tuple[int, ...]] | None:
    if not isinstance(target, dict):
        return None
    try:
        selector = _selector(target)
        grounded = ground_target(asg, selector, 1, start_target=_start_selector(task))
    except (TypeError, ValueError):
        return None
    return grounded.kind, tuple(sorted(grounded.ids))


def _selector(target: dict[str, Any]) -> TargetSelector:
    if target.get("kind") == "start":
        return TargetSelector("pos", "coord", ())
    qualifiers = []
    for item in target.get("qualifiers", []):
        kind = str(item.get("kind", ""))
        if kind == "between_regions":
            regions = item.get("regions", [])
            if len(regions) == 2:
                qualifiers.append(TargetQualifier(kind, _selector(regions[0]), second_target=_selector(regions[1])))
        else:
            anchor = item.get("target")
            if isinstance(anchor, str) and anchor.upper() == "START":
                anchor_selector = TargetSelector("pos", "coord", ())
            elif isinstance(anchor, dict):
                anchor_selector = _selector(anchor)
            else:
                continue
            raw_domain = item.get("comparison_domain")
            domain = (
                raw_domain.get("kind")
                if isinstance(raw_domain, dict)
                else raw_domain
            )
            qualifiers.append(TargetQualifier(
                kind,
                anchor_selector,
                reference_mode=str(item.get("reference_mode", "start" if anchor == "START" or (isinstance(anchor, dict) and anchor.get("kind") == "start") else "unique")),
                comparison_domain=(
                    str(domain)
                    if domain is not None
                    else "reference_level"
                    if kind in {"nearest_to", "farthest_from"}
                    else None
                ),
            ))
    values = tuple(target.get("values", []))
    selector = str(target.get("selector", "category"))
    if selector == "category" and not values:
        selector = "any"
    return TargetSelector(str(target.get("kind", "obj")), selector, values, tuple(qualifiers))


def _start_selector(task: dict[str, Any]) -> TargetSelector | None:
    start = task.get("robot_position")
    if not isinstance(start, dict):
        return None
    return TargetSelector("pos", "coord", (
        float(start.get("x", 0.0)), float(start.get("floor_height", 0.0)), float(start.get("y", 0.0))
    ))


def _start_region(task: dict[str, Any], asg: ASG) -> int | None:
    start = task.get("robot_position")
    if not isinstance(start, dict):
        return None
    point = Point2(float(start.get("x", 0.0)), float(start.get("y", 0.0)))
    return next((region.index for region in asg.regions if point_in_region(point, region)), None)


def _target_regions(resolved: tuple[str, tuple[int, ...]], asg: ASG) -> set[int]:
    kind, ids = resolved
    if kind == "region":
        return set(ids)
    if kind == "obj":
        return {obj.region_index for idx in ids if (obj := asg.get_object(idx)) is not None}
    if kind == "portal":
        return {
            region for idx in ids if (portal := asg.get_portal(idx)) is not None
            for region in (portal.region_a, portal.region_b)
        }
    return set()


def _target(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    target = {
        key: copy.deepcopy(value[key])
        for key in ("kind", "selector", "values", "qualifiers")
        if key in value
    }
    target.pop("selector", None)
    if not target.get("kind") or (
        not target.get("values")
        and not target.get("qualifiers")
        and value.get("selector") != "any"
    ):
        return None
    return target


def _binding(value: Any) -> tuple[str, tuple[int, ...]] | None:
    if not isinstance(value, dict):
        return None
    ids = value.get("grounded_ids", value.get("ids", value.get("id")))
    values = ids if isinstance(ids, list) else [ids]
    try:
        normalized = tuple(sorted(int(item) for item in values if item is not None))
    except (TypeError, ValueError):
        return None
    return (str(value.get("kind", "")).lower(), normalized) if normalized else None


def _target_items(source: dict[str, Any], family: str) -> list[dict[str, Any]]:
    if family == "goal":
        return [source["goal"]] if isinstance(source.get("goal"), dict) else []
    return _items(source, family)


def _event(value: Any, events: dict[str, Any]) -> Any:
    text = str(value)
    return copy.deepcopy(events.get(text, {"symbol": text.upper() if text.upper() in {"START", "GOAL"} else text}))


def _family(items: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [item for item in items if item.get("family") == family]


def _items(source: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = source.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _valid_obligation(item: dict[str, Any]) -> bool:
    return item.get("family") in SEMANTIC_FAMILIES and (
        item.get("target") is not None
        or item.get("family") in {"order", "width", "preference"}
    )


def _field_projection(items: list[dict[str, Any]], keys: tuple[str, ...]) -> set[str]:
    projected = set()
    for item in items:
        values = {key: item[key] for key in keys if key in item}
        if values:
            projected.add(_key({"family": item.get("family"), **values}))
    return projected


def _bidirectional_score(precision_hits: int, predicted: int, recall_hits: int, gold: int) -> dict[str, float | int]:
    precision = precision_hits / predicted if predicted else (1.0 if not gold else 0.0)
    recall = recall_hits / gold if gold else 1.0
    return {
        "precision_hits": precision_hits,
        "recall_hits": recall_hits,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _score(hits: int, predicted: int, gold: int) -> dict[str, float | int]:
    return _bidirectional_score(hits, predicted, hits, gold)


def _key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
