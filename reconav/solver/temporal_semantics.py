"""Pure temporal-scope rules shared by route search and route validation."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from itertools import product
from typing import Any


def transition_scope_active(
    after: str,
    until: str,
    progress: Collection[str],
    target_names: Collection[str],
    *,
    include_until_transition: bool,
) -> bool:
    """Return whether a constraint is active on the next transition."""
    if after not in progress or after in target_names:
        return False
    if until in progress:
        return False
    return include_until_transition or until not in target_names


def logical_scope_is_ordered(after_seq: int, until_seq: int) -> bool:
    return after_seq < until_seq


def transition_in_waypoint_scope(
    destination_index: int,
    after_waypoint: int,
    until_waypoint: int,
    *,
    include_until_transition: bool,
) -> bool:
    if destination_index <= after_waypoint:
        return False
    return (
        destination_index <= until_waypoint
        if include_until_transition
        else destination_index < until_waypoint
    )


def order_satisfied_over_occurrences(
    kind: str,
    first: str,
    second: str,
    names: Collection[str],
    events: Sequence[str],
) -> tuple[bool, dict[str, Any]]:
    """Evaluate an order existentially over all occurrences in one event trace."""
    first_positions = [index for index, name in enumerate(events) if name == first]
    second_positions = [index for index, name in enumerate(events) if name == second]
    required = set(names)
    valid_pairs: list[tuple[int, int]] = []
    for first_position in first_positions:
        for second_position in second_positions:
            if first_position >= second_position:
                continue
            if kind == "before":
                valid_pairs.append((first_position, second_position))
            elif kind == "next" and second_position == first_position + 1:
                valid_pairs.append((first_position, second_position))
            elif kind == "between_only":
                middle = events[first_position + 1 : second_position]
                if required.issubset(middle) and all(
                    name in required for name in middle
                ):
                    valid_pairs.append((first_position, second_position))
            elif kind == "between_without" and all(
                name not in required
                for name in events[first_position + 1 : second_position]
            ):
                valid_pairs.append((first_position, second_position))
    if kind not in {"before", "next", "between_only", "between_without"}:
        raise ValueError(f"unsupported order constraint kind: {kind}")
    relevant = {first, second, *required}
    return bool(valid_pairs), {
        "event_trace": list(events),
        "candidate_occurrences": {
            name: [index for index, event in enumerate(events) if event == name]
            for name in sorted(relevant)
        },
        "valid_witness_pairs": [list(pair) for pair in valid_pairs],
    }


def minimum_soft_order_penalty(
    orders: Sequence[Mapping[str, Any]],
    events: Sequence[str],
    soft_event_names: Collection[str],
) -> dict[str, Any]:
    """Choose one occurrence per soft event and minimize its order penalty."""
    referenced_soft = sorted(
        {
            name
            for order in orders
            for name in _order_references(order)
            if name in soft_event_names and name in events
        }
    )
    positions = {
        name: tuple(index for index, event in enumerate(events) if event == name)
        for name in referenced_soft
    }
    assignments = product(*(positions[name] for name in referenced_soft))
    if not referenced_soft:
        assignments = [()]
    best: dict[str, Any] | None = None
    for selected in assignments:
        assignment = dict(zip(referenced_soft, selected))
        filtered = tuple(
            event
            for index, event in enumerate(events)
            if event not in assignment or assignment[event] == index
        )
        results = []
        penalty = 0.0
        for order in orders:
            references = _order_references(order)
            applicable = all(reference in events for reference in references)
            if applicable:
                satisfied, witness = order_satisfied_over_occurrences(
                    str(order["kind"]),
                    str(order["first"]),
                    str(order["second"]),
                    [str(name) for name in order.get("names", [])],
                    filtered,
                )
            else:
                satisfied = True
                witness = {
                    "event_trace": list(filtered),
                    "candidate_occurrences": {},
                    "valid_witness_pairs": [],
                    "not_penalized_missing_reference": True,
                }
            if not satisfied:
                penalty += 1.0
            results.append(
                {
                    "applicable": applicable,
                    "satisfied": satisfied,
                    "witness": witness,
                }
            )
        candidate = {
            "penalty": penalty,
            "selected_occurrences": assignment,
            "candidate_occurrences": {
                name: list(values) for name, values in positions.items()
            },
            "results": results,
        }
        if best is None or (penalty, selected) < (
            float(best["penalty"]),
            tuple(best["selected_occurrences"][name] for name in referenced_soft),
        ):
            best = candidate
    assert best is not None
    return best


def minimum_soft_event_penalty(
    vias: Sequence[Mapping[str, Any]],
    required_orders: Sequence[Mapping[str, Any]],
    soft_orders: Sequence[Mapping[str, Any]],
    events: Sequence[str],
) -> dict[str, Any]:
    """Choose optional event occurrences under conditional hard orders.

    A hard order that mentions an optional VIA constrains which spatial
    occurrence may represent that VIA.  When no occurrence is admissible, the
    VIA remains unselected and pays one preference penalty; the conditional
    order is then vacuous, matching :class:`PreferRuntime`.
    """
    names = sorted(str(item["name"]) for item in vias)
    positions = {
        name: tuple(index for index, event in enumerate(events) if event == name)
        for name in names
    }
    assignments = product(*((None, *positions[name]) for name in names))
    if not names:
        assignments = [()]
    best: dict[str, Any] | None = None
    for selected in assignments:
        assignment = dict(zip(names, selected))
        filtered = tuple(
            event
            for index, event in enumerate(events)
            if event not in assignment or assignment[event] == index
        )
        required_results = [_evaluate_order(item, filtered) for item in required_orders]
        if any(not item["satisfied"] for item in required_results):
            continue
        soft_results = [_evaluate_order(item, filtered) for item in soft_orders]
        penalty = sum(1.0 for name in names if assignment[name] is None)
        penalty += sum(
            1.0
            for result in soft_results
            if not result["satisfied"]
        )
        candidate = {
            "penalty": penalty,
            "selected_occurrences": assignment,
            "candidate_occurrences": {
                name: list(values) for name, values in positions.items()
            },
            "required_results": required_results,
            "soft_order_results": soft_results,
            "event_trace": list(filtered),
        }
        key = (penalty, tuple(-1 if value is None else value for value in selected))
        if best is None or key < best["_key"]:
            best = {**candidate, "_key": key}
    assert best is not None
    best.pop("_key")
    return best


def _evaluate_order(order: Mapping[str, Any], events: Sequence[str]) -> dict[str, Any]:
    references = _order_references(order)
    applicable = all(reference in events for reference in references)
    if not applicable:
        return {
            "applicable": False,
            "satisfied": True,
            "witness": {
                "event_trace": list(events),
                "candidate_occurrences": {},
                "valid_witness_pairs": [],
                "conditional_on_unselected_optional_event": True,
            },
        }
    satisfied, witness = order_satisfied_over_occurrences(
        str(order["kind"]),
        str(order["first"]),
        str(order["second"]),
        [str(name) for name in order.get("names", [])],
        events,
    )
    return {"applicable": True, "satisfied": satisfied, "witness": witness}


def _order_references(order: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(order["first"]),
        str(order["second"]),
        *(str(name) for name in order.get("names", [])),
    )
