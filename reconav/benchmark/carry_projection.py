"""Project acquisition events from predicted carry specifications."""

from __future__ import annotations

from typing import Any






def acquisition_only_spec_names(specification: dict[str, Any]) -> set[str]:
    """Return predicted VIA names used only by deterministic CARRY elaboration."""
    widths = [
        item
        for item in _items(specification, "width")
        if item.get("derived_from") == "carry"
    ]
    intervals = {
        (_source(item), str(item.get("until", "GOAL"))) for item in widths
    }
    return {
        source
        for source, _until in intervals
        if not _has_independent_event_use(
            specification,
            source,
            intervals,
            ignore_generator_carry_order=False,
        )
    }


def _has_independent_event_use(
    specification: dict[str, Any],
    source: str,
    operational_intervals: set[tuple[str, str]] | frozenset[tuple[str, str]],
    *,
    ignore_generator_carry_order: bool,
) -> bool:
    for order in _items(specification, "order"):
        pair = (str(order.get("first", "")), str(order.get("second", "")))
        if (
            ignore_generator_carry_order
            and pair in operational_intervals
            and order.get("kind") == "before"
        ):
            continue
        if source in _order_names(order):
            return True
    if any(source in _order_names(item) for item in _items(specification, "prefer_order")):
        return True
    for family in ("avoid", "count", "prefer"):
        for item in _items(specification, family):
            if str(item.get("after", "")) == source or str(item.get("until", "")) == source:
                return True
            if item.get("preference") == "via" and str(item.get("name", "")) == source:
                return True
    for width in _items(specification, "width"):
        if width.get("derived_from") == "carry" or (
            _source(width), str(width.get("until", "GOAL"))
        ) in operational_intervals:
            continue
        if source in {
            str(width.get("milestone", "")),
            str(width.get("after", "")),
            str(width.get("until", "")),
        }:
            return True
    return False


def _order_names(item: dict[str, Any]) -> set[str]:
    names = {str(item.get("first", "")), str(item.get("second", ""))}
    values = item.get("names")
    if isinstance(values, list):
        names.update(str(value) for value in values)
    return names


def _source(item: dict[str, Any]) -> str:
    return str(item.get("milestone", item.get("after", "")))


def _items(value: dict[str, Any], field: str) -> list[dict[str, Any]]:
    items = value.get(field)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
