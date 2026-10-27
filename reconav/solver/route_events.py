"""Shared geometric route-event primitives used by runtime and replay."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reconav.scene.asg import ASG
from reconav.scene.physical import (
    OBJECT_ENCOUNTER_CLEARANCE_M,
    SEMANTIC_DISTANCE_EPS_M,
    within_semantic_clearance,
)
from reconav.solver.segment_events import entry_fractions, EVENT_POSITION_EPS
from reconav.solver.cancellation import check_local_search_budget


# Route search asks the same object/region geometry question on many successor
# edges.  Cache only immutable ASG-derived footprints, keyed by ASG identity;
# event state and clearance remain per call.
_OBJECT_FOOTPRINT_CACHE: dict[
    tuple[int, int, int], BaseGeometry | None
] = {}
_OBJECT_ZONE_CACHE: dict[
    tuple[int, int, tuple[int, ...]], BaseGeometry | None
] = {}
_OBJECT_EXPANDED_ZONE_CACHE: dict[
    tuple[int, int, tuple[int, ...], float], BaseGeometry | None
] = {}
_ASG_CACHE_REFS: dict[int, weakref.ReferenceType[ASG]] = {}


def _clear_asg_geometry_cache(asg_identity: int) -> None:
    for cache in (
        _OBJECT_FOOTPRINT_CACHE,
        _OBJECT_ZONE_CACHE,
        _OBJECT_EXPANDED_ZONE_CACHE,
    ):
        for key in [key for key in cache if key[0] == asg_identity]:
            del cache[key]


def _register_cached_asg(asg: ASG) -> int:
    """Make an ``id(asg)`` cache key safe against Python object-id reuse."""

    identity = id(asg)
    current = _ASG_CACHE_REFS.get(identity)
    if current is not None and current() is asg:
        return identity
    if current is not None:
        _clear_asg_geometry_cache(identity)

    def cleanup(reference: weakref.ReferenceType[ASG]) -> None:
        if _ASG_CACHE_REFS.get(identity) is reference:
            _ASG_CACHE_REFS.pop(identity, None)
            _clear_asg_geometry_cache(identity)

    _ASG_CACHE_REFS[identity] = weakref.ref(asg, cleanup)
    return identity


@dataclass(frozen=True)
class ObjectEncounter:
    object_id: int
    waypoint_index: int
    segment_fraction: float


def object_encounter_entries(
    asg: ASG,
    object_ids: Iterable[int],
    waypoints: Sequence[Any],
    regions: Sequence[int],
    *,
    first_segment: int = 1,
    last_segment: int | None = None,
    clearance_m: float = OBJECT_ENCOUNTER_CLEARANCE_M,
    excluded_terminal_object_ids: Iterable[int] = (),
    include_terminal_event: bool = True,
) -> tuple[ObjectEncounter, ...]:
    """Return entries into object encounter zones over complete route segments.

    ``first_segment`` and ``last_segment`` are destination waypoint indices.
    Objects already occupied at the beginning of the interval are not entries;
    leaving and later re-entering produces a new occurrence.  Footprints are
    use the retained collision footprint and are selected using the route's
    current region so XY overlap across levels never becomes an encounter.
    """

    if len(waypoints) < 2 or len(regions) != len(waypoints):
        return ()
    asg_identity = _register_cached_asg(asg)
    first = max(1, int(first_segment))
    last = min(
        len(waypoints) - 1,
        len(waypoints) - 1 if last_segment is None else int(last_segment),
    )
    if first > last:
        return ()

    ids = tuple(sorted({int(value) for value in object_ids}))
    def shape(object_id: int, region_id: int) -> BaseGeometry | None:
        key = asg_identity, object_id, region_id
        if key in _OBJECT_FOOTPRINT_CACHE:
            return _OBJECT_FOOTPRINT_CACHE[key]
        obj = asg.get_object(object_id)
        if obj is None:
            raise ValueError(f"route encounter references missing object {object_id}")
        if (
            int(region_id) != int(obj.region_index)
            and int(region_id) not in obj.body_band_footprint_2d_by_region
        ):
            _OBJECT_FOOTPRINT_CACHE[key] = None
            return None
        polygons = obj.navigation_footprint_2d()
        if not polygons:
            _OBJECT_FOOTPRINT_CACHE[key] = None
            return None
        value = unary_union([polygon.to_shapely() for polygon in polygons])
        _OBJECT_FOOTPRINT_CACHE[key] = value
        return value

    def zone(object_id: int, region_ids: tuple[int, ...]) -> BaseGeometry | None:
        normalized = tuple(dict.fromkeys(region_ids))
        key = asg_identity, object_id, normalized
        if key in _OBJECT_ZONE_CACHE:
            return _OBJECT_ZONE_CACHE[key]
        footprints = [
            value
            for region_id in normalized
            if (value := shape(object_id, region_id)) is not None
        ]
        value = unary_union(footprints) if footprints else None
        _OBJECT_ZONE_CACHE[key] = value
        return value

    def expanded_zone(
        object_id: int,
        region_ids: tuple[int, ...],
    ) -> BaseGeometry | None:
        normalized = tuple(dict.fromkeys(region_ids))
        key = asg_identity, object_id, normalized, round(float(clearance_m), 6)
        if key in _OBJECT_EXPANDED_ZONE_CACHE:
            return _OBJECT_EXPANDED_ZONE_CACHE[key]
        value = zone(object_id, normalized)
        expanded = value.buffer(clearance_m) if value is not None else None
        _OBJECT_EXPANDED_ZONE_CACHE[key] = expanded
        return expanded

    active: set[int] = set()
    start_point = _point(waypoints[first - 1])
    start_region = int(regions[first - 1])
    for object_id in ids:
        check_local_search_budget()
        footprint = shape(object_id, start_region)
        if footprint is not None and within_semantic_clearance(start_point.distance(footprint), clearance_m):
            active.add(object_id)

    entries: list[ObjectEncounter] = []
    for index in range(first, last + 1):
        check_local_search_budget()
        before = _point(waypoints[index - 1])
        after = _point(waypoints[index])
        segment = LineString([before, after])
        source_region_id = int(regions[index - 1])
        target_region_id = int(regions[index])
        source_region = asg.get_region(source_region_id)
        target_region = asg.get_region(target_region_id)
        same_level = (
            source_region is not None
            and target_region is not None
            and int(source_region.level) == int(target_region.level)
        )
        region_ids = (source_region_id, target_region_id)
        current: set[int] = set()
        crossed: list[tuple[float, int]] = []
        for object_id in ids:
            check_local_search_budget()
            # A region transition on one level represents a continuous planar
            # segment, so both body-band footprints participate.  A cross-level
            # transition does not: its XY projection is not physical travel on
            # either floor.  In that case only occupancy at the destination
            # point may create a new target-level encounter; the source-level
            # portion was already handled by the preceding same-level segment.
            encounter_zone = (
                zone(object_id, region_ids)
                if same_level
                else shape(object_id, target_region_id)
            )
            if encounter_zone is None:
                continue
            end_inside = within_semantic_clearance(after.distance(encounter_zone), clearance_m)
            intersects = (
                within_semantic_clearance(segment.distance(encounter_zone), clearance_m)
                if same_level
                else end_inside
            )
            if end_inside:
                current.add(object_id)
            if intersects:
                fractions = (
                    entry_fractions(
                        segment,
                        encounter_zone,
                        clearance_m + SEMANTIC_DISTANCE_EPS_M,
                        initially_inside=object_id in active,
                    )
                    if same_level else ((1.0,) if object_id not in active else ())
                )
                crossed.extend((fraction, object_id) for fraction in fractions)
        entries.extend(
            ObjectEncounter(object_id, index, fraction)
            for fraction, object_id in sorted(crossed)
        )
        active = current
    # Only an occurrence exactly at the boundary may be excluded. Being the
    # eventual target is never permission to erase an earlier entry episode.
    excluded = frozenset(int(value) for value in excluded_terminal_object_ids)
    return tuple(
        event for event in entries
        if not (event.waypoint_index == last
                and event.segment_fraction >= 1.0 - EVENT_POSITION_EPS
                and (not include_terminal_event or event.object_id in excluded))
    )


def object_ids_encountered(
    asg: ASG,
    object_ids: Iterable[int],
    waypoints: Sequence[Any],
    regions: Sequence[int],
    **kwargs: Any,
) -> frozenset[int]:
    return frozenset(
        item.object_id
        for item in object_encounter_entries(
            asg,
            object_ids,
            waypoints,
            regions,
            **kwargs,
        )
    )


def _point(value: Any) -> Point:
    if isinstance(value, dict):
        return Point(float(value["x"]), float(value["z"]))
    if hasattr(value, "x") and hasattr(value, "z"):
        return Point(float(value.x), float(value.z))
    return Point(float(value[0]), float(value[2]))
