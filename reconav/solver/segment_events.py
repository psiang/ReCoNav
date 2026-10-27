"""Exact line/round-offset intersections for semantic encounter events.

This does not use the conservative robot collision buffer.  In particular a
semantic 1 m threshold must not grow with raster or circle-approximation error.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

from shapely.geometry import LineString, Point
from reconav.solver.cancellation import check_local_search_budget

EVENT_POSITION_EPS = 1e-9
EVENT_SEMANTICS_VERSION = "open-occurrence-position-v2"


def _parts(value):
    check_local_search_budget()
    if hasattr(value, "geoms"):
        for child in value.geoms:
            yield from _parts(child)
    else:
        yield value


def _linear_interval(a: float, b: float, lo: float, hi: float):
    if abs(b) <= 1e-15:
        return (0.0, 1.0) if lo - 1e-12 <= a <= hi + 1e-12 else None
    first, second = (lo - a) / b, (hi - a) / b
    first, second = max(0.0, min(first, second)), min(1.0, max(first, second))
    return (first, second) if first <= second + 1e-12 else None


def _circle_interval(px, py, dx, dy, cx, cy, radius):
    ux, uy = px - cx, py - cy
    a = dx * dx + dy * dy
    b = 2.0 * (ux * dx + uy * dy)
    c = ux * ux + uy * uy - radius * radius
    disc = b * b - 4.0 * a * c
    tolerance = 1e-12 * max(1.0, b * b, abs(4.0 * a * c))
    if disc < -tolerance:
        return None
    root = math.sqrt(max(0.0, disc))
    lo, hi = max(0.0, (-b - root) / (2.0 * a)), min(1.0, (-b + root) / (2.0 * a))
    return (lo, hi) if lo <= hi + 1e-12 else None


def merge_intervals(intervals: Iterable[tuple[float, float]]):
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        check_local_search_budget()
        lo, hi = max(0.0, lo), min(1.0, hi)
        if lo > hi + EVENT_POSITION_EPS:
            continue
        if merged and lo <= merged[-1][1] + EVENT_POSITION_EPS:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def distance_intervals(segment: LineString, geometry, radius: float):
    """Closed t-intervals where d(segment(t), geometry) <= radius, 0<=t<=1.

    The round offset is the original polygon interior plus capsules around
    its edges. Analytic circle/strip intersections avoid polygonal-buffer
    endpoint errors. Disconnected pieces and holes can yield several entries.
    """
    check_local_search_budget()
    if geometry is None or geometry.is_empty:
        return ()
    if radius < 0 or not math.isfinite(radius):
        raise ValueError("encounter radius must be finite and nonnegative")
    (px, py), (qx, qy) = segment.coords[0], segment.coords[-1]
    dx, dy = qx - px, qy - py
    if dx * dx + dy * dy <= 1e-24:
        return ((0.0, 1.0),) if Point(px, py).distance(geometry) <= radius else ()
    if segment.distance(geometry) > radius + 1e-10:
        return ()
    intervals: list[tuple[float, float]] = []
    # The portion inside an original polygon is always inside its offset.
    for part in _parts(segment.intersection(geometry)):
        check_local_search_budget()
        if part.is_empty:
            continue
        coords = list(getattr(part, "coords", ()))
        if coords:
            values = [float(segment.project(Point(x, y), normalized=True)) for x, y in coords]
            intervals.append((min(values), max(values)))
    for part in _parts(geometry):
        check_local_search_budget()
        if part.is_empty:
            continue
        if part.geom_type == "Polygon":
            rings = [part.exterior, *part.interiors]
        elif part.geom_type in {"LineString", "LinearRing"}:
            rings = [part]
        elif part.geom_type == "Point":
            value = _circle_interval(px, py, dx, dy, part.x, part.y, radius)
            if value is not None:
                intervals.append(value)
            continue
        else:
            continue
        for ring in rings:
            check_local_search_budget()
            coords = list(ring.coords)
            for (ax, ay), (bx, by) in zip(coords, coords[1:]):
                check_local_search_budget()
                # Each round endpoint cap.
                for cx, cy in ((ax, ay), (bx, by)):
                    value = _circle_interval(px, py, dx, dy, cx, cy, radius)
                    if value is not None:
                        intervals.append(value)
                ex, ey = bx - ax, by - ay
                length = math.hypot(ex, ey)
                if length <= 1e-12:
                    continue
                tx, ty = ex / length, ey / length
                ux, uy = px - ax, py - ay
                along = _linear_interval(ux * tx + uy * ty, dx * tx + dy * ty, 0.0, length)
                normal = _linear_interval(-ux * ty + uy * tx, -dx * ty + dy * tx, -radius, radius)
                if along is not None and normal is not None:
                    lo, hi = max(along[0], normal[0]), min(along[1], normal[1])
                    if lo <= hi + 1e-12:
                        intervals.append((lo, hi))
    return merge_intervals(intervals)


def entry_fractions(segment, geometry, radius, *, initially_inside: bool):
    intervals = distance_intervals(segment, geometry, radius)
    return tuple(lo for index, (lo, _hi) in enumerate(intervals)
                 if not (index == 0 and lo <= EVENT_POSITION_EPS and initially_inside))
