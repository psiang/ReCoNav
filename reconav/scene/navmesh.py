"""Region containment, planar distances and route waypoint serialization."""

from __future__ import annotations

import math

from typing import Any

from reconav.scene.asg import ASGRegionEntity
from reconav.scene.geometry import Point2

OBJECT_GOAL_RADIUS_M = 1.0


def point_in_region(point: Point2, region: ASGRegionEntity) -> bool:
    return any(point_in_ring(point, poly.vertices) for poly in region.polygon_2d)


def distance_to_polygons(point: Point2, polygons) -> float:
    best = float("inf")
    for poly in polygons:
        vertices = list(poly.vertices)
        if len(vertices) < 3:
            continue
        if point_in_ring(point, vertices):
            return 0.0
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            best = min(best, point_to_segment_distance(point, a, b))
    return best


def point_to_segment_distance(point: Point2, a: Point2, b: Point2) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(point.x - a.x, point.y - a.y)
    t = max(0.0, min(1.0, ((point.x - a.x) * dx + (point.y - a.y) * dy) / denom))
    return math.hypot(point.x - (a.x + dx * t), point.y - (a.y + dy * t))


def waypoints(points: list[Any]) -> list[dict[str, float]]:
    out = [{"x": float(p.x), "y": float(p.y) + 1.2, "z": float(p.z)} for p in points]
    for i, point in enumerate(out):
        a = point if i < len(out) - 1 else out[max(i - 1, 0)]
        b = out[i + 1] if i < len(out) - 1 else point
        dx, dz = b["x"] - a["x"], b["z"] - a["z"]
        point["heading_rad"] = math.atan2(dz, dx) if dx * dx + dz * dz > 1e-12 else 0.0
    return out


def point_in_ring(point: Point2, ring) -> bool:
    inside = False
    pts = list(ring)
    for i, current in enumerate(pts):
        previous = pts[i - 1]
        denom = previous.y - current.y
        if abs(denom) <= 1e-12:
            continue
        if (current.y > point.y) != (previous.y > point.y):
            x_at_y = (previous.x - current.x) * (point.y - current.y) / denom + current.x
            if point.x < x_at_y:
                inside = not inside
    return inside
