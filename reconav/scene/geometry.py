"""
Geometric primitives shared by ASGs and route solving.

2D is the XZ plane modeled as x–y math. ``Polygon2.new`` / ``ConvexPolygon2.new``
build a CCW exterior ring (Shapely ``orient`` on a temporary polygon).

``Halfspace2``: ``a*x + b*y + c <= 0`` is the feasible side (inward for a CCW boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

try:
    from shapely import orient as orient_polygon  # Shapely ≥2.1
except ImportError:  # pragma: no cover
    from shapely.geometry.polygon import orient as orient_polygon  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point2:
    x: float
    y: float


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


def distance(first: Point3 | List[Point3], second: Point3 | None = None) -> float:
    """
    Euclidean distance: ``distance(p, q)`` for two ``Point3``, or polyline length
    ``distance([p0, p1, ...])`` for a chain of ``Point3``.
    """
    if second is not None:
        if not isinstance(first, Point3) or not isinstance(second, Point3):
            raise TypeError(
                f"distance: expected two Point3, got {type(first).__name__} and {type(second).__name__}"
            )
        dx, dy, dz = second.x - first.x, second.y - first.y, second.z - first.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    if isinstance(first, list):
        if len(first) < 2:
            return 0.0
        if not first or not isinstance(first[0], Point3):
            raise TypeError("distance: chain must be List[Point3]")
        s = 0.0
        for i in range(len(first) - 1):
            a, b = first[i], first[i + 1]
            dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
            s += math.sqrt(dx * dx + dy * dy + dz * dz)
        return s
    raise TypeError("distance: pass two Point3, or one list of Point3")


# ---------------------------------------------------------------------------
# 2D halfspaces (inward normals for CCW polygons)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Halfspace2:
    """Closed affine halfspace ``a*x + b*y + c <= 0`` (``(a,b)`` unit after ``from_directed_edge``)."""

    a: float
    b: float
    c: float

    def normalized(self) -> Halfspace2:
        n = math.hypot(self.a, self.b)
        if n <= 1e-15:
            raise ValueError("degenerate halfspace normal")
        return Halfspace2(self.a / n, self.b / n, self.c / n)

    @staticmethod
    def from_directed_edge(p: Point2, q: Point2) -> Halfspace2:
        """Halfspace on the CCW left of directed edge ``p → q`` (inward for CCW polygon boundary)."""
        dx, dy = q.x - p.x, q.y - p.y
        a, b = dy, -dx
        c = -(a * p.x + b * p.y)
        return Halfspace2(a, b, c).normalized()


# ---------------------------------------------------------------------------
# Simple polygon (2D) — one ring; Shapely only in ``new`` / ``to_shapely``
# ---------------------------------------------------------------------------


def _polygon_from_vertices(vertices: Sequence[Point2]) -> Polygon:
    shell = _ring_coords_from_vertices(vertices)
    return Polygon(shell)


def _ring_coords_from_vertices(vertices: Sequence[Point2]) -> List[Tuple[float, float]]:
    coords = [(float(p.x), float(p.y)) for p in vertices]
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    return coords


def _vertices_from_ring(poly: Polygon) -> Tuple[Point2, ...]:
    arr = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
    return tuple(Point2(float(arr[i, 0]), float(arr[i, 1])) for i in range(arr.shape[0]))


def _vertices_to_points2(vertices: Union[Sequence[Point2], np.ndarray]) -> List[Point2]:
    if isinstance(vertices, np.ndarray):
        arr = np.asarray(vertices, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("vertices array must have shape (n, 2)")
        return [Point2(float(arr[i, 0]), float(arr[i, 1])) for i in range(arr.shape[0])]
    out: List[Point2] = []
    for p in vertices:
        if isinstance(p, Point2):
            out.append(p)
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            out.append(Point2(float(p[0]), float(p[1])))
        else:
            raise TypeError(f"vertex must be Point2 or length-2 sequence, got {type(p)!r}")
    return out


@dataclass(frozen=True)
class Polygon2:
    """Simple 2D polygon: one closed vertex ring (may be non‑convex)."""

    vertices: Tuple[Point2, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("polygon needs at least 3 vertices")

    @classmethod
    def new(cls, vertices: Union[Sequence[Point2], np.ndarray]) -> Polygon2:
        pts = _vertices_to_points2(vertices)
        shell = _ring_coords_from_vertices(pts)
        g = Polygon(shell)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.geom_type != "Polygon":
            raise ValueError("invalid polygon from vertices")
        g = orient_polygon(g, sign=1.0)
        return cls(vertices=_vertices_from_ring(g))

    def to_shapely(self) -> Polygon:
        return _polygon_from_vertices(self.vertices)

    def edges(self) -> Iterator[Tuple[Point2, Point2]]:
        vs = self.vertices
        n = len(vs)
        for i in range(n):
            yield vs[i], vs[(i + 1) % n]

    def is_convex(self) -> bool:
        """True iff boundary is strictly convex (up to collinear edges); expects CCW from ``new``."""
        vs = self.vertices
        n = len(vs)
        if n < 3:
            return False
        sign = 0
        for i in range(n):
            ax = vs[i].x - vs[(i - 1) % n].x
            ay = vs[i].y - vs[(i - 1) % n].y
            bx = vs[(i + 1) % n].x - vs[i].x
            by = vs[(i + 1) % n].y - vs[i].y
            cr = ax * by - ay * bx
            if abs(cr) <= 1e-12:
                continue
            s = 1 if cr > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
        return sign != 0

    def num_vertices(self) -> int:
        return len(self.vertices)


@dataclass(frozen=True)
class ConvexPolygon2(Polygon2):
    """Convex 2D polygon; same storage as :class:`Polygon2`."""

    @classmethod
    def new(cls, vertices: Union[Sequence[Point2], np.ndarray]) -> ConvexPolygon2:
        base = Polygon2.new(vertices)
        if not base.is_convex():
            raise ValueError("polygon is not convex")
        return cls(vertices=base.vertices)
