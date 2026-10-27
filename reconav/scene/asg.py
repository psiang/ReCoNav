"""ASG core model (source-agnostic) with JSON serialization helpers.

This module contains:
- ASG graph class (`ASG`)
- Entity hierarchy (`ASGEntity` + region/object/portal subclasses)
- Topology edge class (`ASGTopology`)

Geometry operations stay in `reconav.scene.geometry`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from reconav.scene.geometry import ConvexPolygon2, Point2


class TraversalType(str, Enum):
    Passable = "Passable"
    Interactable = "Interactable"
    Obstacle = "Obstacle"
    Structural = "Structural"
    Unknown = "Unknown"

    @classmethod
    def from_json(cls, raw: Any) -> TraversalType:
        if isinstance(raw, cls):
            return raw
        try:
            return cls(str(raw))
        except ValueError:
            return cls.Obstacle


@dataclass(frozen=True)
class ASGEntity:
    index: int
    label: str


@dataclass(frozen=True)
class ASGLevelEntity(ASGEntity):
    description: str = ""
    floor_elevation_mean: float = 0.0
    ceiling_elevation_mean: float = 0.0
    height_mean: float = 0.0


@dataclass(frozen=True)
class ASGRegionEntity(ASGEntity):
    level: int
    height: float
    polygon_2d: List[ConvexPolygon2]
    description: str = ""
    floor_elevation: Optional[float] = None
    # Full boundary rings; polygon_2d holds the convex solver decomposition.
    logical_polygon_2d: List[dict[str, Any]] = field(default_factory=list)

    def get_height(self) -> float:
        """Scene floor height for 3D distance heuristics; uses region ``height`` when present."""
        return float(
            self.floor_elevation
            if self.floor_elevation is not None
            else self.height
        )


@dataclass(frozen=True)
class ASGObjectEntity(ASGEntity):
    region_index: int
    raw_footprint_2d: List[ConvexPolygon2]
    collision_footprint_2d: List[ConvexPolygon2]
    traversal_type: TraversalType = TraversalType.Obstacle
    description: str = ""
    geometry_ref: Optional[str] = None
    level_index: Optional[int] = None
    min_height: Optional[float] = None
    max_height: Optional[float] = None
    body_band_footprint_2d_by_region: dict[int, List[ConvexPolygon2]] = field(
        default_factory=dict
    )

    def navigation_footprint_2d(self) -> List[ConvexPolygon2]:
        """Footprint used for robot approach/navigation: collision band, falling back to raw geometry."""
        return self.collision_footprint_2d or self.raw_footprint_2d

    def get_centroid(self) -> Point2:
        """Mean of raw footprint vertices (xz plane as x-y math)."""
        return self._centroid_of(self.raw_footprint_2d)

    def get_navigation_centroid(self) -> Point2:
        """Mean of navigation footprint vertices (xz plane as x-y math)."""
        return self._centroid_of(self.navigation_footprint_2d())

    def body_footprint_for_region(
        self,
        region_index: int,
    ) -> List[ConvexPolygon2]:
        return self.body_band_footprint_2d_by_region.get(int(region_index), [])

    def _centroid_of(self, footprint: List[ConvexPolygon2]) -> Point2:
        pts: List[Point2] = []
        for poly in footprint:
            pts.extend(poly.vertices)
        if not pts:
            return Point2(0.0, 0.0)
        n = float(len(pts))
        return Point2(sum(p.x for p in pts) / n, sum(p.y for p in pts) / n)


@dataclass(frozen=True)
class ASGPortalEntity(ASGEntity):
    region_a: int
    region_b: int
    object_indices: List[int]
    anchor_a: Point2
    anchor_b: Point2
    status: Optional[str] = None
    clear_width: Optional[float] = None
    description: str = ""
    center: Optional[Point2] = None
    tangent: Optional[Point2] = None
    normal: Optional[Point2] = None
    endpoint_a: Optional[Point2] = None
    endpoint_b: Optional[Point2] = None
    # Manual stair portals may have one measured landing section per side.
    # Existing consumers continue to use endpoint_a/endpoint_b as the
    # representative (bottleneck) clearance section.
    clearance_segments: List[dict[str, Any]] = field(default_factory=list)

    def get_anchor(self, region: int | ASGRegionEntity) -> Point2:
        """Return the anchor on the side of ``region`` (index or entity)."""
        idx = region.index if isinstance(region, ASGRegionEntity) else int(region)
        if idx == self.region_a:
            return self.anchor_a
        if idx == self.region_b:
            return self.anchor_b
        raise ValueError(f"portal {self.index} does not touch region {idx}")

    def get_another_region(self, region_index: int) -> int:
        if region_index == self.region_a:
            return self.region_b
        if region_index == self.region_b:
            return self.region_a
        raise ValueError(f"portal {self.index} does not touch region {region_index}")


@dataclass(frozen=True)
class ASGTopology:
    region_a: int
    region_b: int
    portal_index: int


@dataclass
class ASG:
    scene_id: str
    regions: List[ASGRegionEntity]
    objects: List[ASGObjectEntity]
    portals: List[ASGPortalEntity]
    topology: List[ASGTopology]
    levels: List[ASGLevelEntity] = field(default_factory=list)
    # Runtime provenance used to isolate persistent physical caches.  It is
    # deliberately not serialized as scene semantics.
    data_source: Optional[str] = None
    object_endpoint_clearance_m: Optional[float] = None
    endpoint_region_inset_m: Optional[float] = None
    object_endpoint_reachability_fallback: bool = False

    # ---------- query interfaces ----------
    def get_region(self, region_index: int) -> Optional[ASGRegionEntity]:
        for x in self.regions:
            if x.index == region_index:
                return x
        return None

    def get_level(self, level_index: int) -> Optional[ASGLevelEntity]:
        for x in self.levels:
            if x.index == level_index:
                return x
        return None

    def regions_for_level(self, level_index: int) -> List[ASGRegionEntity]:
        return [region for region in self.regions if region.level == level_index]

    def level_for_object(self, object_index: int) -> Optional[ASGLevelEntity]:
        obj = self.get_object(object_index)
        region = self.get_region(obj.region_index) if obj is not None else None
        return self.get_level(region.level) if region is not None else None

    def get_object(self, object_index: int) -> Optional[ASGObjectEntity]:
        for x in self.objects:
            if x.index == object_index:
                return x
        return None

    def get_portal(self, portal_index: int) -> Optional[ASGPortalEntity]:
        for x in self.portals:
            if x.index == portal_index:
                return x
        return None

    def topology_for_region(self, region_index: int) -> List[ASGTopology]:
        return [e for e in self.topology if e.region_a == region_index or e.region_b == region_index]

    # ---------- serialization ----------
    def to_entity_dict(self) -> dict[str, Any]:
        """Semantic attributes only (for LLM / metadata)."""
        region_levels = {region.index: region.level for region in self.regions}
        return {
            "scene_id": self.scene_id,
            "levels": [
                {
                    "index": level.index,
                    "label": level.label,
                    "description": level.description,
                    "floor_elevation_mean": level.floor_elevation_mean,
                    "ceiling_elevation_mean": level.ceiling_elevation_mean,
                    "height_mean": level.height_mean,
                }
                for level in self.levels
            ],
            "regions": [
                {
                    "index": r.index,
                    "label": r.label,
                    "description": r.description,
                    "level": r.level,
                }
                for r in self.regions
            ],
            "objects": [
                {
                    "index": o.index,
                    "region_index": o.region_index,
                    "level": (
                        o.level_index
                        if o.level_index is not None
                        else region_levels.get(o.region_index)
                    ),
                    "label": o.label,
                    "description": o.description,
                    "traversal_type": o.traversal_type.value,
                    "geometry_ref": o.geometry_ref,
                }
                for o in self.objects
            ],
            "portals": [
                {
                    "index": p.index,
                    "label": p.label,
                    "description": p.description,
                    "region_a": p.region_a,
                    "region_b": p.region_b,
                    "level_a": region_levels.get(p.region_a),
                    "level_b": region_levels.get(p.region_b),
                    "object_indices": list(p.object_indices),
                    "status": p.status,
                }
                for p in self.portals
            ],
        }

    def to_geometry_dict(self) -> dict[str, Any]:
        """Geometric attributes only (for Z3 / spatial reasoning)."""
        policy = {key: value for key, value in (
            ("object_endpoint_clearance_m", self.object_endpoint_clearance_m),
            ("endpoint_region_inset_m", self.endpoint_region_inset_m),
        ) if value is not None}
        if self.object_endpoint_reachability_fallback:
            policy["object_endpoint_reachability_fallback"] = True
        return {
            "scene_id": self.scene_id,
            **({"navigation_policy": policy} if policy else {}),
            "regions": [
                {
                    "index": r.index,
                    "height": r.height,
                    "floor_elevation": r.get_height(),
                    "polygon_2d": _polygon_2d_to_json(r.polygon_2d),
                    **(
                        {"logical_polygon_2d": r.logical_polygon_2d}
                        if r.logical_polygon_2d
                        else {}
                    ),
                }
                for r in self.regions
            ],
            "objects": [
                {
                    "index": o.index,
                    "raw_footprint_2d": _polygon_2d_to_json(o.raw_footprint_2d),
                    "collision_footprint_2d": _polygon_2d_to_json(
                        o.collision_footprint_2d
                    ),
                    "body_band_footprint_2d_by_region": {
                        str(region_index): _polygon_2d_to_json(polygons)
                        for region_index, polygons in sorted(
                            o.body_band_footprint_2d_by_region.items()
                        )
                    },
                    "min_height": o.min_height,
                    "max_height": o.max_height,
                }
                for o in self.objects
            ],
            "portals": [
                {
                    "index": p.index,
                    "anchor_a": [float(p.anchor_a.x), float(p.anchor_a.y)],
                    "anchor_b": [float(p.anchor_b.x), float(p.anchor_b.y)],
                    "clear_width": p.clear_width,
                    "center": _point_to_json(p.center),
                    "tangent": _point_to_json(p.tangent),
                    "normal": _point_to_json(p.normal),
                    "endpoint_a": _point_to_json(p.endpoint_a),
                    "endpoint_b": _point_to_json(p.endpoint_b),
                    **(
                        {"clearance_segments": p.clearance_segments}
                        if p.clearance_segments
                        else {}
                    ),
                }
                for p in self.portals
            ],
        }

    def to_topology_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "topology": [
                {
                    "region_a": e.region_a,
                    "region_b": e.region_b,
                    "portal_index": e.portal_index,
                }
                for e in self.topology
            ],
        }

    @classmethod
    def from_dicts(
        cls,
        entity_data: dict[str, Any],
        geometry_data: dict[str, Any],
        topology_data: dict[str, Any],
    ) -> ASG:
        sid = str(entity_data.get("scene_id", geometry_data.get("scene_id", topology_data.get("scene_id", ""))))
        if not sid:
            raise ValueError("missing scene_id in entity/geometry/topology data")

        region_geom = {int(g["index"]): g for g in geometry_data.get("regions", []) if "index" in g}
        object_geom = {int(g["index"]): g for g in geometry_data.get("objects", []) if "index" in g}
        portal_geom = {int(g["index"]): g for g in geometry_data.get("portals", []) if "index" in g}

        levels = [
            ASGLevelEntity(
                index=int(level["index"]),
                label=str(level.get("label", "")),
                description=str(level.get("description", "")),
                floor_elevation_mean=float(
                    level.get("floor_elevation_mean", level.get("floor_elevation_y", 0.0))
                ),
                ceiling_elevation_mean=float(
                    level.get("ceiling_elevation_mean", level.get("ceiling_elevation_y", 0.0))
                ),
                height_mean=float(level.get("height_mean", 0.0)),
            )
            for level in entity_data.get("levels", [])
        ]

        regions = [
            ASGRegionEntity(
                index=int(r["index"]),
                label=str(r.get("label", "")),
                description=str(r.get("description", "")),
                level=int(r.get("level", 0)),
                height=float(region_geom.get(int(r["index"]), {}).get("height", 0.0)),
                floor_elevation=(
                    float(region_geom.get(int(r["index"]), {})["floor_elevation"])
                    if "floor_elevation" in region_geom.get(int(r["index"]), {})
                    else None
                ),
                polygon_2d=_polygon_2d_from_json(
                    region_geom.get(int(r["index"]), {}).get("polygon_2d", [])
                ),
                logical_polygon_2d=list(
                    region_geom.get(int(r["index"]), {}).get("logical_polygon_2d", [])
                    or []
                ),
            )
            for r in entity_data.get("regions", [])
        ]
        objects = [
            ASGObjectEntity(
                index=int(o["index"]),
                region_index=int(o["region_index"]),
                label=str(o.get("label", "")),
                raw_footprint_2d=_polygon_2d_from_json(
                    object_geom.get(int(o["index"]), {}).get("raw_footprint_2d", [])
                ),
                collision_footprint_2d=_polygon_2d_from_json(
                    object_geom.get(int(o["index"]), {}).get("collision_footprint_2d", [])
                ),
                traversal_type=TraversalType.from_json(o.get("traversal_type")),
                description=str(o.get("description", "")),
                geometry_ref=(
                    None
                    if o.get("geometry_ref") in (None, "", "None")
                    else str(o.get("geometry_ref"))
                ),
                level_index=(
                    None if o.get("level") is None else int(o.get("level"))
                ),
                min_height=(
                    None
                    if object_geom.get(int(o["index"]), {}).get("min_height") is None
                    else float(object_geom[int(o["index"])]["min_height"])
                ),
                max_height=(
                    None
                    if object_geom.get(int(o["index"]), {}).get("max_height") is None
                    else float(object_geom[int(o["index"])]["max_height"])
                ),
                body_band_footprint_2d_by_region={
                    int(region_index): _polygon_2d_from_json(polygons)
                    for region_index, polygons in (
                        object_geom.get(int(o["index"]), {})
                        .get("body_band_footprint_2d_by_region", {})
                        .items()
                    )
                },
            )
            for o in entity_data.get("objects", [])
        ]

        portals: List[ASGPortalEntity] = []
        for p in entity_data.get("portals", []):
            idx = int(p["index"])
            geom = portal_geom.get(idx, {})
            aa = geom.get("anchor_a") or [0.0, 0.0]
            bb = geom.get("anchor_b") or [0.0, 0.0]
            raw_indices = p.get("object_indices", [])
            if not isinstance(raw_indices, list):
                raw_indices = []
            clear_width = geom.get("clear_width")
            portals.append(
                ASGPortalEntity(
                    index=idx,
                    label=str(p.get("label", "")),
                    region_a=int(p["region_a"]),
                    region_b=int(p["region_b"]),
                    object_indices=[int(x) for x in raw_indices if x not in (None, "", "None")],
                    anchor_a=Point2(float(aa[0]), float(aa[1])),
                    anchor_b=Point2(float(bb[0]), float(bb[1])),
                    status=None if p.get("status") in (None, "", "None") else str(p.get("status")),
                    clear_width=None if clear_width is None else float(clear_width),
                    description=str(p.get("description", "")),
                    center=_point_from_json(geom.get("center")),
                    tangent=_point_from_json(geom.get("tangent")),
                    normal=_point_from_json(geom.get("normal")),
                    endpoint_a=_point_from_json(geom.get("endpoint_a")),
                    endpoint_b=_point_from_json(geom.get("endpoint_b")),
                    clearance_segments=list(geom.get("clearance_segments", []) or []),
                )
            )

        topology = [
            ASGTopology(
                region_a=int(e["region_a"]),
                region_b=int(e["region_b"]),
                portal_index=int(e["portal_index"]),
            )
            for e in topology_data.get("topology", [])
        ]
        return cls(
            scene_id=sid,
            regions=regions,
            objects=objects,
            portals=portals,
            topology=topology,
            levels=levels,
            object_endpoint_clearance_m=geometry_data.get("navigation_policy", {}).get("object_endpoint_clearance_m"),
            endpoint_region_inset_m=geometry_data.get("navigation_policy", {}).get("endpoint_region_inset_m"),
            object_endpoint_reachability_fallback=bool(geometry_data.get("navigation_policy", {}).get("object_endpoint_reachability_fallback", False)),
        )


# ---------- JSON wire helpers (private) ----------


def _polygon_2d_from_json(raw: Any) -> List[ConvexPolygon2]:
    """Parse `[[[x,y],...], ...]` into `List[ConvexPolygon2]`."""
    if not isinstance(raw, list):
        return []
    out: List[ConvexPolygon2] = []
    for ring in raw:
        if not isinstance(ring, list) or len(ring) < 3:
            continue
        out.append(
            ConvexPolygon2.new(
                [Point2(float(p[0]), float(p[1])) for p in ring if isinstance(p, (list, tuple)) and len(p) >= 2]
            )
        )
    return out


def _polygon_2d_to_json(polys: Iterable[ConvexPolygon2]) -> List[List[List[float]]]:
    return [[[float(p.x), float(p.y)] for p in poly.vertices] for poly in polys]


def _point_to_json(point: Point2 | None) -> list[float] | None:
    return None if point is None else [float(point.x), float(point.y)]


def _point_from_json(raw: Any) -> Point2 | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    return Point2(float(raw[0]), float(raw[1]))
