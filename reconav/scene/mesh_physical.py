"""Region envelopes and body-band occupancy derived from complete scene mesh.

Region polygons define feasible-space bounds.  Floor mesh is used only to
estimate each region's local height.  Complete scene-mesh triangles intersecting
the robot body-height band define collision occupancy; object labels do not.
"""
from __future__ import annotations
from reconav import paths as release_paths
import hashlib
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import cv2
import numpy as np
from shapely.geometry import GeometryCollection, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from reconav.scene.asg import ASG
DEFAULT_NAV_ROOT = Path(str(release_paths.benchmark_path('scenes')))

@dataclass(frozen=True)
class MeshProjectionConfig:
    raster_resolution_m: float = 0.02
    floor_hint_max_delta_m: float = 0.2
    floor_window_half_width_m: float = 0.04
    height_patch_span_m: float = 0.08
    horizontal_normal_min: float = 0.9
    minimum_floor_evidence_area_m2: float = 0.01

    @property
    def raster_half_diagonal_m(self) -> float:
        return self.raster_resolution_m / math.sqrt(2.0)

@dataclass(frozen=True)
class MeshSource:
    link_path: str
    resolved_path: str
    sha256: str
    byte_size: int
    transform: str
    triangle_count: int

@dataclass(frozen=True)
class MeshFloorEvidence:
    """Auditable local floor-height evidence selected from the scene mesh."""
    region_id: int
    level_id: int
    level_floor_hint_m: float
    annotation_floor_elevation_m: float
    resolved_floor_elevation_m: float
    selected_face_indices: tuple[int, ...]
    selected_face_count: int
    selected_overlap_area_m2: float
    weighted_spread_m: float
    rejected_candidate_area_fraction: float
    candidate_height_range_m: tuple[float, float]
    selection: str = 'maximum overlap-area fixed height window within level hint'

@dataclass(frozen=True)
class HeightPatchSpec:
    patch_id: int
    region_ids: tuple[int, ...]
    floor_elevations_m: tuple[float, ...]
    body_band_bottom_m: float
    body_band_top_m: float

    @property
    def reference_height_m(self) -> float:
        return float(np.median(np.asarray(self.floor_elevations_m, dtype=np.float64)))

@dataclass
class MeshHeightPatch:
    spec: HeightPatchSpec
    support: BaseGeometry
    center_support: BaseGeometry
    raw_occupancy: BaseGeometry
    inflated_occupancy: BaseGeometry
    free_space: BaseGeometry
    support_face_indices: tuple[int, ...]
    occupancy_face_indices: tuple[int, ...]
    metrics: dict[str, object]
_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}

def locate_scene_mesh(asg: ASG, *, nav_root: Path=DEFAULT_NAV_ROOT) -> Path:
    """Resolve the immutable viewer mesh for the ASG's actual data source."""
    if not asg.data_source:
        raise ValueError('ASG data_source is required to locate the scene mesh')
    candidate = Path(os.environ.get('RECONAV_SCENE_ROOT', str(nav_root))) / str(asg.data_source) / str(asg.scene_id) / 'viewer/geometry.glb'
    if not candidate.is_file():
        raise FileNotFoundError(f'scene mesh is missing for {asg.data_source}/{asg.scene_id}: {candidate}')
    return candidate

def scene_mesh_source(path: Path, triangles: np.ndarray) -> MeshSource:
    path = Path(path)
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    digest = mesh_file_sha256(resolved)
    return MeshSource(link_path=str(path), resolved_path=str(resolved), sha256=digest, byte_size=int(stat.st_size), transform='identity (viewer GLB POSITION coordinates are world-space)', triangle_count=int(len(triangles)))

def mesh_file_sha256(path: Path) -> str:
    """Hash an immutable mesh once per path/stat identity, not once per endpoint."""
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _DIGEST_CACHE.get(cache_key)
    if digest is None:
        sha = hashlib.sha256()
        with resolved.open('rb') as stream:
            while (chunk := stream.read(1024 * 1024)):
                sha.update(chunk)
        digest = sha.hexdigest()
        _DIGEST_CACHE[cache_key] = digest
    return digest

def estimate_mesh_floor_elevations(asg: ASG, triangles: np.ndarray, *, config: MeshProjectionConfig) -> dict[int, MeshFloorEvidence]:
    """Resolve each semantic region's local floor from real horizontal faces.

    The level floor is a bounded hint, not the answer.  A fixed-width window is
    used instead of transitive height clustering: a staircase of furniture
    tops can therefore no longer merge the floor peak into a table/ceiling
    peak.  Region outlines select local height evidence here; the caller
    applies the annotation separately as the exact feasible-space envelope.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError('mesh triangles must have shape (N, 3, 3)')
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    norms = np.linalg.norm(normals, axis=1)
    horizontal = np.abs(normals[:, 1]) / np.maximum(norms, 1e-12)
    projected_area = 0.5 * np.abs(normals[:, 1])
    mean_y = triangles[:, :, 1].mean(axis=1)
    candidate_indices = np.flatnonzero((horizontal >= config.horizontal_normal_min) & (projected_area > 1e-09))
    triangle_min_x = triangles[:, :, 0].min(axis=1)
    triangle_max_x = triangles[:, :, 0].max(axis=1)
    triangle_min_z = triangles[:, :, 2].min(axis=1)
    triangle_max_z = triangles[:, :, 2].max(axis=1)
    evidence: dict[int, MeshFloorEvidence] = {}
    for region in sorted(asg.regions, key=lambda value: value.index):
        level = asg.get_level(region.level)
        if level is None:
            raise ValueError(f'region {region.index} references missing level {region.level}')
        hint = float(level.floor_elevation_mean)
        shape = _union((polygon.to_shapely() for polygon in region.polygon_2d))
        min_x, min_z, max_x, max_z = shape.bounds
        local_indices = candidate_indices[(np.abs(mean_y[candidate_indices] - hint) <= config.floor_hint_max_delta_m) & (triangle_max_x[candidate_indices] >= min_x) & (triangle_min_x[candidate_indices] <= max_x) & (triangle_max_z[candidate_indices] >= min_z) & (triangle_min_z[candidate_indices] <= max_z)]
        samples: list[tuple[float, float, int]] = []
        for face_index in local_indices:
            projected = Polygon([(float(vertex[0]), float(vertex[2])) for vertex in triangles[int(face_index)]])
            if projected.area <= 1e-09:
                continue
            overlap = float(projected.intersection(shape).area)
            if overlap > 1e-07:
                samples.append((float(mean_y[int(face_index)]), overlap, int(face_index)))
        if not samples:
            raise ValueError(f'region {region.index} has no horizontal mesh floor evidence within {config.floor_hint_max_delta_m:.3f} m of level hint {hint:.3f}')
        heights = np.asarray([item[0] for item in samples], dtype=np.float64)
        weights = np.asarray([item[1] for item in samples], dtype=np.float64)
        order = np.argsort(heights)
        ordered_heights = heights[order]
        ordered_weights = weights[order]
        prefix = np.concatenate(([0.0], np.cumsum(ordered_weights)))
        half_width = float(config.floor_window_half_width_m)
        best: tuple[float, float, int] | None = None
        for center in np.unique(ordered_heights):
            left = int(np.searchsorted(ordered_heights, center - half_width, side='left'))
            right = int(np.searchsorted(ordered_heights, center + half_width, side='right'))
            score = float(prefix[right] - prefix[left])
            rank = (-score, abs(float(center) - hint), float(center))
            if best is None or rank < best:
                best = rank
        assert best is not None
        center = float(best[2])
        selected_mask = np.abs(heights - center) <= half_width + 1e-12
        selected = [sample for sample, keep in zip(samples, selected_mask) if keep]
        selected_values = heights[selected_mask]
        selected_weights = weights[selected_mask]
        selected_area = float(selected_weights.sum())
        if selected_area < config.minimum_floor_evidence_area_m2:
            raise ValueError(f'region {region.index} local floor evidence area {selected_area:.4f} m2 is insufficient')
        resolved = _weighted_median(selected_values, selected_weights)
        spread = _weighted_median(np.abs(selected_values - resolved), selected_weights)
        total_area = float(weights.sum())
        evidence[int(region.index)] = MeshFloorEvidence(region_id=int(region.index), level_id=int(region.level), level_floor_hint_m=hint, annotation_floor_elevation_m=float(region.get_height()), resolved_floor_elevation_m=resolved, selected_face_indices=tuple(sorted((item[2] for item in selected))), selected_face_count=len(selected), selected_overlap_area_m2=selected_area, weighted_spread_m=spread, rejected_candidate_area_fraction=float(max(0.0, 1.0 - selected_area / total_area)), candidate_height_range_m=(float(heights.min()), float(heights.max())))
    return evidence

def height_patch_specs(asg: ASG, *, bottom_offset_m: float, robot_height_m: float, config: MeshProjectionConfig, region_floor_elevations: dict[int, float] | None=None) -> tuple[HeightPatchSpec, ...]:
    """Cluster only numerically compatible local floor elevations.

    Clustering avoids rebuilding effectively identical projections for noisy
    per-room floor estimates.  The occupancy band conservatively spans every
    local height in the patch, and larger height changes remain separate.
    """
    resolved = region_floor_elevations or {int(region.index): float(region.get_height()) for region in asg.regions}
    ordered = sorted(((float(resolved[int(region.index)]), int(region.index)) for region in asg.regions), key=lambda item: (item[0], item[1]))
    clusters: list[list[tuple[float, int]]] = []
    for height, region_id in ordered:
        if not clusters or height - min((value for value, _region in clusters[-1])) > config.height_patch_span_m:
            clusters.append([])
        clusters[-1].append((height, region_id))
    return tuple((HeightPatchSpec(patch_id=index, region_ids=tuple((region_id for _height, region_id in cluster)), floor_elevations_m=tuple((height for height, _region_id in cluster)), body_band_bottom_m=min((height for height, _region_id in cluster)) + float(bottom_offset_m), body_band_top_m=max((height for height, _region_id in cluster)) + float(robot_height_m)) for index, cluster in enumerate(clusters)))

def build_mesh_height_patches(asg: ASG, triangles: np.ndarray, specs: Sequence[HeightPatchSpec], *, floor_evidence: dict[int, MeshFloorEvidence], robot_radius_m: float, conservative_buffer_radius_m: float, round_quad_segs: int, config: MeshProjectionConfig) -> dict[int, MeshHeightPatch]:
    """Build region-envelope support and body occupancy per height patch."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError('mesh triangles must have shape (N, 3, 3)')
    grid = _Grid.for_triangles(triangles, config.raster_resolution_m)
    min_y = triangles[:, :, 1].min(axis=1)
    max_y = triangles[:, :, 1].max(axis=1)
    region_shapes = {int(region.index): _union((polygon.to_shapely() for polygon in region.polygon_2d)) for region in asg.regions}
    result: dict[int, MeshHeightPatch] = {}
    for spec in specs:
        support = _union((region_shapes[region_id] for region_id in spec.region_ids))
        support_indices = tuple(sorted({face_index for region_id in spec.region_ids for face_index in floor_evidence[region_id].selected_face_indices}))
        occupancy_candidate = (max_y >= spec.body_band_bottom_m) & (min_y <= spec.body_band_top_m)
        occupancy_indices = np.flatnonzero(occupancy_candidate)
        occupancy_mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
        _paint_projected_triangles(occupancy_mask, triangles[occupancy_indices], grid, clip_bottom=spec.body_band_bottom_m, clip_top=spec.body_band_top_m)
        raw_occupancy = _mask_geometry(occupancy_mask, grid)
        raster_error = config.raster_half_diagonal_m
        center_support = support
        inflated_occupancy = _clean(raw_occupancy.buffer(conservative_buffer_radius_m + raster_error, quad_segs=round_quad_segs, join_style=1, cap_style=1))
        free = _clean(center_support.difference(inflated_occupancy))
        metrics: dict[str, object] = {'patch_id': spec.patch_id, 'region_ids': list(spec.region_ids), 'reference_height_m': spec.reference_height_m, 'floor_elevations_m': list(spec.floor_elevations_m), 'body_band_m': [spec.body_band_bottom_m, spec.body_band_top_m], 'floor_evidence_face_count': int(len(support_indices)), 'occupancy_face_count': int(len(occupancy_indices)), 'support_area_m2': float(support.area), 'center_support_area_m2': float(center_support.area), 'raw_occupancy_area_m2': float(raw_occupancy.area), 'inflated_occupancy_area_m2': float(inflated_occupancy.area), 'free_area_m2': float(free.area), 'free_component_count': len(_polygons(free)), 'raster_resolution_m': config.raster_resolution_m, 'raster_half_diagonal_m': raster_error, 'nominal_robot_radius_m': float(robot_radius_m), 'region_boundary_center_inset_m': 0.0, 'effective_occupancy_buffer_m': float(conservative_buffer_radius_m + raster_error), 'support_selection': 'union of annotated/generated region polygons on the height sheet; floor mesh is height evidence only', 'occupancy_selection': 'all complete-mesh triangles clipped to the patch body band; no face2obj or traversal filter'}
        result[spec.patch_id] = MeshHeightPatch(spec=spec, support=support, center_support=center_support, raw_occupancy=raw_occupancy, inflated_occupancy=inflated_occupancy, free_space=free, support_face_indices=support_indices, occupancy_face_indices=tuple((int(value) for value in occupancy_indices)), metrics=metrics)
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    return float(values[order[index]])

@dataclass(frozen=True)
class _Grid:
    min_x: float
    max_z: float
    resolution: float
    width: int
    height: int

    @classmethod
    def for_triangles(cls, triangles: np.ndarray, resolution: float) -> _Grid:
        projected = triangles[:, :, [0, 2]]
        minimum = projected.min(axis=(0, 1)).astype(float)
        maximum = projected.max(axis=(0, 1)).astype(float)
        padding = max(0.5, 3.0 * resolution)
        min_x = math.floor((minimum[0] - padding) / resolution) * resolution
        max_x = math.ceil((maximum[0] + padding) / resolution) * resolution
        min_z = math.floor((minimum[1] - padding) / resolution) * resolution
        max_z = math.ceil((maximum[1] + padding) / resolution) * resolution
        return cls(min_x=min_x, max_z=max_z, resolution=resolution, width=int(round((max_x - min_x) / resolution)) + 1, height=int(round((max_z - min_z) / resolution)) + 1)

    def pixel(self, x: float, z: float) -> tuple[int, int]:
        return (int(round((float(x) - self.min_x) / self.resolution)), int(round((self.max_z - float(z)) / self.resolution)))

    def world(self, column: float, row: float) -> tuple[float, float]:
        return (self.min_x + float(column) * self.resolution, self.max_z - float(row) * self.resolution)

def _paint_projected_triangles(mask: np.ndarray, triangles: np.ndarray, grid: _Grid, *, clip_bottom: float | None, clip_top: float | None) -> None:
    polygons: list[np.ndarray] = []
    for triangle in triangles:
        vertices = [np.asarray(vertex, dtype=np.float64) for vertex in triangle]
        if clip_bottom is not None:
            vertices = _clip_y(vertices, clip_bottom, keep_above=True)
        if clip_top is not None and vertices:
            vertices = _clip_y(vertices, clip_top, keep_above=False)
        if not vertices:
            continue
        pixels = np.asarray([grid.pixel(vertex[0], vertex[2]) for vertex in vertices], dtype=np.int32)
        polygons.append(pixels.reshape((-1, 1, 2)))
    if not polygons:
        return
    areas = np.asarray([abs(float(cv2.contourArea(polygon))) for polygon in polygons], dtype=np.float64)
    area_polygons = [polygon for polygon, area in zip(polygons, areas) if area >= 0.5]
    line_polygons = [polygon for polygon, area in zip(polygons, areas) if area < 0.5]
    if area_polygons:
        cv2.fillPoly(mask, area_polygons, 255, lineType=cv2.LINE_8)
    if line_polygons:
        cv2.polylines(mask, line_polygons, True, 255, thickness=1, lineType=cv2.LINE_8)

def _clip_y(vertices: Sequence[np.ndarray], boundary: float, *, keep_above: bool) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    if not vertices:
        return output

    def inside(vertex: np.ndarray) -> bool:
        return bool(vertex[1] >= boundary) if keep_above else bool(vertex[1] <= boundary)
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = float(current[1] - previous[1])
            if abs(delta) > 1e-12:
                fraction = (boundary - float(previous[1])) / delta
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output

def _mask_geometry(mask: np.ndarray, grid: _Grid) -> BaseGeometry:
    contours, hierarchy = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return GeometryCollection()
    hierarchy = hierarchy[0]
    polygons: list[Polygon] = []
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1 or len(contour) < 3:
            continue
        exterior = [grid.world(column, row) for column, row in contour[:, 0, :]]
        holes: list[list[tuple[float, float]]] = []
        child = int(hierarchy[index][2])
        while child != -1:
            hole_contour = contours[child]
            if len(hole_contour) >= 3:
                holes.append([grid.world(column, row) for column, row in hole_contour[:, 0, :]])
            child = int(hierarchy[child][0])
        polygon = Polygon(exterior, holes)
        if not polygon.is_empty and polygon.area > 1e-08:
            polygons.extend(_polygons(_clean(polygon)))
    return _union(polygons)

def _union(values: Iterable[BaseGeometry]) -> BaseGeometry:
    geometries = [value for value in values if value is not None and (not value.is_empty)]
    return _clean(unary_union(geometries)) if geometries else GeometryCollection()

def _clean(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return geometry
    return geometry if geometry.is_valid else geometry.buffer(0)

def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return [value for value in getattr(geometry, 'geoms', ()) if isinstance(value, Polygon) and (not value.is_empty)]
