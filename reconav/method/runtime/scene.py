"""Load scene ASGs and construct model-facing scene views."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

from shapely.ops import unary_union
from shapely.geometry import mapping

from reconav.scene.asg import ASG
from reconav.scene.io import load_asg
from reconav.scene.vocabulary import default_object_categories, object_confusion_groups, load_semantic_policy, region_confusion_groups, ORDINARY_PORTAL_LABELS
from reconav.method.grounding.label_grounding import GENERIC_KIND_HEADS


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (tuple, list)):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


@dataclass
class SceneContext:
    asg: ASG
    data_source: str
    source_path: str
    fingerprint: str
    _views: dict | None = None

    @classmethod
    def from_asg(cls, asg: ASG, *, data_source: str, source_path: str = "memory"):
        if not data_source:
            raise ValueError("a real ASG source name is required for cache isolation")
        if asg.data_source not in {None, data_source}:
            raise ValueError("ASG source and requested data_source disagree")
        asg.data_source = data_source
        vocabulary = default_object_categories()
        vocabulary_id = digest({key: [value.canonical, sorted(value.mpcat40_all)] for key, value in vocabulary.items()})
        return cls(asg, data_source, source_path, digest({
            "source": data_source, "entity": asg.to_entity_dict(), "vocabulary": vocabulary_id,
            "semantic_policy": load_semantic_policy(),
            "geometry": asg.to_geometry_dict(), "topology": asg.to_topology_dict()}))

    def public_view(self, geometry_mode: str = "compact") -> dict:
        """Return all scene entities in the requested geometry representation.

        Compact is the model-facing view. Bounds uses bounding boxes, and
        footprints serializes polygon rings to six decimal places. The solver
        uses the full-precision ASG in every mode.
        """
        if self._views is None:
            self._views = {}
        if geometry_mode in self._views:
            return self._views[geometry_mode]
        if geometry_mode not in {"bounds", "compact", "footprints"}:
            raise ValueError("unknown scene geometry representation")
        objects = []
        object_rows = (sorted(self.asg.objects, key=lambda value: value.index)
                       if geometry_mode == "compact" else self.asg.objects)
        for obj in object_rows:
            item = {"id": obj.index, "label": obj.label, "region": obj.region_index,
                    "level": obj.level_index}
            parts = obj.navigation_footprint_2d()
            if parts:
                shape = unary_union([p.to_shapely() for p in parts])
                item.update(bounds_xz=list(shape.bounds), center_xz=[shape.centroid.x, shape.centroid.y])
                if geometry_mode == "footprints":
                    item["footprint_xz"] = mapping(shape)
            else:
                item["geometry_missing"] = True
            objects.append(item)
        regions = []
        region_rows = (sorted(self.asg.regions, key=lambda value: value.index)
                       if geometry_mode == "compact" else self.asg.regions)
        for region in region_rows:
            item = {"id": region.index, "label": region.label, "level": region.level,
                    "floor_height": region.get_height()}
            if geometry_mode == "compact":
                if region.polygon_2d:
                    shape = unary_union([p.to_shapely() for p in region.polygon_2d])
                    item.update(bounds_xz=list(shape.bounds), center_xz=[shape.centroid.x, shape.centroid.y])
                else:
                    item["geometry_missing"] = True
            if geometry_mode == "footprints":
                item["polygon_xz"] = mapping(unary_union([p.to_shapely() for p in region.polygon_2d]))
            regions.append(item)
        entity = self.asg.to_entity_dict()
        portals = []
        portal_rows = (sorted(entity.get("portals", []), key=lambda value: value["index"])
                       if geometry_mode == "compact" else entity.get("portals", []))
        for item in portal_rows:
            # Only semantic and relation information from the actual scene.
            if geometry_mode == "compact":
                portal = {"id": item["index"]}
                portal.update({k: item[k] for k in (
                    "label", "region_a", "region_b", "level_a", "level_b",
                    "object_indices", "description", "status") if k in item})
            else:
                portal = {k: v for k, v in item.items() if k in {
                    "index", "label", "region_a", "region_b", "level_a", "level_b",
                    "object_indices", "description", "status"}}
            portals.append(portal)
        # Public label semantics are supplied to both learned baselines, too.
        # This is scene-independent policy restricted to labels present in the
        # Include levels and label denotations for the whole scene.
        present = {obj.label for obj in self.asg.objects}
        denotations = {}
        for head, labels in object_confusion_groups(default_object_categories()).items():
            available = sorted(present.intersection(labels))
            if available and available != [head]:
                denotations[head] = available
        # The topology payload is shared in full, retaining parallel portals.
        metadata = {"representation": "compact-instance-asg-v1"} if geometry_mode == "compact" else {}
        view = _rounded({"scene_id": self.asg.scene_id, "data_source": self.data_source,
                         "geometry_mode": geometry_mode, **metadata,
                         "coordinate_system": "x/z horizontal; y vertical",
                         "objects": objects, "regions": regions, "portals": portals,
                         "levels": entity.get("levels", []), "topology": self.asg.to_topology_dict(),
                         "object_label_denotations": denotations,
                         "generic_kind_heads": {k: sorted(v) for k, v in GENERIC_KIND_HEADS.items()},
                         "generic_kind_label": "any (all instances of the selected kind)",
                         "distance_relation_defaults": {"near": "within 2m in the reference object's room",
                             "nearest_to": "compare across rooms on the reference level",
                             "farthest_from": "compare across rooms on the reference level",
                             "start_reference": "original task START and its level"},
                         "region_label_denotations": {k: sorted(v) for k, v in region_confusion_groups(frozenset(x.label for x in self.asg.regions)).items() if v != {k}},
                         "portal_label_denotations": {"portal": sorted(ORDINARY_PORTAL_LABELS)},
                         "surface_head_aliases": load_semantic_policy().get("surface_head_aliases", {})})
        self._views[geometry_mode] = view
        return view


class DirectorySceneProvider:
    def __init__(self, root: Path, data_source: str):
        self.root = Path(root).resolve()
        self.data_source = data_source
        self._scenes: dict[str, SceneContext] = {}

    def get(self, scene_id: str) -> SceneContext:
        if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
            raise ValueError("scene ID must be a directory basename")
        if scene_id not in self._scenes:
            scene_dir = self.root / scene_id / "asg"
            asg = load_asg(scene_dir, data_source=self.data_source)
            if asg.scene_id != scene_id:
                raise ValueError(f"scene identity mismatch in {scene_dir}")
            self._scenes[scene_id] = SceneContext.from_asg(
                asg, data_source=self.data_source, source_path=str(scene_dir))
        return self._scenes[scene_id]
