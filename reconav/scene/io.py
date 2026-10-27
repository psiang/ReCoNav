"""Disk IO helpers for plain text/json and ASG persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reconav.scene.asg import ASG


def _save_json(path: Path, data: Any, *, ensure_ascii: bool = False, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent) + "\n", encoding="utf-8")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_asg(base_dir: Path, asg: ASG) -> tuple[Path, Path, Path]:
    """
    Save ASG into:
    - ``entity.json`` (semantic)
    - ``geometry.json`` (spatial)
    - ``topology.json``
    """
    base = Path(base_dir)
    entity_path = base / "entity.json"
    geometry_path = base / "geometry.json"
    topology_path = base / "topology.json"
    _save_json(entity_path, asg.to_entity_dict())
    _save_json(geometry_path, asg.to_geometry_dict())
    _save_json(topology_path, asg.to_topology_dict())
    return entity_path, geometry_path, topology_path


def load_asg(base_dir: Path, *, data_source: str | None = None) -> ASG:
    """
    Load ASG from:
    - ``entity.json``
    - ``geometry.json``
    - ``topology.json``
    """
    base = Path(base_dir)
    entity_data = _load_json(base / "entity.json")
    geometry_data = _load_json(base / "geometry.json")
    topology_data = _load_json(base / "topology.json")
    if not isinstance(entity_data, dict) or not isinstance(topology_data, dict):
        raise ValueError("entity.json/topology.json must be JSON objects")
    if not isinstance(geometry_data, dict):
        raise ValueError("geometry.json must be a JSON object")
    asg = ASG.from_dicts(entity_data, geometry_data, topology_data)
    # Layout: scenes/<data_source>/<scene_id>/asg. Pass data_source explicitly
    # when loading a scene from a different directory layout.
    inferred = (
        base.parent.parent.name
        if base.name == "asg" and len(base.parents) >= 2
        else None
    )
    asg.data_source = str(data_source or inferred) if (data_source or inferred) else None
    return asg
